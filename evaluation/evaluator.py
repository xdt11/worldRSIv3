"""Pinned 4DCodeBench jobs and artifact adaptation; no reimplemented metrics.

Scoring is a separate Docker job with private references. Inference sees only
the selected video and public checker. Newton/Warp is an explicit renderer variant.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import uuid
import runpy
import time
from fractions import Fraction
from functools import lru_cache
from harness.sandbox.docker import gpu_argument

def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def inside(root, path):
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Path escapes its scope')
    return target

REVISION = '6c466be34107d7f388f30f97bf059b68ba1a21b0'
PROTOCOL = '4dcodebench-newton-v1'
SPLITS = {'smoke', 'evolution', 'validation', 'held-out'}
STATES = {'not_applicable', 'excluded', 'error', 'uncomputable'}


def source_identity(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != REVISION:
        raise ValueError('4DCodeBench source must match the pinned commit')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('Pinned 4DCodeBench source has tracked modifications')
    return {'commit': commit, 'protocol': PROTOCOL, 'renderer': 'Python Warp Render', 'official_cycles_render': False}


def video_contract(video):
    info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True, timeout=120)
    stream = json.loads(info.stdout)['streams'][0]
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'fps': stream['r_frame_rate'], 'frame_count': int(stream['nb_read_frames'])}


@lru_cache(maxsize=4096)
def _resource_hash(path, size, modified, changed):
    return sha(path)


def scoring_identity(config, case):
    """Freeze only resources used by the selected official metric profile."""
    root = Path(config['root']).resolve()
    types = config.get('types')
    types = OFFICIAL_TYPES if types in (None, [], ['all']) else types
    source_identity(config['source'])
    active = _applicable_types(config, case)
    trees = {}
    if config.get('runtime', 'cpu') == 'cuda':
        trees['runtime_manifest'] = root.parent / 'scorer/identity.json'
        trees['base_runtime_manifest'] = root.parent / 'identity.json'
    if case.startswith('synthetic/') and set(active) & {'scene_3d', 'trajectory', 'dynamics', 'occupancy', 'uni3d', 'dynamic_iou'}:
        trees['reference/world'] = inside(root / 'data', case + '/world')
    if case.startswith('real/') and 'dynamic_iou' in active:
        trees['reference/annotation'] = inside(root / 'data', case + '/annotation')
    for metric in active:
        if case.startswith('synthetic/') and metric in {'flow', 'track2d', 'depth'}:
            continue
        for name in ESTIMATES.get(metric, ()):
            if metric == 'uni3d' and case.startswith('synthetic/'):
                continue
            trees['reference/estimates/' + name] = inside(root / 'data', case + '/estimates/' + name)
        for name in CHECKPOINTS.get(metric, ()):
            directory = name.split('/')[0]
            trees['checkpoints/' + directory] = root / 'checkpoints' / directory
    if config.get('python_packages'):
        trees['dependencies'] = Path(config['python_packages']).resolve()
    files = {}
    for label, tree in trees.items():
        paths = [tree] if tree.is_file() else sorted(tree.rglob('*'))
        for path in paths:
            if '__pycache__' in path.parts or path.suffix in {'.pyc', '.partial'}:
                continue
            if path.is_symlink():
                raise ValueError('Scorer resources cannot contain symlinks')
            if path.is_file():
                stat = path.stat()
                name = label if tree.is_file() else label + '/' + str(path.relative_to(tree))
                files[name] = (sha(path) if stat.st_size < 8 * 1024**2 else
                    _resource_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return {'source': source_identity(config['source']), 'runtime': config.get('runtime', 'cpu'), 'types': list(types), 'files': files,
            'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def metrics_from_reward(reward, detail):
    """Preserve every official reading and cause; do not silently drop failures."""
    result = {}
    for key, value in reward.items():
        if type(value) in (int, float) and math.isfinite(value):
            row = {'status': 'ok', 'value': value}
        elif isinstance(value, str) and value in STATES:
            row = {'status': value, 'value': None, 'reason': detail.get('causes', {}).get(key)}
        else:
            raise ValueError(f'Unrecognized official reward reading: {key}={value!r}')
        result['4dcodebench/' + key] = {**row, 'protocol': PROTOCOL, 'source': 'reward.json', 'raw': value}
    return result



def _scorer_environment(config, resources):
    """One explicit CPU or workspace-owned CUDA interpreter for scoring jobs."""
    resources = Path(resources)
    runtime = config.get('runtime', 'cpu')
    if runtime == 'cpu':
        mounts = [(resources / 'perception', '/opt/resources/perception', True)]
        interpreter = '/opt/resources/perception/python/bin/python3.12'
        pythonpath = '/opt/scorer-packages:/opt/resources/perception/.venv/lib/python3.12/site-packages:/benchmark'
    elif runtime == 'cuda':
        mounts = [(resources / 'perception', '/opt/resources/perception', True)]
        toolchain = resources / 'scorer/toolchain'
        if toolchain.is_dir():
            mounts.append((toolchain, '/opt/scorer-toolchain', True))
        interpreter = '/opt/resources/perception/python/bin/python3.12'
        pythonpath = '/opt/scorer-packages:/opt/resources/perception/.venv/lib/python3.12/site-packages:/benchmark'
    else:
        raise ValueError('Scorer runtime must be cpu or cuda')
    if config.get('python_packages'):
        packages = Path(config['python_packages']).resolve()
        if not packages.is_dir():
            raise ValueError('Configured scorer dependency overlay is missing')
        mounts.append((packages, '/opt/scorer-packages', True))
    return interpreter, pythonpath, mounts

class FourDCodeBench:
    def __init__(self, config, resources):
        self.config = config
        self.resources = Path(resources).resolve()
        self.source = Path(config['source']).resolve()
        self.root = Path(config['root']).resolve()
        self.identity = source_identity(self.source)

    def job(self, case, workspace, destination, *, stage='score', types=None, checkpoint=None, reference=None):
        if stage not in {'prepare', 'score', 'viz'}:
            raise ValueError('Stage must be prepare, score or viz')
        case_path = PurePosixPath(case)
        if len(case_path.parts) != 2 or case_path.parts[0] not in {'real', 'synthetic'} or '..' in case_path.parts:
            raise ValueError('Invalid case path')
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        workspace = Path(workspace).resolve()
        case_root = Path(reference['case_root']).resolve() if reference else inside(self.root / 'cases', case)
        if not (case_root / 'reference.mp4').is_file():
            raise ValueError('Case reference.mp4 is missing')
        private = Path(reference['data_root']).resolve() if reference else inside(self.root / 'data', case)
        private.mkdir(parents=True, exist_ok=True)
        checkpoints = self.root / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        manifest = [{'case': case} if stage == 'prepare' else
                    {'case': case, 'world': 'runs/job/workspace/world', 'out': 'runs/job'}]
        save(destination / 'manifest.json', manifest)
        interpreter, pythonpath, runtime_mounts = _scorer_environment(self.config, self.resources)
        name = 'worldrsi-score-' + uuid.uuid4().hex
        args = ['docker', 'run', '--rm', '--pull=never', '--name', name, '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
            '--cpus', str(self.config.get('cpus', 4)), '--memory', self.config.get('memory', '12g'),
            '--pids-limit', '256', '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g,mode=1777',
            '--env', 'HOME=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'HF_HUB_OFFLINE=1',
            '--env', 'TRANSFORMERS_OFFLINE=1', '--env', 'PYTHONPATH=' + pythonpath, '--workdir', '/benchmark']
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'OPENCV_FOR_THREADS_NUM', 'CV_NUM_THREADS'):
            args += ['--env', variable + '=' + str(max(1, int(float(self.config.get('cpus', 4)))))]
        if self.config.get('runtime') == 'cuda':
            args += ['--tmpfs', '/runtime-cache:rw,nosuid,nodev,exec,size=256m,mode=1777',
                '--env', 'CC=/opt/scorer-toolchain/cc', '--env', 'TRITON_CACHE_DIR=/runtime-cache/triton']
        mounts = [*runtime_mounts, (self.source, '/benchmark', True),
            (case_root, '/cases/' + case, True), (private, '/data/' + case, stage != 'prepare'),
            (checkpoints, '/checkpoints', True), (destination, '/runs/job', False)]
        if stage != 'prepare':
            for part in ('world', 'solution'):
                path = workspace / part
                if path.is_dir():
                    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
                        raise ValueError('Scoring artifacts cannot contain symlinks')
                    mounts.append((path, '/runs/job/workspace/' + part, True))
        for src, dst, readonly in mounts:
            args += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
        if self.config.get('gpus') is not None:
            args += ['--gpus', gpu_argument(self.config['gpus'])]
        args += [self.config.get('image', 'worldrsi:local'), interpreter, '-m',
                 {'prepare': 'scorer.prepare', 'score': 'scorer', 'viz': 'visualizer'}[stage],
                 '--manifest', '/runs/job/manifest.json']
        if types:
            args += [{'prepare': '--only', 'score': '--type', 'viz': '--media'}[stage], ','.join(types)]
        receipt = {**self.identity, 'stage': stage, 'case': case, 'command': args,
            'reference_sha256': sha(case_root / 'reference.mp4'), 'requested_types': types,
            'runtime': self.config.get('runtime', 'cpu'),
            'runtime_manifest_sha256': sha(self.resources / 'scorer/identity.json') if self.config.get('runtime') == 'cuda' else None,
            'scope': 'Official code and metric states; Newton/Warp renderer variant; no alternate metric substitution'}
        process=None
        try:
            if checkpoint:checkpoint()
            with (destination/'stdout.log').open('w') as stdout, (destination/'stderr.log').open('w') as stderr:
                process=subprocess.Popen(args,stdout=stdout,stderr=stderr)
                deadline=time.monotonic()+self.config.get('timeout_s',1800)
                while True:
                    if checkpoint:checkpoint()
                    if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(args,self.config.get('timeout_s',1800))
                    try:code=process.wait(timeout=.2);break
                    except subprocess.TimeoutExpired:pass
                receipt.update(exit_code=code,status='ok' if code==0 else 'error')
        except subprocess.TimeoutExpired:
            receipt.update(status='error',reason='Scorer timeout')
        except InterruptedError as exc:
            receipt.update(status='interrupted',reason=str(exc));raise
        finally:
            try:subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
            except (OSError,subprocess.SubprocessError) as exc:receipt['cleanup_error']=str(exc)
            finally:
                try:
                    if process and process.poll() is None:
                        process.kill();process.wait(timeout=5)
                finally:save(destination / 'job.json', receipt)
        reward = load(destination / 'results/reward.json', {})
        detail_path=destination/'results/reward.detail.json'
        # The official scorer emits NaN diagnostics for degenerate/short clips.
        # Preserve its raw file while keeping archived/API results strict JSON.
        detail=json.loads(detail_path.read_text(),parse_constant=lambda _:None) if detail_path.is_file() else {}
        return {'protocol': receipt, 'reward': reward, 'detail': detail,
                'metrics': metrics_from_reward(reward, detail)}

    def evaluate(self, case, workspace, destination, expected_reference_sha256=None, expected_scoring_identity=None, *, checkpoint=None):
        if expected_reference_sha256 and sha(inside(self.root / 'cases', case + '/reference.mp4')) != expected_reference_sha256:
            raise ValueError('Scorer reference differs from the task-frozen public video')
        if expected_scoring_identity is not None and scoring_identity(self.config, case) != expected_scoring_identity:
            raise ValueError('Private reference, weights or scorer dependencies changed after task creation')
        return self.job(case, workspace, destination, types=self.config.get('types'),checkpoint=checkpoint)




OFFICIAL_TYPES = ('dynamic_iou', 'scene_3d', 'trajectory', 'dynamics', 'occupancy',
                  'interpenetration', 'mesh', 'uni3d', 'flow', 'track2d', 'semantic', 'geophys', 'depth')
TYPE_KEYS = {
    'dynamic_iou': ('dynamic_iou',), 'scene_3d': ('scene_3d',), 'trajectory': ('trajectory_dtw',),
    'dynamics': ('emd_step',), 'occupancy': ('occupancy_dtw',), 'interpenetration': ('interpenetration',),
    'mesh': ('mesh_watertight', 'mesh_manifold', 'mesh_clean_faces', 'mesh_no_self_intersection'),
    'uni3d': ('uni3d_point_scene', 'uni3d_moge_scene'), 'flow': ('flow_distribution',),
    'track2d': ('track2d_dtw',), 'semantic': ('semantic_dinov3', 'semantic_tips'),
    'geophys': ('geophys_dinov3',), 'depth': ('depth_error',)}
OFFICIAL_KEYS = tuple(key for name in OFFICIAL_TYPES for key in TYPE_KEYS[name])
CHECKPOINTS = {
    'semantic': ('dinov3-vitl16-pretrain-lvd1689m/model.safetensors', 'dinov3-vitl16-pretrain-lvd1689m/config.json', 'dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json',
                 'tipsv2-l14/model.safetensors', 'tipsv2-l14/config.json', 'tipsv2-l14/configuration_tips.py',
                 'tipsv2-l14/modeling_tips.py', 'tipsv2-l14/image_encoder.py', 'tipsv2-l14/text_encoder.py'),
    'geophys': ('dinov3-vitl16-pretrain-lvd1689m/model.safetensors', 'dinov3-vitl16-pretrain-lvd1689m/config.json', 'dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json'),
    'uni3d': ('uni3d-l/model.pt',),
}
PREPARATION_CHECKPOINTS = {
    'flow': ('raft-large/raft_large_C_T_SKHT_V2.pth',),
    'track2d': ('cotracker3/scaled_offline.pth',),
    'depth': ('video-depth-anything-large/video_depth_anything_vitl.pth',),
    'uni3d': ('moge-3-vitl/model.pt',),
}
ESTIMATES = {'flow': ('flow.h5',), 'track2d': ('tracks.npz',), 'depth': ('depth.h5',),
             'semantic': ('semantic_dinov3.npz', 'semantic_tips.npz'),
             'geophys': ('geophys_dinov3.npz',), 'uni3d': ('moge.npz',)}
TASK_ROLES = frozenset({'agent', 'reconstruction', 'run_critic'})


def _applicable_types(config, case):
    """Reuse the pinned official applicability/exclusion rules, not local heuristics."""
    types = config.get('types')
    types = OFFICIAL_TYPES if types in (None, [], ['all']) else types
    rules = runpy.run_path(str(Path(config['source']) / 'scorer/outcome.py'))
    return [name for name in types if any(rules['applies'](key, case)
        and not rules['exclusion'](key, case) for key in TYPE_KEYS[name])]


@lru_cache(maxsize=16)
def _runtime_probe(image, resources, overlay, identity_sha256, gpus, runtime='cpu'):
    """Inspect the actual isolated scoring interpreter without loading any weights."""
    code = """import importlib.util,json,torch
names=['h5py','hdf5plugin','trimesh','scipy','nvdiffrast','pytorch3d','timm','transformers','easydict','igl','rtree','cv2','sentencepiece','triton','moge','flex_gemm','utils3d_moge','torchvision','cotracker']
result={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'modules':{n:importlib.util.find_spec(n) is not None for n in names}}
if result['cuda_available']:
 try:
  import triton
  triton.runtime.driver.active.get_current_device()
  result['triton_driver_ready']=True
 except Exception as exc:
  result['triton_driver_ready']=False
  result['triton_driver_error']=type(exc).__name__+': '+str(exc)
print(json.dumps(result))"""
    try:
        interpreter, pythonpath, mounts = _scorer_environment({'runtime': runtime, 'python_packages': overlay}, Path(resources))
    except (OSError, ValueError) as exc:
        return {'status': 'error', 'reason': str(exc)}
    name='worldrsi-probe-'+uuid.uuid4().hex
    args = ['docker', 'run', '--rm', '--pull=never', '--name',name,'--network=none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
        '--memory', '2g', '--pids-limit', '128', '--tmpfs', '/tmp:rw,nosuid,nodev,size=128m,mode=1777',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1',
        '--env', 'HOME=/tmp', '--env', 'PYTHONPATH=' + pythonpath]
    if runtime == 'cuda':
        args += ['--tmpfs', '/runtime-cache:rw,nosuid,nodev,exec,size=128m,mode=1777',
            '--env', 'CC=/opt/scorer-toolchain/cc', '--env', 'TRITON_CACHE_DIR=/runtime-cache/triton']
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'OPENCV_FOR_THREADS_NUM', 'CV_NUM_THREADS'):
        args += ['--env', variable + '=1']
    for source, target, _ in mounts:
        args += ['--mount', f'type=bind,src={source},dst={target},readonly']
    if gpus:
        args += ['--gpus', gpu_argument(gpus)]
    try:
        done = subprocess.run([*args, image, interpreter, '-c', code], capture_output=True, text=True, timeout=60)
        if done.returncode:
            return {'status': 'error', 'reason': done.stderr[-2000:]}
        return {'status': 'ok', **json.loads(done.stdout)}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'status': 'error', 'reason': str(exc)}
    finally:
        try:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
        except (OSError,subprocess.SubprocessError):pass


def task_usage(calls, task_id=None):
    """Only reconstruction and run critics belong to evaluated task cost.

    Request kind (generation/summary/retry) never changes its invocation role.
    RSI, candidate trials and unrelated roles are deliberately outside this scope.
    """
    selected, seen = [], {}
    for call in calls:
        if task_id is not None and call.get('task_id') is not None and call['task_id'] != task_id:
            continue
        role = call.get('role', call.get('stage'))
        if role not in TASK_ROLES:
            continue
        identity = call.get('id')
        if identity and identity in seen:
            if call.get('usage') != seen[identity].get('usage'):
                raise ValueError('Conflicting usage receipts for one request: ' + identity)
            continue
        if identity:
            seen[identity] = call
        selected.append(call)
    rows, totals = {}, dict(input_tokens=0, output_tokens=0, total_tokens=0,
                            cached_input_tokens=0, api_calls=0, complete=True, cache_split_complete=True)
    for call in selected:
        role = 'reconstruction' if call.get('role', call.get('stage')) == 'agent' else call.get('role', call.get('stage'))
        usage = call.get('usage') or {}
        valid = usage.get('complete', True) is True and all(type(usage.get(k)) is int and usage[k] >= 0
                                                          for k in ('input_tokens', 'output_tokens'))
        row = rows.setdefault(role, dict(input_tokens=0, output_tokens=0, total_tokens=0,
                                        cached_input_tokens=0, api_calls=0, complete=True, cache_split_complete=True))
        details = usage.get('input_tokens_details') or {}
        cached = usage.get('cached_input_tokens', details.get('cached_tokens'))
        cache_valid = valid and usage.get('cache_split_complete', True) is True and type(cached) is int and 0 <= cached <= usage['input_tokens']
        for target in (row, totals):
            target['api_calls'] += 1
            target['complete'] &= valid
            target['cache_split_complete'] &= cache_valid
            if valid:
                target['input_tokens'] += usage['input_tokens']
                target['output_tokens'] += usage['output_tokens']
                target['total_tokens'] += usage['input_tokens'] + usage['output_tokens']
                if cache_valid:
                    target['cached_input_tokens'] += cached
    for row in [totals, *rows.values()]:
        row['uncached_input_tokens'] = row['input_tokens'] - row['cached_input_tokens'] if row['cache_split_complete'] else None
    return {**totals, 'stages': rows, 'scope': 'all_reconstruction_and_run_critic_requests',
            'task_id': task_id, 'excluded_roles': ['task_critic', 'task_rsi_critic'],
            'request_ids': [row['id'] for row in selected if row.get('id')]}


class Evaluator:
    """Full official task quality plus actual tokens, independent of RSI policy.

    Task evaluation cannot be reduced by a legacy ``types`` setting. Explicit
    subset diagnostics use ``FourDCodeBench.job(types=...)`` instead.
    """
    def __init__(self, config, resources):
        self.resources = Path(resources).resolve()
        project = self.resources.parent.parent
        self.config = dict(config)
        if self.config.get('runtime', 'cpu') not in {'cpu', 'cuda'}:
            raise ValueError('Scorer runtime must be cpu or cuda')
        for key in ('source', 'root', 'python_packages'):
            if self.config.get(key):
                value = Path(self.config[key])
                self.config[key] = str(value.resolve() if value.is_absolute() else (project / value).resolve())
        types = self.config.get('types')
        if types not in (None, [], ['all']) and (not isinstance(types, list) or set(types) - set(OFFICIAL_TYPES)):
            raise ValueError('Evaluation types must be official 4DCodeBench types or [all]')
        self.config['types'] = None

    @property
    def metrics(self):
        return ('tokens', *(f'4dcodebench/{key}' for key in OFFICIAL_KEYS))

    def identity(self, config=None):
        return {'protocol': PROTOCOL, 'source': source_identity(self.config['source']),
                'evaluator_sha256': sha(__file__), 'config': dict(self.config)}

    def validate(self, config, policy, task_ids):
        return validate_protocol(self.config, policy, task_ids)

    compare = staticmethod(lambda baseline, candidate, policy: compare_versions(baseline, candidate, policy))

    def timestamps(self, video):
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=time_base:frame=best_effort_timestamp,best_effort_timestamp_time',
            '-of', 'json', str(video)],
            capture_output=True, text=True, check=True, timeout=120)
        decoded = json.loads(result.stdout)
        streams = decoded.get('streams', [])
        time_base = Fraction(streams[0]['time_base']) if streams and streams[0].get('time_base') else None
        times = []
        for frame in decoded.get('frames', []):
            pts = frame.get('best_effort_timestamp')
            if pts is not None and time_base is not None:
                value = float(int(pts) * time_base)
            elif frame.get('best_effort_timestamp_time') is not None:
                value = float(frame['best_effort_timestamp_time'])
            else:
                raise ValueError('Decoded reference frame has no presentation timestamp')
            if not math.isfinite(value) or (times and value < times[-1]):
                raise ValueError('Reference presentation timestamps must be finite and ordered')
            times.append(value)
        if not times:
            raise ValueError('Reference video has no decoded frames')
        return times

    def execution_config(self, allocated=None):
        """Physical allocation affects execution, never the frozen score protocol."""
        config=dict(self.config)
        if allocated:
            for key in ('cpus','memory','gpus','image','timeout_s'):
                if key in allocated:config[key]=allocated[key]
        if config.get('gpus')=='auto':
            raise ValueError('Automatic scorer GPU allocation requires a scheduler resource lease')
        return config

    def readiness(self, case, *, execution_config=None, reference=None, types=None, estimates_required=True):
        execution=self.execution_config(execution_config)
        source = Path(self.config['source'])
        root = Path(self.config['root'])
        types = tuple(types or self.config.get('types') or OFFICIAL_TYPES)
        missing = []
        try:
            identity = source_identity(source)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            missing.append({'kind': 'source', 'reason': str(exc)})
            identity = None
        case_root=Path(reference['case_root']) if reference else root/'cases'/case
        data_root=Path(reference['data_root']) if reference else root/'data'/case
        for path in (case_root / 'reference.mp4',):
            if not path.is_file():
                missing.append({'kind': 'public_reference', 'path': str(path)})
        active = ([name for name in types if name in _applicable_types(self.config,case)] if identity else [])
        for metric in active:
            if case.startswith('synthetic/') and metric in {'flow', 'track2d', 'depth'}:
                continue
            weights=CHECKPOINTS.get(metric, ())+(PREPARATION_CHECKPOINTS.get(metric, ()) if reference and not estimates_required else ())
            for filename in weights:
                path = root / 'checkpoints' / filename
                if not path.is_file():
                    missing.append({'kind': 'checkpoint', 'metric_type': metric, 'path': str(path)})
            for filename in ESTIMATES.get(metric, ()):
                if not estimates_required:continue
                # Uni3D's point reference is synthetic; its MoGe reference is real.
                if metric == 'uni3d' and case.startswith('synthetic/'):
                    continue
                path = data_root / 'estimates' / filename
                if not path.is_file():
                    missing.append({'kind': 'reference_estimate', 'metric_type': metric, 'path': str(path)})
        if case.startswith('synthetic/') and set(active) & {'scene_3d', 'trajectory', 'dynamics', 'occupancy', 'uni3d', 'dynamic_iou'}:
            path = root / 'data' / case / 'world'
            if not (path / 'camera.json').is_file():
                missing.append({'kind': 'private_world', 'path': str(path)})
        if 'dynamic_iou' in active and case.startswith('real/'):
            path = root / 'data' / case / 'annotation/dynamic_mask.npz'
            if not path.is_file():
                missing.append({'kind': 'annotation', 'metric_type': 'dynamic_iou', 'path': str(path)})
        runtime = None
        if set(active) - {'mesh', 'interpenetration'} or self.config.get('runtime') == 'cuda':
            identity_path = self.resources / 'identity.json'
            overlay = Path(self.config.get('python_packages', ''))
            stamp = str(overlay.stat().st_mtime_ns) if overlay.is_dir() else 'missing'
            signature = (sha(identity_path) if identity_path.is_file() else 'unfrozen') + ':' + stamp
            if self.config.get('runtime') == 'cuda':
                manifest = self.resources / 'scorer/identity.json'
                signature += ':' + (sha(manifest) if manifest.is_file() else 'missing-cuda-manifest')
            runtime = _runtime_probe(execution.get('image', 'worldrsi:local'),
                str(self.resources), str(overlay),
                signature, str(execution.get('gpus') or ''), self.config.get('runtime', 'cpu'))
            if runtime.get('status') != 'ok':
                missing.append({'kind': 'runtime', 'reason': runtime.get('reason')})
            else:
                required = {'h5py', 'hdf5plugin', 'trimesh', 'scipy', 'cv2'}
                if 'mesh' in active:
                    required.update({'igl', 'rtree'})
                if set(active) & {'scene_3d', 'trajectory', 'dynamics', 'occupancy', 'uni3d'}:
                    required.add('pytorch3d')
                if set(active) & {'semantic', 'geophys'}:
                    required.update({'transformers', 'sentencepiece'})
                if 'uni3d' in active:
                    required.update({'timm', 'easydict'})
                    if reference and not estimates_required:required.update({'moge','flex_gemm','utils3d_moge'})
                if reference and not estimates_required and 'flow' in active:required.add('torchvision')
                if reference and not estimates_required and 'track2d' in active:required.add('cotracker')
                if set(active) & {'dynamic_iou', 'flow', 'track2d', 'depth'}:
                    required.add('nvdiffrast')
                    if not runtime.get('cuda_available'):
                        missing.append({'kind': 'runtime', 'reason': 'Official raster metrics require a CUDA-enabled scoring interpreter'})
                if set(active) & {'trajectory', 'track2d'}:
                    required.add('triton')
                    if not runtime.get('triton_driver_ready'):
                        missing.append({'kind': 'runtime', 'reason': runtime.get('triton_driver_error',
                            'Official DTW kernels require a working CUDA Triton driver and C compiler')})
                for name in sorted(required):
                    if not runtime.get('modules', {}).get(name):
                        missing.append({'kind': 'runtime_module', 'name': name})
        return {'status': 'not_ready' if missing else 'ready', 'missing': missing, 'runtime': runtime,
                'requested_types': list(types), 'full_profile': set(types) == set(OFFICIAL_TYPES),
                'protocol': identity, 'runtime_dependencies': 'Validated by actual isolated scorer execution; resource readiness alone is not proof'}

    def evaluate(self, task_dir, task_record, calls, artifact_dir=None, artifact_source=None, *, execution_config=None, checkpoint=None):
        """Score an explicitly selected archive even if its run later failed.

        Orchestration validates and selects archives, then supplies their source
        metadata. This evaluator never silently searches older runs. Scores
        describe that snapshot; they do not change the run or task outcome.
        """
        task_dir = Path(task_dir).resolve()
        execution=self.execution_config(execution_config)
        usage = task_usage(calls, task_record.get('id'))
        token_metric = {'status': 'ok' if usage['complete'] else 'missing',
                        'value': usage['total_tokens'] if usage['complete'] else None,
                        'known_value': usage['total_tokens'], 'scope': usage['scope']}
        result = {'schema_version': 1, 'protocol': PROTOCOL, 'usage': usage,
                  'metrics': {'tokens': token_metric}, 'quality_status': 'unavailable',
                  'scope': 'one complete task, all reconstruction runs and run critics, before task RSI',
                  'full_profile': True, 'config': dict(self.config),
                  'execution_config':{key:execution.get(key) for key in ('cpus','memory','gpus','image','timeout_s')}}
        benchmark = task_record.get('benchmark')
        if not benchmark:
            from evaluation.user_media import evaluate_media
            return evaluate_media(self,task_dir,task_record,result,artifact_dir,artifact_source,execution,checkpoint)
        expected = tuple('4dcodebench/' + key for key in OFFICIAL_KEYS)
        result['metric_coverage'] = {'expected_count': len(expected), 'reported_count': 0,
                                     'missing': list(expected), 'complete': False}

        def unavailable(reason, status='unavailable'):
            result.update(quality_status=status, reason=reason)
            for name in expected:
                if name not in result['metrics']:
                    result['metrics'][name] = {'status': status, 'value': None, 'reason': reason,
                        'protocol': PROTOCOL, 'source': 'task_evaluator'}
            return result

        ready = self.readiness(benchmark['case'],**({'execution_config':execution} if execution_config is not None else {}))
        result['readiness'] = ready
        result['protocol_identity'] = ready['protocol']
        result['evaluator_sha256'] = sha(__file__)
        result['resource_identity'] = load(task_dir / 'scoring_identity.json')
        identity = {'config': self.config, 'source': ready['protocol'], 'benchmark': benchmark}
        result['protocol_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        runs = task_record.get('runs', [])
        last = runs[-1] if runs else {}
        explicit_artifact = artifact_dir is not None
        if not explicit_artifact and last and (last.get('failure') or last.get('status') not in {None, 'completed'}):
            return unavailable('The final reconstruction run failed or was interrupted; no validated archive was explicitly selected')
        if artifact_dir is None:
            path = last.get('render_dir')
            artifact_dir = inside(task_dir, path) if path else None
        elif not Path(artifact_dir).is_absolute():
            artifact_dir = inside(task_dir, artifact_dir)
        if artifact_dir is None or not (Path(artifact_dir) / 'world').is_dir():
            return unavailable('The selected task artifact has no exported world')
        artifact_dir = Path(artifact_dir).resolve()
        if not artifact_dir.is_relative_to(task_dir):
            raise ValueError('Task evaluation artifacts must remain inside their task')
        relative = str(artifact_dir.relative_to(task_dir))
        selected = next((run for run in reversed(runs) if run.get('render_dir') == relative), {})
        source = dict(artifact_source or {})
        if source.get('render_dir', relative) != relative:
            raise ValueError('Artifact source does not match the selected render directory')
        if selected and source.get('run_id', selected.get('id')) != selected.get('id'):
            raise ValueError('Artifact source run does not own the selected render')
        is_current = selected == last and not last.get('failure') and last.get('status') == 'completed'
        source = {'run_id': selected.get('id'), 'is_current': is_current,
                  'fallback': not is_current, **source, 'render_dir': relative}
        result['artifact_source'] = source
        result['task_outcome'] = {'status': task_record.get('status'), 'run_id': last.get('id'),
            'run_status': last.get('status'), 'failure': last.get('failure')}
        native_export=load(artifact_dir/'world_export.json',{})
        if native_export.get('protocol') in {'newton-camera-trajectory-v1','newton-native-render-v1'} or native_export.get('status')=='render_only':
            return unavailable('The selected native camera/material profile is outside the fixed-camera complete official benchmark contract')
        if ready['status'] != 'ready':
            return unavailable('Official metric resources are incomplete', 'not_ready')
        destination = task_dir / 'evaluation' / ('score_' + uuid.uuid4().hex)
        result['evidence_path'] = str(destination.relative_to(task_dir))
        try:
            scorer = FourDCodeBench(execution, self.resources)
            scored = scorer.evaluate(benchmark['case'], artifact_dir, destination,
                expected_reference_sha256=benchmark.get('sha256'),
                expected_scoring_identity=load(task_dir / 'scoring_identity.json'),checkpoint=checkpoint)
        except InterruptedError:
            raise
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            return unavailable('Official scorer failed: ' + str(exc), 'error')
        result['fourd'] = scored
        result['metrics'].update(scored['metrics'])
        result['quality_status'] = scored['protocol']['status']
        missing = [name for name in expected if name not in scored['metrics']]
        result['metric_coverage'] = {'expected_count': len(expected),
            'reported_count': len(expected) - len(missing), 'missing': missing, 'complete': not missing}
        if missing:
            return unavailable('Official scorer did not emit the complete task profile', 'error')
        if any(row['status'] == 'error' for row in scored['metrics'].values()):
            result['quality_status'] = 'error'
        return result


def validate_protocol(config, policy, task_ids):
    """No fixed five-task composition or implicit adoption thresholds."""
    errors = []
    if not isinstance(policy, dict) or not policy:
        return ['Adoption policy has not been configured']
    required = policy.get('task_ids')
    if not isinstance(required, list) or not required or len(set(required)) != len(required):
        errors.append('Policy requires an explicit nonempty unique task_ids list')
    elif set(required) != set(task_ids):
        errors.append('Policy task_ids must match the frozen validation samples')
    rules = policy.get('metrics')
    if not isinstance(rules, dict) or not rules:
        errors.append('Policy requires explicit metric rules')
    else:
        for name, rule in rules.items():
            if name != 'tokens' and not name.startswith('4dcodebench/'):
                errors.append('Only official 4DCodeBench metrics and tokens can select versions')
            if not isinstance(rule, dict) or rule.get('direction') not in {'higher', 'lower'}:
                errors.append('Metric direction must be explicitly higher or lower: ' + name)
                continue
            for key in ('max_regression', 'min_mean_gain'):
                value = rule.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or (key == 'max_regression' and value < 0):
                    errors.append('Invalid metric threshold: ' + name + '.' + key)
    return errors


def compare_versions(baseline, candidate, policy):
    result = {'decision': 'pending', 'reasons': [], 'comparison': {}, 'policy': policy}
    errors = validate_protocol({}, policy, policy.get('task_ids', []) if isinstance(policy, dict) else [])
    if errors:
        return {**result, 'reasons': errors}
    required = set(policy['task_ids'])
    before = {row.get('task_id'): row for row in baseline}
    after = {row.get('task_id'): row for row in candidate}
    if len(before) != len(baseline) or len(after) != len(candidate) or set(before) != required or set(after) != required:
        return {**result, 'decision': 'incomparable', 'reasons': ['Versions require the same frozen validation samples']}
    for task in required:
        a, b = before[task], after[task]
        for key in ('source_sha256', 'seed', 'protocol_sha256'):
            if a.get(key) is None or a.get(key) != b.get(key):
                return {**result, 'decision': 'incomparable', 'reasons': ['Missing or different ' + key + ': ' + task]}
        if a.get('status') != 'completed' or b.get('status') != 'completed':
            return {**result, 'decision': 'incomparable', 'reasons': ['Validation task is incomplete: ' + task]}
    passed = True
    for metric, rule in policy['metrics'].items():
        gains = []
        for task in sorted(required):
            a, b = (row[task].get('metrics', {}).get(metric, {}) for row in (before, after))
            if any(x.get('status') != 'ok' or type(x.get('value')) not in (int, float) or not math.isfinite(x['value']) for x in (a, b)):
                return {**result, 'reasons': ['Metric unavailable for paired comparison: ' + metric]}
            gains.append((b['value'] - a['value']) * (1 if rule['direction'] == 'higher' else -1))
        mean = sum(gains) / len(gains)
        ok = min(gains) >= -rule['max_regression'] and mean >= rule['min_mean_gain']
        result['comparison'][metric] = {'gains': gains, 'mean_gain': mean, 'passed': ok}
        passed &= ok
    result['decision'] = 'accepted' if passed else 'rejected'
    return result
