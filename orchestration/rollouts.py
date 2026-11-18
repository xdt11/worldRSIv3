"""Independent experiment archives around immutable safeRSI scene deliveries."""
from __future__ import annotations
import copy
from functools import partial
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading

from orchestration.models import TaskContext
from orchestration.tasks import serialized_task
from harness.core.budget import Budget
from harness.evidence.archive import inside, identifier, load, save, sha, now, uid
from harness.evidence.artifacts import archived_render
from harness.evidence.context import tool, image_item
from harness.evidence.usage import call_usage
from harness.tools.go2.rollout import VIDEO_ROLES, encode_images


def _ordinary_evidence(root, name):
    """An ordinary archive file; never follow a source or log symlink."""
    if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Evidence needs a rollout-relative path')
    original=root/name
    target=inside(root,name)
    for parent in (original,*original.parents):
        if parent.is_symlink(): raise ValueError('Rollout evidence cannot contain symlinks')
        if parent==root: break
    return target if target.is_file() else None


def rollout_review_enabled(task):
    return (task.get('split') not in {'test','validation','heldout','held-out','held_out','trial','smoke'}
            and task['config'].get('outer_rsi',True))


def _archived_proxy_frames(root, result):
    """Recover an older episode's synchronized raw camera only from its recorded frames."""
    path = _ordinary_evidence(root, 'trajectory.json')
    fps = result.get('render_fps')
    if path is None or type(fps) is not int or fps <= 0:
        return []
    rows = load(path)
    if not isinstance(rows,list) or len(rows) != result.get('frame_count') or not rows:
        return []
    origin = rows[0].get('time_s') if isinstance(rows[0],dict) else None
    if type(origin) not in (int,float): return []
    frames = []
    for index, row in enumerate(rows):
        name = f'work/native/frames/{index:06}/first_proxy.png'
        if (not isinstance(row, dict) or row.get('index') != index or row.get('first_proxy') != '/'+name
                or type(row.get('time_s')) not in (int,float)
                or not math.isclose(row['time_s']-origin, index/fps, rel_tol=0, abs_tol=1e-6)):
            return []
        frame = _ordinary_evidence(root, name)
        if frame is None: return []
        frames.append(frame)
    return frames


def _evidence_preview(value, secret='', limit=1800):
    """Small readable tool receipts; complete archived JSON remains on demand."""
    private={'authorization','api_key','access_token','password','client_secret'}
    def clean(item):
        if isinstance(item,dict):
            return {key:'[REDACTED]' if key.lower() in private else clean(child) for key,child in item.items()}
        if isinstance(item,list): return [clean(child) for child in item]
        if isinstance(item,str):
            if item.startswith('data:image/'): return '[Inline image; open archived JSON to inspect]'
            return item.replace(secret,'[REDACTED]') if secret else item
        return item
    result=clean(value)
    serialized=json.dumps(result,ensure_ascii=False)
    return result if len(serialized)<=limit else {'preview':serialized[:limit], 'truncated':True}


def _archived_instruction(context):
    """Read both the original JSON payload and newer literal instruction messages."""
    messages=context.get('input',[]) if isinstance(context,dict) else []
    contents=[]
    for message in messages:
        if not isinstance(message,dict) or message.get('role')!='user': continue
        content=message.get('content')
        if isinstance(content,list):
            content=next((part.get('text') for part in content if isinstance(part,dict)
                          and part.get('type') in {'input_text','text'}),None)
        if isinstance(content,str): contents.append(content)
    if not contents: return None
    if len(contents)>1:
        try: metadata=json.loads(contents[1])
        except (ValueError,TypeError): metadata=None
        if isinstance(metadata,dict) and 'instruction_source' in metadata:
            return contents[0]  # The literal user instruction may itself be JSON.
    try: payload=json.loads(contents[0])
    except (ValueError,TypeError): return contents[0]
    if isinstance(payload,dict) and isinstance(payload.get('instruction'),str): return payload['instruction']
    return contents[0]


class RolloutRunner:
    def __init__(self, app):
        self.app, self.store = app, app.store
        self._legacy_video_lock = threading.Lock()

    def list(self, task_id=None):
        paths = ([self.store.task_path(task_id) / 'task.json'] if task_id else self.store.task_records(kind='threat'))
        rows = []
        for path in paths:
            task = load(path, {})
            for index in task.get('rollouts', []):
                record = load(inside(path.parent, index['report_path']), {})
                if record: rows.append(record)
        return sorted(rows, key=lambda row: row.get('created_at', ''), reverse=True)

    def get(self, identity):
        identifier(identity)
        record = next((row for row in self.list() if row['id'] == identity), None)
        if record is None: raise FileNotFoundError(identity)
        videos = record.get('videos') or {}
        if record['status'] == 'completed' and 'first_person_proxy' not in videos:
            root = self.store.task_path(record['task_id'])/'evaluation/rollouts'/identity
            result = record.get('result') or {}
            path = 'videos/first_person_proxy.webm'
            original = 'videos/first_person_proxy.mp4'
            if not isinstance(result,dict) or not all(type(result.get(key)) is int and result[key] > 0
                    for key in ('frame_count','render_fps')):
                return record
            try:
                cached = _ordinary_evidence(root,path) and _ordinary_evidence(root,original)
                available = cached or _archived_proxy_frames(root,result)
            except (ValueError,OSError):
                available = False
            if available:
                record['videos'] = {**videos, 'first_person_proxy':{'path':path, 'original_path':original,
                    'status':'completed', 'source':'archived_warp_frames',
                    'frame_count':result['frame_count'], 'fps':result['render_fps']}}
        return record

    def artifact(self, identity, path):
        record = self.get(identity)
        root = self.store.task_path(record['task_id']) / 'evaluation/rollouts' / identity
        target = _ordinary_evidence(root,path)
        proxy = (record.get('videos') or {}).get('first_person_proxy') or {}
        if (target is None and proxy.get('source') == 'archived_warp_frames'
                and path in ('videos/first_person_proxy.webm','videos/first_person_proxy.mp4')):
            with self._legacy_video_lock:
                target = _ordinary_evidence(root,path)
                if target is None:
                    frames = _archived_proxy_frames(root, record['result'])
                    if frames:
                        target = root/path
                        encode_images(frames, proxy['fps'], target)
        if target is None: raise FileNotFoundError(path)
        return target

    def evidence(self, identity):
        """Index actual evaluator calls and appearance inputs without inline API bodies."""
        record=self.get(identity)
        task=self.store.task(record['task_id'])
        root=self.store.task_path(record['task_id'])/'evaluation/rollouts'/identifier(identity)
        secret=self.store.api_key()
        def path(name):
            try:
                target=_ordinary_evidence(root,str(name))
                return str(target.relative_to(root)) if target is not None else None
            except (OSError,ValueError): return None
        def read(name,default):
            relative=path(name)
            if relative is None: return default
            try: return load(root/relative,default)
            except (OSError,ValueError): return default
        def image(name):
            relative=path(name)
            return {'status':'available' if relative else 'unavailable','path':relative}
        reference={'proxy':image('work/native/frames/000000/first_proxy.png'),
            'appearance':image('original_reference.png'), 'generated':image('reference.png'),
            'used':image('work/reference.png'), 'request_path':path('reference.request.json'),
            'receipt_path':path('reference.receipt.json'), 'provenance_path':path('appearance_source.json')}
        invocations,events=[],[]
        for meta_file in sorted((root/'invocations').glob('*/invocation.json')):
            prefix=str(meta_file.parent.relative_to(root))
            metadata=read(prefix+'/invocation.json',{})
            if not isinstance(metadata,dict) or not metadata: continue
            invocation_id=meta_file.parent.name
            entry={key:metadata[key] for key in ('role','stage','status','created_at','finished_at','error','usage') if key in metadata}
            entry.update(id=invocation_id,meta_path=path(prefix+'/invocation.json'),
                         context_path=path(prefix+'/context.json'),result_path=path(prefix+'/result.json'))
            entry['archived_instruction']=_archived_instruction(read(prefix+'/context.json',{}))
            invocations.append(entry)
            for call_file in sorted((meta_file.parent/'calls').glob('*/meta.json')):
                call_prefix=str(call_file.parent.relative_to(root))
                call=read(call_prefix+'/meta.json',{})
                if not isinstance(call,dict) or not call: continue
                event={key:call[key] for key in ('model','role','stage','status','created_at','finished_at','error','elapsed_s','http_status') if key in call}
                event.update(kind='call',id=call_file.parent.name,invocation_id=invocation_id,
                    usage=call_usage(call),meta_path=path(call_prefix+'/meta.json'),
                    request_path=path(call_prefix+'/request.json'),response_path=path(call_prefix+'/response.json'))
                events.append(event)
            tools=read(prefix+'/tools.json',[])
            for event in tools if isinstance(tools,list) else []:
                if not isinstance(event,dict): continue
                try: event_id=identifier(event.get('id',''))
                except (TypeError,ValueError): continue
                row={key:event[key] for key in ('id','name','stage','status','created_at','finished_at','call_id','api_call_id') if key in event}
                row.update(kind='tool',invocation_id=invocation_id,role=metadata.get('role'),
                    meta_path=path(prefix+'/tools.json'),result_path=path(prefix+'/tools/'+event_id+'.json'))
                if 'arguments' in event: row['arguments']=_evidence_preview(event['arguments'],secret,8000)
                row['result']=_evidence_preview(event.get('feedback') or event.get('result'),secret)
                row['logs']=[]
                for channel in ('stdout','stderr'):
                    result=event.get('result') or event.get('feedback') or {}
                    value=result.get(channel+'_path') if isinstance(result,dict) else None
                    if not isinstance(value,str): continue
                    if value.startswith('/logs/'): value='tool_logs/'+value.removeprefix('/logs/')
                    relative=path(value)
                    if relative: row['logs'].append({'label':channel,'path':relative})
                events.append(row)
        events.sort(key=lambda event:(str(event.get('created_at','')),event['kind'],event['id']))
        invocations.sort(key=lambda item:(str(item.get('created_at','')),item['id']))
        archived=next((item['archived_instruction'] for item in invocations if item.get('archived_instruction') is not None),None)
        return {'instruction':task.get('instruction',''),'instruction_source':'scene_task.instruction',
            'archived_instruction':archived,'instruction_matches_archive':archived==task.get('instruction','') if archived is not None else None,
            'reference':reference,'invocations':invocations,'events':events}

    def recover(self):
        for row in self.list():
            if row['status']=='queued' and self.store.task(row['task_id']).get('automatic_rollout_id')==row['id']:
                if not self.app.scheduler.is_busy(key='rollout:'+row['id']):
                    self._schedule(row['task_id'],row['id'],review=row.get('task_critic_status')=='pending')
                continue
            if row['status'] in {'running', 'queued'}:
                reason='Controller restarted'
                root=self.store.task_path(row['task_id'])/'evaluation/rollouts'/row['id']
                if (root/'result.json').is_file():
                    try:
                        self.complete(row['task_id'],row['id'])
                        continue
                    except (OSError,KeyError,ValueError,StopIteration) as exc:
                        reason=f'Controller restarted before the archive was verified: {exc}'
                self._update(row['task_id'],row['id'],status='interrupted',phase='finished',task_critic_status='skipped',
                    error=reason,finished_at=now())
        for row in self.list():
            if row.get('status') in {'completed','failed'} and row.get('task_critic_status') in {'pending','running'}:
                self.app.scheduler.schedule(self.review,row['task_id'],row['id'],kind='rollout_review',
                    key='review:'+row['id'],task_ids=(row['task_id'],))
        for path in self.store.task_records(kind='threat'):
            task=load(path)
            identity=task.get('automatic_rollout_id')
            if identity and not any(row['id']==identity for row in task.get('rollouts',[])):
                # A crash between freezing the scene and publishing the queued experiment.
                self.enqueue(task['id'],identity,review=task.get('task_critic_skipped')=='awaiting_rollout')

    def enqueue(self, task_id, identity, *, review):
        task = self.store.task(task_id)
        self._update(task_id,identity,id=identity,task_id=task_id,title=task['title'],
            status='queued',phase='queued',created_at=now(),task_critic_status='pending' if review else 'skipped')
        self._schedule(task_id,identity,review=review)

    def _schedule(self, task_id, identity, *, review):
        try:
            self.app.scheduler.schedule(partial(self.run,identity=identity,review=review),task_id,
                key='rollout:'+identity,kind='rollout',task_ids=(task_id,),handoff=True)
        except Exception as exc:
            self._update(task_id,identity,status='failed',phase='finished',failed_phase='queued',
                error=str(exc),task_critic_status='skipped',finished_at=now())
            if review:
                self.store.update_task(task_id,task_critic_skipped='rollout_not_scheduled')

    def _update(self, task_id, identity, /, **changes):
        directory = self.store.task_path(task_id)
        path = directory / 'evaluation/rollouts' / identity / 'report.json'
        record = {**load(path, {}), **changes, 'updated_at': now()}; save(path, record)
        with self.store.lock:
            task = self.store.task(task_id)
            index = [row for row in task.get('rollouts', []) if row['id'] != identity]
            index.append({'id': identity, 'status': record['status'],
                          'report_path': str(path.relative_to(directory))})
            self.store.update_task(task_id, rollouts=index)
        return record

    def _archive_episode(self, task_id, identity):
        """Verify native evidence and publish a stable receipt outside the writable sandbox."""
        task=self.store.task(task_id)
        task_dir=self.store.task_path(task_id)
        directory=task_dir/'evaluation/rollouts'/identifier(identity)
        record=load(directory/'report.json')
        result=load(directory/'result.json',{})
        if result.get('status')!='completed' or not result.get('source_unchanged'):
            raise ValueError('A genuine completed rollout receipt is required')
        source=record['scene_source']
        delivery=None
        if record['status']!='completed':
            run=next(row for row in task['runs'] if row['id']==source['run_id'])
            archive=archived_render(task_dir,run,for_replay=True)
            if not archive or archive['render_revision']!=source['render_revision']:
                raise ValueError('The immutable scene changed during evaluation')
            delivery=inside(task_dir,archive['render_dir'])/'delivery'
        for item in source['files']:
            if (sha(inside(directory/'source/scene',item['path']))!=item['sha256']
                    or (delivery is not None and sha(inside(delivery,item['path']))!=item['sha256'])):
                raise ValueError('The immutable scene changed during evaluation')
        videos={}
        for key in VIDEO_ROLES:
            if key == 'first_person_proxy' and key not in record.get('videos', {}) and 'render_window_s' not in result:
                continue  # A historical two-video receipt.
            path=_ordinary_evidence(directory,f'videos/{key}.mp4')
            if path is None or not path.stat().st_size: raise ValueError('A required rollout video is missing: '+key)
            videos[key]={'status':'completed','path':f'videos/{key}.mp4','sha256':sha(path),
                         'frame_count':result.get('frame_count'),'fps':result.get('render_fps')}
            compatible=_ordinary_evidence(directory,f'videos/{key}.webm')
            if compatible is not None and compatible.stat().st_size:
                videos[key].update(path=f'videos/{key}.webm',sha256=sha(compatible),
                    original_path=f'videos/{key}.mp4',original_sha256=sha(path))
            previous=record.get('videos',{}).get(key,{})
            for field in ('path','sha256','original_path','original_sha256'):
                if previous.get(field) and videos[key].get(field)!=previous[field]:
                    raise ValueError('Archived video identity changed: '+key)
        evidence={}
        for name in ('setup.json','result.json','trajectory.json','decisions.json','observations.json','render_chunks.json'):
            path=_ordinary_evidence(directory,name)
            if path is None: raise ValueError('A required rollout artifact is missing: '+name)
            evidence[name]=sha(path)
        for name in ('physics_preflight.json','appearance_source.json','reference.png','reference.receipt.json'):
            path=_ordinary_evidence(directory,name)
            if path is not None: evidence[name]=sha(path)
        metrics={key:result[key] for key in ('simulated_seconds','goal_reached','goal_distance_m','fallen','travel_distance_m')}
        receipt={'rollout_id':identity,'task_id':task_id,'scene_source':source,'metrics':metrics,
                 'videos':videos,'evidence_sha256':evidence,'path_base':'rollout_directory'}
        receipt_path=directory/'receipt.json'
        if receipt_path.exists():
            if load(receipt_path)!=receipt: raise ValueError('Rollout archive differs from its verified receipt')
        else:
            save(receipt_path,receipt)
        prefix=str(directory.relative_to(task_dir))
        artifacts=[{'path':f'{prefix}/videos/{key}.mp4','role':role,'simulation_start_s':0}
                   for key,role in VIDEO_ROLES.items() if key in videos]
        artifacts += [{'path':f'{prefix}/{name}','role':'execution_evidence'} for name in ('receipt.json',*evidence)]
        return {'result':result,'metrics':metrics,'videos':videos,'artifacts':artifacts,'receipt_path':'receipt.json'}

    def complete(self, task_id, identity):
        verified=self._archive_episode(task_id,identity)
        return self._update(task_id,identity,**verified,status='completed',phase='finished',error=None,finished_at=now())

    def _prepare_scene(self, task, directory):
        task_id = task['id']
        task_dir = self.store.task_path(task_id)
        frozen = (task.get('evaluation') or {}).get('artifact_source') or {}
        source_run, archive = None, None
        for run in reversed(task.get('runs', [])):
            if run.get('status')!='completed' or (frozen and run['id']!=frozen['run_id']):
                continue
            candidate = archived_render(task_dir, run, for_replay=True)
            if (candidate and candidate['is_current'] and
                    (not frozen or (not frozen.get('fallback') and candidate['render_revision']==frozen['render_revision']))):
                source_run, archive = run, candidate; break
        if archive is None: raise ValueError('The task has no completed current native scene delivery')
        scene = inside(task_dir, archive['render_dir']) / 'delivery'
        manifest = load(scene / 'dependency_manifest.json')
        baseline = directory / 'source/scene'; baseline.mkdir(parents=True)
        files = manifest['files']
        for item in files:
            original = inside(scene, item['path'])
            if original.is_symlink() or sha(original) != item['sha256']: raise ValueError('Scene delivery identity changed')
            target = inside(baseline, item['path']); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target); target.chmod(0o444)
        preview = inside(task_dir, archive['preview_path'])
        shutil.copyfile(preview, directory / 'source/preview.png')
        (directory / 'work').mkdir()
        for entry in task.get('inputs', []):
            src = inside(task_dir, entry['local_path'])
            dest = inside(directory, entry['local_path']); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        self.app.versions.materialize(task['version'], directory / 'skills')
        return {'task_id': task_id, 'run_id': source_run['id'], 'render_revision': archive['render_revision'],
                  'files': files, 'world_sha256': archive['world_sha256'], 'code_sha256': archive['code_sha256']}

    @serialized_task
    def run(self, task_id, *, review=True, identity=None):
        """Only a finalized scene can enter an independent experiment."""
        from harness.agents import evaluation
        from harness.tools.go2 import rollout
        task = self.store.task(task_id)
        if task.get('kind')!='threat' or task['status']!='completed':
            raise ValueError('Choose a finalized, completed safeRSI scene for evaluation')
        task_dir = self.store.task_path(task_id)
        automatic = identity is not None
        if automatic:
            if identity!=task.get('automatic_rollout_id'):
                raise ValueError('The automatic rollout does not belong to this scene delivery')
        else:
            identity = uid('rollout')
        directory = task_dir/'evaluation/rollouts'/identity
        queued = load(directory/'report.json',{})
        if queued and queued['status']!='queued': return queued
        self._update(task_id,identity,id=identity,task_id=task_id,title=task['title'],
            status='running',phase='preparing',created_at=queued.get('created_at') or now(),
            task_critic_status='pending' if review and rollout_review_enabled(task) else 'skipped',
            videos={key:{'status':'unavailable'} for key in VIDEO_ROLES},
            scope='Independent Go2 experiment; source task only produces a scene')
        record = None
        try:
            self.app.verify_runtime()
            source=self._prepare_scene(task,directory)
            self._update(task_id,identity,scene_source=source)
            cfg = copy.deepcopy(task['config'])
            if not automatic:
                current=self.store.config()
                for key in ('base_url','models','model','critic_model'):
                    if current.get(key) is not None: cfg[key]=copy.deepcopy(current[key])
                cfg['rollout']=current['rollout']
            limits = {**self.store.config()['rollout'], **cfg.get('rollout', {})}
            resources_path = self.store.root / 'data/resources/go2/manifest.json'
            resources = load(resources_path)
            if not resources: raise ValueError('Prepare Go2 resources with scripts/prepare_go2_resources.py')
            for name, digest in resources['files'].items():
                if sha(inside(resources_path.parent, name)) != digest: raise ValueError('Go2 resource changed: ' + name)
            cfg['harness'] = {}  # Evaluation uses the current fixed harness, never rewrites old task execution.
            cfg['runtime']['parallel_tool_calls'] = False
            cfg['sandbox']['memory'] = limits.get('memory', '16g')
            cfg['sandbox'].pop('resource_identity_sha256', None)
            context = TaskContext({**task, 'config': cfg}, task_dir, self.app.bind(cfg))
            self._update(task_id,identity,harness=self.app.harness_snapshot(context.services),
                resources_manifest_sha256=sha(resources_path),
                limits={key:limits[key] for key in ('duration_s','max_duration_s','agent_max_calls','agent_max_tokens')},
                execution_model=cfg.get('critic_model') or cfg['model'])
            checkpoint = lambda: self.app.scheduler.checkpoint(self.store.task(task_id))
            experiment = {'rollout_id': identity, 'scene': '/source/scene', 'preview': '/source/preview.png',
                'world': '/source/scene/world.yaml', 'resources': '/opt/resources/go2/manifest.json',
                'work': '/work', 'scene_identity': source,
                'setup_contract': {'start_position_m': '[x,y,z] metres, root position', 'start_yaw_rad': 'radians',
                    'goal_position_m': '[x,y,z] metres, resolved from instruction', 'task_description': 'resolved task and assumptions',
                    'duration_s': min(limits['duration_s'],limits['max_duration_s']), 'render_fps': limits['render_fps'], 'navigation_hz': .5,
                    'camera': {'width': 320, 'height': 256, 'horizontal_fov_deg': 70},
                    'reference_frame_index': 'optional decoded original-video frame, default 0'},
                'max_duration_s':limits['max_duration_s'],
                'execution':'Submit setup with execute_rollout. The controller runs the episode and archives all three videos and the receipt.'}
            with self.app.tasks.resources(context, 'sandbox', rollout_directory=directory) as allocated:
                context.record['config'] = allocated
                gpu = str(allocated['sandbox'].get('gpus') or '').removeprefix('device=').strip('"')
                selected = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                renderer_gpu = next((int(row.split(',')[0]) for row in selected.stdout.splitlines()
                    if gpu in [part.strip() for part in row.split(',')]), 'auto')
                with context.services.sandbox_factory(directory, allocated, purpose='rollout') as box:
                    def execute(name, args):
                        if name == 'bash': return box.bash(**args), None
                        if name == 'view_image':
                            path=args['path']
                            pixels=(inside(directory/'source',path[8:]).read_bytes()
                                    if path.startswith('/source/') else box.image(path))
                            return {'path':path}, [image_item(pixels)]
                        raise ValueError('Unknown evaluator tool')
                    def read_setup(args):
                        setup_path = args['setup_path']
                        if not setup_path.startswith('/work/'): raise ValueError('Write setup JSON under /work')
                        setup = load(inside(directory/'work', setup_path[6:]))
                        return rollout.validate_setup(setup, limits)
                    tools = [tool('bash', 'Inspect the immutable scene/assets and write the experiment setup under /work.',
                                  {'command': {'type':'string'}, 'timeout_s': {'type':'integer','minimum':1,'maximum':300}}, ['command']),
                             tool('view_image', 'Inspect actual image pixels, including /source/preview.png.', {'path': {'type':'string'}}, ['path'])]
                    self._update(task_id, identity, phase='evaluation_agent')
                    prepared = evaluation.prepare(context.run(source['run_id']), box, directory=directory,
                        experiment=experiment, tools=tools, execute=execute, finish_validate=read_setup,
                        budget=Budget(limits['agent_max_calls'], limits['agent_max_tokens']),
                        before_call=lambda: self.app.verify_runtime())
                    agent={'invocation_path':str(prepared.archive.directory.relative_to(task_dir)),
                           'context_path':str((prepared.archive.directory/'context.json').relative_to(task_dir))}
                    try:
                        submission=self.app.tasks._invoke(prepared)
                        setup=read_setup(submission)
                    except Exception as exc:
                        status='interrupted' if isinstance(exc,InterruptedError) else 'failed'
                        self._update(task_id,identity,agent={**agent,'status':status,'usage':prepared.archive.usage()},
                                     agent_error=str(exc))
                        raise
                    self._update(task_id,identity,agent={**agent,'status':'completed','usage':prepared.archive.usage()},
                                 agent_result=submission,setup=setup,phase='executing')
                    checkpoint()
                    rollout.execute(box, directory, setup, task={**task,'config':allocated}, resources=resources,
                        config={**limits, 'task_directory':str(task_dir), 'assigned_renderer_gpu':renderer_gpu},
                        api_key=self.store.api_key(), checkpoint=checkpoint,
                        update=lambda **state: self._update(task_id, identity, **state))
                    self._update(task_id,identity,phase='archiving')
                    record=self.complete(task_id,identity)
        except InterruptedError as exc:
            if record is None:
                self._update(task_id, identity, status='interrupted', phase='finished', error=str(exc),
                             task_critic_status='skipped',finished_at=now())
            raise
        except Exception as exc:
            previous=load(directory/'report.json')
            if previous['status']=='completed':
                record=self._update(task_id,identity,cleanup_error=str(exc))
            else:
                record=self._update(task_id,identity,status='failed',phase='finished',failed_phase=previous['phase'],
                    error=str(exc),result=load(directory/'result.json'),finished_at=now())
        if review:
            self.review(task_id, identity, raise_on_error=False)
            record = load(directory/'report.json')
        return record

    @serialized_task
    def review(self, task_id, identity, *, raise_on_error=True):
        """Review the selected archived episode and its source scene."""
        from harness.agents.review import prepare
        task = self.store.task(task_id)
        if task.get('kind')!='threat' or task['status'] not in {'completed','failed'}:
            raise ValueError('Task critique requires a finished scene task and a terminal rollout report')
        if not rollout_review_enabled(task): return
        directory=self.store.task_path(task_id)
        report=load(directory/'evaluation/rollouts'/identifier(identity)/'report.json')
        if not report or report.get('task_id')!=task_id: raise FileNotFoundError(identity)
        if report.get('status') not in {'completed','failed','interrupted'}: raise ValueError('Rollout is still active')
        cfg = copy.deepcopy(task['config']); cfg['harness'] = {}
        current=self.store.config()
        for key in ('base_url','models','model','critic_model'):
            if current.get(key) is not None: cfg[key]=copy.deepcopy(current[key])
        context = TaskContext({**task,'config':cfg}, directory, self.app.bind(cfg))
        self._update(task_id,identity,task_critic_status='running')
        try:
            if report['status']=='completed':
                verified=self._archive_episode(task_id,identity)
                if not report.get('receipt_path'):
                    self._update(task_id,identity,receipt_path=verified['receipt_path'],artifacts=verified['artifacts'])
            run_id=(report.get('scene_source') or {}).get('run_id') or task['runs'][-1]['id']
            prepared = prepare(context.run(run_id), final=True, propose=True, rollout_id=identity,
                budget=Budget(cfg.get('max_calls_per_run'), cfg.get('max_total_tokens_per_run')),
                before_call=lambda: self.app.verify_runtime())
            critique = self.app.tasks._invoke(prepared)
            candidate = None
            if critique.get('changes'):
                candidate = self.app.versions.propose(task['version'], critique['changes'], critique.get('summary','Go2 rollout lessons'),
                    provenance={**critique.get('provenance', {}), 'task_id':task_id, 'task_kind':'threat', 'rollout_id':identity})
                if candidate == task['version']: candidate = None
                if candidate:
                    save(self.store.data/'skills/candidates'/f'{candidate}.json', {
                        'id': candidate, 'parent': task['version'], 'task_id': task_id, 'task_kind':'threat',
                        'rollout_id':identity, 'status':'awaiting_confirmation', 'created_at':now(),
                        'summary':critique.get('summary'), 'adoption':{'decision':'pending','mode':'manual'}})
            invocation=str(prepared.archive.directory.relative_to(directory))
            self._update(task_id,identity,task_critique=critique,candidate_version=candidate,
                         task_critic_error=None,task_critic_status='completed',task_critic_invocation=invocation)
            feedback={'critique':critique,'task_critic_error':None,'outer_rsi_performed':True,
                      'task_critic_skipped':None,'task_critic_rollout_id':identity,
                      'evolution_usage':self.app.tasks.usage(task_id,stages={'task_critic','task_rsi_critic'})}
            if candidate:
                feedback['candidate_version']=candidate
            self.store.update_task(task_id,**feedback)
        except InterruptedError:
            self._update(task_id,identity,task_critic_status='interrupted')
            raise
        except Exception as exc:
            self._update(task_id, identity, task_critic_error=str(exc),task_critic_status='failed')
            self.store.update_task(task_id,task_critic_error=str(exc),task_critic_skipped=None)
            if raise_on_error: raise
