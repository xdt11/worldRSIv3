"""Per-task Docker boundary. Credentials and evaluation never enter the container."""
from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import uuid


class SandboxError(RuntimeError):
    pass


class _RenderProgress:
    phases = ('startup', 'initialize', 'render', 'validate', 'export', 'package')

    def __init__(self, started, expected=None):
        self.phase = 'startup'
        self.completed = 0
        self.expected = expected
        self.updated = started
        self.offset = 0

    def read(self, log, now):
        size = os.fstat(log.fileno()).st_size
        # Include a short overlap for a record that straddled the previous poll.
        start = min(size, max(0, self.offset - 96, size - 65536))
        data = os.pread(log.fileno(), size - start, start)
        self.offset = size
        for match in re.finditer(rb'^WORLD_RSI_RENDER (initialize|render|validate|export|package) (\d{1,9}) (\d{1,9})\n', data, re.M):
            phase = match[1].decode()
            completed, expected = int(match[2]), int(match[3])
            if expected and self.expected is None:
                self.expected = expected
            if expected and expected != self.expected:
                continue
            if (self.phases.index(phase) < self.phases.index(self.phase) or
                    completed < self.completed or completed > (self.expected or 0)):
                continue
            if phase in ('validate','export','package') and completed != self.expected:
                continue
            if phase != self.phase or completed > self.completed:
                self.phase, self.completed, self.updated = phase, completed, now

    def diagnostic(self, reason, limit):
        return {'code':'render_timeout' if reason != 'stop_requested' else 'render_cancelled',
                'message':f'Formal render stopped ({reason}) in {self.phase}, '
                          f'{self.completed}/{self.expected if self.expected is not None else "unknown"} frames completed.',
                'reason':reason, 'phase':self.phase, 'completed_frames':self.completed,
                'expected_frames':self.expected, 'limit_s':limit, 'archive_valid':False}


def gpu_argument(value):
    """Quote one Docker CSV field when a device list itself contains commas."""
    value=str(value)
    if value.startswith('device=') and ',' in value:return '"'+value+'"'
    return value


class Sandbox:
    _verified_resources = {}
    project_label = "org.worldrsi.project=" + hashlib.sha256(str(Path(__file__).resolve().parents[2]).encode()).hexdigest()

    @classmethod
    def for_store(cls, store):
        namespace = str(store.root)
        if store.data.resolve() != store.root / 'data': namespace += ':' + str(store.data.resolve())
        label = 'org.worldrsi.project=' + hashlib.sha256(namespace.encode()).hexdigest()
        return type('ScopedSandbox', (cls,), {'project_label':label})

    def __init__(self, task_dir: Path, config: dict, *, run_id: str | None = None, cache_dir: Path | None = None,
                 purpose: str = 'task', replay_workers: dict[str, bytes] | None = None):
        if purpose not in {'task','viewer','rollout'}:raise ValueError('Unknown sandbox purpose')
        if replay_workers is not None and (purpose != 'viewer' or not replay_workers or any(
                not re.fullmatch(r'harness/(?:(?:tools/(?:newton/)?)?__init__|tools/newton/workers/[A-Za-z_]\w*)\.py', name)
                for name in replay_workers)):
            raise ValueError('Recorded workers are restricted to the native viewer package')
        self.replay_workers = replay_workers
        self.purpose=purpose
        self.task_dir = Path(task_dir).resolve()
        if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid sandbox run ID")
        self.run_id = run_id
        self.config = config.get("sandbox", config)
        evaluation = config.get('evaluation') or {}
        self.benchmark = evaluation if purpose=='task' and evaluation.get('source') else None
        self.seed = config.get("seed")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1):
            raise ValueError("seed must be null or an integer in [0, 2**32-1]")
        self.work = self.task_dir / "work"
        self.inputs = self.task_dir / "inputs"
        self.logs = self.task_dir / "tool_logs"
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.name = "worldrsi-" + uuid.uuid4().hex
        scope = Path("runs") / run_id if run_id else Path()
        self.receipt_dir = self.task_dir / scope / "sandboxes" / self.name
        self.receipt_path = str((self.receipt_dir / "startup.json").relative_to(self.task_dir))
        self.cleanup_path = str((self.receipt_dir / "cleanup.json").relative_to(self.task_dir))
        self.container_id = None
        self.started = False
        self.cleanup = None
        self.scene_session = None
        self._closing = False
        self._foreign_container = False
        for directory in (self.work, self.inputs, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def health(image="worldrsi:newton") -> dict:
        try:
            daemon = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                    capture_output=True, text=True, timeout=10)
            if daemon.returncode:
                return {"available": False, "error": daemon.stderr.strip(), "image": image}
            local = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                                   capture_output=True, text=True, timeout=10)
            return {"available": local.returncode == 0, "server_version": daemon.stdout.strip(),
                    "image": image, "image_id": local.stdout.strip() or None,
                    "error": local.stderr.strip() or None}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "error": str(exc), "image": image}

    @classmethod
    def reap_orphans(cls) -> dict:
        """Call only while holding the project's exclusive controller lock."""
        report = {"label": cls.project_label, "containers": [], "removed": [], "errors": []}
        try:
            listed = subprocess.run(["docker", "ps", "-aq", "--no-trunc", "--filter", "label=" + cls.project_label],
                                    capture_output=True, text=True, timeout=10)
            if listed.returncode:
                report["errors"].append(listed.stderr.strip())
                return report
            identities = list(dict.fromkeys(listed.stdout.split()))
            if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in identities):
                report["errors"].append("Docker returned an invalid full container ID; nothing was removed")
                return report
            report["containers"] = identities
            for identity in identities:
                result = subprocess.run(["docker", "rm", "--force", identity], capture_output=True,
                                        text=True, timeout=15)
                if result.returncode:
                    report["errors"].append({"container_id": identity, "error": result.stderr.strip()})
                else:
                    report["removed"].append(identity)
        except (OSError, subprocess.SubprocessError) as exc:
            report["errors"].append(str(exc))
        return report

    def start(self):
        if self.started:
            return
        if self.cleanup is not None:
            raise SandboxError("A closed sandbox cannot be restarted; create a new Sandbox instance")
        image = self.config.get("image", "worldrsi:newton")
        health = self.health(image)
        if not health["available"]:
            raise SandboxError("Docker sandbox unavailable; host execution is disabled: " + str(health["error"]))
        if os.getuid() == 0:
            raise SandboxError("Run the controller as a non-root user; the sandbox uses its UID for task files")
        resources = Path(self.config.get("resources_path") or Path(__file__).resolve().parents[2] / "data/resources").resolve()
        newton_environment = resources / "newton_env"
        perception = resources / "perception"
        if not (newton_environment / "bin/python").is_file() or (self.purpose=='task' and not (perception / "python/bin/python3.12").is_file()):
            raise SandboxError("Independent Newton and perception resources are incomplete: " + str(resources))
        identity = self._resources_identity(resources,newton_only=self.purpose=='viewer')
        expected_identity = self.config.get('resource_identity_sha256')
        if expected_identity and identity['manifest_sha256'] != expected_identity:
            raise SandboxError('Task resource identity changed after execution configuration was pinned')
        args = ["docker", "run", "--detach", "--pull=never", "--name", self.name,
                "--label", self.project_label, "--label", "org.worldrsi.task=" + self.task_dir.name,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
                "--cpus", str(self.config.get("cpus", 4)),
                "--memory", str(self.config.get("memory", "8g")),
                "--memory-swap", str(self.config.get("memory", "8g")),
                "--pids-limit", str(self.config.get("pids_limit", 256)),
                "--shm-size", "256m", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
                "--mount", f"type=bind,src={self.work},dst=/work",
                "--mount", f"type=bind,src={self.inputs},dst=/input,readonly",
                "--mount", f"type=bind,src={self.logs},dst=/logs,readonly",
                "--mount", f"type=bind,src={newton_environment},dst=/opt/newton-env,readonly",
                "--workdir=/work", "--env=HOME=/tmp/home", "--env=XDG_CACHE_HOME=/work/.cache",
                "--env=HF_HUB_OFFLINE=1", "--env=TRANSFORMERS_OFFLINE=1",
                "--env=PYOPENGL_PLATFORM=egl", "--env=PYTHONPATH=/opt/worldrsi", "--env=WORLD_RSI_DEVICE=cuda:0" if self.config.get("gpus") else "--env=WORLD_RSI_DEVICE=cpu"]
        if self.purpose=='task':
            args += ['--mount',f'type=bind,src={perception},dst=/opt/resources/perception,readonly']
        # Thin fixed workers are mounted from this harness revision. Reuse the
        # provisioned base image without silently executing its older workers.
        harness = Path(__file__).resolve().parents[1]
        worker_sources = [harness / '__init__.py', harness / 'tools/__init__.py']
        worker_sources.append(harness / 'tools/newton/__init__.py')
        worker_sources += list((harness / 'tools/newton/workers').glob('*.py'))
        if self.replay_workers is not None:
            worker_sources = [harness.parent / name for name in self.replay_workers]
        if self.purpose == 'rollout':
            go2 = resources / 'go2'
            manifest = json.loads((go2 / 'manifest.json').read_text())
            torch_packages = Path(manifest['host_paths']['torch_site_packages'])
            if not (torch_packages / 'torch/__init__.py').is_file():
                raise SandboxError('The reused Go2 policy runtime is incomplete')
            args += ['--mount', f'type=bind,src={go2},dst=/opt/resources/go2,readonly',
                     '--mount', f'type=bind,src={torch_packages},dst=/opt/go2-torch,readonly']
            worker_sources += list((harness / 'tools/go2').glob('*.py'))
        if self.purpose == 'task':
            worker_sources += [harness / f'tools/{name}.py' for name in
                ('_media', 'probe_media', 'extract_frames', 'crop_image', 'estimate_depth', 'segment_image')]
        workers = {}
        worker_snapshot = self.receipt_dir / 'workers'
        worker_snapshot.mkdir(parents=True, exist_ok=True)
        for source in sorted(set(worker_sources)):
            relative = source.relative_to(harness.parent)
            content = source.read_bytes() if self.replay_workers is None else self.replay_workers[str(relative)]
            pinned = worker_snapshot / relative
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_bytes(content)
            pinned.chmod(0o444)
            if self.replay_workers is None:
                args += ['--mount', f'type=bind,src={pinned},dst=/opt/worldrsi/{relative},readonly']
            # Media CLI names remain stable for the independent perception tools.
            if source.parent == harness / 'tools' and source.stem in {
                    '_media','probe_media','extract_frames','crop_image','estimate_depth','segment_image'}:
                args += ['--mount', f'type=bind,src={pinned},dst=/opt/worldrsi/{source.name},readonly']
            workers[str(relative)] = hashlib.sha256(content).hexdigest()
        if self.replay_workers is not None:
            # Only the verified bytes recopied into this fresh viewer receipt
            # are exposed. Mount as a package to exclude extra image modules.
            args += ['--mount', f'type=bind,src={worker_snapshot / "harness"},dst=/opt/worldrsi/harness,readonly']
        references = [harness.parent / 'docs/newton-api.json',
                      harness / 'tools/world-state-v3.schema.json']
        references += [harness.parent / path for path in self.config.get('context_documents',
                        ['docs/world-state.md', 'docs/newton-runtime.md'])]
        for source in sorted(set(references)):
            if not source.is_file():
                raise SandboxError('Missing task reference: ' + str(source))
            content = source.read_bytes()
            pinned = worker_snapshot / 'references' / source.name
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_bytes(content)
            pinned.chmod(0o444)
            args += ['--mount', f'type=bind,src={pinned},dst=/opt/worldrsi/{source.name},readonly']
            workers[str(source.relative_to(harness.parent))] = hashlib.sha256(content).hexdigest()
        source = self.task_dir / 'source'
        if self.purpose in {'task','rollout'} and source.is_dir():
            if source.is_symlink() or any(path.is_symlink() for path in source.rglob('*')):
                raise SandboxError('Source scene archive cannot contain symlinks')
            args += ['--mount', f'type=bind,src={source},dst=/source,readonly']
            if self.purpose == 'rollout':
                # Older scenes open assets relative to /work even when their
                # entry point is /source/scene/scene.py. Expose the same bytes.
                for dependency in sorted((source / 'scene').iterdir()):
                    target = self.work / dependency.name
                    if target.is_symlink():
                        raise SandboxError('Invalid rollout dependency mount target')
                    if dependency.is_dir():
                        target.mkdir(exist_ok=True)
                    else:
                        target.touch(exist_ok=True)
                    args += ['--volume', f'{dependency}:/work/{dependency.name}:ro']
        skills = self.task_dir / 'skills'
        if skills.is_dir():
            if skills.is_symlink() or any(p.is_symlink() for p in skills.rglob('*')):
                raise SandboxError('Pinned skills cannot contain symlinks')
            args += ['--mount', f'type=bind,src={skills},dst=/skills,readonly']
        checker_identity = None
        if self.benchmark:
            from evaluation.evaluator import source_identity
            source = Path(self.benchmark['source'])
            if not source.is_absolute():
                source = Path(__file__).resolve().parents[2] / source
            source = source.resolve()
            checker_identity = source_identity(source)
            for name in ('checker','config'):
                public = source / name
                if not public.is_dir() or public.is_symlink() or any(p.is_symlink() for p in public.rglob('*')):
                    raise SandboxError('Public benchmark code must be a directory without symlinks')
                args += ['--mount', f'type=bind,src={public},dst=/opt/4dcodebench/{name},readonly']
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for name in ('warp', 'nvidia'):
                (self.cache_dir / name).mkdir(exist_ok=True)
            args += ['--mount', f'type=bind,src={self.cache_dir},dst=/cache',
                     '--env=XDG_CACHE_HOME=/cache', '--env=WARP_CACHE_PATH=/cache/warp',
                     '--env=__GL_SHADER_DISK_CACHE_PATH=/cache/nvidia',
                     '--env=__GL_SHADER_DISK_CACHE=1']
        threads = str(max(1, int(float(self.config.get("cpus", 4)))))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            args += ["--env", key + "=" + threads]
        if self.seed is not None:
            args += ["--env", f"WORLD_RSI_SEED={self.seed}", "--env", f"PYTHONHASHSEED={self.seed}"]
        if self.config.get("gpus"):
            # The NVIDIA runtime defaults to compute/utility only. Graphics
            # capability injects the host EGL driver for native camera rendering.
            args += ["--gpus", gpu_argument(self.config["gpus"]), "--env",
                     "NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,video"]
        args += [health["image_id"], "sleep", "infinity"]
        startup = {"command": args, "health": health, "container_name": self.name,
                   "worker_sha256":workers,
                   "public_checker":checker_identity,
                   "run_id": self.run_id, "purpose":self.purpose, "resources_path": str(resources),
                   "resources_identity": identity,
                   "labels": dict(label.split("=", 1) for label in (
                       self.project_label, "org.worldrsi.task=" + self.task_dir.name)),
                   "started_at": datetime.now(timezone.utc).isoformat()}
        timeout = self.config.get('startup_timeout_s', 120)
        startup['timeout_s'] = timeout
        begin = time.monotonic()
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=timeout)
            self.container_id = result.stdout.strip() if result.returncode == 0 else None
            startup.update(stdout=result.stdout, stderr=result.stderr,
                           container_id=self.container_id, exit_code=result.returncode)
        except subprocess.TimeoutExpired as exc:
            diagnostics = self._startup_diagnostics()
            state = diagnostics.get('container', {})
            owned = (state.get('labels', {}).get('org.worldrsi.project') == self.project_label.split('=', 1)[1]
                     and state.get('labels', {}).get('org.worldrsi.task') == self.task_dir.name
                     and state.get('image') == health['image_id'])
            recovered = owned and state.get('state', {}).get('Status') == 'running'
            self._foreign_container = bool(state) and not owned
            self.container_id = state.get('id') if owned else None
            def output(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            startup.update(exit_code=None, container_id=self.container_id, cli_timed_out=True,
                           stdout=output(exc.stdout), stderr=output(exc.stderr), diagnostics=diagnostics,
                           error=f'Docker run exceeded {timeout:g}s',
                           status='recovered' if recovered else 'failed',
                           elapsed_s=time.monotonic()-begin,
                           finished_at=datetime.now(timezone.utc).isoformat())
            try:
                self._write_receipt('startup.json', startup)
            except OSError:
                self.close()
                raise
            if recovered:
                self.started = True
                return
            self.close()
            failure = SandboxError(f'Docker sandbox startup exceeded {timeout:g}s; '
                f'container state: {state.get("state", {}).get("Status", "unconfirmed")}. '
                f'Diagnostics: {self.receipt_path}')
            failure.reason = {'code':'sandbox_startup_timeout', 'message':str(failure),
                              'evidence_path':self.receipt_path}
            raise failure from exc
        except (OSError, subprocess.SubprocessError) as exc:
            startup.update(container_id=None, exit_code=None, error=str(exc), status='failed',
                           elapsed_s=time.monotonic()-begin,
                           finished_at=datetime.now(timezone.utc).isoformat())
            try:
                self._write_receipt("startup.json", startup)
            finally:
                self.close()
            raise
        startup.update(status='started' if result.returncode == 0 else 'failed',
                       elapsed_s=time.monotonic()-begin,
                       finished_at=datetime.now(timezone.utc).isoformat())
        if result.returncode:
            startup['diagnostics'] = self._startup_diagnostics()
        try:
            self._write_receipt("startup.json", startup)
        except OSError:
            # A receipt failure must not leave a detached container running.
            self.close()
            raise
        if result.returncode:
            self.close()
            raise SandboxError("Docker failed to create task sandbox: " + result.stderr)
        self.started = True

    def _startup_diagnostics(self):
        """Inspect this unique container only; never archive its environment."""
        result = {}
        try:
            inspected = subprocess.run(['docker', 'inspect', self.name],
                                       capture_output=True, text=True, timeout=10)
            result['inspect_exit_code'] = inspected.returncode
            if inspected.returncode == 0:
                value = json.loads(inspected.stdout)[0]
                result['container'] = {'id':value['Id'], 'image':value['Image'],
                    'state':value['State'], 'labels':value.get('Config', {}).get('Labels') or {}}
            else:
                result['inspect_error'] = inspected.stderr[-2000:]
            logs = subprocess.run(['docker', 'logs', '--tail', '40', self.name],
                                  capture_output=True, text=True, timeout=10)
            result['logs'] = {'exit_code':logs.returncode, 'stdout':logs.stdout[-4000:],
                              'stderr':logs.stderr[-4000:]}
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
            result['error'] = str(exc)
        return result

    def _write_receipt(self, filename: str, record: dict):
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        # Each container has its own immutable receipts. Legacy task-level
        # sandbox.json files remain untouched, including in resumed tasks.
        with (self.receipt_dir / filename).open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)

    @classmethod
    def _resources_identity(cls, resources, *, newton_only=False):
        manifest = resources / "newton-identity.json"
        if manifest.is_file():
            content = manifest.read_bytes()
            identity = json.loads(content)
            artifacts=[a for a in identity['artifacts'] if not newton_only or a['path'].startswith('newton_env/')]
            if not artifacts:raise SandboxError('Resource manifest has no artifacts for this sandbox purpose')
            for artifact in artifacts:
                path = (resources / artifact["path"]).resolve()
                if not path.is_relative_to(resources):
                    raise SandboxError("Resource identity artifact escapes independent resource directory")
                stat = path.stat()
                signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                # Rehash small sources every time; some filesystems retain the
                # same timestamp for multiple writes inside one clock tick.
                digest = cls._verified_resources.get(signature) if stat.st_size > 64 * 1024 * 1024 else None
                if digest is None:
                    value = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            value.update(block)
                    digest = value.hexdigest()
                    if stat.st_size > 64 * 1024 * 1024:
                        cls._verified_resources[signature] = digest
                if digest != artifact["sha256"]:
                    raise SandboxError("Independent resource SHA-256 mismatch: " + artifact["path"])
            return {"manifest_sha256": hashlib.sha256(content).hexdigest(), "manifest_path": str(manifest),
                    "artifacts_verified": len(artifacts), "verification_scope":
                    ('Newton-only viewer; perception weights are not mounted or used. '+identity['verification_scope']
                     if newton_only else identity['verification_scope'])}
        raise SandboxError("Missing resources/newton-identity.json; prepare independent resource provenance before tasks")

    def close(self):
        if self.cleanup is not None:
            return self.cleanup
        self._closing = True
        session = self.close_scene()
        begin = time.monotonic()
        timeout = self.config.get('cleanup_timeout_s', 60)
        try:
            if self._foreign_container:
                cleanup = {'exit_code':None, 'skipped':'Inspected container does not belong to this sandbox'}
            else:
                result = subprocess.run(["docker", "rm", "--force", self.name],
                                        capture_output=True, text=True, timeout=timeout)
                cleanup = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup = {"exit_code": None, "error": str(exc)}
        if session is not None and session.process is not None:
            try:
                session.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=5)
        cleanup.update(container_name=self.name, container_id=self.container_id,
                       run_id=self.run_id, timeout_s=timeout, elapsed_s=time.monotonic()-begin,
                       finished_at=datetime.now(timezone.utc).isoformat())
        self._write_receipt("cleanup.json", cleanup)
        self.cleanup = cleanup
        self.started = False
        self._closing = False
        return cleanup

    def close_scene(self):
        session, self.scene_session = self.scene_session, None
        if session is not None: session.close()
        return session

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    def open_process(self, args: list[str], *, stderr, own_group=False):
        """Start a streaming worker through this sandbox's execution boundary."""
        if not self.started:
            raise SandboxError("Task sandbox has not been started")
        if own_group:
            # Establish the group inside Docker, before importing any scene code.
            launch = """import os, sys
if os.getpgrp() != os.getpid():
    os.setpgid(0, 0)
print(f'WORLD_RSI_GROUP {os.getpid()}', file=sys.stderr, flush=True)
os.execvp(sys.argv[1], sys.argv[1:])
"""
            args = ["python", "-I", "-S", "-c", launch, *args]
        return subprocess.Popen(["docker", "exec", "--interactive", self.name, *args],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr, bufsize=0)

    def kill_worker_group(self, log, args):
        """Signal only the in-container group whose leader still has the expected argv."""
        header = os.pread(log.fileno(), 64, 0)
        match = re.match(rb'WORLD_RSI_GROUP ([1-9]\d{0,8})\n', header)
        if not match:
            return False
        check_and_kill = """import os, signal, sys
pid = int(sys.argv[1])
with open(f'/proc/{pid}/cmdline', 'rb') as stream:
    argv = stream.read().split(b'\\0')[:-1]
if pid <= 1 or os.getpgid(pid) != pid or argv != list(map(os.fsencode, sys.argv[2:])):
    sys.exit(1)
os.killpg(pid, signal.SIGKILL)
"""
        try:
            result = subprocess.run(["docker", "exec", self.name, "python", "-I", "-S", "-c",
                                     check_and_kill, match[1].decode(), *args], capture_output=True, timeout=4)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _wait_render(self, process, log, args, started, limit, explicit, idle_s=600):
        expected = None
        if self.run_id is not None:
            contract_path = self.task_dir/'runs'/self.run_id/'output_contract.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text())
                if contract.get('kind') == 'image':
                    expected = 1
                elif contract.get('kind') == 'video':
                    expected = len(contract['timestamps_s'])
        progress = _RenderProgress(started, expected)
        task_path = self.task_dir / 'task.json'
        last_task_update = None
        reason = None
        while True:
            try:
                code = process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                code = None
            now = time.monotonic()
            progress.read(log, now)
            if code is not None:
                if code in (124,137) and now - started >= limit:
                    configured_limit = self.config.get('newton_timeout_s')
                    reason = ('configured_wall_limit' if explicit and int(configured_limit or limit) <= limit
                              else 'max_tool_timeout')
                break
            if now - started >= limit + 15:
                reason = 'docker_exec_unresponsive'
                self.close()
                process.kill()
                process.wait()
                break
            if task_path.is_file():
                stamp = task_path.stat().st_mtime_ns
                if stamp != last_task_update:
                    last_task_update = stamp
                    if json.loads(task_path.read_text()).get('stop_requested'):
                        reason = 'stop_requested'
            if reason is None and not explicit and now - progress.updated >= idle_s and now - started < limit:
                reason = 'no_render_progress'
            if reason is not None:
                # GNU timeout owns a process group inside the container. End the
                # verified group, including encoders that ignore a parent's TERM.
                stopped = False
                if self.kill_worker_group(log, args):
                    try:
                        process.wait(timeout=7)
                        stopped = True
                    except subprocess.TimeoutExpired:
                        pass
                if not stopped and process.poll() is None:
                    self.close()
                    process.kill()
                    process.wait()
                break
        if reason:
            return (130 if reason == 'stop_requested' else 124), progress.diagnostic(
                reason, idle_s if reason == 'no_render_progress' else limit)
        return code, None

    def _exec(self, args: list[str], *, stdin: bytes | None = None, timeout_s=120, formal_render=False) -> dict:
        if not self.started:
            raise SandboxError("Task sandbox has not been started")
        timeout_s = int(timeout_s)
        if not 1 <= timeout_s <= int(self.config.get("max_tool_timeout_s", 1800)):
            raise ValueError("Tool timeout is outside the configured limit")
        identity = uuid.uuid4().hex
        prefix = self.logs / identity
        signal = "KILL" if formal_render else "TERM"
        bounded = ["timeout",f"--signal={signal}","--kill-after=5s",str(timeout_s),*args]
        command = ["docker","exec","--interactive",self.name]
        if formal_render:
            command += ["sh","-c",'echo "WORLD_RSI_GROUP $$" >&2; exec "$@"',"formal-render"]
        command += bounded
        error = None
        began = time.monotonic()
        with prefix.with_suffix(".stdout").open("wb") as stdout_file, prefix.with_suffix(".stderr").open("w+b") as stderr_file:
            try:
                if formal_render:
                    with subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout_file,stderr=stderr_file) as proc:
                        try:
                            code,error = self._wait_render(proc,stderr_file,bounded,began,timeout_s,
                                                         self.config.get('newton_timeout_s') is not None)
                        except BaseException:
                            if proc.poll() is None:
                                self.close()
                                proc.kill()
                            raise
                else:
                    result = subprocess.run(command, input=stdin, stdout=stdout_file, stderr=stderr_file,
                                            timeout=timeout_s + 15)
                    code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                self.close()  # Never leave unbounded work running after a lost docker exec.
                stderr_file.write(b"\nHost timeout: task sandbox was stopped.")
        stdout_path, stderr_path = prefix.with_suffix(".stdout"), prefix.with_suffix(".stderr")
        stdout_size, stderr_size = stdout_path.stat().st_size, stderr_path.stat().st_size
        limit = int(self.config.get("tool_output_chars", 24000))
        def preview(path):
            with path.open('rb') as stream:
                if path.stat().st_size <= limit: return stream.read()
                half = limit // 2
                head = stream.read(half)
                stream.seek(-half,2)
                return head + b'\n[... archived log middle omitted ...]\n' + stream.read()
        stdout, stderr = preview(stdout_path), preview(stderr_path)
        record = {"id": identity, "command": command, "exit_code": code,
                  "elapsed_s": time.monotonic() - began,
                  "stdout_path": str(prefix.with_suffix(".stdout").relative_to(self.task_dir)),
                  "stderr_path": str(prefix.with_suffix(".stderr").relative_to(self.task_dir)),
                  "stdout_bytes": stdout_size, "stderr_bytes": stderr_size}
        if error:
            record['error'] = error
        if formal_render:
            # Keep frame progress in its raw archived log, outside LLM context.
            stderr = re.sub(rb'^WORLD_RSI_(?:GROUP|RENDER) [^\n]*\n',b'',stderr,flags=re.M)
        prefix.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return {**record, "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "output_truncated": stdout_size > limit or stderr_size > limit}

    def _full_stdout(self, result, maximum=32000000) -> bytes:
        if result["stdout_bytes"] > maximum:
            raise ValueError(f"Tool result exceeds the {maximum} byte read limit; full output remains in its log")
        return (self.task_dir / result["stdout_path"]).read_bytes()

    def bash(self, command: str, timeout_s=120) -> dict:
        # Login shells reset Docker's PATH and select the system Python instead
        # of the independently provisioned Newton environment.
        result = self._exec(["bash", "-o", "pipefail", "-c", command], timeout_s=timeout_s)
        return result

    @staticmethod
    def _path(path: str, *, write=False) -> str:
        p = PurePosixPath(path)
        if not p.is_absolute():
            p = PurePosixPath("/work") / p
        if ".." in p.parts or str(p) == "/work" or not (
            str(p).startswith("/work/") or (not write and str(p).startswith("/input/"))
        ):
            raise ValueError("Path must be a file under /work" + (" or /input" if not write else ""))
        return str(p)

    def _file(self, path: str, content: str | bytes | None = None) -> dict:
        path = self._path(path, write=content is not None)
        # Resolve inside the container, so symlink races cannot reach host files.
        script = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); "
                  "assert p.is_relative_to('/work')" + ("" if content is not None else " or p.is_relative_to('/input')") + "; ")
        if content is None:
            script += "sys.stdout.buffer.write(p.read_bytes())"
        else:
            script += "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
        result = self._exec(["python", "-c", script, path],
                            stdin=content.encode() if isinstance(content,str) else content,
                            timeout_s=min(120,int(self.config.get('max_tool_timeout_s',1800))))
        if result["exit_code"]:
            raise SandboxError(result["stderr"])
        return result

    def write_file(self, path: str, content: str) -> dict:
        result = self._file(path, content)
        return {**result, "path": self._path(path, write=True)}

    def write_bytes(self, path: str, content: bytes) -> dict:
        return {**self._file(path, content), 'path':self._path(path, write=True)}

    def read_file(self, path: str) -> str:
        return self._full_stdout(self._file(path)).decode("utf-8")

    def image(self, path: str) -> bytes:
        result = self._file(path)
        maximum = int(self.config.get("max_image_bytes", 20000000))
        return self._full_stdout(result, maximum=maximum)

    def execute_newton(self, code: str, *, inspection=None) -> dict:
        self.write_file("scene.py", code)
        output_dir = "/work/renders/" + uuid.uuid4().hex
        maximum = int(self.config.get('max_tool_timeout_s',1800))
        explicit = self.config.get('newton_timeout_s')
        command = ["env","WORLD_RSI_RENDER_PROGRESS=1","python","-m",
                   "harness.tools.newton.workers.entry","/work/scene.py",output_dir]
        if inspection is not None:
            command.append(json.dumps(inspection, allow_nan=False))
        result = self._exec(command,
                            timeout_s=maximum if explicit is None else min(int(explicit),maximum), formal_render=True)
        result["output_dir"] = output_dir
        result["manifest_path"] = output_dir + "/render_manifest.json"
        if result["exit_code"] == 0:
            result["manifest"] = json.loads(self.read_file(result["manifest_path"]))
        return result

    def run_media_tool(self, name: str, arguments: dict, schema: dict, *, interpreter: str = 'python') -> dict:
        from harness.tools._media import validate_arguments
        if name not in {'probe_media','extract_frames','crop_image','estimate_depth','segment_image'}:
            raise ValueError('Unknown media tool: '+str(name))
        validate_arguments(name,arguments,schema)
        if interpreter not in {'python','perception-python'}:
            raise ValueError('Unsupported media worker interpreter')
        payload = json.dumps(arguments, ensure_ascii=False,allow_nan=False).encode()
        result = self._exec([interpreter, f'/opt/worldrsi/{name}.py'], stdin=payload,
                            timeout_s=int(self.config.get("vision_timeout_s", 600)))
        if result["exit_code"] == 0:
            try:
                result["result"] = json.loads(self._full_stdout(result))
            except ValueError:
                result["parse_error"] = "Media worker did not return JSON; see full stdout log"
        return result
