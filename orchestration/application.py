"""Application composition. Runtime and tools never own task scheduling policy."""
from dataclasses import dataclass
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

from orchestration.models import TaskContext
from orchestration.config import HarnessSpec, validate_config
from orchestration.storage.repository import ArtifactStore, sha
from orchestration.scheduler import Scheduler
from orchestration.evolution.versions import VersionStore
from orchestration.evolution.acceptance import Acceptance
from harness.core.contracts import EnvironmentChanged
from harness.sandbox.docker import Sandbox
from harness.tools import builtin, world_state
from orchestration.viewer import LivePool
from evaluation.evaluator import Evaluator


@dataclass
class Services:
    app: object
    spec: HarnessSpec
    sandbox_factory: object
    evaluator_factory: object
    tools: object = builtin
    world: object = None
    renderer: object = None
    acceptance: object = None

    @property
    def root(self): return self.app.store.root
    @property
    def store(self): return self.app.store
    @property
    def versions(self): return self.app.versions
    def stop_requested(self): return self.app.scheduler.stop.is_set()
    def checkpoint(self, record): self.app.scheduler.checkpoint(record)


class Application:
    components = [{'kind': k, 'name': n} for k,n in (
        ('runtime','stirrup-source'),('sandbox','docker'),('renderer','newton-warp'),
        ('evaluator','4dcodebench-and-task-tokens'))]

    def __init__(self, store=None, *, versions=None):
        self.store = store or ArtifactStore()
        self._runtime_code = self.code_identity()
        self._runtime_dependencies = self.runtime_identity()
        self.versions = versions or VersionStore(self.store.data/'skills/repository')
        self.scheduler = Scheduler(self.store)
        from orchestration.tasks import TaskRunner
        from orchestration.campaigns import CampaignRunner
        from orchestration.evolution.trials import Trials
        from orchestration.threats import ThreatRunner
        from orchestration.rollouts import RolloutRunner
        self.tasks, self.campaigns, self.trials = TaskRunner(self), CampaignRunner(self), Trials(self)
        self.threats = ThreatRunner(self)
        self.rollouts = RolloutRunner(self)
        self.live = LivePool(self.store, self.bind(self.store.config()).sandbox_factory)

    def bind(self, cfg, frozen=None):
        spec = HarnessSpec.parse(frozen if frozen is not None else cfg.get('harness'))
        provider = Sandbox.for_store(self.store)
        def sandbox(directory, config, **kwargs):
            config = {**config, 'sandbox': {**config['sandbox'],
                'resources_path': str(self.store.root/'data/resources'),
                'context_documents': list(spec.documents)}}
            return provider(directory, config, **kwargs)
        sandbox.health, sandbox.reap_orphans = provider.health, provider.reap_orphans
        return Services(self, spec, sandbox,
            lambda context: Evaluator(context.config['evaluation'], self.store.root/'data/resources'),
            world=SimpleNamespace(validate=world_state.validate_world, align=world_state.validate_code),
            renderer=SimpleNamespace(execute=lambda box,code,**options:box.execute_newton(code,**options)),
            acceptance=Acceptance())

    def context(self, task_id):
        record = self.store.task(task_id)
        return TaskContext(record, self.store.task_path(task_id),
            self.bind(record['config'],record.get('harness',{}).get('spec')))

    def requirements(self, cfg, campaign=False):
        missing = validate_config(cfg)
        if not self.store.api_key(): missing.append('WORLD_RSI_API_KEY')
        return missing

    def code_identity(self):
        return {str(p.relative_to(self.store.root)):sha(p)
                for folder in ('harness','evaluation','orchestration','web')
                for p in (self.store.root/folder).rglob('*.py')}

    def runtime_identity(self):
        paths = list((self.store.root/'harness/core').glob('*.py')) + list((self.store.root/'harness/prompts').glob('*.txt'))
        files = {str(p.relative_to(self.store.root)):sha(p) for p in paths}
        lock = self.store.root/'requirements-controller.lock'
        return {'framework':'stirrup-source', 'upstream_revision':'c092cee80ce09cd1dfb93aef053296a9ecd9dbb6',
            'source_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'python':platform.python_version(), 'lock_sha256':sha(lock) if lock.is_file() else None}

    def verify_runtime(self, record=None):
        if self.code_identity()!=self._runtime_code or self.runtime_identity()!=self._runtime_dependencies:
            raise EnvironmentChanged('Harness changed after controller startup; restart before executing tasks')
        if record is None: return
        frozen = record.get('harness') or {}
        if frozen.get('code')!=self._runtime_code or frozen.get('runtime')!=self._runtime_dependencies:
            raise EnvironmentChanged('Task belongs to a different frozen harness; create a new task')
        for path,digest in frozen.get('documents',{}).items():
            if not (self.store.root/path).is_file() or sha(self.store.root/path)!=digest:
                raise EnvironmentChanged('Frozen document changed: '+path)

    def prepare_execution(self, cfg, campaign=False, *, kind='reconstruction'):
        self.verify_runtime()
        missing = self.requirements(cfg,campaign)
        if missing: raise ValueError('Configuration required: '+', '.join(missing))
        box = self.bind(cfg).sandbox_factory
        health = box.health(cfg['sandbox']['image'])
        if not health['available']:raise ValueError('Sandbox unavailable: '+str(health.get('error')))
        identity = sha(self.store.root/'data/resources/newton-identity.json')
        expected = cfg['sandbox'].get('resource_identity_sha256')
        if expected and expected!=identity:raise EnvironmentChanged('Frozen resources changed')
        if kind == 'threat':
            return {**cfg, 'sandbox':{**cfg['sandbox'],'image':health['image_id'],'resource_identity_sha256':identity}}
        if kind != 'reconstruction':raise ValueError('Unknown task kind')
        scoring = box.health(cfg['evaluation'].get('image','worldrsi:local'))
        if not scoring['available']:raise ValueError('Scorer image unavailable')
        return {**cfg, 'evaluation':{**cfg['evaluation'],'image':scoring['image_id']},
            'sandbox':{**cfg['sandbox'],'image':health['image_id'],'resource_identity_sha256':identity}}

    def initialize(self):
        if self.scheduler.stop.is_set():
            raise RuntimeError('A shut-down Application cannot be reused; create a new Application')
        self.store.data.mkdir(parents=True,exist_ok=True)
        self.versions.initialize()
        self.tasks.recover()
        self.rollouts.recover()

    def harness_snapshot(self, services):
        self.verify_runtime()
        documents = {p:sha(self.store.root/p) for p in
            (*services.spec.documents,*services.spec.prompts.values())}
        if (self.store.root/'docs/newton-api.json').is_file():documents['docs/newton-api.json']=sha(self.store.root/'docs/newton-api.json')
        if (self.store.root/'harness/tools/world-state-v3.schema.json').is_file():documents['harness/tools/world-state-v3.schema.json']=sha(self.store.root/'harness/tools/world-state-v3.schema.json')
        identity = {'spec':services.spec.to_dict(),'documents':documents,
            'code':dict(self._runtime_code),'runtime':dict(self._runtime_dependencies)}
        result = subprocess.run(['git','-C',str(self.store.root),'rev-parse','HEAD'],
            capture_output=True,text=True,timeout=5)
        identity['repository_revision'] = result.stdout.strip() if result.returncode==0 else None
        return {**identity,'id':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
