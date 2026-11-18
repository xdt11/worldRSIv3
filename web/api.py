"""HTTP composition boundary; business state belongs to the injected application."""
from contextlib import asynccontextmanager
import asyncio
import fcntl
from functools import lru_cache, wraps
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, urlencode
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from orchestration.application import Application
from orchestration.config import DEFAULT_CONFIG
from orchestration.config import normalize_config
from orchestration.config import normalize_threat_config
from orchestration.config import merge_config
from orchestration.config import validate_config
from orchestration.storage.repository import identifier
from orchestration.storage.repository import inside
from orchestration.storage.repository import load
from orchestration.storage.repository import now
from orchestration.storage.repository import save
from orchestration.storage.repository import uid
from orchestration.storage.repository import sha
from harness.evidence.usage import summarize_usage
from harness.tools.newton.session import FrameRequestError, render_command
from web.media import encode_video, public_input_path, thumbnail_revision, input_thumbnail
from orchestration.reports import build_report
from orchestration.rollouts import rollout_review_enabled
from evaluation.groups import metric_groups
from harness.evidence.artifacts import archived_render
from orchestration.threats import SourceRevisionChanged


def create_app(application=None):
    application = application if application is not None else Application()
    store = application.store
    sandbox = application.bind(store.config()).sandbox_factory
    BIND_HOST = os.environ.get('WORLD_RSI_HOST', '127.0.0.1')
    extra_hosts = {host.strip() for host in os.environ.get('WORLD_RSI_LISTEN_HOSTS', '').split(',') if host.strip()}
    extra_hosts.update(host.strip() for host in os.environ.get('WORLD_RSI_ALLOWED_HOSTS', '').split(',') if host.strip())
    allowed_hosts = {'localhost', '127.0.0.1', '[', 'testserver', BIND_HOST} | extra_hosts
    PREVIEW_LOCK = threading.Lock()


    @asynccontextmanager
    async def lifespan(app):
        store.data.mkdir(parents=True, exist_ok=True)
        controller_lock = (store.data / 'controller.lock').open('a')
        try:
            fcntl.flock(controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            controller_lock.close()
            raise RuntimeError('Another worldRSI controller is using this data directory')
        try:
            application.initialize()
            sandbox.reap_orphans()
            application.live.start()
            yield
        finally:
            application.scheduler.shutdown(wait=True)
            application.live.close()
            sandbox.reap_orphans()
            controller_lock.close()


    app = FastAPI(title='worldRSIv3', lifespan=lifespan)
    app.state.application = application
    app.mount('/static', StaticFiles(directory=store.root / 'web'), name='static')


    @app.middleware('http')
    async def local_boundary(request: Request, call_next):
        # Allow the workbench address while retaining Host and same-origin checks.
        host = request.headers.get('host', '').split(':')[0]
        origin = request.headers.get('origin')
        if host not in allowed_hosts:
            return JSONResponse({'detail': 'Workbench host required'}, status_code=403)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin:
            parsed = urlparse(origin)
            if parsed.netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Same-origin request required'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'no-store' if request.url.path.startswith('/api') else 'no-cache'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
        return response


    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)


    @app.exception_handler(FileNotFoundError)
    async def not_found(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=404)


    @app.get('/')
    def index():
        return FileResponse(store.root / 'web/index.html')

    @app.get('/safe')
    @app.get('/safe.html')
    def safe_index():
        return FileResponse(store.root / 'web/safe.html')

    @app.get('/rollouts')
    @app.get('/rollouts.html')
    def rollout_index():
        return FileResponse(store.root / 'web/rollouts.html')


    @lru_cache(maxsize=4)
    def health_snapshot(bucket, image):
        return {'docker': sandbox.health(image),
                'disk': {'path': str(store.root), 'free_bytes': shutil.disk_usage(store.root).free},
                'api_key_configured': bool(store.api_key())}


    def versions():
        return [{**row, 'id': row['commit'], 'parent': next(iter(row['parents']), None),
                 **load(store.data / 'skills/candidates' / f'{row["commit"]}.json', {}),
                 'status': 'active' if row['active'] else load(store.data / 'skills/candidates' / f'{row["commit"]}.json', {}).get('status', 'initial')}
                for row in application.versions.history()]


    @app.get('/api/state')
    def state():
        return workspace_state('reconstruction')

    @app.get('/api/safe/state')
    def safe_state():
        return workspace_state('threat')

    def workspace_state(kind):
        scheduling=application.scheduler.snapshot()
        tasks = []
        for path in store.task_records(kind):
            value = application.tasks.accounting(load(path))
            summary = {key: value.get(key) for key in ('id', 'title', 'split', 'status', 'version',
                                                     'created_at', 'updated_at', 'error', 'parent_task_id', 'trial_id', 'usage',
                                                     'phase','resource_state','resource_phase','stop_requested','kind','source_scene')}
            summary.update(job=task_job(value['id'],scheduling['jobs']))
            summary['runs'] = [{key: run.get(key) for key in ('id', 'status', 'created_at', 'updated_at',
                                                            'usage', 'call_count', 'phase')} | {'render_revision': (run_archive(value['id'],run) or {}).get('render_revision')} for run in value['runs']]
            tasks.append(summary)
        campaigns = ([application.campaigns.accounting(load(path)) for path in (store.data / 'campaigns').glob('*.json')]
                     if kind == 'reconstruction' else [])
        cfg = (normalize_threat_config(defaults=store.config()) if kind == 'threat' else store.config())
        return {'workspace':kind, 'tasks': sorted(tasks, key=lambda r: r['created_at'], reverse=True),
                'versions': [{k: row.get(k) for k in ('id', 'parent', 'status', 'active', 'created_at',
                                                    'updated_at', 'summary', 'task_id', 'adoption', 'scope', 'campaign_id')}
                             for row in versions()],
                'campaigns': [{k: row.get(k) for k in ('id', 'status', 'created_at', 'updated_at', 'error', 'usage','job_id','job_key')}
                              for row in campaigns], 'config': cfg,
                'plugins': application.components,
                'metric_families':metric_groups({}),
                'harness': application.bind(cfg).spec.to_dict(),
                'scheduler': scheduling,
                'health': {**health_snapshot(int(time.time() // 30), cfg['sandbox']['image']),
                           'busy': scheduling['busy'], 'configuration_required': application.requirements(cfg),
                           'live_sessions': application.live.snapshot()}}


    @app.get('/api/inputs')
    def list_inputs():
        return [input_card(row) for row in store.inputs()]


    def input_card(row):
        benchmark = row.get('benchmark') or {}
        case = benchmark.get('case') or ''
        category = case.partition('/')[0] if case else 'manual'
        card = {**row,'display_name':case.rsplit('/',1)[-1] if case else Path(row['path']).name,
            'case_id':case or None,'category':category,
            'dataset':benchmark.get('dataset'),'thumbnail_url':None}
        if row['kind'] in {'image','video'}:
            try:
                revision=thumbnail_revision(public_input_path(store.data/'inputs',row['path']))
                card.update(thumbnail_revision=revision,thumbnail_url='/api/inputs/thumbnail?'+urlencode(
                    {'path':row['path'],'revision':revision}))
            except (OSError,ValueError):
                pass  # An unavailable preview never removes a selectable input.
        return card


    @app.get('/api/inputs/thumbnail')
    def public_thumbnail(path: str, revision: str | None = None):
        source = public_input_path(store.data/'inputs',path)
        actual = thumbnail_revision(source)
        if revision and revision != actual:
            raise HTTPException(status_code=409,detail='Input changed; refresh the sample list')
        try:
            target=input_thumbnail(source,store.data/'thumbnails',actual)
        except TimeoutError as exc:
            raise HTTPException(status_code=503,detail=str(exc),headers={'Retry-After':'1'}) from exc
        return FileResponse(target,media_type='image/jpeg',headers={
            'ETag':'"'+actual+'"','Cache-Control':'private, max-age=31536000, immutable' if revision else 'no-cache, no-store'})


    @app.post('/api/inputs')
    def upload_input(file: UploadFile = File(...)):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov', '.webm', '.mkv'}:
            raise ValueError('Upload an image or video')
        target = store.data / 'inputs/manual' / (uid('media') + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            with target.open('wb') as out:
                for block in iter(lambda: file.file.read(1024 * 1024), b''):
                    count += len(block)
                    if count > 512 * 1024 * 1024:
                        raise ValueError('File exceeds the 512 MiB upload limit')
                    out.write(block)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return input_card(next(r for r in store.inputs() if r['path'] == str(target.relative_to(store.data / 'inputs'))))


    class NewTask(BaseModel):
        title: str = ''
        instruction: str = ''
        inputs: list[str] = Field(default_factory=list)
        split: str = 'manual'
        reference: str | None = None
        config: dict | None = None

    class NewThreatTask(BaseModel):
        title: str = ''
        instruction: str
        source_task_id: str
        source_run_id: str
        source_revision: str
        config: dict | None = None


    def serialized(action):
        @wraps(action)
        def guarded(*args, **kwargs):
            with store.lock:
                return action(*args, **kwargs)
        return guarded


    def settings(base, patch=None):
        patch = patch or {}
        if not isinstance(patch, dict) or set(patch) - set(DEFAULT_CONFIG):
            raise ValueError('Unknown configuration keys; credentials belong in the private .env')
        value = merge_config(base, patch)
        validate_config(value)
        application.bind(value)  # Validate selected plugin names before persisting settings.
        return value


    def task_job(task_id,jobs=None):
        if jobs is None:jobs=application.scheduler.snapshot()['jobs']
        return next((row for row in reversed(jobs) if row['status'] in {'queued','running'} and
            (row['key']=='task:'+task_id or row['kind'] in {'rollout','rollout_review'} and task_id in row['task_ids'])),None)


    def candidate_guard(version, metadata):
        task_ids=(metadata['task_id'],) if metadata.get('task_id') else ()
        return application.scheduler.mutation_guard(task_ids=task_ids,conflict_keys=('candidate:'+version,))


    def managed_candidate(version, metadata):
        campaign_id=metadata.get('campaign_id')
        if not campaign_id and metadata.get('task_id'):
            try: campaign_id=store.task(metadata['task_id']).get('campaign_id')
            except FileNotFoundError: pass
        record=load(store.data/'campaigns'/f'{campaign_id}.json',{}) if campaign_id else {}
        return bool(campaign_id and record.get('pending_candidate')==version and
            application.scheduler.is_busy(key='campaign:'+campaign_id))


    def require_candidate_decision(version, metadata):
        if managed_candidate(version,metadata):
            raise ValueError('This candidate belongs to a live experiment; use its confirmation control')


    @app.get('/api/jobs')
    def jobs():
        return application.scheduler.snapshot()


    def cancel_job(identity):
        job=next((row for row in reversed(application.scheduler.snapshot()['jobs']) if
            (row['id']==identity or row['key']==identity) and row['status'] in {'queued','running'}),None)
        if job is None:raise FileNotFoundError(identity)
        result=application.scheduler.cancel(job['id'])
        if result['kind']=='task':
            for task_id in result['task_ids']:
                store.update_task(task_id,stop_requested=True)
        if result['kind']=='rollout' and result['status']=='queued' and result['key'].startswith('rollout:'):
            identity=result['key'].removeprefix('rollout:')
            application.rollouts._update(result['task_ids'][0],identity,status='interrupted',phase='finished',
                task_critic_status='skipped',error='Job cancelled before experiment startup',finished_at=now())
        for path in (store.data/'campaigns').glob('*.json'):
            record=load(path,{})
            if record.get('job_id')!=job['id']:continue
            if result['status'] in {'queued','cancelled'}:
                record.update(status='stopped',updated_at=now());save(path,record)
            elif record['id'] in application.scheduler.controls:
                application.campaigns.action(record['id'],'stop')
        if result['kind']=='trial' and result['status'] in {'queued','cancelled'}:
            for task_id in result['task_ids']:
                source=store.task(task_id)
                for trial in source.get('trials',[]):
                    if trial.get('job_id')==job['id']:
                        trial.update(status='stopped',updated_at=now())
                        application.trials.save(task_id,trial['candidate_version'],trial)
        return result


    @app.post('/api/jobs/{job_id}/cancel')
    @serialized
    def cancel_scheduled_job(job_id: str):
        return cancel_job(job_id)


    def latest_scene(value):
        for run in reversed(value['runs']):
            archive = run.get('archived_render')
            if archive:
                return {**archive, 'run_id':run['id'],
                        'is_latest_run':run['id'] == value['runs'][-1]['id']}
        return None


    def run_archive(task_id, run):
        try:
            return archived_render(store.task_path(task_id), run)
        except (ValueError, OSError, KeyError, TypeError):
            return None


    @app.post('/api/tasks')
    @serialized
    def new_task(body: NewTask):
        if body.split != 'manual':
            raise ValueError('Evolution/test split is controlled by the fixed campaign')
        return application.tasks.create(body.title, body.instruction, body.inputs, reference=body.reference,
                                  cfg=settings(store.config(), body.config))

    @app.get('/api/safe/archives')
    def source_archives():
        rows = application.threats.list_archives()
        for row in rows:
            row['thumbnail_url'] = '/api/tasks/' + row['source_task_id'] + '/artifact?' + urlencode({'path':row['preview_path']})
        return rows

    @app.post('/api/safe/tasks')
    @serialized
    def new_threat_task(body: NewThreatTask):
        cfg = settings(normalize_threat_config(body.config, defaults=store.config()))
        cfg['trial_enabled'] = False
        try:
            return application.threats.create(body.title, body.instruction,
                body.source_task_id, body.source_run_id, body.source_revision, cfg=cfg)
        except SourceRevisionChanged as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


    @app.get('/api/rollouts')
    def list_rollouts(task_id: str | None = None):
        if task_id is not None:
            task_id = identifier(task_id)
        return {'rollouts': application.rollouts.list(task_id=task_id)}

    @app.get('/api/rollouts/{rollout_id}')
    def rollout_detail(rollout_id: str):
        return application.rollouts.get(identifier(rollout_id))

    @app.get('/api/rollouts/{rollout_id}/evidence')
    def rollout_evidence(rollout_id: str):
        return application.rollouts.evidence(identifier(rollout_id))

    @app.post('/api/rollouts/{rollout_id}/review')
    @serialized
    def review_rollout(rollout_id: str):
        record=application.rollouts.get(identifier(rollout_id))
        task_id=record['task_id']
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            if record['status'] not in {'completed','failed','interrupted'}:
                raise ValueError('Rollout is still active')
            task=store.task(task_id)
            if not rollout_review_enabled(task):
                raise ValueError('Task RSI is disabled for this task')
            application.verify_runtime()
            job=application.scheduler.schedule(application.rollouts.review,task_id,rollout_id,
                key='task:'+task_id,kind='rollout_review',task_ids=(task_id,))
            return {'status':'scheduled','rollout_id':rollout_id,'job':job}

    @app.get('/api/rollouts/{rollout_id}/artifact')
    def rollout_artifact(rollout_id: str, path: str):
        target = application.rollouts.artifact(identifier(rollout_id), path)
        if not target.is_file():
            raise FileNotFoundError(path)
        mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if mime in {'text/html', 'image/svg+xml', 'application/javascript'}:
            mime = 'text/plain'
        return FileResponse(target, media_type=mime)

    @app.post('/api/tasks/{task_id}/reassess-scene')
    @serialized
    def reassess_scene(task_id: str):
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            store.task(task_id)
            job=application.scheduler.schedule(application.tasks.reassess_scene,task_id,
                key='task:'+task_id,kind='scene_review',task_ids=(task_id,))
            return {'status':'scheduled','task_id':task_id,'job':job}

    @app.post('/api/tasks/{task_id}/rollout')
    @serialized
    def run_rollout(task_id: str):
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            value = store.task(task_id)
            if value.get('kind') != 'threat' or value.get('status') != 'completed':
                raise ValueError('Choose a completed safeRSI scene for Go2 evaluation')
            archive = next((candidate for run in reversed(value.get('runs', []))
                            if run.get('status') == 'completed'
                            and (candidate := archived_render(store.task_path(task_id), run, for_replay=True))
                            and candidate.get('engine') == 'newton' and candidate.get('is_current')), None)
            if archive is None:
                raise ValueError('This scene has no current native Newton archive to evaluate')
            # Evaluation uses the current environment independently of the frozen
            # generation harness; never rewrite or re-run the source task.
            current = store.config()
            cfg = merge_config(value['config'], {key: current[key]
                for key in ('base_url', 'models', 'model', 'critic_model') if current.get(key) is not None})
            cfg['rollout'] = current['rollout']
            cfg['harness'] = {}
            cfg['sandbox'].pop('resource_identity_sha256', None)
            application.prepare_execution(cfg, kind='threat')
            job = application.scheduler.schedule(application.rollouts.run, task_id,
                key='task:'+task_id, kind='rollout', task_ids=(task_id,))
            return {'status':'scheduled', 'task_id':task_id, 'job':job}


    def invocation_tools(task_id, *, run_id=None, rsi=False):
        root = store.task_path(task_id)
        events = []
        for path in (root / 'invocations').glob('*/invocation.json'):
            record = load(path, {})
            role = record.get('role') or record.get('stage')
            if (role in {'task_critic', 'task_rsi_critic'}) != rsi:
                continue
            if run_id is not None and record.get('run_id') != run_id:
                continue
            events.extend(load(path.with_name('tools.json'), []))
        return sorted(events, key=lambda event: event.get('created_at', ''))

    @app.get('/api/tasks/{task_id}')
    def task_detail(task_id: str):
        value = store.task(task_id)
        # Render current task accounting from immutable invocation receipts.
        calls = store.calls(task_id)
        value = application.tasks.accounting(value,calls)
        value['runs'] = [{**run, 'archived_render':run_archive(task_id, run)} for run in value['runs']]
        value['trials'] = [application.trials.accounting(row) for row in value.get('trials',[])]
        value['task_calls'] = sorted([c for c in calls if c['stage'] == 'task_critic'],
                                    key=lambda c: c['created_at'])
        value['task_tools'] = invocation_tools(task_id, rsi=True)
        return {**value, 'metric_groups':[] if value.get('kind')=='threat' else metric_groups(value.get('metrics') or {}), 'job':task_job(task_id), 'config': normalize_config(value['config']) if not value['runs'] else value['config'],
                'latest_scene': latest_scene(value)}


    @app.patch('/api/tasks/{task_id}/config')
    @serialized
    def task_settings(task_id: str, body: dict):
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            value = store.task(task_id)
            if value['status'] in {'running', 'finishing', 'completed'} or value.get('finalized_at'):
                raise ValueError('Task settings are frozen while running or after completion')
            if value['runs'] and set(body) - {'outer_rsi', 'trial_enabled'}:
                raise ValueError('Execution settings are fixed after the first run; RSI and trial switches remain editable')
            base = value['config'] if value['runs'] else normalize_config(value['config'])
            cfg = settings(base, body)
            if value.get('kind') == 'threat':
                if cfg.get('trial_enabled'):
                    raise ValueError('Threat-scene rollout trials belong to the next stage')
                cfg['trial_enabled'] = False
            changes = {} if value['runs'] else {'harness':application.harness_snapshot(application.bind(cfg))}
            return store.update_task(task_id, config=cfg, **changes)


    @app.get('/api/tasks/{task_id}/runs/{run_id}')
    def run_detail(task_id: str, run_id: str):
        value = store.task(task_id)
        run = next((r for r in value['runs'] if r['id'] == run_id), None)
        if run is None:
            raise FileNotFoundError(run_id)
        calls = store.calls(task_id)
        selected = [c for c in calls if c['run_id'] == run_id and c['stage'] != 'task_critic']
        tools = invocation_tools(task_id, run_id=run_id)
        # Keep raw events immutable while presenting explicit execution errors.
        tools = [{**event, 'status': 'failed'} if event.get('result', {}).get('exit_code', 0) != 0
                 or event.get('result', {}).get('error') else event for event in tools]
        return {**run, 'calls': sorted(selected, key=lambda r: r['created_at']),
                'archived_render':run_archive(task_id, run),
                'usage': summarize_usage(selected), 'call_count': len(selected),
                'usage_scope': 'agent_and_run_critic',
                'tools': sorted(tools, key=lambda e: e.get('created_at', ''))}


    @app.get('/api/tasks/{task_id}/calls/{call_id}/{kind}')
    def call_detail(task_id: str, call_id: str, kind: str):
        store.task(task_id)
        if kind not in {'request', 'response', 'meta'}:
            raise ValueError('kind must be request, response or meta')
        record = next((row for row in store.calls(task_id) if row['id'] == call_id), None)
        if record is None:
            raise FileNotFoundError(call_id)
        relative = record.get(kind + '_path')
        if relative is None:
            raise FileNotFoundError(kind + ' receipt path')
        path = inside(store.task_path(task_id), relative)
        if not path.is_file():
            raise FileNotFoundError(str(path.relative_to(store.task_path(task_id))))
        return FileResponse(path, media_type='application/json')


    @app.websocket('/api/tasks/{task_id}/runs/{run_id}/live')
    async def native_live(websocket: WebSocket, task_id: str, run_id: str):
        host = websocket.headers.get('host', '')
        origin = websocket.headers.get('origin')
        if host.split(':')[0] not in allowed_hosts or (
                origin and urlparse(origin).netloc != host):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = None
        latest = asyncio.Queue(maxsize=1)
        disconnected = asyncio.Event()
        async def receive_latest():
            try:
                while True:
                    message = await websocket.receive()
                    if message['type'] == 'websocket.disconnect':
                        break
                    try:
                        raw = message.get('text')
                        if not isinstance(raw,str):
                            raise ValueError('Render requests must be JSON text')
                        if len(raw) > 16384:
                            raise ValueError('Render request exceeds 16 KiB')
                        request = render_command(json.loads(raw), store.config()['live'])
                    except (ValueError, TypeError) as exc:
                        request = exc
                    if latest.full():
                        latest.get_nowait()
                    latest.put_nowait(request)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                disconnected.set()
                if latest.empty():
                    latest.put_nowait(None)
        async def reject_frame(exc):
            await websocket.send_json({'type':'frame_error','status':'rejected',
                'recoverable':True,'detail':str(exc)})
        reader = asyncio.create_task(receive_latest())
        try:
            await websocket.send_json({'type': 'status', 'status': 'starting',
                'detail': '正在载入 Newton 世界和 Warp 渲染器；首次使用需要编译预热。'})
            session = await asyncio.to_thread(application.live.reserve, task_id, run_id)
            expected_revision = websocket.query_params.get('revision')
            if expected_revision and expected_revision != getattr(session,'archive_revision',None):
                raise ValueError('The run has a newer native output; refresh this view')
            initialization = asyncio.create_task(asyncio.to_thread(session.initialize))
            departure = asyncio.create_task(disconnected.wait())
            async def progress():
                began = time.monotonic()
                while not initialization.done() and not disconnected.is_set():
                    await asyncio.sleep(2)
                    if not initialization.done() and not disconnected.is_set():
                        try:
                            await websocket.send_json({'type':'status','status':'preparing',
                                'detail':f'Newton/Warp: {getattr(session,"phase","building")} · {time.monotonic()-began:.0f}s. 场景构建会在后台复用。'})
                        except (WebSocketDisconnect,RuntimeError):
                            return
            progress_task = asyncio.create_task(progress())
            done, _ = await asyncio.wait([initialization, departure], return_when=asyncio.FIRST_COMPLETED)
            progress_task.cancel()

            if departure in done:
                # The thread finishes cancellation and its own container cleanup.
                initialization.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
                return
            departure.cancel()
            await initialization
            await websocket.send_json(session.ready)
            last_frame = 0
            while not disconnected.is_set():
                command = await latest.get()
                if command is None:
                    break
                if isinstance(command, Exception):
                    await reject_frame(command)
                    continue
                await asyncio.sleep(max(0, 1 / session.settings['max_fps'] - (time.monotonic() - last_frame)))
                # Drain stale timestamps/camera positions after the frame-rate wait.
                while not latest.empty():
                    command = latest.get_nowait()
                if disconnected.is_set() or command is None:
                    break
                if isinstance(command, Exception):
                    await reject_frame(command)
                    continue
                try:
                    metadata, jpeg = await asyncio.to_thread(session.render, command)
                except FrameRequestError as exc:
                    await reject_frame(exc)
                    continue
                if disconnected.is_set():
                    break
                await websocket.send_json(metadata)
                await websocket.send_bytes(jpeg)
                last_frame = time.monotonic()
        except (WebSocketDisconnect, RuntimeError) as exc:
            if not disconnected.is_set():
                try:
                    await websocket.send_json({'type': 'status', 'status': 'failed', 'detail': str(exc)})
                except (RuntimeError, WebSocketDisconnect):
                    pass
        except Exception as exc:
            if not disconnected.is_set():
                await websocket.send_json({'type': 'status', 'status': 'failed', 'detail': str(exc)})
        finally:
            reader.cancel()
            if session:
                application.live.release(session)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass


    @app.get('/api/tasks/{task_id}/runs/{run_id}/video')
    def download_video(task_id: str, run_id: str, format: str = 'mp4', revision: str | None = None):
        if format not in {'mp4', 'webm'}:
            raise ValueError('Archive video format must be mp4 or webm')
        value = store.task(task_id)
        run = next((r for r in value['runs'] if r['id'] == run_id), None)
        archive = run_archive(task_id, run)
        if not archive:
            raise ValueError('This run has no native renders to export')
        if revision and revision != archive['render_revision']:
            return JSONResponse({'detail':'This run has a newer native output; reload the archive'}, status_code=409)
        render_dir = inside(store.task_path(task_id), archive['render_dir'])
        frames = load(render_dir / 'render_manifest.json', {}).get('frames', [])
        video, receipt = encode_video(render_dir, store.data / 'videos', duration_s=1 if len(frames) == 1 else None,
                                      video_format=format)
        return FileResponse(video, media_type='video/' + format, filename=f'{task_id}-{run_id}.{format}',
            headers={'Cache-Control': 'private, max-age=31536000, immutable' if revision else 'no-cache, no-store',
                     'X-Render-Revision': archive['render_revision'],
                     'X-Render-Directory': archive['render_dir'],
                     'X-Video-Scope': 'Archived native camera; original renderer timestamps',
                     'X-Render-Current': str(archive['is_current']).lower()})


    @app.get('/api/tasks/{task_id}/artifact')
    def artifact(task_id: str, path: str):
        value = store.task(task_id)
        target = inside(store.task_path(task_id), path)
        relative = target.relative_to(store.task_path(task_id))
        source = value.get('kind') == 'threat' and (
            relative.as_posix() in {'source/preview.png', 'source/manifest.json', 'source/runtime_state.json', 'source/render_frames.json'}
            or relative.parts[:2] == ('source', 'scene'))
        if not relative.parts or (not source and relative.parts[0] not in {'work', 'runs', 'input', 'inputs', 'invocations', 'tool_logs', 'evaluation', 'evaluations'}):
            raise ValueError('Artifact path is outside task evidence')
        if not target.is_file():
            raise FileNotFoundError(path)
        mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if mime in {'text/html', 'image/svg+xml', 'application/javascript'}:
            mime = 'text/plain'
        immutable_media = source or relative.parts[0] == 'inputs' or (relative.parts[0] == 'runs'
                          and target.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.webm', '.glb'})
        return FileResponse(target, media_type=mime,
                            headers={'Cache-Control': 'private, max-age=31536000, immutable'} if immutable_media else None)


    @app.get('/api/tasks/{task_id}/preview')
    def browser_preview(task_id: str, path: str):
        """Codec fallback only; original inputs and evaluator frames stay untouched."""
        value = store.task(task_id)
        source = next((o for o in value['inputs'] if o['local_path'] == path and o['kind'] == 'video'), None)
        if source is None:
            raise ValueError('Select an archived video input')
        original = inside(store.task_path(task_id), path)
        cache = store.data / 'previews' / (sha(original) + '-vp9-960-v1.webm')
        with PREVIEW_LOCK:
            if not cache.is_file():
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix('.tmp.webm')
                try:
                    result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(original),
                        '-map', '0:v:0', '-vf', 'scale=min(960\\,iw):-2', '-an', '-c:v', 'libvpx-vp9',
                        '-deadline', 'realtime', '-cpu-used', '6', '-threads', '2', '-crf', '32', '-b:v', '0',
                        str(temporary)], capture_output=True, timeout=180)
                    if result.returncode:
                        raise ValueError('Browser preview encoding failed: ' + result.stderr.decode(errors='replace')[-2000:])
                    os.replace(temporary, cache)
                finally:
                    temporary.unlink(missing_ok=True)
        return FileResponse(cache, media_type='video/webm',
                            headers={'Cache-Control': 'private, max-age=31536000, immutable'})


    require_execution = application.prepare_execution

    @app.post('/api/tasks/{task_id}/run')
    @serialized
    def start_run(task_id: str):
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            value = store.task(task_id)
            application.verify_runtime(value if value['runs'] else None)
            if value['status'] not in {'ready', 'interrupted'}:
                raise ValueError('Task is not ready for another run')
            # A task's form choices take priority over global defaults.
            chosen = value['config'] if value['runs'] else normalize_config(value['config'])
            pinned = (require_execution(chosen, kind='threat') if value.get('kind') == 'threat' else require_execution(chosen))
            if not value['runs']:
                value = store.update_task(task_id, config=pinned,
                    harness=application.harness_snapshot(application.bind(pinned)))
            if len(value['runs']) >= value['config']['max_runs']:
                raise ValueError('Task reached max_runs')
            store.update_task(task_id,stop_requested=False)
            job=application.scheduler.schedule(application.tasks.execute,task_id,key='task:'+task_id,kind='task',task_ids=(task_id,))
            return {'status': 'scheduled', 'task_id': task_id, 'execution': 'remaining_runs_then_full_evaluation','job':job}


    @app.post('/api/tasks/{task_id}/stop')
    @serialized
    def stop_run(task_id: str):
        value = store.task(task_id)
        job=task_job(task_id)
        if job:
            result=cancel_job(job['id'])
            return {'status':'cancelled' if result['status'] in {'queued','cancelled'} else 'stopping','task_id':task_id,'job':result}
        if value['status'] not in {'running','finishing'} and not application.scheduler.is_busy(task_ids=(task_id,)):
            raise ValueError('This task has no active agent run')
        store.update_task(task_id, stop_requested=True)
        application.scheduler.resources.cancel(task_id)
        return {'status': 'stopping', 'task_id': task_id,
                'detail': 'The current provider call may finish before the run stops'}


    class FinishTask(BaseModel):
        outer_rsi: bool | None = None
        trial_enabled: bool | None = None


    @app.post('/api/tasks/{task_id}/finish')
    @serialized
    def finish(task_id: str, body: FinishTask = FinishTask()):
        with application.scheduler.mutation_guard(task_ids=(task_id,)):
            value = store.task(task_id)
            application.verify_runtime(value)
            if not value['runs'] or value['status'] in {'running', 'finishing', 'completed'} or value.get('finalized_at'):
                raise ValueError('Task cannot be finished in its current state')
            changes = body.model_dump(exclude_none=True)
            if changes:
                if value.get('kind') == 'threat' and changes.get('trial_enabled'):
                    raise ValueError('Threat-scene rollout trials belong to the next stage')
                store.update_task(task_id, config=settings(value['config'], changes))
            store.update_task(task_id,stop_requested=False)
            job=application.scheduler.schedule(application.tasks.finish,task_id,key='task:'+task_id,kind='task',task_ids=(task_id,))
            return {'status':'scheduled','task_id':task_id,'job':job}


    @app.get('/api/versions/{version}')
    def version_detail(version: str):
        snapshot = application.versions.snapshot(version)
        row = next((v for v in versions() if v['id'] == version), None)
        if row is None:
            raise FileNotFoundError(version)
        return {**row, 'campaign_managed':managed_candidate(version,row),
                'trials':[application.trials.accounting(trial) for trial in row.get('trials',[])],
                'components': snapshot,
                'diff': application.versions.diff(row['parent'], version) if row['parent'] else '',
                'empty_initial_components': ['skills'] if not row['parent'] else []}


    @app.get('/api/campaigns/{campaign_id}')
    def campaign_detail(campaign_id: str):
        result = load(store.data / 'campaigns' / f'{identifier(campaign_id)}.json')
        if result is None:
            raise FileNotFoundError(campaign_id)
        return application.campaigns.accounting(result)

    @app.get('/api/campaigns/{campaign_id}/report')
    def campaign_report(campaign_id: str):
        return build_report(application, campaign_id)

    @app.post('/api/campaigns/{campaign_id}/{action}')
    @serialized
    def campaign_control(campaign_id: str, action: str):
        if action not in {'pause', 'resume', 'stop', 'confirm', 'continue-baseline'}: raise ValueError('Unknown experiment action')
        identity = identifier(campaign_id)
        record=load(store.data / 'campaigns' / f'{identity}.json')
        if not record: raise FileNotFoundError(identity)
        if action=='continue-baseline':
            cfg,receipt=application.campaigns.baseline_continuation(identity)
            cfg=require_execution(cfg,campaign=True)
            record.update(status='scheduled',error=None,updated_at=now())
            return schedule_campaign(record,application.campaigns.run,cfg,receipt,
                                     key='campaign:'+identity,kind='campaign')
        job=next((row for row in application.scheduler.snapshot()['jobs'] if
            row['id']==record.get('job_id') and row['status'] in {'queued','running'}),None)
        if job and action=='stop':
            cancel_job(job['id'])
            return load(store.data/'campaigns'/f'{identity}.json')
        if job and job['status']=='queued':
            raise ValueError('This experiment is queued; it can be stopped before execution')
        return application.campaigns.action(identity,action)


    @app.post('/api/versions/{version}/evaluate')
    @serialized
    def evaluate_version(version: str):
        row=candidate_record(version)
        with candidate_guard(version,row):
            cfg = require_execution(store.config(), campaign=True)
            identity = uid('evaluation')
            record = {'id': identity, 'kind': 'external_evaluation', 'status': 'scheduled',
                      'created_at': now(), 'updated_at': now(), 'config': cfg, 'rounds': [], 'progress': []}
            return schedule_campaign(record,application.campaigns.evaluate,version,cfg,
                key='evaluation:'+version,kind='evaluation',conflict_keys=('candidate:'+version,))


    class VersionAction(BaseModel):
        task_id: str | None = None
        reason: str = ''


    def candidate_record(version):
        application.versions.snapshot(version)
        value = load(store.data / 'skills/candidates' / f'{version}.json')
        if not value:
            raise ValueError('This version is not an RSI candidate')
        return value


    @app.post('/api/versions/{version}/trial')
    @serialized
    def trial_version(version: str, body: VersionAction = VersionAction()):
        metadata = candidate_record(version)
        with candidate_guard(version,metadata):
            task_id = body.task_id or metadata.get('task_id')
            if not task_id or task_id != metadata.get('task_id'):
                raise ValueError('Use the original task that proposed this candidate')
            source = store.task(task_id)
            if source.get('kind') == 'threat':
                raise ValueError('Threat-scene rollout trials belong to the next stage')
            if source['status'] != 'completed' or not source['runs'] or metadata['parent'] != source['version']:
                raise ValueError('Complete the original parent-version task before trying its candidate')
            require_execution(source['config'])
            identity = uid('trial')
            record = {'id': identity, 'candidate_version': version, 'status': 'scheduled', 'created_at': now()}
            store.update_task(task_id, trials=[*source.get('trials', []), record])
            metadata.update(trials=[*metadata.get('trials', []), record], updated_at=now())
            save(store.data / 'skills/candidates' / f'{version}.json', metadata)
            try:
                job=application.scheduler.schedule(application.trials.run,task_id,version,identity,key='trial:'+version,kind='trial',task_ids=(task_id,),conflict_keys=('candidate:'+version,))
                record.update(job_id=job['id'],job_key=job['key'])
                application.trials.save(task_id,version,record)
            except Exception as exc:
                record.update(status='failed', error=str(exc), updated_at=now())
                store.update_task(task_id, trials=[*source.get('trials', []), record])
                metadata.update(trials=[*metadata.get('trials', [])[:-1], record], updated_at=now())
                save(store.data / 'skills/candidates' / f'{version}.json', metadata)
                raise
            return {'status': 'scheduled', 'trial_id': identity, 'task_id': task_id,'job':job}


    def record_decision(version, metadata, decision, status):
        metadata.update(status=status, adoption=decision, updated_at=now(),
                        decisions=[*metadata.get('decisions', []), decision])
        if decision.get('scope'):metadata['scope']=decision['scope']
        save(store.data / 'skills/candidates' / f'{version}.json', metadata)
        if metadata.get('task_id'):
            store.update_task(metadata['task_id'], candidate_decision=decision)
        return metadata


    @app.post('/api/versions/{version}/adopt')
    @serialized
    def adopt_version(version: str, body: VersionAction = VersionAction()):
        metadata = candidate_record(version)
        with candidate_guard(version,metadata):
            require_candidate_decision(version,metadata)
            decision = application.bind(store.config()).acceptance.manual(application.versions, version, body.reason or 'User confirmed in the workbench')
            return record_decision(version, metadata, decision, 'accepted')


    @app.post('/api/versions/{version}/reject')
    @serialized
    def reject_version(version: str, body: VersionAction = VersionAction()):
        metadata = candidate_record(version)
        with candidate_guard(version,metadata):
            require_candidate_decision(version,metadata)
            if application.versions.active() == version:
                raise ValueError('The active version cannot be rejected')
            decision = {'decision': 'rejected', 'mode': 'manual', 'candidate_version': version,
                        'baseline_version': metadata['parent'], 'reason': body.reason or 'User rejected in the workbench',
                        'created_at': now(),'scope':metadata.get('scope',{'kind':'global','ref':'refs/worldrsi/active'})}
            return record_decision(version, metadata, decision, 'rejected')


    class NewCampaign(BaseModel):
        config: dict | None = None
        evolution_inputs: list[str] = Field(default_factory=list)
        test_inputs: list[str] = Field(default_factory=list)


    def schedule_campaign(record, action, *args, key, kind, conflict_keys=()):
        path=store.data/'campaigns'/f'{record["id"]}.json'
        save(path,record)
        try:
            job=application.scheduler.schedule(action,record['id'],*args,key=key,kind=kind,conflict_keys=conflict_keys)
        except Exception as exc:
            record.update(status='failed',error=str(exc),updated_at=now());save(path,record)
            raise
        record=load(path)
        record.update(job_id=job['id'],job_key=job['key']);save(path,record)
        return {**record,'job':job}


    @app.post('/api/campaigns')
    @serialized
    def new_campaign(body: NewCampaign = NewCampaign()):
        cfg = settings(store.config(), body.config)
        cfg['campaign']['auto_adopt'] = True
        if body.evolution_inputs or body.test_inputs:
            selected=application.campaigns.select_inputs(body.evolution_inputs,body.test_inputs)
            cfg['campaign'].update(input_selection=selected,benchmark_manifest_sha256=None,trial_enabled=True,outer_rsi=True)
        elif not cfg['campaign'].get('benchmark_manifest_sha256'):
            raise ValueError('Select evolution and test samples before starting an experiment')
        cfg = require_execution(cfg, campaign=True)
        identity = uid('campaign')
        record = {'id': identity, 'status': 'scheduled','start_version':application.versions.active(),
                  'schedule_policy':'pipeline_v1', 'created_at': now(), 'updated_at': now(),
                  'config': cfg, 'rounds': [], 'progress': []}
        return schedule_campaign(record,application.campaigns.run,cfg,key='campaign:'+identity,kind='campaign')


    @app.get('/api/config')
    def get_config():
        return store.config()


    @app.patch('/api/config')
    @serialized
    def edit_config(body: dict):
        value = settings(store.config(), body)
        value['campaign']['auto_adopt'] = True
        save(store.data / 'config.json', value)
        return value


    return app

def main():
    import uvicorn
    host = os.environ.get('WORLD_RSI_HOST','127.0.0.1')
    hosts = list(dict.fromkeys([host, *(item.strip() for item in
        os.environ.get('WORLD_RSI_LISTEN_HOSTS','').split(',') if item.strip())]))
    port = int(os.environ.get('WORLD_RSI_PORT', '8789'))
    config = uvicorn.Config(create_app(), host=host, port=port, ws_per_message_deflate=False)
    if len(hosts)==1:
        uvicorn.Server(config).run()
        return
    sockets = []
    try:
        for address in hosts:
            sockets.append(uvicorn.Config(config.app,host=address,port=port).bind_socket())
        uvicorn.Server(config).run(sockets=sockets)
    finally:
        for sock in sockets: sock.close()


if __name__ == '__main__':
    main()
