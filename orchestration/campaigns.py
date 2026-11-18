"""Sequential evolution, immutable version benchmarks, bounded concurrent execution."""
from concurrent.futures import ThreadPoolExecutor, Future, wait
from contextlib import contextmanager, nullcontext
import hashlib
import json
import threading

from orchestration.models import TaskContext
from harness.evidence.usage import summarize_usage
from harness.evidence.usage import combine_usage
from harness.evidence.usage import compare_charges
from orchestration.config import normalize_config
from orchestration.storage.repository import load
from orchestration.storage.repository import save
from orchestration.storage.repository import sha
from orchestration.storage.repository import now
from orchestration.storage.repository import uid
from orchestration.scheduler import Control
from orchestration.continuation import completed_baseline


class _BenchmarkExecutor:
    """One bounded stream of case futures across all immutable version batches."""
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='benchmark')
        self.preparer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='benchmark-prepare')
        self.limit, self.active = workers, 0
        self.condition = threading.Condition()
        self.groups, self.closed = [], False
        self.thread = threading.Thread(target=self._dispatch, daemon=True, name='benchmark-queue')
        self.thread.start()

    def submit(self, function, arguments):
        future = Future()
        with self.condition:
            if self.closed: raise RuntimeError('Benchmark queue is closed')
            self.groups.append({'future':future, 'function':function, 'arguments':iter(arguments),
                                'running':0, 'exhausted':False, 'error':None})
            self.condition.notify_all()
        return future

    def submit_prepared(self, prepare):
        """Prepare a frozen batch without blocking the evolution coordinator."""
        result=Future()
        def transfer(completed):
            try:result.set_result(completed.result())
            except BaseException as exc:result.set_exception(exc)
        def run():
            try:prepare().add_done_callback(transfer)
            except BaseException as exc:result.set_exception(exc)
        self.preparer.submit(run)
        return result

    def _finish(self, group):
        if group['exhausted'] and not group['running'] and not group['future'].done():
            if group['error'] is not None: group['future'].set_exception(group['error'])
            else: group['future'].set_result(None)

    def _done(self, future, group):
        with self.condition:
            self.active -= 1; group['running'] -= 1
            try: future.result()
            except BaseException as exc:
                group.update(error=exc, exhausted=True)
            self._finish(group)
            self.condition.notify_all()

    def _dispatch(self):
        while True:
            with self.condition:
                self.groups = [g for g in self.groups if not g['future'].done()]
                available = next((g for g in self.groups if not g['exhausted']), None)
                if available is not None and self.active < self.limit:
                    try: arguments = next(available['arguments'])
                    except StopIteration:
                        available['exhausted'] = True; self._finish(available); continue
                    self.active += 1; available['running'] += 1
                    future = self.pool.submit(available['function'], *arguments)
                    future.add_done_callback(lambda done, group=available: self._done(done, group))
                    # Give another version batch a turn without creating an
                    # unbounded executor queue or one thread per batch.
                    self.groups.remove(available); self.groups.append(available)
                    continue
                if self.closed and not self.active: return
                self.condition.wait(.1)

    def shutdown(self, wait=True, cancel_futures=False):
        # Preparations must finish or observe cancellation before closing dispatch.
        self.preparer.shutdown(wait=wait, cancel_futures=False)
        with self.condition:
            self.closed = True
            if cancel_futures:
                for group in self.groups:
                    if not group['exhausted']:
                        group.update(exhausted=True, error=InterruptedError('Benchmark queue stopped'))
                        self._finish(group)
            self.condition.notify_all()
        if wait: self.thread.join()
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class CampaignRunner:
    def __init__(self, app): self.app, self.store = app, app.store

    def path(self, identity): return self.store.data / 'campaigns' / f'{identity}.json'

    def accounting(self, record):
        """Derive review status and costs without overwriting archived experiment records."""
        values, calls = {}, []
        for path in (self.store.data/'tasks').glob('*/task.json'):
            task = load(path)
            if task.get('campaign_id')==record['id'] or task['id'] in record.get('task_ids',[]):
                selected = self.store.calls(task['id'])
                calls.extend(selected)
                values[task['id']] = self.app.tasks.accounting(task,selected)
        def reviewed(row, status):
            result = {**row, 'status':status}
            if row.get('status') and row['status'] != status:
                result['recorded_status'] = row.get('recorded_status',row['status'])
            return result
        def arm(row):
            task = values.get(row.get('record_id',row.get('task_id')))
            if not task: return row
            return {**reviewed(row,task['status']),'error':task.get('error'),'failure':task.get('failure'),
                    'usage':task.get('usage',{}),'metrics':task.get('metrics') or {}}
        batches = {}
        for key, batch in record.get('benchmarks',{}).items():
            result = {**batch,'results':[arm(row) for row in batch['results']]}
            if batch.get('status')=='completed' and any(row['status']!='completed' for row in result['results']):
                result = reviewed(result,'failed')
            batches[key] = result
        rounds = []
        for row in record.get('rounds',[]):
            evolution = values.get(row.get('task_id'))
            if evolution and row.get('evolution_status') != evolution['status']:
                row = {**row,'recorded_evolution_status':row.get('evolution_status'),
                       'evolution_status':evolution['status'],'error':evolution.get('error')}
            before, after = [arm(v) for v in row.get('baseline',[])],[arm(v) for v in row.get('candidate_results',[])]
            if row.get('status')=='completed' and (row.get('evolution_status')=='failed' or
                    any(batches.get(row.get(key),{}).get('status')=='failed'
                        for key in ('baseline_batch_id','test_batch_id')) or
                    any(case.get('status') in {'failed','interrupted','stopped'} for case in before+after)):
                row = reviewed(row,'completed_with_errors')
            rounds.append({**row,'baseline':before,'candidate_results':after,
                'cost_comparison':compare_charges(combine_usage(v.get('usage',{}) for v in before),
                                                 combine_usage(v.get('usage',{}) for v in after))})
        if record.get('status')=='completed' and (any(b.get('status')=='failed' for b in batches.values()) or
                any(r.get('status')=='completed_with_errors' for r in rounds)):
            record = reviewed(record,'completed_with_errors')
        return {**record,'benchmarks':batches,'rounds':rounds,'usage':summarize_usage(calls),
                'usage_scope':'all_campaign_calls_counted_once'}

    def update(self, identity, change):
        def update(record):
            change(record)
            record['updated_at'] = now()
        return self.store.update_json(self.path(identity), update)

    def control(self, identity):
        with self.store.lock:
            existing = self.app.scheduler.controls.get(identity)
            if existing and not existing.closed:
                raise ValueError('This experiment already has a live controller')
        try:
            self.app.scheduler.checkpoint()
        except InterruptedError as exc:
            self.update(identity, lambda row: row.update(status=self.failure_status(exc), error=str(exc), phase='finished'))
            raise
        def changed():
            with self.store.lock:
                if control.closed: return
                record = load(self.path(identity))
                status = control.status
                if record['status'] != status:
                    record.update(status=status, updated_at=now())
                    save(self.path(identity), record)
        with self.store.lock:
            existing = self.app.scheduler.controls.get(identity)
            if existing and not existing.closed:
                raise ValueError('This experiment already has a live controller')
            control = Control(changed)
            self.app.scheduler.controls[identity] = control
        try:
            self.app.scheduler.checkpoint({'campaign_id':identity})
        except InterruptedError as exc:
            control.closed = True
            self.app.scheduler.controls.pop(identity, None)
            self.update(identity, lambda row: row.update(status=self.failure_status(exc), error=str(exc), phase='finished'))
            raise
        return control

    def action(self, identity, action):
        control = self.app.scheduler.controls.get(identity)
        if not control: raise ValueError('This experiment has no live controller; interrupted work is not replayed')
        if action == 'confirm':
            record = load(self.path(identity))
            candidate = record.get('pending_candidate')
            if not control.waiting or not candidate: raise ValueError('No experiment version awaits confirmation')
            decision = self.app.bind(record['config']).acceptance.manual(self.app.versions, candidate,
                'User confirmed experiment version', lineage=identity)
            self._adoption(identity, candidate, decision)
            control.waiting = False
            self.update(identity, lambda row: row.update(pending_candidate=None))
            control.changed()
        else: control.request(action)
        return load(self.path(identity))

    def checkpoint(self, control):
        self.app.scheduler.checkpoint()
        control.checkpoint()

    @contextmanager
    def slot(self, semaphore, control):
        self.checkpoint(control)
        while not semaphore.acquire(timeout=.1): self.checkpoint(control)
        try:
            self.checkpoint(control)
            yield
        finally: semaphore.release()

    def evaluator(self, cfg):
        context = TaskContext({'config':cfg}, self.store.data, self.app.bind(cfg))
        return context.services.evaluator_factory(context)

    def protocol_hash(self, cfg):
        services = self.app.bind(cfg)
        material = {'config':cfg, 'harness':self.app.harness_snapshot(services),
                    'evaluation_identity':self.evaluator(cfg).identity(cfg['evaluation']),
                    'resource_identity_sha256':sha(self.store.root/'data/resources/identity.json'),
                    'sandbox_image':services.sandbox_factory.health(cfg['sandbox']['image']).get('image_id')}
        return hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()

    def manifest(self, split, cfg=None):
        selection=(cfg or {}).get('campaign',{}).get('input_selection')
        if selection:
            return selection['evolution' if split=='evolution' else 'test']
        inputs = self.store.inputs()
        selected = (cfg or {}).get('campaign', {}).get('benchmark_manifest_sha256')
        if selected:
            registry = load(self.store.data / 'benchmarks/registry.json', {})
            manifest = registry.get('manifests', {}).get(selected)
            if manifest is None and registry.get('sha256') == selected:
                manifest = registry.get('manifest')
            if not manifest:
                raise ValueError('The requested benchmark manifest has not been imported')
            # Repeated version selection uses validation, never the final held-out.
            wanted = 'validation' if split == 'test' else split
            by_case = {r['benchmark']['case']: r for r in inputs if r.get('benchmark')}
            sources = []
            for case in manifest['cases']:
                if case['split'] != wanted: continue
                source = by_case.get(case['case'])
                if source is None or sha(self.store.data / 'inputs' / source['path']) != case['sha256']:
                    raise ValueError('Frozen benchmark input missing or changed: ' + case['case'])
                sources.append({**source, 'benchmark': {**source['benchmark'], 'manifest_sha256': selected}})
        else:
            sources = [r for r in inputs if r['split'] == split and not r.get('benchmark')]
        return [{**source, 'sha256':sha(self.store.data/'inputs'/source['path'])}
                for source in sources if source['kind'] in {'image','video'}]

    def select_inputs(self, evolution, tests):
        """Freeze experiment-local roles and order without relabeling the catalog."""
        if not evolution or not tests:raise ValueError('Select at least one evolution and one test sample')
        if len(set(evolution))!=len(evolution) or len(set(tests))!=len(tests):raise ValueError('Duplicate selected sample')
        available={row['path']:row for row in self.store.inputs()}
        result={};seen={}
        for role,paths in (('evolution',evolution),('test',tests)):
            result[role]=[]
            for path in paths:
                row=available.get(path)
                if not row or not row.get('benchmark'):raise ValueError('Only registered 4DCodeBench samples may be selected')
                if row['split'] in {'held-out','heldout','held_out'}:raise ValueError('Held-out samples cannot be used for repeated selection')
                if role=='evolution' and row['split'] not in {'smoke','evolution'}:raise ValueError('Validation samples cannot enter evolution')
                digest=sha(self.store.data/'inputs'/path)
                if digest!=row['benchmark']['sha256']:raise ValueError('Selected benchmark content changed')
                if digest in seen:raise ValueError('Evolution and test sets must be disjoint, including duplicate videos')
                seen[digest]=role;result[role].append({**row,'sha256':digest})
        return result

    def create(self, identity, version, source, cfg, label, split):
        if sha(self.store.data/'inputs'/source['path']) != source['sha256']:
            raise ValueError('Campaign input changed after manifest was frozen: ' + source['path'])
        value = self.app.tasks.create(f'{label} · {source["path"]}',
            'Reconstruct the observed world from the supplied media using the configured renderer. '
            'Account for visible geometry, appearance, camera and motion where identifiable.',
            [source['path']], split=split, version=version, cfg=cfg)
        if value['inputs'][0]['sha256'] != source['sha256']: raise ValueError('Input changed during task creation')
        self.store.update_task(value['id'], campaign_id=identity, benchmark_label=label)
        self.update(identity,lambda r:r.setdefault('task_ids',[]).append(value['id']))
        return value

    def start_batch(self, identity, version, cfg, label, sources, protocol, pool, admission, control):
        self.checkpoint(control)
        # Both the task snapshot and the configuration explicitly disable held-out evolution/trials.
        test_cfg = {**cfg, 'outer_rsi':False, 'trial_enabled':False}
        batch_id = uid('benchmark')
        batch = {'id':batch_id, 'label':label, 'version':version, 'status':'running', 'created_at':now(),
                 'task_ids':[], 'expected_tasks':len(sources), 'results':[], 'protocol_sha256':protocol}
        self.update(identity,lambda r:r.setdefault('benchmarks',{}).update({batch_id:batch}))
        def prepare():
            values=[]
            try:
                for source in sources:
                    self.checkpoint(control)
                    value=self.create(identity,version,source,test_cfg,label,'test')
                    values.append(value)
                    self.update(identity,lambda r:r['benchmarks'][batch_id]['task_ids'].append(value['id']))
                arguments=((identity,batch_id,value,source,version,cfg,protocol,admission,control)
                           for value,source in zip(values,sources))
                return pool.submit(self.case,arguments)
            except BaseException as exc:
                self.update(identity,lambda r:r['benchmarks'][batch_id].update(
                    status='interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),finished_at=now()))
                raise
        return batch_id, [pool.submit_prepared(prepare)]

    def case(self, identity, batch_id, value, source, version, cfg, protocol, admission, control):
        error = None
        try:
            with control.worker(), self.slot(admission,control):
                self.checkpoint(control)
                self.app.tasks.execute(value['id'], evolve=False, run_trial=False)
        except Exception as exc:
            error = exc
            self.store.update_task(value['id'],status='interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        result = self.store.task(value['id'])
        if result['version'] != version or load(self.store.task_path(value['id'])/'snapshot.json') != self.app.versions.snapshot(version):
            error = ValueError('Benchmark task did not use the exact requested version snapshot')
            result = self.store.update_task(value['id'],status='failed',error=str(error))
        arm = self.app.tasks.arm(result)
        row = {**arm, 'task_id':source['path'], 'record_id':value['id'], 'version':version,
               'snapshot_sha256':result.get('snapshot_sha256'), 'source_sha256':source['sha256'],
               'seed':cfg['seed'], 'protocol_sha256':protocol}
        if error or result.get('error'): row['error'] = str(error) if error else result['error']
        if result.get('failure'): row['failure'] = result['failure']
        def append(record):
            batch = record['benchmarks'][batch_id]
            batch['results'].append(row)
            batch['results'].sort(key=lambda r:r['task_id'])
            if len(batch['results']) == len(batch['task_ids']):
                batch.update(status='failed' if any(r['status']!='completed' for r in batch['results']) else 'completed',finished_at=now())
            record.setdefault('progress',[]).append({'label':batch['label'],'task_id':value['id'],
                'source':source['path'],'version':version,'status':row['status']})
        self.update(identity,append)
        # Publish completed comparisons as soon as both immutable batches land,
        # even if the campaign is still executing a later evolution task.
        self.settle(identity,cfg)
        if isinstance(error,InterruptedError): raise error
        return row

    def await_futures(self, futures, control, identity, cfg):
        pending = set(futures)
        while pending:
            self.checkpoint(control)
            done, pending = wait(pending, timeout=.2)
            for future in done: future.result()
            self.settle(identity,cfg)
        self.checkpoint(control)

    def settle(self, identity, cfg):
        """Pair already completed immutable batches; never consult the moving active ref."""
        with self.store.lock:
            record = load(self.path(identity))
            for round in record.get('rounds',[]):
                if round.get('evaluation') or not round.get('test_batch_id'): continue
                before = record['benchmarks'][round['baseline_batch_id']]
                after = record['benchmarks'][round['test_batch_id']]
                if any(b['status']=='running' for b in (before,after)): continue
                decision = self.evaluator(cfg).compare(before['results'],after['results'],cfg.get('adoption_policy'))
                decision.update(baseline_version=round['parent'],candidate_version=round['tested_version'],
                                reference_only=True,trials_enabled=True,
                                cost_comparison=compare_charges(combine_usage(v['usage'] for v in before['results']),
                                                               combine_usage(v['usage'] for v in after['results'])))
                round.update(baseline=before['results'],candidate_results=after['results'],evaluation=decision,
                             status='completed' if before['status']==after['status']=='completed'
                                 and round.get('evolution_status')!='failed' else 'completed_with_errors')
                if round.get('candidate'):
                    path = self.store.data/'skills/candidates'/f'{round["candidate"]}.json'
                    metadata = load(path)
                    metadata.update(evaluation=decision,updated_at=now())
                    save(path,metadata)
            record['updated_at'] = now()
            save(self.path(identity),record)

    def _adoption(self, identity, candidate, decision):
        with self.store.lock:
            path = self.store.data/'skills/candidates'/f'{candidate}.json'
            metadata = load(path)
            metadata.update(status='accepted' if decision['decision']=='accepted' else 'awaiting_confirmation',
                            adoption=decision,updated_at=now(),campaign_id=identity,
                            scope={'kind':'campaign','id':identity,'ref':self.app.versions.active_ref(identity)})
            metadata.setdefault('decisions', []).append(decision)
            save(path,metadata)
            self.update(identity,lambda r:next(row for row in r['rounds'] if row.get('candidate')==candidate).update(decision=decision))
            if decision['decision']=='accepted':
                self.update(identity,lambda r:r.update(active_version=candidate))

    def benchmark(self, version, cfg, campaign_id, label):
        control = self.app.scheduler.controls[campaign_id]
        maximum = cfg['campaign']['max_parallel_tasks']
        admission = threading.BoundedSemaphore(maximum)
        sources = self.manifest('test', cfg)
        if not sources: raise ValueError('External evaluation requires test inputs')
        pool = _BenchmarkExecutor(min(cfg['campaign']['test_workers'], maximum))
        try:
            batch_id, futures = self.start_batch(campaign_id,version,cfg,label,sources,
                self.protocol_hash(cfg),pool,admission,control)
            self.await_futures(futures,control,campaign_id,cfg)
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        return load(self.path(campaign_id))['benchmarks'][batch_id]['results']

    def evaluate(self, campaign_id, candidate, cfg):
        control = self.control(campaign_id)
        try:
            with control.worker():
                cfg = normalize_config(cfg)
                metadata = load(self.store.data/'skills/candidates'/f'{candidate}.json')
                if not metadata: raise ValueError('Candidate metadata is missing')
                parent = metadata['parent']
                self.app.versions.snapshot(parent)
                self.preflight(self.manifest('test',cfg),cfg,campaign_id=campaign_id,control=control)
                baseline = self.benchmark(parent,cfg,campaign_id,'Baseline')
                proposed = self.benchmark(candidate,cfg,campaign_id,'Candidate')
                decision = self.evaluator(cfg).compare(baseline,proposed,cfg['adoption_policy'])
                decision.update(baseline_version=parent,candidate_version=candidate,reference_only=True,
                    cost_comparison=compare_charges(combine_usage(v.get('usage',{}) for v in baseline),
                                                   combine_usage(v.get('usage',{}) for v in proposed)))
                adoption = {'decision':'pending','mode':'manual','reason':'User confirmation is required',
                            'baseline_version':parent,'candidate_version':candidate}
                def update_metadata(current):
                    current.update(evaluation=decision,updated_at=now())
                    if current.get('status') not in {'accepted','rejected'}:
                        current.update(status='awaiting_confirmation',adoption=adoption)
                self.store.update_json(self.store.data/'skills/candidates'/f'{candidate}.json',update_metadata)
                self.update(campaign_id,lambda r:r.update(rounds=[{'parent':parent,'candidate':candidate,
                    'baseline':baseline,'candidate_results':proposed,'evaluation':decision,'decision':adoption}]))
            status, error = 'completed', None
        except Exception as exc:
            status, error = self.failure_status(exc), str(exc)
        finally:
            control.closed = True
            self.update(campaign_id,lambda r:r.update(status=status,error=error))
            self.app.scheduler.controls.pop(campaign_id,None)

    def failure_status(self, exc):
        return ('interrupted' if self.app.scheduler.stop.is_set() else 'stopped') if isinstance(exc,InterruptedError) else 'failed'

    def baseline_continuation(self, identity):
        record=load(self.path(identity))
        if not record:raise FileNotFoundError(identity)
        if record['status'] not in {'interrupted','stopped','failed'}:
            raise ValueError('Continuation requires a stopped controller after the baseline')
        control=self.app.scheduler.controls.get(identity)
        if (control and not control.closed) or self.app.scheduler.is_busy(key='campaign:'+identity) or \
                self.app.scheduler.is_busy(task_ids=record.get('task_ids',[])):
            raise ValueError('This campaign still has live work')
        return completed_baseline(self.app,record)

    def preflight(self, sources, cfg, *, campaign_id=None, control=None):
        """Refuse batch model spending when the frozen official evaluator is unavailable."""
        resources=getattr(getattr(self.app,'scheduler',None),'resources',None)
        checkpoint=(lambda:self.checkpoint(control)) if control is not None else (
            lambda:self.app.scheduler.checkpoint())
        lease=(resources.lease(cfg,phase='evaluation',owner=f'campaign:{campaign_id or uid("preflight")}:preflight',
                               checkpoint=checkpoint) if resources is not None else nullcontext(cfg))
        checked={}
        with lease as resolved:
            evaluator=self.evaluator(cfg)
            for source in sources:
                benchmark=source.get('benchmark')
                if not benchmark:raise ValueError('Campaigns require registered 4DCodeBench samples')
                case=benchmark['case']
                if case in checked:continue
                result=(evaluator.readiness(case,execution_config=resolved['evaluation']) if resources is not None
                        else evaluator.readiness(case))
                if result['status']!='ready':
                    raise ValueError('Evaluator not_ready for '+case+': '+json.dumps(result.get('missing',[])))
                checked[case]=result
        return checked

    def run(self, campaign_id, cfg, baseline=None):
        control = self.control(campaign_id)
        pool, all_futures = None, []
        status, error = 'failed', None
        try:
            with control.worker():
                cfg = normalize_config(cfg)
                # Each campaign advances its private skill lineage automatically.
                cfg['campaign']['auto_adopt'] = True
                options, sources, tests = cfg['campaign'], self.manifest('evolution', cfg), self.manifest('test', cfg)
                if not sources or not tests: raise ValueError('Campaign requires explicit evolution and validation inputs')
                frozen=load(self.path(campaign_id))
                if baseline is not None:
                    checked_cfg,checked=completed_baseline(self.app,frozen)
                    if checked!=baseline or checked_cfg!=cfg:
                        raise ValueError('The baseline changed after continuation was scheduled')
                start=frozen.get('start_version') or frozen.get('baseline_version') or self.app.versions.active()
                parent=self.app.versions.fork_lineage(campaign_id,start)
                self.update(campaign_id,lambda r:r.update(start_version=start,active_version=parent))
                readiness=self.preflight([*sources,*tests],cfg,campaign_id=campaign_id,control=control)
                self.update(campaign_id,lambda r:r.update(evaluator_readiness=readiness))
                maximum=options['max_parallel_tasks']
                plan={'task_slots':maximum,'test_workers':min(options['test_workers'],max(1,maximum-1)),
                      'scope':'Campaign request concurrency only; CPU/RAM/GPU leases are globally arbitrated per task phase.'}
                self.update(campaign_id,lambda r:r.update(config=cfg,baseline_version=parent,resource_plan=plan,
                    start_version=start,active_version=parent,version_scope={'kind':'campaign','id':campaign_id,
                        'ref':self.app.versions.active_ref(campaign_id)},
                    schedule_policy='pipeline_v1',input_manifest={'evolution':sources,'test':tests},
                    benchmarks=frozen['benchmarks'] if baseline is not None else {},
                    phase='evolution' if baseline is not None else 'baseline',error=None))
                admission = threading.BoundedSemaphore(plan['task_slots'])
                pool = _BenchmarkExecutor(plan['test_workers'])
                protocol, previous = self.protocol_hash(cfg), None
                if baseline is not None:
                    previous,futures=baseline['baseline_batch_id'],[]
                    self.update(campaign_id,lambda r:r.setdefault('continuations',[]).append({**baseline,'at':now()}))
                    for task_id in baseline['superseded_unstarted_tasks']:
                        self.store.update_task(task_id,status='stopped',continuation_superseded=True)
                else:
                    # Freeze V0 before evolution, but do not wait for independent tests.
                    previous, futures = self.start_batch(campaign_id,parent,cfg,'V0 · Initial baseline',tests,protocol,
                                                         pool,admission,control)
                    all_futures.extend(futures)
                self.update(campaign_id,lambda r:r.update(baseline_batch_id=previous,total_rounds=len(sources)))
                if maximum==1:self.await_futures(futures,control,campaign_id,cfg)
                for index, source in enumerate(sources):
                    self.checkpoint(control)
                    for completed in all_futures:
                        if completed.done():completed.result()
                    parent=self.app.versions.active(campaign_id)
                    value = self.create(campaign_id,parent,source,
                        {**cfg,'outer_rsi':options['outer_rsi'],'trial_enabled':False},f'Evolution {index+1}','evolution')
                    self.update(campaign_id,lambda r:r.update(current_task_id=value['id'],phase='evolution'))
                    try:
                        with self.slot(admission,control):
                            self.checkpoint(control)
                            result = self.app.tasks.execute(value['id'],evolve=options['outer_rsi'],run_trial=False)
                    except InterruptedError: raise
                    except Exception as exc:
                        result = self.store.update_task(value['id'],status='failed',error=str(exc))
                    self.checkpoint(control)
                    candidate = result.get('candidate_version')
                    row = {'round':index+1,'task_id':value['id'],'parent':parent,'candidate':candidate,
                           'tested_version':candidate or parent,'status':'testing' if previous else 'completed',
                           'evolution_status':result['status'],'error':result.get('error'),
                           'baseline_batch_id':previous,'trials_enabled':options['trial_enabled']}
                    self.update(campaign_id,lambda r:r['rounds'].append(row))
                    # Every round gets a fresh full test batch, including unchanged/failed proposals.
                    tested=candidate or parent
                    batch, futures=self.start_batch(campaign_id,tested,cfg,f'Round {index+1} · Full test',
                        tests,protocol,pool,admission,control)
                    all_futures.extend(futures)
                    self.update(campaign_id,lambda r:r['rounds'][-1].update(test_batch_id=batch,trials_enabled=True))
                    self.update(campaign_id,lambda r:r.update(phase='round_testing'))
                    previous=batch
                    if candidate:
                        # This is campaign-local policy, not a metric-based verdict.
                        decision=self.app.bind(cfg).acceptance.experiment(self.app.versions,candidate,True,lineage=campaign_id)
                        self._adoption(campaign_id,candidate,decision)
                    else:
                        self.update(campaign_id,lambda r:r['rounds'][-1].update(decision={
                            'decision':'pending','mode':'none','reason':'No evidence-backed skill change proposed; full test rerun retained'}))
                    parent = self.app.versions.active(campaign_id)
                    self.settle(campaign_id,cfg)
                self.update(campaign_id,lambda r:r.update(phase='draining_tests',current_task_id=None))
                self.await_futures(all_futures,control,campaign_id,cfg)
                record = load(self.path(campaign_id))
                status = ('completed_with_errors' if any(b['status']!='completed' for b in record['benchmarks'].values())
                          or any(r.get('evolution_status')=='failed' for r in record['rounds']) else 'completed')
        except Exception as exc:
            status, error = self.failure_status(exc), str(exc)
            if not control.stopped: control.request('stop')
        finally:
            if pool: pool.shutdown(wait=True,cancel_futures=True)
            control.closed = True
            def finish(record):
                record.update(status=status,error=error,phase='finished',current_task_id=None)
                for batch in record.get('benchmarks',{}).values():
                    if batch['status']=='running': batch['status']='stopped' if status=='stopped' else 'interrupted'
                for task_id in record.get('task_ids',[]):
                    task = self.store.task(task_id)
                    if task['status']=='ready': self.store.update_task(task_id,status='stopped')
                record['usage'] = summarize_usage(call for task_id in set(record.get('task_ids',[]))
                    for call in self.store.calls(task_id))
                record['usage_scope'] = 'all_campaign_calls_counted_once'
            self.update(campaign_id,finish)
            self.app.scheduler.controls.pop(campaign_id,None)
