"""Task lifecycle composes selected plugins; it contains no provider/tool dispatch."""
import shutil
import hashlib
import json
import subprocess
from contextlib import contextmanager
from functools import wraps

from orchestration.models import RunContext
from harness.core.contracts import EnvironmentChanged
from orchestration.config import normalize_config
from orchestration.storage.repository import inside
from orchestration.storage.repository import load
from orchestration.storage.repository import save
from orchestration.storage.repository import sha
from orchestration.storage.repository import now
from orchestration.storage.repository import uid
from harness.evidence.usage import summarize_usage
from harness.evidence.usage import charge_metric
from harness.core.budget import Budget
from harness.core.budget import Stagnation
from harness.core.budget import LoopFailure
from harness.agents.reconstruction import output_contract
from harness.evidence.artifacts import archived_render


def completion_failure(run):
    if run.get('failure'):
        return run['failure']
    if run.get('status') == 'completed' and run.get('render_dir'):
        return None
    if run.get('status') in {'completed', 'no_render'}:
        return {'code':'no_render', 'message':'The final world has no validated render. '
                'Any earlier render belongs to an archived revision; submit the final scene again.'}
    return {'code':'run_failed', 'message':run.get('error') or
            f"The final run ended with status {run.get('status', 'unknown')}"}


def serialized_task(action):
    """Campaign children and HTTP jobs share the same reentrant task guard."""
    @wraps(action)
    def execute(self, task_id, *args, **kwargs):
        with self.app.scheduler.execute_guard(task_id, self.store.task(task_id)):
            return action(self, task_id, *args, **kwargs)
    return execute


class TaskRunner:
    def __init__(self, app): self.app, self.store = app, app.store

    def recover(self):
        # Interrupted work is visible and explicitly retryable, never silently resumed.
        for path in self.store.task_records():
            value = load(path)
            if value['status'] in {'running', 'finishing'}:
                for run in value['runs']:
                    if run['status'] == 'running':
                        run.update(status='interrupted', error='Controller restarted')
                value.update(status='interrupted', error='Controller restarted', updated_at=now())
                save(path, value)
            for trial in value.get('trials',[]):
                if trial['status'] in {'scheduled','running'}:
                    trial.update(status='interrupted',error='Controller restarted',updated_at=now())
                    self.app.trials.save(value['id'],trial['candidate_version'],trial)
        for path in (self.store.data / 'campaigns').glob('*.json'):
            value = load(path)
            if value['status'] in {'scheduled', 'running', 'pausing', 'paused', 'stopping', 'awaiting_confirmation'}:
                value.update(status='interrupted', error='Controller restarted', updated_at=now())
                save(path, value)

    def create(self, title, instruction, selected, *, split='manual', version=None, cfg=None, reference=None,
                    source_task=None, allow_nonbenchmark=False):
        self.app.verify_runtime()
        cfg = normalize_config(cfg if cfg is not None else self.store.config())
        available = {r['path']: r for r in (source_task['inputs'] if source_task else self.store.inputs())}
        if len(selected) != len(set(selected)):
            raise ValueError('Input files must be unique')
        if any(p not in available for p in selected):
            raise ValueError('Unknown input path')
        if reference is not None and reference not in selected:
            raise ValueError('reference must be one of the selected inputs')
        benchmarks = [available[p]['benchmark'] for p in selected if available[p].get('benchmark')]
        if any(available[p]['kind'] not in {'image','video'} for p in selected):
            raise ValueError('Task inputs must be images or videos')
        if benchmarks:
            if len(selected) != 1:
                raise ValueError('A benchmark case has exactly one public reference video')
            benchmark = benchmarks[0]
            manifest_id = cfg.get('campaign', {}).get('benchmark_manifest_sha256')
            if manifest_id and not source_task and split in {'evolution', 'test', 'validation'}:
                registry = load(self.store.data / 'benchmarks/registry.json', {})
                manifest = registry.get('manifests', {}).get(manifest_id, {})
                if not any(all(row.get(k) == benchmark.get(k) for k in ('case', 'split', 'sha256'))
                           for row in manifest.get('cases', [])):
                    raise ValueError('Task input is outside the frozen campaign manifest')
                benchmark = benchmarks[0] = {**benchmark, 'manifest_sha256': manifest_id}
                available[selected[0]] = {**available[selected[0]], 'benchmark': benchmark}
            if split == 'evolution' and benchmark['split'] not in {'evolution','smoke'}:
                raise ValueError('Evolution cannot read a smoke/validation/held-out benchmark case')
            if split not in {'evolution','trial', 'test', 'validation', 'heldout', 'held-out', 'held_out'}:
                split = benchmark['split']
            if split != 'evolution':
                cfg = {**cfg, 'outer_rsi': False, 'trial_enabled': False}
        if split == 'evolution' and any(available[p]['split'] not in {'evolution','smoke'} for p in selected):
            raise ValueError('Evolution tasks cannot use held-out test inputs')
        if not instruction.strip() and not selected:
            raise ValueError('Provide text or media')
        identity = uid('task')
        directory = self.store.task_path(identity)
        (directory / 'inputs').mkdir(parents=True)
        (directory / 'work').mkdir()
        observations = []
        for index, path in enumerate(selected):
            source = (inside(self.store.task_path(source_task['id']), available[path]['local_path']) if source_task
                      else inside(self.store.data / 'inputs', path))
            name = source.name if source_task else f'{index:02d}_{source.name}'
            target = directory / 'inputs' / name
            shutil.copy2(source, target)
            target.chmod(0o444)
            if source_task and sha(target) != available[path]['sha256']:
                raise ValueError('Archived task input changed; candidate trial refused')
            source_id=available[path].get('source_id') or 'media_'+hashlib.sha256(
                f'{index}:/input/{name}:{sha(target)}'.encode()).hexdigest()[:16]
            observations.append({**available[path], 'source_id':source_id,'sandbox_path': '/input/' + name,
                                 'local_path': 'inputs/' + name, 'sha256': sha(target)})
        if reference is None and len(observations) == 1:
            reference = observations[0]['path']
        services = self.app.bind(cfg)
        pinned = version or self.app.versions.active()
        snapshot = self.app.versions.snapshot(pinned)
        save(directory / 'snapshot.json', snapshot)
        self.app.versions.materialize(pinned, directory / 'skills')
        # Configuration is frozen per task and contains no credentials.
        record = {'schema_version': 3, 'kind': 'reconstruction', 'id': identity, 'title': title.strip() or identity, 'instruction': instruction,
                  'split': split, 'status': 'ready', 'version': pinned, 'created_at': now(),
                  'updated_at': now(), 'runs': [], 'inputs': observations, 'reference': reference,
                  'config': cfg, 'artifacts': [], 'metrics': None, 'critique': None, 'trials': [],
                  'harness': self.app.harness_snapshot(services)}
        record['snapshot_sha256'] = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
        if benchmarks:
            if observations[0]['sha256'] != benchmarks[0]['sha256']:
                raise ValueError('Benchmark input changed after its manifest was frozen')
            from evaluation.evaluator import source_identity
            from evaluation.evaluator import video_contract
            from evaluation.evaluator import scoring_identity
            from evaluation.evaluator import Evaluator
            fourd = Evaluator(cfg['evaluation'], self.store.root/'data/resources').config
            if not fourd:
                raise ValueError('Configure the official evaluator before creating benchmark tasks')
            record['benchmark'] = {**benchmarks[0], 'protocol': source_identity(fourd['source']),
                'video': video_contract(directory / observations[0]['local_path'])}
            times = Evaluator(cfg['evaluation'], self.store.root/'data/resources').timestamps(directory / observations[0]['local_path'])
            record['config'] = {**cfg, 'evaluation':{**cfg['evaluation'],'video_timestamps_s':times}}
            save(directory / 'scoring_identity.json', scoring_identity(fourd, benchmarks[0]['case']))
        elif reference is not None:
            from evaluation.media_reference import prepare_reference
            try:media=prepare_reference(directory,next(o for o in observations if o['path']==reference))
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
                shutil.rmtree(directory)
                raise ValueError('Cannot decode the selected reference image or video') from exc
            record['media_reference']=media
            record['config']={**cfg,'evaluation':{**cfg['evaluation'],'video_timestamps_s':media['timestamps_s']}}
        save(directory / 'task.json', record)
        return record

    def artifacts(self, directory):
        result = []
        for path in sorted(directory.rglob('*')):
            if (path.is_file() and not path.is_symlink() and not {'.cache', 'viewer_cache', 'live'}.intersection(path.parts)
                    and path.relative_to(directory).parts[0] in {'runs', 'work'}
                    and path.suffix.lower() in {'.yaml', '.py', '.md', '.png', '.jpg', '.mp4', '.json', '.obj', '.npz'}):
                result.append({'path': str(path.relative_to(directory)), 'name': str(path.relative_to(directory)),
                               'size': path.stat().st_size})
        return result

    def _invoke(self, prepared):
        """Start the native session; the application owns result application."""
        import asyncio
        async def execute():
            async with prepared.agent.session() as session:
                result, _, _ = await session.run(prepared.messages)
                return prepared.parse_result(result)
        try:
            return asyncio.run(execute())
        except BaseException as exc:
            prepared.archive.update(status='interrupted' if isinstance(exc,InterruptedError) else 'failed',
                error=str(exc),finished_at=now(),usage=prepared.archive.usage())
            raise

    def _ensure_edit_scope(self, run, budget, checkpoint, *, role):
        from harness.agents.threat import prepare_edit_scope
        path=run.task.directory/'scene_edit_requirements.json'
        digest=hashlib.sha256(run.task.record['instruction'].encode()).hexdigest()
        if path.exists():
            if load(path)['instruction_sha256']!=digest:
                raise ValueError('The frozen scene requirements belong to another instruction')
            return
        scope=prepare_edit_scope(run,budget=budget,before_call=checkpoint,role=role)
        requirements=self._invoke(scope)['requirements']
        save(path,{'instruction_sha256':digest,'requirements':requirements,'invocation_id':scope.archive.id})

    def _critic(self, run, *, final=False, propose=True, budget=None, before_call=None):
        from harness.agents import run_critic, task_critic
        role = task_critic if final else run_critic
        checkpoint = before_call or (lambda:self.app.verify_runtime(run.task.record))
        # A malformed review gets one fresh attempt within the same metered budget.
        for attempt in range(2):
            try:
                prepared = role.prepare(run, budget=budget,
                    **({'propose':propose} if final else {}),
                    before_call=checkpoint)
                result=self._invoke(prepared)
                if not final and run.task.record.get('kind')=='threat':
                    result['invocation_id']=prepared.archive.directory.name
                return result
            except (InterruptedError,EnvironmentChanged):raise
            except Exception:
                if attempt:raise

    def _vqa(self, context, artifact):
        from evaluation.vqa import evaluate
        from harness.agents.vqa_judge import judge
        def checkpoint():self.app.scheduler.checkpoint(self.store.task(context.id))
        source=context.services.evaluator_factory(context).config['source']
        return evaluate(source,context.record['benchmark']['case'],
            artifact/'world/render.mp4' if artifact else None,context.directory/'evaluation/vqa',
            lambda frames,prompt:judge(context,frames,prompt),checkpoint=checkpoint)

    @contextmanager
    def resources(self, context, phase, run=None, *, rollout_directory=None):
        """Lease only the active container phase, retaining the frozen policy."""
        scheduler=self.app.scheduler
        def checkpoint(): scheduler.checkpoint(self.store.task(context.id))
        business_phase='reconstruction' if phase=='sandbox' else 'evaluation'
        if rollout_directory is None:
            self.store.update_task(context.id,phase='waiting_resources',resource_phase=phase,resource_state='queued')
        if run:self.store.run_update(context.id,run.id,phase='waiting_resources')
        receipt=None
        try:
            priority=1 if context.record.get('campaign_id') and context.record.get('split')=='evolution' else 0
            with scheduler.resources.lease(context.config,phase=phase,owner=context.id,checkpoint=checkpoint,
                                           priority=priority) as allocated:
                directory=rollout_directory or (run.directory if run else context.directory/'evaluation')
                path=directory/'resource_leases'/f'{uid("lease")}.json'
                fields=('image','cpus','memory','gpus','runtime')
                receipt={'task_id':context.id,'run_id':run.id if run else None,'phase':phase,'priority':priority,
                    'requested':{k:context.config[phase].get(k) for k in fields},
                    'assigned':{k:allocated[phase].get(k) for k in fields},'started_at':now(),'status':'active'}
                save(path,receipt)
                if rollout_directory is None:
                    self.store.update_task(context.id,phase=business_phase,resource_state='active',
                        resource_lease_path=str(path.relative_to(context.directory)))
                if run:self.store.run_update(context.id,run.id,phase=business_phase,
                    resource_lease_path=str(path.relative_to(context.directory)))
                try:yield allocated
                finally:
                    receipt.update(status='released',finished_at=now());save(path,receipt)
        finally:
            if rollout_directory is None:
                self.store.update_task(context.id,resource_state='released' if receipt else 'cancelled')

    @serialized_task
    def run(self, task_id):
        from harness.agents import reconstruction
        context = self.app.context(task_id)
        if context.record.get('finalized_at'):
            raise ValueError('Task is finalized; create a new task for further execution')
        self.app.verify_runtime(context.record)
        value, cfg, services = context.record, context.config, context.services
        self.app.scheduler.checkpoint(value)
        if services.stop_requested(): raise InterruptedError('Controller is shutting down')
        missing = self.app.requirements(cfg)
        if missing: raise ValueError('Configuration required: '+', '.join(missing))
        if len(value['runs']) >= cfg['max_runs']: raise ValueError('Task reached configured max_runs')
        record = {'id':uid('run'), 'status':'running','created_at':now(),'usage':None}
        value['runs'].append(record)
        self.store.update_task(task_id,status='running',phase='reconstruction',runs=value['runs'],error=None)
        run = RunContext(context, record, context.directory/'runs'/record['id'])
        run.directory.mkdir(parents=True)
        def usage(): return self.usage(task_id,run.id,{'agent','reconstruction','run_critic'})
        budget = Budget(cfg.get('max_calls_per_run'),cfg.get('max_total_tokens_per_run'),
            meter=lambda:self.usage(task_id,run.id,{'agent','reconstruction'}))
        failure, final_text, status = None, None, 'completed'
        try:
            if value.get('kind')=='threat':
                self._ensure_edit_scope(run,budget,lambda:self.app.verify_runtime(context.record),role='reconstruction')
            with self.resources(context,'sandbox',run) as allocated, \
                    services.sandbox_factory(context.directory,allocated,run_id=run.id) as box:
                self.store.run_update(task_id,run.id,sandbox_startup_path=box.receipt_path,
                    sandbox_cleanup_path=box.cleanup_path)
                memory = context.directory/'local_memory.md'
                if memory.exists():box.write_file('local_memory.md',memory.read_text())
                contract = output_contract(context)
                run.state['output_contract'] = contract
                save(run.directory/'output_contract.json',contract)
                if contract['kind'] != 'scene_edit':
                    box.write_file('output_contract.json',json.dumps(contract))
                def archive_result(event):
                    path='tool_results/'+event.id+'.json'
                    save(context.directory/'work'/path,event.result)
                    return '/work/'+path
                actor = reconstruction
                if value.get('kind') == 'threat':
                    from harness.agents import threat
                    actor = threat
                prepared = actor.prepare(run,box,budget=budget,archive_result=archive_result,
                    stagnation=Stagnation(cfg.get('max_repeated_tool_outcomes',3),
                        progress=lambda:tuple(run.state.get(k) for k in ('world_hash','code_hash','render_hash'))),
                    before_call=lambda:self.app.verify_runtime(context.record))
                outcome = self._invoke(prepared)
                final_text = outcome if isinstance(outcome,str) else getattr(outcome,'text',str(outcome))
            if not run.state.get('rendered'):status='no_render'
        except InterruptedError as exc:
            status, failure = 'interrupted', {'code':'interrupted','message':str(exc)}
        except Exception as exc:
            status, failure = 'failed', getattr(exc,'reason',{'code':'invocation_failed','message':str(exc)})
        self.store.run_update(task_id,run.id,status=status,error=failure.get('message') if failure else None,
            failure=failure,render_dir=run.state.get('render_dir'),final_text=final_text,
            usage=usage(),call_count=usage()['api_calls'],agent_finished_at=now())
        if status=='interrupted':
            self.store.run_update(task_id,run.id,finished_at=now(),phase='interrupted',
                critic_status='skipped',critic_skipped='explicit_interruption')
            self.store.update_task(task_id,status='interrupted')
            raise InterruptedError('Task interrupted')
        # Failure analysis must still run after reconstruction exhausts its budget.
        # Both invocations remain part of this run's measured task consumption.
        if not context.stopped():
            self.store.run_update(task_id,run.id,phase='run_critic',critic_status='running')
            self.store.update_task(task_id,phase='run_critic')
            try:
                critic_budget=Budget(cfg.get('run_critic_max_calls',6),cfg.get('run_critic_max_total_tokens',200_000),
                    meter=lambda:self.usage(task_id,run.id,{'run_critic'}))
                critique=self._critic(run,budget=critic_budget)
                text=critique['local_memory']
                (run.directory/'local_memory.md').write_text(text)
                (context.directory/'local_memory.md').write_text(text)
                self.store.run_update(task_id,run.id,critique=critique,critic_status='completed')
                if (status == 'completed' and run.state.get('output_contract', {}).get('kind') == 'scene_edit'
                        and critique.get('edit_verdict') != 'supported'):
                    self.store.run_update(task_id,run.id,status='failed',failure={
                        'code':'edit_effect_unverified','message':critique.get('edit_reason',critique['summary'])})
                self.store.update_task(task_id,local_memory=text)
            except InterruptedError:
                self.store.run_update(task_id,run.id,status='interrupted',critic_status='interrupted',
                    phase='interrupted',finished_at=now(),usage=usage())
                self.store.update_task(task_id,status='interrupted')
                raise
            except Exception as exc:
                self.store.run_update(task_id,run.id,critique_error=str(exc),critic_status='failed',
                    critic_failure=getattr(exc,'reason',{'code':'critic_failed','message':str(exc)}))
                if status == 'completed' and run.state.get('output_contract',{}).get('kind') == 'scene_edit':
                    self.store.run_update(task_id,run.id,status='failed',failure={
                        'code':'edit_review_failed','message':str(exc)})
        if context.stopped():
            self.store.run_update(task_id,run.id,status='interrupted',finished_at=now(),usage=usage())
            self.store.update_task(task_id,status='interrupted')
            raise InterruptedError('Task interrupted')
        measured=usage()
        self.store.run_update(task_id,run.id,usage=measured,call_count=measured['api_calls'],
            finished_at=now(),phase='finished',usage_scope='task_execution')
        self.store.update_task(task_id,status='running' if value.get('execution_mode')=='continuous' else 'ready',
            phase='between_runs',artifacts=self.artifacts(context.directory))
        return run.id

    @serialized_task
    def execute(self, task_id, *, evolve=None, run_trial=True, run_limit=None):
        """Execute the remaining runs and finalize once, without another UI action."""
        value=self.store.task(task_id)
        if value.get('finalized_at'):
            raise ValueError('Task is finalized; create a new task for further execution')
        self.app.verify_runtime(value)
        limit=value['config']['max_runs'] if run_limit is None else min(run_limit,value['config']['max_runs'])
        self.store.update_task(task_id,execution_mode='continuous',status='running',stop_requested=False)
        try:
            while len(self.store.task(task_id)['runs'])<limit:
                current=self.store.task(task_id)
                self.app.scheduler.checkpoint(current)
                if current.get('stop_requested'):
                    raise InterruptedError('Task stop requested')
                self.run(task_id)
            self.store.update_task(task_id,status='ready',phase='evaluation')
            self.finish(task_id,evolve=evolve,run_trial=run_trial)
        except Exception as exc:
            if not self.store.task(task_id).get('finalized_at'):
                self.store.update_task(task_id,status='interrupted' if isinstance(exc,InterruptedError) else 'failed',
                    error=str(exc),phase='interrupted' if isinstance(exc,InterruptedError) else 'failed')
            raise
        finally:
            self.store.update_task(task_id,execution_mode=None)
        return self.store.task(task_id)

    def usage(self, task_id, run_id=None, stages=None):
        records = self.store.calls(task_id)
        return summarize_usage(r for r in records if (run_id is None or r.get('run_id') == run_id)
                               and (stages is None or r.get('stage') in stages))

    def accounting(self, value, calls=None):
        from evaluation.vqa import effective_task
        value=effective_task(value,self.store.task_path(value['id']))
        calls=self.store.calls(value['id']) if calls is None else calls
        execution=[c for c in calls if c.get('role',c.get('stage')) in {'agent','reconstruction','run_critic'}]
        evolution=[c for c in calls if c.get('role',c.get('stage')) in {'task_critic','task_rsi_critic'}]
        evaluation=[c for c in calls if c.get('role',c.get('stage'))=='evaluation_vqa']
        runs=[{**run,'usage':summarize_usage(c for c in execution if c.get('run_id')==run['id'])}
              for run in value.get('runs',[])]
        return {**value,'runs':runs,'usage':summarize_usage(execution),
            'evolution_usage':summarize_usage(evolution),'evaluation_usage':summarize_usage(evaluation),'usage_scope':'task_execution'}

    @serialized_task
    def finish(self, task_id, evolve=None, run_trial=True):
        value=self.store.task(task_id)
        self.app.verify_runtime(value)
        held_out=value['split'] in {'test','validation','heldout','held-out','held_out','trial','smoke'}
        evolve=bool(value['config'].get('outer_rsi',True) if evolve is None else evolve) and not held_out
        if value['status'] in {'completed','running','finishing'} or value.get('finalized_at'):
            raise ValueError('Task is already completed or busy')
        if not value['runs']:raise ValueError('Run the task before finishing')
        self.store.update_task(task_id,status='finishing',phase='evaluation')
        candidate=None
        try:
            context=self.app.context(task_id)
            self.app.scheduler.checkpoint(value)
            last=value['runs'][-1]
            failure=completion_failure(last)
            artifact,artifact_source=None,None
            for source_run in reversed(value['runs']):
                snapshot=archived_render(context.directory,source_run,for_scoring=value.get('kind')!='threat')
                if snapshot:
                    artifact=inside(context.directory,snapshot['render_dir'])
                    artifact_source={'run_id':source_run['id'],**snapshot,
                        'fallback':snapshot.get('fallback',False) or source_run['id']!=last['id'] or source_run.get('status')!='completed'
                                   or not snapshot.get('is_current',False),
                        'run_status':source_run.get('status'),
                        'world_matches_final':snapshot.get('world_sha256')==sha(context.directory/'runs'/last['id']/'world.yaml')
                            if (context.directory/'runs'/last['id']/'world.yaml').is_file() else False,
                        'code_matches_final':snapshot.get('code_sha256')==sha(context.directory/'runs'/last['id']/'scene.py')
                            if (context.directory/'runs'/last['id']/'scene.py').is_file() else False,
                        'scope':'Quality of this validated native artifact snapshot; task/run outcome is recorded separately.'}
                    break
            if value.get('kind') == 'threat':
                result=self.app.threats.evaluate(context.directory,value,self.store.calls(task_id),
                    artifact_dir=artifact,artifact_source=artifact_source)
                if not failure and result['edit_status']=='unchanged':
                    failure={'code':'no_scene_edit','message':'No declared physical, native-state or rendered scene change was found.'}
                if not failure and result['edit_status']=='unverified':
                    failure={'code':'edit_effect_unverified','message':'The matched inspections did not establish a scene effect.'}
                result['task_outcome']={'status':'failed' if failure else 'completed','run_id':last['id'],
                                        'run_status':last['status'],'failure':failure}
            else:
                evaluator=context.services.evaluator_factory(context)
                # Freeze evaluation before RSI. No later evolution receipt may alter this result.
                try:
                    with self.resources(context,'evaluation') as allocated:
                        result=evaluator.evaluate(context.directory,{**value,'status':'failed' if failure else 'completed'},self.store.calls(task_id),
                            artifact_dir=artifact,artifact_source=artifact_source,execution_config=allocated['evaluation'],
                            checkpoint=lambda:self.app.scheduler.checkpoint(self.store.task(task_id)))
                except InterruptedError:raise
                except Exception as exc:
                    from evaluation.evaluator import task_usage, OFFICIAL_KEYS
                    measured=task_usage(self.store.calls(task_id),task_id)
                    result={'quality_status':'error','reason':str(exc),'full_profile':True,'usage':measured,
                        'metrics':{'tokens':{'status':'ok' if measured['complete'] else 'missing',
                            'value':measured['total_tokens'] if measured['complete'] else None}}}
                    expected=['4dcodebench/'+key for key in OFFICIAL_KEYS]
                    if not value.get('benchmark'):
                        from evaluation.user_media import initialize_result
                        _,expected=initialize_result(value,result)
                    result['metrics'].update({key:{'status':'error','value':None,'reason':str(exc)} for key in expected})
                if value.get('benchmark') and value['config']['evaluation'].get('vqa_enabled',True):
                    self.store.update_task(task_id,phase='vqa')
                    try:result['vqa']=self._vqa(context,artifact)
                    except InterruptedError:raise
                    except Exception as exc:result['vqa']={'status':'error','error':str(exc),'metrics':{
                        '4dcodebench/vqa_'+key:{'status':'error','value':None,'reason':str(exc)} for key in ('accuracy','balanced_accuracy')}}
                    result['metrics'].update(result['vqa']['metrics'])
                    result['vqa']['artifact_source']=artifact_source
                elif not value.get('benchmark'):
                    result['vqa']={'status':'not_applicable','reason':'User media has no official VQA question bank','metrics':{}}
            from evaluation.groups import metric_groups
            result['metric_groups']=[] if value.get('kind')=='threat' else metric_groups(result['metrics'])
            result['finalized_before_task_rsi']=True
            result['created_at']=now()
            self.app.scheduler.checkpoint(self.store.task(task_id))
            if value.get('kind')=='threat':
                return self._finish_scene(value,result,failure,evolve)
            save(context.directory/'evaluation/result.json',result)
            save(context.directory/'evaluation.json',result)
            self.store.update_task(task_id,evaluation=result,metrics=result['metrics'],
                usage=result['usage'],usage_scope='task_execution')
            self.app.scheduler.checkpoint(self.store.task(task_id))
            critique,critic_error=None,None
            if evolve:
                self.store.update_task(task_id,phase='task_critic')
                budget=Budget(value['config'].get('max_calls_per_run'),value['config'].get('max_total_tokens_per_run'),
                    meter=lambda:self.usage(task_id,stages={'task_critic','task_rsi_critic'}))
                try:
                    critique=self._critic(context.run(last['id']),final=True,propose=True,budget=budget)
                except InterruptedError:
                    raise
                except Exception as exc:
                    critic_error=str(exc)
                    self.store.update_task(task_id,task_critic_error=critic_error)
                critique=critique or {}
                changes=critique.get('changes',{})
                if changes:
                    candidate=self.app.versions.propose(value['version'],changes,
                        critique.get('summary','Skill candidate'),provenance={**(critique.get('provenance') or {}),
                            'task_id':task_id,'task_kind':value.get('kind','reconstruction'),
                            'run_ids':[row['id'] for row in value['runs']],
                            'campaign_id':value.get('campaign_id')})
                    if candidate==value['version']:candidate=None
                    if candidate:
                        save(self.store.data/'skills/candidates'/f'{candidate}.json',{
                            'id':candidate,'parent':value['version'],'task_id':task_id,
                            'task_kind':value.get('kind','reconstruction'),
                            'campaign_id':value.get('campaign_id'),
                            'status':'awaiting_confirmation','created_at':now(),
                            'summary':critique.get('summary'),'adoption':{'decision':'pending','mode':'manual'}})
            self.store.update_task(task_id,status='failed' if failure else 'completed',
                error=failure['message'] if failure else None,failure=failure,finalized_at=now(),
                critique=critique,candidate_version=candidate,outer_rsi_performed=bool(evolve),
                task_critic_skipped='scope_excluded' if not evolve else None,
                task_critic_error=critic_error,phase='finished',
                evolution_usage=self.usage(task_id,stages={'task_critic','task_rsi_critic'}))
        except Exception as exc:
            if not self.store.task(task_id).get('finalized_at'):
                self.store.update_task(task_id,status='interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
            raise
        if candidate and run_trial and value.get('kind')!='threat' and not failure and value['config'].get('trial_enabled',False):
            self.app.trials.run(task_id,candidate)
        return self.store.task(task_id)

    def _finish_scene(self, task, result, failure, evolve, **updates):
        task_id = task['id']
        directory = self.store.task_path(task_id)
        automatic = not failure and task['config'].get('rollout',{}).get('enabled',True)
        identity = uid('rollout') if automatic else None
        save(directory/'evaluation/result.json',result)
        save(directory/'evaluation.json',result)
        self.store.update_task(task_id, status='failed' if failure else 'completed', phase='finished',
            finalized_at=now(), evaluation=result, metrics=result['metrics'], usage=result['usage'],
            usage_scope='task_execution', failure=failure, error=failure['message'] if failure else None,
            automatic_rollout_id=identity, outer_rsi_performed=False, task_critic_error=None,
            task_critic_skipped='scope_excluded' if not evolve else 'scene_failed' if failure else 'awaiting_rollout',
            **updates)
        if identity:
            self.app.rollouts.enqueue(task_id,identity,review=evolve)
        return self.store.task(task_id)

    @serialized_task
    def reassess_scene(self, task_id):
        """Reassess an archived delivery rejected by the former unscoped scene critic."""
        from orchestration.rollouts import rollout_review_enabled
        task=self.store.task(task_id)
        last=(task.get('runs') or [{}])[-1]
        if (task.get('kind')!='threat' or task['status']!='failed' or not task.get('finalized_at')
                or (task.get('failure') or {}).get('code')!='edit_effect_unverified'
                or (last.get('failure') or {}).get('code')!='edit_effect_unverified'
                or 'scene_checks' in (last.get('critique') or {}) or task.get('rollouts')):
            raise ValueError('This is not a finalized scene rejected by the legacy run critic')
        self.app.verify_runtime()
        from orchestration.models import TaskContext
        ctx=TaskContext(task,self.store.task_path(task_id),self.app.bind({**task['config'],'harness':{}}))
        run=ctx.run(last['id'])
        archive=archived_render(ctx.directory,last,for_replay=True)
        original=(task.get('evaluation') or {}).get('artifact_source') or {}
        if (not archive or not archive['is_current'] or original.get('run_id')!=last['id']
                or original.get('render_revision')!=archive['render_revision']
                or archive['code_sha256']!=sha(run.directory/'scene.py')
                or archive['world_sha256']!=sha(run.directory/'world.yaml')):
            raise ValueError('The rejected delivery does not match the final archived scene')
        audit_dir=ctx.directory/'evaluation/reassessments'/uid('review')
        save(audit_dir/'previous_task.json',task)
        old_calls={c['id'] for c in self.store.calls(task_id)}
        def usage():
            return summarize_usage(c for c in self.store.calls(task_id) if c['id'] not in old_calls
                                   and c.get('run_id')==run.id and c.get('stage')=='run_critic')
        budget=Budget(ctx.config.get('run_critic_max_calls'),ctx.config.get('run_critic_max_total_tokens'),meter=usage)
        self._ensure_edit_scope(run,budget,self.app.verify_runtime,role='run_critic')
        critique=self._critic(run,budget=budget,before_call=self.app.verify_runtime)
        audit={'run_id':run.id,'render_revision':archive['render_revision'],'reviewed_at':now(),
            'review_harness_id':self.app.harness_snapshot(ctx.services)['id'],'critique':critique,'usage':usage()}
        save(audit_dir/'review.json',audit)
        measured=self.usage(task_id,run.id,{'agent','reconstruction','run_critic'})
        failure=None if critique['edit_verdict']=='supported' else {
            'code':'edit_effect_unverified','message':critique['edit_reason']}
        last.update(critique=critique,critic_status='completed',status='failed' if failure else 'completed',
            error=None,failure=failure,usage=measured,call_count=measured['api_calls'])
        source={**original,**archive,'run_status':last['status'],'fallback':bool(failure)}
        result=self.app.threats.evaluate(ctx.directory,task,self.store.calls(task_id),
            artifact_dir=inside(ctx.directory,archive['render_dir']),artifact_source=source)
        if not failure and result['edit_status']!='changed':
            raise ValueError('The archived inspection does not establish a scene change')
        result.update(task_outcome={'status':last['status'],'run_id':run.id,'run_status':last['status'],'failure':failure},
            finalized_before_task_rsi=True,metric_groups=[],created_at=now())
        self.app.scheduler.checkpoint(task)
        memory=critique['local_memory']
        (run.directory/'local_memory.md').write_text(memory)
        (ctx.directory/'local_memory.md').write_text(memory)
        return self._finish_scene(task,result,failure,rollout_review_enabled(task),runs=task['runs'],
            critique=None,local_memory=memory,scene_reassessment=str((audit_dir/'review.json').relative_to(ctx.directory)))

    def arm(self, value):
        """A small truthful scene/metric summary, without loading API bodies."""
        calls = self.store.calls(value['id'])
        value = self.accounting(value,calls)
        usage = value.get('usage') or self.app.tasks.usage(value['id'])
        archive=archived_render(self.store.task_path(value['id']),value['runs'][-1]) if value.get('runs') else None
        from evaluation.groups import metric_groups
        return {**(archive or {}),'task_id': value['id'], 'version': value['version'], 'status': value['status'],
                'metric_groups':[] if value.get('kind')=='threat' else metric_groups(value.get('metrics') or {}),
                'run_id': value['runs'][-1]['id'] if value.get('runs') else None,
                'run_count': len(value.get('runs', [])), 'metrics': value.get('metrics') or {},
                'usage': usage, 'reconstruction_usage':summarize_usage(c for c in calls if c.get('role',c.get('stage')) in {'agent','reconstruction','run_critic'}) if calls else None,
                'render_dir':(archive or {}).get('render_dir'), 'preview_path':(archive or {}).get('preview_path'),
                'frame_count':(archive or {}).get('frame_count',0),
                'reconstruction_tokens': (sum(usage.get('by_stage', {}).get(stage, 0)
                                              for stage in ('agent', 'run_critic')) if usage.get('complete') else None)}
