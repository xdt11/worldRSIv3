"""Candidate trials are separate tasks, with no recursive RSI or automatic adoption."""
from orchestration.config import normalize_config
from orchestration.storage.repository import load
from orchestration.storage.repository import save
from orchestration.storage.repository import now
from orchestration.storage.repository import uid
from harness.evidence.usage import compare_charges


class Trials:
    def __init__(self, app): self.app, self.store = app, app.store

    def accounting(self, record):
        arms = {}
        for name in ('baseline','candidate'):
            arm = record.get(name)
            if arm and arm.get('task_id') and self.store.task_path(arm['task_id']).joinpath('task.json').is_file():
                current = self.app.tasks.arm(self.store.task(arm['task_id']))
                arms[name] = {**arm,'usage':current['usage'],'reconstruction_usage':current['reconstruction_usage'],
                              'metrics':{**arm.get('metrics',{}),'charged_amount':current['metrics'].get('charged_amount')}}
        result = {**record,**arms}
        if all(result.get(name) for name in ('baseline','candidate')):
            result['comparison'] = {**result.get('comparison',{}),'charged_amount':
                compare_charges(result['baseline'].get('usage',{}),result['candidate'].get('usage',{}))}
        return result

    def save(self, source_task_id, candidate, record):
        with self.store.lock:
            source = self.store.task(source_task_id)
            records = [r for r in source.get('trials', []) if r.get('id') != record['id']]
            self.store.update_task(source_task_id, trials=[*records, record])
            path = self.store.data / 'skills/candidates' / f'{candidate}.json'
            metadata = load(path)
            records = [r for r in metadata.get('trials', []) if r.get('id') != record['id']]
            metadata.update(trials=[*records, record], updated_at=now())
            if metadata.get('status') not in {'accepted', 'rejected'}:
                metadata['status'] = 'testing' if record['status'] in {'running', 'scheduled'} else 'awaiting_confirmation'
            save(path, metadata)

    def run(self, source_task_id, candidate, trial_id=None):
        """Reconstruct the same archived observations with a candidate, without RSI/adoption."""
        with self.store.lock:
            source,record = self._claim(source_task_id,candidate,trial_id)
        return self._run_claimed(source_task_id,candidate,source,record)

    def _claim(self, source_task_id, candidate, trial_id):
        """Claim under the store lock, then release it before agent execution."""
        source = self.store.task(source_task_id)
        if source.get('kind') == 'threat':
            raise ValueError('Threat-scene policy/rollout trials belong to the next stage')
        metadata = load(self.store.data / 'skills/candidates' / f'{candidate}.json')
        if not metadata or metadata.get('parent') != source['version'] or metadata.get('task_id') != source_task_id:
            raise ValueError('Candidate must have been proposed by this exact source task/version')
        if source['status'] != 'completed' or not source['runs']:
            raise ValueError('Finish the source task before testing its candidate')
        self.app.versions.snapshot(candidate)
        trial_id = trial_id or uid('trial')
        scheduled = next((r for r in source.get('trials', []) if r.get('id') == trial_id), {})
        if scheduled and scheduled.get('status') != 'scheduled':
            raise ValueError('This trial has already started or finished')
        record = {'id': trial_id, 'source_task_id': source_task_id, 'candidate_version': candidate,
                  'status': 'running', 'created_at': scheduled.get('created_at') or now(),
                  'updated_at': now(), 'baseline': self.app.tasks.arm(source),
                  'candidate': None, 'decision': {'decision': 'pending', 'mode': 'manual'},
                  'cost_scope': 'Total tokens include each task summary. Reconstruction tokens include agent + run critic; '
                                'task scores exclude global RSI proposal cost; evolution overhead is recorded separately.'}
        self.save(source_task_id, candidate, record)
        return source,record

    def _run_claimed(self, source_task_id, candidate, source, record):
        trial_id = record['id']
        try:
            cfg = normalize_config(source['config'])
            cfg.update(outer_rsi=False, trial_enabled=False)
            runs = min(len(source['runs']), cfg['max_runs'])
            value = self.app.tasks.create('候选试验 · ' + source['title'], source['instruction'],
                                [o['path'] for o in source['inputs']], split='trial', version=candidate,
                                cfg=cfg, reference=source.get('reference'), source_task=source)
            self.store.update_task(value['id'], parent_task_id=source_task_id, trial_id=trial_id)
            record['candidate'] = self.app.tasks.arm(value)
            self.save(source_task_id, candidate, record)
            result = self.app.tasks.execute(value['id'],evolve=False,run_trial=False,run_limit=runs)
            record['candidate'] = self.app.tasks.arm(result)
            limits = []
            if source['config'].get('seed') is None:
                limits.append('No seed was configured; equal random initialization is not established.')
            if any(run.get('status') in {'failed', 'interrupted', 'no_render'} for run in source['runs']):
                limits.append('The baseline contains failed, interrupted or unrendered runs.')
            if record['baseline']['run_count'] != record['candidate']['run_count']:
                limits.append('Agent run counts differ.')
            identities = [(v.get('harness') or {}).get('code',{}).get('evaluation/evaluator.py') for v in (source, result)]
            if not all(identities) or identities[0] != identities[1]:
                limits.append('The same recorded evaluator implementation is not established for both arms.')
            identities = [(v.get('evaluation') or {}).get('protocol_identity') for v in (source, result)]
            if not all(identities) or identities[0] != identities[1]:
                limits.append('The same metric model checkpoints, source and runtime identities are not established.')
            protocols = [(v.get('evaluation') or {}).get('config') for v in (source, result)]
            if any(p is None for p in protocols) or protocols[0] != protocols[1]:
                limits.append('The same metric preprocessing, resize and video alignment configuration is not established.')
            comparison = {}
            for metric in sorted(set(record['baseline']['metrics']) | set(record['candidate']['metrics'])):
                before, after = [record[arm]['metrics'].get(metric, {'status': 'missing', 'value': None})
                                 for arm in ('baseline', 'candidate')]
                delta = None
                if before.get('status') == after.get('status') == 'ok':
                    a, b = before.get('value'), after.get('value')
                    if a == b:
                        delta = 0
                    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        delta = b - a
                    elif a == 'Infinity':
                        delta = '-Infinity'
                    elif b == 'Infinity':
                        delta = 'Infinity'
                else:
                    limits.append(f'{metric} is missing or failed for at least one arm.')
                comparison[metric] = {'baseline': before, 'candidate': after, 'delta': delta}
            comparison['charged_amount'] = compare_charges(record['baseline']['usage'],record['candidate']['usage'])
            record.update(status='completed', updated_at=now(), comparison=comparison,
                          comparability={'strict': not limits, 'limits': limits})
        except Exception as exc:
            record.update(status='interrupted' if isinstance(exc, InterruptedError) else 'failed',
                          updated_at=now(), error=str(exc))
            if record.get('candidate'):
                record['candidate'] = self.app.tasks.arm(self.store.task(record['candidate']['task_id']))
            self.save(source_task_id, candidate, record)
            if isinstance(exc, InterruptedError):
                raise
            return record
        self.save(source_task_id, candidate, record)
        return record
