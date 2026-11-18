"""Explicit campaign handoff at a completed baseline, never replaying paid work."""
import hashlib
import json

from evaluation.evaluator import scoring_identity
from orchestration.config import normalize_config
from orchestration.storage.repository import load, sha


# These modules were reviewed for the pipeline migration. Agent execution,
# prompts, tools, evaluator, dependency lock and source media must still match.
SCHEDULING_FILES = {
    'orchestration/campaigns.py', 'orchestration/continuation.py',
    'orchestration/config.py', 'orchestration/resources.py',
    'orchestration/scheduler.py', 'orchestration/tasks.py',
    'orchestration/storage/repository.py', 'web/api.py',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed_baseline(app, record):
    """Validate the frozen cohort and retain its original protocol and evidence."""
    if record.get('kind') or record.get('rounds'):
        raise ValueError('Continuation requires a campaign before its first evolution round')
    cfg = normalize_config(record['config'])
    frozen = record.get('input_manifest')
    if not frozen or any(app.campaigns.manifest(split, cfg) != frozen[split] for split in ('evolution', 'test')):
        raise ValueError('Frozen campaign input selection changed or is missing')
    for source in [*frozen['evolution'], *frozen['test']]:
        if sha(app.store.data/'inputs'/source['path']) != source['sha256']:
            raise ValueError('Frozen campaign input content changed')
    baseline_id = record.get('baseline_batch_id')
    batch = record.get('benchmarks', {}).get(baseline_id, {})
    if batch.get('status') not in {'completed', 'failed'} or len(record['benchmarks']) != 1:
        raise ValueError('The full baseline must finish before continuation')
    parent = record.get('baseline_version')
    if not parent or batch.get('version') != parent or app.versions.active(record['id']) != parent:
        raise ValueError('The campaign skill lineage has already advanced')
    sources = {row['path']: row for row in frozen['test']}
    rows = {row['task_id']: row for row in batch['results']}
    if len(rows) != len(batch['results']) or set(rows) != set(sources):
        raise ValueError('Baseline results do not cover the exact frozen test set')
    if set(batch['task_ids']) != {row['record_id'] for row in rows.values()}:
        raise ValueError('Baseline task membership changed')
    current = app.harness_snapshot(app.bind(cfg))
    task_hashes, changes, old_harness = {}, set(), None
    for source_id, row in rows.items():
        task = app.store.task(row['record_id'])
        if not task.get('finalized_at') or task.get('execution_mode') or task['status'] not in {'completed', 'failed'}:
            raise ValueError('A baseline task is still executing or was interrupted')
        if task.get('campaign_id') != record['id'] or task['version'] != parent or task['split'] != 'test':
            raise ValueError('Baseline task scope or skill version changed')
        snapshot = load(app.store.task_path(task['id'])/'snapshot.json')
        if snapshot != app.versions.snapshot(parent) or task.get('snapshot_sha256') != digest(snapshot):
            raise ValueError('Baseline skill snapshot changed')
        if len(task['inputs']) != 1 or task['inputs'][0]['sha256'] != sources[source_id]['sha256'] or \
                row.get('source_sha256') != sources[source_id]['sha256']:
            raise ValueError('Baseline source identity changed')
        item = task['inputs'][0]
        if sha(app.store.task_path(task['id'])/item['local_path']) != item['sha256']:
            raise ValueError('Archived baseline input changed')
        original_cfg = {**cfg, 'outer_rsi': False, 'trial_enabled': False}
        if sources[source_id].get('benchmark'):
            if task.get('benchmark',{}).get('case') != sources[source_id]['benchmark']['case']:
                raise ValueError('Baseline benchmark case changed')
            evaluator=app.campaigns.evaluator(cfg)
            times=evaluator.timestamps(app.store.task_path(task['id'])/item['local_path'])
            original_cfg={**original_cfg,'evaluation':{**cfg['evaluation'],'video_timestamps_s':times}}
            frozen_scoring=load(app.store.task_path(task['id'])/'scoring_identity.json')
            if frozen_scoring != scoring_identity(evaluator.config,task['benchmark']['case']):
                raise ValueError('Baseline scoring resources changed')
        if task['config'] != original_cfg:
            raise ValueError('Baseline execution configuration differs from the frozen campaign')
        harness = task['harness']
        if old_harness is not None and harness != old_harness:
            raise ValueError('Baseline tasks used different harnesses')
        old_harness = harness
        if any(digest(harness.get(key)) != digest(current.get(key)) for key in ('spec', 'documents', 'runtime')):
            raise ValueError('Agent protocol, prompts or dependencies changed; baseline cannot be reused')
        changed = {path for path in harness['code'].keys() | current['code'].keys()
                   if harness['code'].get(path) != current['code'].get(path)}
        if changed - SCHEDULING_FILES:
            raise ValueError('Non-scheduling source changed: ' + ', '.join(sorted(changed - SCHEDULING_FILES)))
        changes.update(changed)
        expected = task['config']['sandbox'].get('resource_identity_sha256')
        if not expected or sha(app.store.root/'data/resources/identity.json') != expected:
            raise ValueError('Frozen resources changed')
        task_hashes[task['id']] = {'task_sha256': sha(app.store.task_path(task['id'])/'task.json'),
                                 'calls_sha256': digest(sorted(app.store.calls(task['id']),
                                                              key=lambda row:json.dumps(row,sort_keys=True)))}
    # The old coordinator might have prepared its next task before reaching the
    # pause boundary. Retain that record, but never replay any paid invocation.
    unstarted = []
    for task_id in set(record.get('task_ids', [])) - set(batch['task_ids']):
        task = app.store.task(task_id)
        if task.get('runs') or app.store.calls(task_id) or task.get('candidate_version'):
            raise ValueError('An evolution invocation already started; automatic baseline continuation is unsafe')
        unstarted.append(task_id)
    cfg['campaign']['auto_adopt'] = True
    return cfg, {'baseline_batch_id': baseline_id, 'baseline_sha256': digest(batch),
                 'baseline_tasks': task_hashes, 'superseded_unstarted_tasks': sorted(unstarted),
                 'original_config_sha256': digest(record['config']),
                 'from_harness': old_harness['id'], 'to_harness': current['id'],
                 'scheduling_source_changes': sorted(changes),
                 'comparison_note': 'Original protocol hashes are retained. Different frozen protocols remain incomparable; '
                                    'test scores are reference only and never block campaign adoption.'}
