"""Read-only campaign reports derived from v3 receipts and official metrics."""
from datetime import datetime
from statistics import mean
from urllib.parse import urlencode

from harness.evidence.archive import identifier, load, now
from harness.evidence.usage import combine_usage, summarize_usage
from evaluation.groups import metric_groups
from harness.evidence.artifacts import archived_render
from orchestration.report_analysis import enrich_report


def _seconds(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() if value else None


def build_report(application, campaign_id):
    store = application.store
    campaign = load(store.data / 'campaigns' / (identifier(campaign_id) + '.json'))
    if campaign is None:
        raise FileNotFoundError(campaign_id)
    tasks = []
    calls = {}
    for path in (store.data / 'tasks').glob('*/task.json'):
        value = load(path)
        if value.get('campaign_id') != campaign_id and value['id'] not in campaign.get('task_ids', []):
            continue
        receipts = store.calls(value['id'])
        for receipt in receipts:
            calls[(value['id'], receipt['id'])] = receipt
        tasks.append(application.tasks.accounting(value, receipts))
    task_by_id = {task['id']: task for task in tasks}
    versions = list(dict.fromkeys([campaign.get('baseline_version'),
        *[row.get('tested_version') or row.get('candidate') for row in campaign.get('rounds', [])],
        *[task['version'] for task in tasks]]))
    versions = [version for version in versions if version]
    if not versions:
        versions = [application.versions.active()]

    def case(task):
        source = next((row for row in task['inputs'] if row['path'] == task.get('reference')), None)
        key = (task.get('benchmark') or {}).get('case') or (source or {}).get('path') or task['id']
        raw = task.get('metrics') or {}
        measured = {key: row['value'] for key, row in raw.items()
                    if key.startswith('4dcodebench/') and row.get('status') == 'ok'
                    and type(row.get('value')) in (int, float)}
        run = task.get('runs', [])[-1] if task.get('runs') else None
        artifact = archived_render(store.task_path(task['id']),run) if run else None
        prefix = '/api/tasks/' + task['id']
        render = load(store.task_path(task['id']) / artifact['render_dir'] / 'render_manifest.json', {}) if artifact else {}
        frame = next(iter(render.get('frames', [])), None)
        image = prefix + '/artifact?' + urlencode({'path':artifact['render_dir']+'/'+frame['path']}) if frame else None
        public = prefix + '/artifact?path=' + source['local_path'] if source else None
        evaluation = task.get('evaluation') or {}
        return {'id': task['id'], 'case_key': key, 'title': task['title'], 'kind': (source or {}).get('kind'),
            'metric_groups':metric_groups(raw),
            'status': task['status'], 'split': task['split'], 'metrics': measured, 'metric_results': raw,
            'usage': task.get('usage', {}), 'evolution_usage': task.get('evolution_usage', {}),
            'evaluable': evaluation.get('quality_status') == 'ok',
            'quality_status': evaluation.get('quality_status', 'unavailable'),
            'artifact_source':artifact, 'vqa':evaluation.get('vqa'),
            'evaluation': {'frames': len(render.get('frames', [])),
                'expected_frames': (task.get('benchmark') or {}).get('video', {}).get('frame_count'),
                'issues': evaluation.get('readiness', {}).get('missing', [])},
            'source_dataset': '4DCodeBench', 'input_url': public,
            'reference_url': '/api/inputs/thumbnail?'+urlencode({'path':source['path']}) if source else None,
            'reference_video_url': public if (source or {}).get('kind') == 'video' else None,
            'reference_preview_url': prefix + '/preview?path=' + source['local_path'] if source else None,
            'preview_url': image, 'preview_current': artifact.get('is_current',False) if artifact else False,
            'render_video_webm_url': prefix+'/runs/'+run['id']+'/video?'+urlencode({'revision':artifact['render_revision'],'format':'webm'}) if artifact else None,
            'render_video_url': prefix + '/runs/' + run['id'] + '/video?'+urlencode({'revision':artifact['render_revision']}) if artifact else None,
            'frame_count': len(render.get('frames', [])),
            'reference_duration_s': (task.get('benchmark') or {}).get('video', {}).get('duration_s'),
            'render_duration_s': render.get('frames', [{}])[-1].get('timestamp_s') if render.get('frames') else None}

    evaluation_tasks = [task for task in tasks if task['split'] in {'test', 'validation', 'held-out', 'heldout'}]
    rows = []
    batch_ids=list(dict.fromkeys([campaign.get('baseline_batch_id'),*[r.get('test_batch_id') for r in campaign.get('rounds',[])]]))
    batches=[campaign.get('benchmarks',{}).get(key) for key in batch_ids if key]
    batches=[batch for batch in batches if batch]
    entries=[(batch['version'],set(batch['task_ids'])) for batch in batches] or [(version,None) for version in versions]
    for index, (version, task_ids) in enumerate(entries):
        selected = [task for task in evaluation_tasks if (task['id'] in task_ids if task_ids is not None else task['version']==version)]
        cases = [case(task) for task in selected]
        keys = sorted({key for row in cases for key in row['metrics']})
        rows.append({'index': index, 'label': 'V' + str(index), 'version': version, 'cases': cases,
            'total': len(cases), 'completed': sum(row['status'] == 'completed' for row in cases),
            'evaluable': sum(row['evaluable'] for row in cases),
            'usage': combine_usage(task.get('usage', {}) for task in selected),
            'metrics': {key: mean(row['metrics'][key] for row in cases if key in row['metrics']) for key in keys}})
        current=rows[-1]
        judged=[t for t in selected if (t.get('evaluation') or {}).get('vqa')]
        if judged:
            from evaluation.vqa import aggregate
            source=judged[0]['config']['evaluation']['source']
            source=str((store.root/source).resolve())
            pooled=aggregate(source,[(t.get('benchmark') or {})['case'] for t in selected],
                [store.task_path(t['id']) for t in selected])
            current['vqa']=pooled
            if pooled:
                for name in ('accuracy','balanced_accuracy'):current['metrics']['4dcodebench/vqa_'+name]=pooled[name]
        measured={key:{'status':'ok','value':value} for key,value in current['metrics'].items()}
        current['metric_groups']=metric_groups(measured)
        for group in current['metric_groups']:
            per_case=[next(g for g in metric_groups(t.get('metrics') or {}) if g['id']==group['id']) for t in selected]
            applicable=[g for g in per_case if g['status']!='not_applicable']
            if not applicable:group.update(status='not_applicable',value=None)
            elif any(g['status']!='ok' for g in applicable):group.update(status='partial',value=None)
            elif applicable and group['id']!='vqa':group.update(status='ok',value=mean(g['value'] for g in applicable))
    common = set.intersection(*[{row['case_key'] for row in version['cases'] if row['evaluable']}
                               for version in rows]) if rows else set()
    common_rows = []
    for version in rows:
        cases = [row for row in version['cases'] if row['case_key'] in common]
        keys = {key for row in cases for key in row['metrics']}
        common_rows.append({'index': version['index'], 'metrics': {
            key: mean(row['metrics'][key] for row in cases if key in row['metrics']) for key in keys}})
    start = campaign.get('created_at')
    beginning = _seconds(start)
    timeline, failures, evolution = [], [], []
    for task in tasks:
        created = next((row.get('created_at') for row in task.get('runs', []) if row.get('created_at')), task['created_at'])
        end = task.get('finalized_at') or task.get('updated_at') or created
        index = versions.index(task['version'])
        timeline.append({'id': task['id'], 'label': task['title'], 'split': task['split'],
            'status': task['status'], 'version_index': index,
            'start_s': max(0, (_seconds(created) or beginning) - beginning),
            'end_s': max(0, (_seconds(end) or beginning) - beginning)})
        if task['status'] in {'failed', 'interrupted', 'no_render'} or task.get('error'):
            failures.append({'id': task['id'], 'split': task['split'], 'version_index': index,
                'case_key': (task.get('benchmark') or {}).get('case'),
                'code': (task.get('failure') or {}).get('code', task['status']),
                'message': task.get('error') or 'No accepted final reconstruction', **task.get('usage', {})})
        if task.get('evolution_usage', {}).get('api_calls'):
            evolution.append(task['evolution_usage'])
    rounds = []
    for index, row in enumerate(campaign.get('rounds', []), 1):
        task = task_by_id.get(row.get('task_id'))
        critique = (task or {}).get('critique') or {}
        candidate = row.get('candidate')
        patch = application.versions.diff(row['parent'], candidate) if candidate and row.get('parent') else ''
        changes = [{'path': 'skills/', 'patch': patch,
            'added': sum(line.startswith('+') and not line.startswith('+++') for line in patch.splitlines()),
            'removed': sum(line.startswith('-') and not line.startswith('---') for line in patch.splitlines())}] if patch else []
        rounds.append({**row, 'round': index, 'case': case(task) if task else {},
            'version': row.get('tested_version') or row.get('candidate'),
            'usage': (task or {}).get('evolution_usage', {}), 'diagnosis': critique.get('summary'),
            'lessons': critique.get('lessons', []), 'changes': changes})
    all_usage = summarize_usage(calls.values())
    execution = [row for row in calls.values() if row.get('role', row.get('stage')) in {'agent', 'reconstruction', 'run_critic'}]
    task_usage = summarize_usage(execution)
    end = campaign.get('updated_at') or start
    definitions = sorted({key for version in rows for row in version['cases'] for key in row['metric_results'] if key.startswith('4dcodebench/')})
    datasets = [{'split': split, 'label': split, 'items': [case(task) for task in tasks if task['split'] == split]}
                for split in sorted({task['split'] for task in tasks})]
    report = {'schema_version': 1, 'id': campaign_id, 'title': '4DCodeBench RSI 实验',
        'status': application.campaigns.accounting(campaign)['status'], 'config': campaign.get('config', {}),
        'start': start, 'end': end, 'duration_s': max(0, _seconds(end) - beginning),
        'versions': rows, 'rounds': rounds, 'datasets': datasets,'metric_families':metric_groups({}),
        'totals': {**all_usage, 'tasks': len(tasks), 'evolution': sum(t['split'] == 'evolution' for t in tasks),
            'tests': len(evaluation_tasks), 'evaluable': sum(row['evaluable'] for version in rows for row in version['cases'])},
        'task_usage': task_usage, 'evolution_usage': combine_usage(evolution),
        'evaluation_usage':summarize_usage(r for r in calls.values() if r.get('role',r.get('stage'))=='evaluation_vqa'),
        'stages': [{'stage': key, **value} for key, value in all_usage.get('stages', {}).items()],
        'timeline': timeline, 'failures': failures, 'findings': [], 'pairwise': [],
        'common_cohort': {'case_keys': sorted(common), 'versions': common_rows},
        'caveats': ['Task 评测 tokens 仅包含重建与 run critic；task RSI critic 是独立进化开销。',
            '质量结果采用官方指标及状态；版本图表仅是所展示用例的描述性均值，不是新的采纳评分。',
            '资源缺失、不适用和失败状态保留在原始 task 评测，未以零填充。'],
        'sources': {'snapshot_at': now(), 'campaign_url': '/api/campaigns/' + campaign_id,
            'definitions': [{'label': key, 'direction': '官方定义', 'description': '以task原始评测协议与reward/detail为准'} for key in definitions],
            'notes': ['报告只读取本v3 campaign关联的任务与调用原件，不读取旧工作区。']}}
    return enrich_report(report,application,campaign,task_by_id)
