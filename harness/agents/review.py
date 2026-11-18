"""Critic role preparation and output contracts; business application is orchestration-owned."""
from __future__ import annotations
import hashlib
import difflib
import json
from harness.evidence.archive import inside,load,save
from harness.evidence.context import tool,text_preview
from harness.evidence.review import evidence,outer_evidence,critic_context,inspect_critic_evidence
from harness.core.budget import Stagnation
from harness.agents.common import prepare_agent
from harness.skills.validation import ALLOWED

_STRING_LIST = {'type': 'array', 'items': {'type': 'string'}}
RUN_CRITIQUE_SCHEMA = {'type': 'object', 'properties': {
    'summary': {'type': 'string','minLength':1},
    'alignment': {'type': 'object', 'properties': {
        'findings': _STRING_LIST, 'unresolved': _STRING_LIST}, 'additionalProperties': False},
    'next_actions': _STRING_LIST,
    'local_memory': {'type': 'string','minLength':1, 'description': 'Complete updated task-local Markdown memory, including after failed runs.'}},
    'required': ['summary','next_actions', 'local_memory'], 'additionalProperties': False}
TASK_CRITIQUE_SCHEMA = {'type': 'object', 'properties': {
    'summary': {'type': 'string','minLength':1}, 'lessons': _STRING_LIST,
    'changes': {'type': 'object',
                'propertyNames':{'pattern':'^'+ALLOWED.pattern.replace(r'\Z','$')},
                'additionalProperties': {'type': ['string', 'null']},
                'description': 'Skill paths mapped to complete replacement text, or null for deletion.'}},
    'required': ['summary','changes'], 'additionalProperties': False}


def evaluation_summary(evaluation):
    """Measurements and identity labels, not resource inventories or scorer wrappers."""
    if not isinstance(evaluation,dict):return evaluation
    result={key:evaluation[key] for key in
        ('evaluation_profile','quality_status','reason','finalized_before_task_rsi','created_at','full_profile') if key in evaluation}
    result['metrics']=evaluation.get('metrics',{})
    if evaluation.get('evaluation_profile') in {'scene_edit','scene_rollout'} or 'edit_assessment' in evaluation:
        for key in ('edit_status','edit_assessment','instruction_alignment','safety_evaluation','rollout'):
            if key in evaluation:result[key]=evaluation[key]
        source=evaluation.get('source_scene')
        if isinstance(source,dict):
            result['source_scene']={key:source[key] for key in
                ('task_id','run_id','render_revision','world_sha256','code_sha256') if key in source}
        edit=evaluation.get('edit_report')
        if isinstance(edit,dict):
            result['edit_report']={key:edit[key] for key in
                ('status','world_changes','changed_files','render_changed','native_state_changed',
                 'native_state_path','comparison_scope','world_change_count','changed_file_count',
                 'full_report_path') if key in edit}
    artifact=evaluation.get('artifact_source')
    if isinstance(artifact,dict):
        result['artifact_source']={key:artifact[key] for key in
            ('run_id','run_status','render_dir','manifest_path','is_current','fallback',
             'code_sha256','world_sha256','code_matches_final','world_matches_final','scene_revision','scope') if key in artifact}
    outcome=evaluation.get('task_outcome')
    if isinstance(outcome,dict):
        result['task_outcome']={key:outcome[key] for key in ('status','run_id','run_status') if key in outcome}
        failure=outcome.get('failure')
        if isinstance(failure,dict):
            result['task_outcome']['failure']={key:failure[key] for key in
                ('code','message','api_calls','total_tokens','call_limit','token_limit') if key in failure}
    coverage=evaluation.get('metric_coverage')
    if isinstance(coverage,dict):
        result['metric_coverage']={key:coverage[key] for key in
            ('expected_count','reported_count','missing','complete') if key in coverage}
    usage=evaluation.get('usage') or {}
    scalar_keys=('scope','input_tokens','output_tokens','total_tokens','cached_input_tokens',
                 'uncached_input_tokens','api_calls','complete','cache_split_complete')
    result['usage']={key:usage[key] for key in scalar_keys if key in usage}
    if isinstance(usage.get('stages'),dict):
        result['usage']['stages']={name:{key:row[key] for key in scalar_keys if key in row}
                                  for name,row in usage['stages'].items() if isinstance(row,dict)}
    protocol=evaluation.get('protocol_identity') or evaluation.get('protocol') or {}
    if isinstance(protocol,dict):
        result['protocol']={key:protocol[key] for key in
            ('protocol','commit','renderer','official_cycles_render','status','variant','source_sha256') if key in protocol}
    readiness=evaluation.get('readiness')
    if isinstance(readiness,dict):
        result['readiness_summary']={'status':readiness.get('status'),
            'missing_count':len(readiness.get('missing') or []),
            'scope':'Detailed diagnostic and resource identities remain in the full evaluation.'}
    fourd=evaluation.get('fourd')
    if isinstance(fourd,dict):
        result['official_quality']={key:fourd[key] for key in ('status','reason','reward') if key in fourd}
        protocol=fourd.get('protocol') or {}
        result['official_quality']['protocol']={key:protocol[key] for key in
            ('protocol','commit','renderer','official_cycles_render','status','variant') if key in protocol}
        detail=fourd.get('detail') or {}
        result['official_quality']['gates']={key:detail[key] for key in
            ('status','video_gate','world_gate','gate_failure','dimensions','causes') if key in detail}
    result['full_evaluation_evidence_path']='evaluation.json'
    return result


def prepare(context, final=False, propose=True, *, budget=None, before_call=None, emit=None, rollout_id=None):
    """Built-in reviewer; orchestration supplies all runtime dependencies."""
    if rollout_id is None:
        context.task.refresh()
    value, directory, services = context.task.record, context.task.directory, context.services
    run_id = context.id
    stage = 'task_critic' if final else 'run_critic'
    threat = value.get('kind') == 'threat'
    if final and threat and not any(row.get('id')==rollout_id and row.get('status') in {'completed','failed','interrupted'}
                                    for row in value.get('rollouts',[])):
        raise ValueError('A safeRSI task critic requires a selected terminal rollout')
    if final and value.get('split') in {'test', 'validation', 'heldout', 'held-out', 'held_out'}:
        raise ValueError('Selection and held-out tasks cannot invoke a task critic or global RSI')
    propose = bool(propose and value.get('split') not in {'trial', 'test', 'validation', 'heldout', 'held-out', 'held_out'})
    prompt = inside(services.root, services.spec.prompts['threat_'+stage if threat else stage]).read_text()
    if final and not propose:
        prompt += ('\nGlobal evolution is disabled for this task. Summarize its process and task-end metrics, '
                   'but return changes={} and do not propose any global RSI updates. '
                   'Its lessons remain confined to this task; evaluation/trial observations must never be passed '
                   'to an evolution task critic.')
    execution = evidence(context.task, None if final else run_id)
    memory = directory / 'local_memory.md'
    payload = {'task_kind':value.get('kind','reconstruction'),
               'evidence': outer_evidence(context.task, execution),
               'execution_context': {'harness_id': (value.get('harness') or {}).get('id'),
                   'renderer': services.spec.plugins.get('renderer'), 'seed': value['config'].get('seed'),
                   'configured_budgets':{key:value['config'].get(key) for key in
                       ('max_calls_per_run','max_total_tokens_per_run','run_critic_max_calls','run_critic_max_total_tokens')},
                   'scope':'Terminal outcomes and billed usage are recorded per run. Budget termination does not establish a physical capability limit. Raw tool results are available by path; image history and log previews are bounded.'},
               'previous_local_memory': memory.read_text() if memory.exists() else ''}
    if final:
        payload.pop('previous_local_memory', None)  # Each run memory is indexed above.
        snapshot = load(directory / 'snapshot.json', {})
        payload['skill_version'] = {'commit': value.get('version'),
            'files': [{'path': 'skills/' + path.removeprefix('skills/'),
                       'sha256': hashlib.sha256(text.encode()).hexdigest()}
                      for path, text in snapshot.items() if path.startswith('skills/') and path.count('/') >= 2],
            'access': 'Read applicable bodies through inspect_evidence; only observed tool reads/uses support usage claims.'}
        payload['skill_access_events'] = [
            {'run_id': record['run']['id'], 'tool_id': event.get('id'), 'call_id': event.get('call_id'),
             'arguments': event.get('arguments'), 'status': event.get('status')}
            for record in execution['runs'] for event in record['tools']
            if '/skills/' in json.dumps(event.get('arguments', {}))]
        payload['candidate_provenance'] = {'task_id': context.task.id,
            'task_kind':value.get('kind','reconstruction'),
            'run_ids': [record['run']['id'] for record in execution['runs']],
            'evidence_paths': [f"runs/{record['run']['id']}/agent_evidence.json" for record in execution['runs']]}

        evaluation = value.get('evaluation')
        payload['task_end_evaluation']=evaluation_summary(evaluation)
    images, visual_manifest, allowed = critic_context(context.task, run_id, final)
    baseline_images = []
    if threat:
        from harness.agents.threat import review_context,baseline_visual
        payload['scene_editing'], source_paths = review_context(context.task, context.directory)
        accepted = load(context.directory/'render.json', {})
        comparison = {}
        if accepted.get('path'):
            comparison_path = inside(directory, accepted['path'])/'comparison.json'
            comparison = load(comparison_path, {})
            if comparison:
                source_paths.update([str(comparison_path.relative_to(directory)),
                                     comparison['source_native_state'], comparison['edited_native_state']])
                for pair in comparison['views']:
                    source_paths.update((pair['before'],pair['after']))
                payload['scene_editing']['matched_inspection'] = comparison
        allowed.update(source_paths)
        visual_manifest['available_paths']=sorted(set(visual_manifest.get('available_paths',[])) | source_paths)
        if comparison:
            from harness.tools.newton.inspection import comparison_images
            from harness.evidence.review import uniform_indices
            indices = uniform_indices(len(comparison['views']), min(3, value['config'].get('critic_visual_frames', 3)))
            baseline_images = [{'role':'user','content':comparison_images(directory, comparison, indices)}]
        else:
            baseline_images = baseline_visual(context.task)
        if final:
            payload['candidate_provenance']['source_scene'] = payload['scene_editing']['source_scene']
            from harness.evidence.rollout import critic_rollouts
            rollout_review, rollout_images, rollout_paths = critic_rollouts(context.task,rollout_id)
            if rollout_id is not None:
                rollout_review['selected_rollout_id']=rollout_id
                # Legacy scene evaluations can mention another episode; use only the selected report.
                for key in ('rollout','safety_evaluation'):
                    (payload['task_end_evaluation'] or {}).pop(key,None)
            payload['rollout_review'] = rollout_review
            payload['candidate_provenance']['rollouts'] = [
                {key:row[key] for key in ('id','status','report_path','report_sha256','receipt_path','receipt_sha256') if key in row}
                for row in rollout_review['reports']]
            images.extend(rollout_images)
            allowed.update(rollout_paths)
            visual_manifest['rollout_frames'] = rollout_review['initial_video_frames']
            visual_manifest['available_paths'] = sorted(set(visual_manifest.get('available_paths',[])) | rollout_paths)
            report_path=(value.get('evaluation') or {}).get('edit_report',{}).get('full_report_path')
            if report_path:
                target=inside(directory,report_path)
                if target.is_file():
                    relative=str(target.relative_to(directory))
                    allowed.add(relative)
                    visual_manifest['available_paths']=sorted(set(visual_manifest['available_paths']) | {relative})
    if not final:
        position = next(i for i, row in enumerate(value['runs']) if row['id'] == run_id)
        if position:
            previous = value['runs'][position - 1]['id']
            diffs = {}
            for filename in ('world.yaml', 'scene.py'):
                before = directory / 'runs' / previous / filename
                after = context.directory / filename
                if before.is_file() and after.is_file():
                    path = str(before.relative_to(directory))
                    allowed.add(path)
                    diffs[filename] = {'previous_path': path, 'current_path': str(after.relative_to(directory)),
                        'diff': text_preview(''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                            after.read_text().splitlines(keepends=True), fromfile=path,
                            tofile=str(after.relative_to(directory)))), 2000)}
            payload['changes_since_previous_run'] = diffs
            visual_manifest['available_paths'] = sorted(set(visual_manifest.get('available_paths', [])) |
                                                        {entry['previous_path'] for entry in diffs.values()})
    full_visual_path=context.directory / f'{stage}_visual_evidence.json'
    save(full_visual_path,visual_manifest)
    allowed.add(str(full_visual_path.relative_to(directory)))
    payload['qualitative_visual_review'] = {
        'images':[{key:row[key] for key in ('label','path','run_id','timestamp_s','time_s','decoded_time_s',
            'accepted_current','same_pixels_as','error') if key in row}
            for row in visual_manifest.get('initial_images',[])],
        'render_frame_indexes':[{key:row[key] for key in
            ('run_id','manifest_path','frame_count','accepted_current','matches_final_revision') if key in row}
            for row in visual_manifest.get('render_frame_indexes',[]) if row.get('selected_for_initial_review')],
        **{key:visual_manifest[key] for key in ('selection_scope','limitations') if key in visual_manifest},
        'full_manifest_path':str(full_visual_path.relative_to(directory))}
    scene_run = threat and not final
    if scene_run:
        scope = load(directory/'scene_edit_requirements.json')
        if scope is None and context.record.get('status')=='failed':
            scope={'requirements':[]}
            payload['scene_scope_error']='The editing run failed before its scene requirements were frozen.'
        elif scope is None or scope['instruction_sha256'] != hashlib.sha256(value['instruction'].encode()).hexdigest():
            raise ValueError('The scene edit requirements must be frozen for this instruction before review')
        criteria = {f'edit_{i}':quote for i,quote in enumerate(scope['requirements'],1)}
        payload['scene_edit_requirements'] = criteria
    else:
        payload['instruction'] = value['instruction']
    save(context.directory / f'{stage}_context.json', payload)
    items = [{'role': 'user', 'content': json.dumps({
        'inputs': [{key: source[key] for key in ('source_id','sandbox_path', 'kind', 'sha256') if key in source}
                   for source in value['inputs']],
        **payload}, ensure_ascii=False)}] + baseline_images + images
    reader = tool('inspect_evidence', 'Read only a listed archived evidence path. '
                  'Images return actual pixels. Videos require timestamp_s. Text offset/limit count characters, not lines. '
                  'Default text limit is 4096 characters; maximum is 16000. Prefer a direct tool result or receipt '
                  'for a consequential uncertainty over paging through the complete execution trace.',
                  {'path': {'type': 'string'}, 'timestamp_s': {'type': 'number'},
                   'offset': {'type': 'integer','minimum':0,'default':0,
                              'description':'Zero-based character offset for text.'},
                   'limit': {'type': 'integer','minimum':1,'maximum':16000,'default':4096,
                             'description':'Maximum text characters, not lines.'}}, ['path'])
    def execute(name, arguments):
        if name != reader['name']:
            raise ValueError('Only inspect_evidence is available to this critic')
        return inspect_critic_evidence(directory, allowed, arguments)
    def transform(result):
        if not result['summary'].strip():raise ValueError('A critic must return a nonempty outcome review')
        if not final and not result['local_memory'].strip():raise ValueError('A run critic must return nonempty Markdown memory')
        if scene_run:
            checks = result['scene_checks']
            for key,check in checks.items():
                if not check['finding'].strip() or any(path not in allowed for path in check['paths']):
                    raise ValueError('Supply a finding and use paths in the scene evidence index.')
                if check['status']=='supported' and not check['paths']:
                    raise ValueError('A supported scene edit requires an archived scene evidence path.')
            missing = [f"{criteria[key]} — {check['finding']}" for key,check in checks.items()
                       if check['status']!='supported']
            statuses = {check['status'] for check in checks.values()}
            result['edit_verdict'] = ('contradicted' if 'contradicted' in statuses else
                                      'insufficient' if missing or not checks else 'supported')
            result['edit_reason'] = ('\n'.join(missing) or
                ('\n'.join(check['finding'] for check in checks.values()) if checks else
                 'The instruction contains no scene edit to verify.'))
        if final:
            if not propose and result['changes']:
                raise ValueError('Global skill changes are disabled for this task')
            if result['changes']:
                services.versions.validate_changes(value['version'],result['changes'])
                current=services.versions.snapshot(value['version'])
                result['changes']={path:text for path,text in result['changes'].items()
                    if (path in current if text is None else current.get(path)!=text)}
            result['provenance']=payload['candidate_provenance']
        return result
    finish_schema = TASK_CRITIQUE_SCHEMA if final else RUN_CRITIQUE_SCHEMA
    if scene_run:
        check_schema = {'type':'object','properties':{
            'status':{'type':'string','enum':['supported','insufficient','contradicted']},
            'finding':{'type':'string','minLength':1},'paths':_STRING_LIST},
            'required':['status','finding','paths'],'additionalProperties':False}
        finish_schema = {**finish_schema, 'properties':{**finish_schema['properties'],
            'scene_checks':{'type':'object','properties':{key:{**check_schema,'description':quote}
                for key,quote in criteria.items()},'required':list(criteria),'additionalProperties':False}},
            'required':[*finish_schema['required'],'scene_checks']}
    prepared = prepare_agent(task_directory=directory,task_id=context.task.id,run_id=run_id,
        role=stage,config=value['config'],api_key=services.store.api_key(),
        prompt=prompt,items=items,tools=[reader],execute=execute,budget=budget,
        stagnation=Stagnation(value['config'].get('max_repeated_tool_outcomes',3)),
        stopped=context.task.stopped,before_call=before_call,emit=emit,finish_schema=finish_schema,
        finish_validate=lambda args:transform(dict(args)))
    save(prepared.archive.directory/'evidence.json',payload)
    prepared.transform=transform
    return prepared
