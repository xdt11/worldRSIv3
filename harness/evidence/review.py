"""Deterministic, task-scoped evidence assembly and read-only inspection."""
from __future__ import annotations
from typing import Any
import hashlib
import difflib
import io
import json
import subprocess
from pathlib import PurePosixPath
from PIL import Image
from harness.evidence.archive import inside,load,save,sha
from harness.evidence.context import image_item,tool,text_preview,tool_feedback,public_source_id,evidence_label
from harness.evidence.artifacts import archived_render
TaskContext=Any


def archive_run_implementation(context,run_id):
    """Freeze current-run Python sources even when native submission failed.

    This is an implementation snapshot, not a claim that these files executed.
    A task critic only reads an existing snapshot; it never snapshots later work
    under an earlier run's identity.
    """
    runs=context.record.get('runs',[])
    if not run_id or not runs or runs[-1]['id']!=run_id or runs[-1].get('status')=='running':return
    root=context.directory/'work';output=context.directory/'runs'/run_id/'implementation'
    if (output/'manifest.json').is_file():return
    excluded={'renders','tool_results','solution','delivery','snapshots','__pycache__'}
    files=[];skipped=[];total=0
    for source in sorted(root.rglob('*.py')):
        relative=source.relative_to(root)
        if any(part.startswith('.') or part in excluded for part in relative.parts):continue
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p!=root.parent):
            skipped.append({'path':str(relative),'reason':'symlink'});continue
        size=source.stat().st_size
        if len(files)>=514 or size>64*1024*1024 or total+size>256*1024*1024:
            skipped.append({'path':str(relative),'reason':'bounded_source_snapshot'});continue
        raw=source.read_bytes();target=output/'files'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        total+=len(raw)
        files.append({'source_path':'/work/'+str(relative),
            'archive_path':str(target.relative_to(context.directory)),
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    receipts=[];receipt_skipped=[];receipt_bytes=0
    finishes=[event for event in run_tools(context.directory,run_id)
              if event.get('name')=='finish' and event.get('status')=='completed']
    declared=(finishes[-1].get('result') or {}).get('paths',[]) if finishes else []
    for name in declared[:32] if isinstance(declared,list) else []:
        try:
            if not isinstance(name,str):raise ValueError('Receipt path must be text')
            relative=PurePosixPath(name)
            if relative.is_absolute():relative=relative.relative_to('/work')
            if not relative.parts or any(part.startswith('.') or part in excluded for part in relative.parts):
                raise ValueError('Receipt must be an ordinary task-local file')
            if str(relative) in {'world.yaml','scene.py','output_contract.json','local_memory.md'}:continue
            if relative.suffix.lower() not in {'.json','.txt','.md','.yaml','.yml','.csv'}:continue
            original=root/str(relative);source=inside(root,str(relative))
            if not source.is_file() or original.is_symlink() or any(p.is_symlink() for p in original.parents if p!=root.parent):
                raise ValueError('Receipt is missing or contains a symlink')
            size=source.stat().st_size
            if len(receipts)>=16 or size>65536 or receipt_bytes+size>262144:
                raise ValueError('Receipt exceeds the bounded small-text archive')
            raw=source.read_bytes();raw.decode('utf-8')
            target=output/'receipts'/str(relative);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            receipt_bytes+=len(raw)
            receipts.append({'source_path':'/work/'+str(relative),'archive_path':str(target.relative_to(context.directory)),
                'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'scope':'Agent-authored local measurement/check; not formal evaluation or independent validation.'})
        except (OSError,ValueError) as exc:
            receipt_skipped.append({'path':str(name),'reason':str(exc)})
    save(output/'manifest.json',{'run_id':run_id,'files':files,'skipped':skipped,
        'agent_authored_receipts':receipts,'receipt_skipped':receipt_skipped,
        'scope':'Workspace Python sources frozen before this run critic. These files may be unexecuted; join native dependency manifests and tool receipts for execution attribution.'})

def run_tools(directory,run_id):
    events=[]
    for path in (directory/'invocations').glob('*/invocation.json'):
        invocation=load(path,{})
        if invocation.get('run_id')==run_id and invocation.get('role')=='reconstruction':
            for event in load(path.parent/'tools.json',[]):
                event=dict(event)
                result_file=inside(path.parent/'tools',str(event.get('id'))+'.json')
                if result_file.is_file() and not result_file.is_symlink():
                    event['result_evidence_path']=str(result_file.relative_to(directory))
                events.append(event)
    return sorted(events,key=lambda row:row.get('created_at',''))


def reference_timestamps(path):
    from harness.tools.probe_media import video_timeline
    return [frame['decoded_time_s'] for frame in video_timeline(path)[2]]


def decode_reference_frame(path,timestamp):
    from harness.tools.probe_media import video_timeline
    _,_,frames=video_timeline(path)
    if timestamp<frames[0]['decoded_time_s']-1e-9 or timestamp>frames[-1]['decoded_time_s']+1e-9:
        raise ValueError('Requested time is outside the decoded source timeline')
    frame=min(frames,key=lambda row:(abs(row['decoded_time_s']-timestamp),row['frame_index']))
    decoded=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-vf',
        f'select=eq(n\\,{frame["frame_index"]})','-vsync','0','-frames:v','1',
        '-threads','1','-filter_threads','1','-f','image2pipe','-vcodec','png','-'],
        capture_output=True,timeout=30)
    if decoded.returncode or not decoded.stdout:
        raise ValueError('Video frame decoding failed: '+decoded.stderr.decode(errors='replace'))
    return decoded.stdout,{**frame,'requested_time_s':timestamp,
                           'timestamp_error_s':frame['decoded_time_s']-timestamp,
                           'alignment':'measured_decoded_pts'}

def usage_summary(usage):
    if not isinstance(usage,dict):return usage
    keys=('input_tokens','output_tokens','total_tokens','api_calls','complete','scope')
    result={key:usage[key] for key in keys if key in usage}
    if isinstance(usage.get('stages'),dict):
        result['stages']={name:{key:row[key] for key in keys if key in row}
                         for name,row in usage['stages'].items() if isinstance(row,dict) and row.get('api_calls',0)}
    return result


def camera_summary(camera):
    """Keep view identity while complete calibration stays in its exact receipt."""
    if not isinstance(camera,dict):return camera
    keys=('id','role','width','height','time_s','camera_time_s','status','reason')
    if not camera.get('id'):keys+=('position','target','up','fov','fov_deg')
    return {key:camera[key] for key in keys if key in camera}


def bounded_summary(value,limit=4096):
    text=json.dumps(value,ensure_ascii=False)
    return value if len(text)<=limit else {'text_preview':text_preview(text,limit),
        'truncated':True,'note':'Complete values remain in this tool result_evidence_path.'}


def state_summary(result):
    """Preserve actual physical measurements, not only a camera and clock."""
    view={key:result[key] for key in ('tick','dt_s','contacts','particle_count','configuration') if key in result}
    for field in ('entities','shapes'):
        if isinstance(result.get(field),list):
            rows=result[field];view[field]=bounded_summary(rows[:8])
            view[field+'_count']=len(rows)
            if len(rows)>8:view[field+'_omitted']=len(rows)-8
    for key in ('metadata','custom_diagnostics'):
        if key in result:view[key]=bounded_summary(result[key])
    return view


def evidence(context: TaskContext, run_id=None):
    """A content-addressed execution view; raw provider envelopes remain archived."""
    value, directory = context.record, context.directory
    if run_id:archive_run_implementation(context,run_id)
    records, documents = [], {}
    locations = {sha(path):str(path.relative_to(directory)) for run in value['runs']
                 for pattern in ('scene_code_*.py','world_state_*.yaml')
                 for path in (directory/'runs'/run['id']).glob(pattern)}
    calls_by_run = {}
    for path in (directory / 'invocations').glob('*/calls/*/meta.json'):
        meta = load(path)
        if meta.get('stage') == 'agent':
            calls_by_run.setdefault(meta['run_id'], []).append((meta, path.parent))

    def document(text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        documents.setdefault(digest, text)
        return {'document_sha256': digest, **({'path':locations[digest]} if digest in locations else {})}

    def preview(text, limit=4000):
        if not isinstance(text, str) or len(text) <= limit:
            return text
        return {'head': text[:limit // 2], 'tail': text[-limit // 2:],
                'omitted_characters': len(text) - limit,
                'note': 'Use inspect_evidence on the archived log to inspect the omitted middle.'}

    for run in value['runs']:
        if run_id and run['id'] != run_id:
            continue
        run_dir = directory / 'runs' / run['id']
        events = run_tools(directory,run['id'])
        executed = {event.get('call_id') for event in events}
        calls = []
        for meta, folder in sorted(calls_by_run.get(run['id'], []),
                                   key=lambda row: row[0].get('created_at', '')):
            response = load(folder / 'response.json', {})
            content = []
            for item in response.get('output', []):
                if item.get('type') == 'message':
                    content.extend(part['text'] for part in item.get('content', [])
                                   if part.get('type') == 'output_text' and part.get('text'))
                elif item.get('type') == 'reasoning':
                    content.extend(part['text'] for part in item.get('summary', []) if part.get('text'))
                elif item.get('type') == 'function_call' and item.get('call_id') not in executed:
                    content.append({'unexecuted_tool_call': {key: item.get(key)
                                   for key in ('call_id', 'name', 'arguments')}})
            calls.append({'call_id': meta['id'], 'created_at': meta.get('created_at'),
                          'status': meta['status'], 'error': meta.get('error'), 'output': content})
        tools = []
        for event in events:
            raw_args = event.get('arguments', '{}')
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    raise ValueError('Tool arguments must be an object')
            except (ValueError, TypeError):
                args = {'invalid_json_arguments': raw_args}
            for field in ('yaml', 'code'):
                if isinstance(args.get(field), str):
                    args[field] = document(args[field])
            result = dict(event.get('result') or {})
            if isinstance(result.get('world_export'), dict):
                result['world_export'] = tool_feedback({'world_export': result['world_export']})['world_export']
            result.pop('command', None)  # Docker/timeout wrapper repeats the actual tool arguments.
            result.pop('id', None)
            manifest = result.pop('manifest', None)
            if isinstance(manifest, dict):
                frames = manifest.get('frames', [])
                result['render_receipt'] = {key: val for key, val in manifest.items()
                                            if key not in {'frames', 'native_renders'}}
                result['render_receipt'].update(frame_count=len(frames),
                    timestamp_range_s=[frames[0].get('timestamp_s'), frames[-1].get('timestamp_s')] if frames else [],
                    distinct_pixel_count=len({frame.get('pixel_sha256') for frame in frames
                                              if frame.get('pixel_sha256')}))
            for channel in ('stdout', 'stderr'):
                # Structured vision output otherwise repeats byte-for-byte as stdout JSON.
                if channel == 'stdout' and isinstance(result.get('result'), dict):
                    result[channel] = '[Structured output is provided once in result; full stdout is archived.]'
                elif channel in result:
                    result[channel] = preview(result[channel])
            tools.append({key: event.get(key) for key in ('id', 'call_id', 'created_at', 'status', 'name','result_evidence_path') if key in event} |
                         {'arguments': args, 'result': result})
        record = {'run': {key: run.get(key) for key in ('id', 'status', 'error', 'failure', 'usage', 'phase', 'created_at', 'agent_finished_at', 'finished_at')},
                  'agent_calls': calls, 'tools': tools}
        if (run_dir/'input_preparation.json').is_file():
            record['input_preparation']={'path':str((run_dir/'input_preparation.json').relative_to(directory)),
                'role':'host_input_preparation','model_api_calls':0}
        if (run_dir/'implementation/manifest.json').is_file():
            record['implementation']={'manifest_path':str((run_dir/'implementation/manifest.json').relative_to(directory)),
                'scope':'Run-local frozen Python sources, including unexecuted or failed custom solver revisions.'}
            record['agent_authored_receipts']=load(run_dir/'implementation/manifest.json',{}).get('agent_authored_receipts',[])
        if record['run'].get('usage') is not None:
            record['run']['usage']=usage_summary(record['run']['usage'])
        # Full WSD/code appear exactly once per unique content, including rejected revisions.
        for field, filename in (('world', 'world.yaml'), ('code', 'scene.py'), ('memory', 'local_memory.md')):
            path = run_dir / filename
            record[field] = {**document(path.read_text()),'path':str(path.relative_to(directory))} if path.exists() else None
        if run_id is None:
            record['qualitative_critique'] = {key: val for key, val in (run.get('critique') or {}).items()
                                               if key != 'local_memory'}
            record['critique_error'] = run.get('critique_error')
        records.append(record)
    return {'runs': records, 'documents': documents}


def submission_summary(event,evidence_path):
    result=event['result']
    summary={'tool_id':event['id'],**{key:result[key] for key in
        ('exit_code','error','stdout_path','stderr_path','archive_dir') if result.get(key) is not None},
        'full_execution_evidence_path':evidence_path}
    alignment=result.get('alignment')
    if isinstance(alignment,dict):
        summary['alignment']={key:alignment[key] for key in
            ('syntax','semantic_alignment','world_sha256','code_sha256') if key in alignment}
    render=result.get('render_receipt')
    if isinstance(render,dict):
        summary['render_receipt']={key:render[key] for key in
            ('engine','engine_version','frame_count','timestamp_range_s','distinct_pixel_count') if key in render}
    exported=result.get('world_export')
    if isinstance(exported,dict):
        summary['world_export']={key:exported[key] for key in
            ('status','reason','protocol','renderer','official_cycles_render','frame_count',
             'camera_profile','metric_scope','unsupported_material_trajectories') if key in exported}
        if isinstance(exported.get('solution'),dict):
            summary['world_export']['solution']={key:exported['solution'][key] for key in
                ('status','verification','asset_count','file_count') if key in exported['solution']}
        if result.get('archive_dir'):
            summary['world_export']['receipt_path']=result['archive_dir']+'/world_export.json'
    return summary


def outer_evidence(context: TaskContext, complete, *, archive=True):
    """Whole-task attribution needs outcomes and failures, not repeated deliberation."""
    records, needed, previous = [], set(), {}
    directory = context.directory
    for record in complete['runs']:
        run_id = record['run']['id']
        path = f'runs/{run_id}/agent_evidence.json'
        if archive:
            # This is a new derived view, never a replacement for original traces.
            used = {record[field]['document_sha256'] for field in ('world', 'code', 'memory') if record.get(field)}
            used.update(arg['document_sha256'] for event in record['tools']
                        for arg in event['arguments'].values()
                        if isinstance(arg, dict) and 'document_sha256' in arg)
            save(directory / path, {'runs': [record],
                 'documents': {key: complete['documents'][key] for key in sorted(used)}})
        counts, failures, submissions, last_state = {}, [], [], None
        for event in record['tools']:
            counts[event['name']] = counts.get(event['name'], 0) + 1
            result = event['result']
            if event['status'] == 'failed' or result.get('error') or result.get('exit_code', 0) != 0:
                failure = {key:event[key] for key in ('id','name','status','result_evidence_path') if key in event}
                failure['full_execution_evidence_path'] = path
                failure['result'] = {key:result[key] for key in ('exit_code','stdout_path','stderr_path') if key in result}
                error = result.get('error')
                if isinstance(error,dict):
                    failure['result']['error'] = {key:error[key] for key in ('code','message','world_path','status') if key in error}
                    if isinstance(error.get('issues'),list):
                        failure['result']['error']['issue_count'] = len(error['issues'])
                elif error:
                    failure['result']['error'] = error
                failures.append(failure)
            if event['name'] == 'submit_scene':
                submissions.append(submission_summary(event,path))
            if (event['name']=='scene_state' and event['status']=='completed'
                    and not result.get('error') and 'time_s' in result):
                summary={'time_s':result['time_s'],**state_summary(result)}
                if 'camera' in result:summary['camera']=camera_summary(result['camera'])
                last_state={key:event[key] for key in ('id','name','status','result_evidence_path') if key in event}
                last_state['result']=summary
        final_outputs = [text for call in record['agent_calls'] for text in call['output'] if isinstance(text, str)]
        compact = {'run': record['run'], 'agent_call_count': len(record['agent_calls']),
                   'tool_counts': counts, 'failures': failures, 'render_submissions': submissions,
                   'last_agent_text': text_preview(final_outputs[-1],4000) if final_outputs else None,
                   'qualitative_critique': record.get('qualitative_critique'),
                   'critique_error': record.get('critique_error'),
                   'world': record['world'], 'code': record['code'], 'memory': record.get('memory'),
                   'full_execution_evidence_path': path}
        if record.get('input_preparation'):
            compact['input_preparation']=record['input_preparation']
        if record.get('implementation'):
            compact['implementation']=record['implementation']
        if record.get('agent_authored_receipts'):
            compact['agent_authored_receipts']=record['agent_authored_receipts']
        if last_state:compact['last_scene_state']=last_state
        compact['revision_diffs'] = {}
        for field in ('world', 'code', 'memory'):
            if not compact.get(field):
                continue
            digest = compact[field]['document_sha256']
            needed.add(digest)
            if field in previous and previous[field] != digest:
                before = complete['documents'][previous[field]]
                after = complete['documents'][digest]
                diff = ''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                                   fromfile=previous[field], tofile=digest))
                compact['revision_diffs'][field] = {'from_sha256': previous[field], 'to_sha256': digest,
                    'diff': text_preview(diff, 2000), 'full_current_evidence_path': path}
            previous[field] = digest
        records.append(compact)
    return {'runs': records, 'documents': {key:text_preview(complete['documents'][key],6000) for key in sorted(needed)},
            'scope': 'Per-run final WSD/code/memory, revision diffs, latest native state, failures and render receipts. '
                     'Use result_evidence_path for one exact tool result and the listed local receipts for measurements. '
                     'The full execution trace is a fallback for a consequential question not resolved by these direct sources.'}


def uniform_indices(count, limit):
    """Deterministic temporal coverage with endpoints, independent of image quality."""
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return sorted({round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def critic_context(context: TaskContext, run_id, final):
    value, directory = context.record, context.directory
    selected = value['runs'] if final else [run for run in value['runs'] if run['id'] == run_id]
    allowed, frame_paths, images, seen, visual, render_index = set(), set(), [], {}, [], []
    limit = value['config'].get('critic_visual_frames', 6)

    def append_image(raw, label, path=None, **metadata):
        if 'camera' in metadata:metadata['camera']=camera_summary(metadata['camera'])
        with Image.open(io.BytesIO(raw)) as source:
            rgb = source.convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        entry = {'label': label, 'path': path, 'rgb_sha256': digest, **metadata}
        if digest in seen:
            entry['same_pixels_as'] = seen[digest]
        else:
            seen[digest] = label
            pixel_metadata={**evidence_label(metadata),**{key:metadata[key] for key in
                ('timestamp_s','accepted_current','matches_final_revision','run_id','source','time_alignment','time_status') if key in metadata}}
            image_label = label + ('\n' + json.dumps(pixel_metadata, ensure_ascii=False) if pixel_metadata else '')
            images.append({'role': 'user', 'content': [
                {'type': 'input_text', 'text': image_label}, image_item(raw)]})
        visual.append(entry)

    def reference_frame(source, timestamp):
        path = inside(directory, source['local_path'])
        try:
            raw,metadata=decode_reference_frame(path,timestamp)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            visual.append({'path': source['local_path'], 'timestamp_s': timestamp, 'error': str(exc)})
            return
        append_image(raw,f'Reference {source["path"]} at actual {metadata["decoded_time_s"]}s',
                     source['local_path'],source_id=source.get('source_id'),**metadata)

    for source_index,original in enumerate(value['inputs']):
        source={**original,'source_id':public_source_id(original,source_index)}
        allowed.add(source['local_path'])
        if source['kind'] == 'image' and value.get('kind') != 'threat':
            append_image(inside(directory, source['local_path']).read_bytes(),
                         'Reference image ' + source['path'], source['local_path'],source_id=source['source_id'])
    for run in selected:
        run_dir = directory / 'runs' / run['id']
        if (run_dir/'input_preparation.json').is_file():
            allowed.add(str((run_dir/'input_preparation.json').relative_to(directory)))
        observations = []
        for filename in ('world.yaml', 'scene.py', 'local_memory.md', 'agent_evidence.json', 'render.json'):
            if (run_dir / filename).exists():
                allowed.add(str((run_dir / filename).relative_to(directory)))
        implementation=run_dir/'implementation'
        if (implementation/'manifest.json').is_file():
            allowed.add(str((implementation/'manifest.json').relative_to(directory)))
            for path in (implementation/'files').rglob('*.py'):
                if path.is_file() and not path.is_symlink():allowed.add(str(path.relative_to(directory)))
            for receipt in load(implementation/'manifest.json',{}).get('agent_authored_receipts',[]):
                path=inside(directory,receipt['archive_path'])
                if path.is_file() and not path.is_symlink():allowed.add(receipt['archive_path'])
        allowed.update(str(path.relative_to(directory)) for path in run_dir.glob('renders/*/world_export.json'))
        for pattern in ('scene_code_*.py','world_state_*.yaml'):
            allowed.update(str(path.relative_to(directory)) for path in run_dir.glob(pattern))
        for event in run_tools(directory,run['id']):
            result = event.get('result', {})
            if event.get('result_evidence_path'):allowed.add(event['result_evidence_path'])
            allowed.update(row['archive_path'] for row in result.get('images',[]) if row.get('archive_path'))
            for key in ('stdout_path', 'stderr_path', 'archive_path'):
                if result.get(key):
                    allowed.add(result[key])
            if (event.get('name') in {'view_image', 'observe_scene', 'extract_frames', 'crop_image'} and event.get('status') == 'completed'
                    and not result.get('error')):
                for row in result.get('images', [result]):
                    if row.get('archive_path'):
                        if event['name']=='observe_scene':
                            receipt=inside(directory,row['archive_path']).with_suffix('.json')
                            if receipt.is_file():
                                allowed.add(str(receipt.relative_to(directory)))
                                row={**load(receipt,{}),**row}
                        observations.append((event, row))
        # Observations can expose failed hypotheses or alternate cameras. They are
        # sampled independently from formal outputs and never labelled as current.
        current = {key: sha(run_dir / filename) if (run_dir / filename).is_file() else None
                   for key, filename in (('code_sha256', 'scene.py'), ('world_sha256', 'world.yaml'))}
        for index in uniform_indices(len(observations), limit):
            event, observation = observations[index]
            path = observation['archive_path']
            metadata = {'source': 'agent_observation', 'accepted_current': False,
                'run_id': run['id'], 'tool_id': event.get('id'), 'tool_name': event['name'],
                'tool_call_id': event.get('call_id'), 'source_path': observation.get('source_path',observation.get('path')),
                'observed_path':observation.get('path'),
                'scope': 'Archived agent observation; not the final accepted reconstruction or an evaluation frame.',
                **{key: observation[key] for key in ('time_s', 'requested_time_s', 'time_status', 'camera',
                     'scene_revision', 'code_sha256', 'world_sha256', 'sha256', 'kind', 'timestamp_s', 'time_alignment', 'region_xyxy') if key in observation},
                'matches_final_revision': {key: observation[key] == digest if digest and observation.get(key) else None
                                           for key, digest in current.items()}}
            label = f'Run {run["id"]} archived {event["name"]} observation {index}; not formal output'
            try:
                target = inside(directory, path)
                if observation.get('sha256') and sha(target) != observation['sha256']:
                    raise ValueError('Archived observation SHA256 differs from its tool receipt')
                append_image(target.read_bytes(), label, path, **metadata)
            except (OSError, ValueError) as exc:
                visual.append({'label': label, 'path': path, **metadata, 'error': str(exc)})
        # Review the latest verified native RGB even when formal submission or the run failed.
        # Other revisions remain indexed, and no historical image becomes a current output.
        try:
            native=archived_render(directory,run)
        except (OSError,ValueError,TypeError) as exc:
            native=None
            visual.append({'run_id':run['id'],'source':'native_render','error':str(exc)})
        chosen=native.get('render_dir') if native else None
        for manifest_file in sorted(run_dir.glob('renders/*/render_manifest.json')):
            relative=str(manifest_file.parent.relative_to(directory))
            frames=load(manifest_file,{}).get('frames',[])
            if not isinstance(frames,list):continue
            manifest_path=str(manifest_file.relative_to(directory))
            allowed.add(manifest_path)
            artifact_paths=[]
            for filename in ('runtime_state.json','render_receipt.json','world_export.json','rebuild_receipt.json'):
                path=manifest_file.parent/filename
                if path.is_file() and not path.is_symlink():
                    relative_path=str(path.relative_to(directory));allowed.add(relative_path);artifact_paths.append(relative_path)
            # The trusted submission worker freezes the complete ordinary-file
            # closure here; custom kernels/helpers must be readable to the RSI
            # critic before it can generalize them into skill support code.
            for path in (manifest_file.parent/'delivery').rglob('*'):
                if path.is_file() and not path.is_symlink():
                    relative_path=str(path.relative_to(directory));allowed.add(relative_path);artifact_paths.append(relative_path)
            index_entry={'run_id':run['id'],'manifest_path':manifest_path,'base_path':relative,
                'frame_count':len(frames),'selected_for_initial_review':relative==chosen,
                'runtime_and_delivery_paths':sorted(artifact_paths),
                'accepted_current':False,
                'access':'Archived native output; join base_path and frame path to inspect a frame. Formal acceptance and task completion are separate.'}
            if relative==chosen:
                current_match={key:native.get(key)==digest if digest and native.get(key) else None
                               for key,digest in current.items()}
                accepted=bool(native.get('is_current') and run.get('status')=='completed'
                              and not run.get('failure') and all(value is not False for value in current_match.values()))
                index_entry.update(accepted_current=accepted,selection='latest_native_output',
                    code_sha256=native.get('code_sha256'),world_sha256=native.get('world_sha256'),
                    code_path=native.get('code_path'),world_path=native.get('world_path'),matches_final_revision=current_match)
                for key in ('code_path','world_path'):
                    if native.get(key):allowed.add(native[key])
            render_index.append(index_entry)
            for frame in frames:
                try:image=inside(manifest_file.parent,frame['path'])
                except (ValueError,KeyError,TypeError):continue
                if image.is_file():
                    path=str(image.relative_to(directory));allowed.add(path);frame_paths.add(path)
            if relative!=chosen:continue
            for index in uniform_indices(len(frames),limit):
                frame=frames[index]
                path=str(inside(manifest_file.parent,frame['path']).relative_to(directory))
                timestamp=frame.get('timestamp_s')
                label=f'Run {run["id"]} latest native output frame {index} at {timestamp}s; accepted_current={accepted}'
                append_image(inside(directory,path).read_bytes(),label,path,source='native_render',
                    run_id=run['id'],time_s=timestamp,render_source_ref=manifest_path,
                    accepted_current=accepted,matches_final_revision=current_match)
    # Reference videos are sampled independently on the reference timeline, not
    # chosen by the agent's preferred render times. Additional evidence is readable.
    for source_index,original in enumerate(value['inputs']):
        source={**original,'source_id':public_source_id(original,source_index)}
        if source['kind'] != 'video' or value.get('kind') == 'threat':
            continue
        try:
            timestamps = reference_timestamps(inside(directory, source['local_path']))
        except Exception as exc:
            visual.append({'path': source['local_path'], 'error': 'Reference timeline unavailable: ' + str(exc)})
            continue
        selected_times = {timestamps[index] for index in uniform_indices(len(timestamps), limit)}
        for timestamp in sorted(selected_times):
            reference_frame(source, timestamp)
    for path in (directory / 'skills').rglob('*'):
        if path.is_file() and not path.is_symlink():
            allowed.add(str(path.relative_to(directory)))
    if final and (directory / 'evaluation.json').is_file():
        allowed.add('evaluation.json')
    manifest = {'protocol': 'uniform_index_including_endpoints; limit=1 selects the first frame', 'max_frames_per_run_or_reference': limit,
                'max_observation_images_per_run': limit,
                'image_presentation': 'RGB JPEG maximum 1280px; originals retained',
                'selection_scope': 'Qualitative review only; evaluator uses its separate frozen protocol.',
                'limitations': 'Initial samples do not prove unsampled temporal behavior. Request more '
                              'evidence with inspect_evidence when needed; do not infer unseen details.',
                'initial_images': visual, 'render_frame_indexes': render_index,
                'available_paths': sorted(allowed - frame_paths)}
    return images, manifest, allowed


def inspect_critic_evidence(directory, allowed, arguments):
    path = arguments['path']
    if path not in allowed:
        raise ValueError('Path is not in this critic evidence allowlist')
    target = inside(directory, path)
    suffix = target.suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.image'}:
        return {'path': path, 'presentation': 'Original archived image, shown as RGB JPEG thumbnail'}, image_item(target.read_bytes())
    if suffix in {'.mp4', '.mov', '.mkv', '.webm'}:
        timestamp = arguments.get('timestamp_s')
        if type(timestamp) not in (int, float) or not 0 <= timestamp < float('inf'):
            raise ValueError('Video evidence requires a finite nonnegative timestamp_s')
        raw,metadata=decode_reference_frame(target,timestamp)
        return {'path':path,**metadata},image_item(raw)
    offset, limit = arguments.get('offset', 0), arguments.get('limit', 4096)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16000:
        raise ValueError('Text offset must be nonnegative and limit must be between 1 and 16000 characters')
    text = target.read_text(errors='replace')
    return {'path': path, 'offset': offset, 'total_characters': len(text),
            'text': text[offset:offset + limit], 'next_offset': offset + limit if offset + limit < len(text) else None}, None
