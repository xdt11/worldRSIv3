"""Bounded evidence presentation, image encoding and tool feedback."""
import base64
import hashlib
import io
import json
import re
from PIL import Image, ImageOps
from harness.evidence.archive import inside
from harness.evidence.archive import load


def public_source_id(source,index):
    """One identity rule for current records and older records without an ID."""
    return source.get('source_id') or 'media_'+hashlib.sha256(
        f'{index}:{source.get("sandbox_path")}:{source.get("sha256")}'.encode()).hexdigest()[:16]


def text_preview(text, limit=4096):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    if len(text) <= limit: return text
    head,tail = (limit+1)//2,limit//2
    return text[:head] + f'\n[... {len(text)-limit} characters omitted; read the archived log ...]\n' + (text[-tail:] if tail else '')


def evidence_label(record):
    """Small identity beside pixels; full transformations stay in the receipt."""
    keys=('source_id','source_path','path','kind','frame_id','camera_id','shot_id',
          'decoded_time_s','time_s','actual_pts','time_base','region_xyxy','parent_path',
          'original_size_px','presented_size_px','scene_revision')
    label={key:record[key] for key in keys if record.get(key) is not None}
    actual=record.get('decoded_time_s',record.get('time_s'))
    if record.get('requested_time_s') is not None and record['requested_time_s']!=actual:
        label['requested_time_s']=record['requested_time_s']
    if not record.get('frame_id') and record.get('evidence_id'):label['evidence_id']=record['evidence_id']
    return label


def tool_feedback(result, limit=4096):
    """Present receipts, not whole video manifests; raw results remain archived."""
    view = {k:v for k,v in result.items() if k not in {'id','command','elapsed_s'}}
    for field in ('entities','shapes'):
        if isinstance(view.get(field),list):
            view[field+'_count'] = len(view[field])
            view[field] = view[field][:8]
    if isinstance(view.get('bindings'),dict):
        bindings = view['bindings']
        view['bindings_count'] = len(bindings)
        view['bindings'] = {name:{'count':len(rows),'examples':rows[:3]} if isinstance(rows,list) else rows
                            for name,rows in list(bindings.items())[:8]}
    for field in ('metadata','custom_diagnostics'):
        if field in view and len(json.dumps(view[field],ensure_ascii=False)) > limit:
            view[field] = {'text_preview':text_preview(json.dumps(view[field],ensure_ascii=False),limit),
                           'note':'Complete measurements remain in full_result_path.'}
    if isinstance(view.get('images'),list):
        view['image_count']=len(view['images'])
        view['images']=[{'path':row['path']} for row in view['images'] if row.get('path')]
        # Media worker output and adjacent image labels already identify these
        # frames. Preserve the complete raw result in its archive, once.
        if isinstance(view.get('result'),dict):
            view['result']={key:value for key,value in view['result'].items() if key not in {'frames'}}
    elif view.get('evidence_id') and view.get('path'):
        view=evidence_label(view)
    for channel in ('stdout','stderr'):
        if isinstance(view.get(channel),str):
            if channel == 'stdout' and isinstance(view.get('result'),dict):
                view.pop(channel)
            else:
                raw = view[channel]
                view[channel] = text_preview(raw,limit)
                view[channel+'_truncated'] = len(raw) > limit
                if len(raw) > limit:
                    view[channel+'_total_chars'] = len(raw)
        path = view.get(channel+'_path')
        if isinstance(path,str) and path.startswith('tool_logs/'):
            view[channel+'_read_path'] = '/logs/' + path.removeprefix('tool_logs/')
    manifest = view.get('manifest')
    if isinstance(manifest,dict):
        frames = manifest.get('frames',[])
        if not isinstance(frames,list) or any(not isinstance(f,dict) for f in frames): return view
        view['manifest'] = {k:manifest[k] for k in ('engine','engine_version','code_sha256','effective_seed') if k in manifest}
        view['manifest'].update(frame_count=len(frames),
            timestamp_range_s=[frames[0].get('timestamp_s'),frames[-1].get('timestamp_s')] if frames else [],
            native_render_count=len(manifest.get('native_renders',[])),
            frame_examples=[{k:f.get(k) for k in ('path','timestamp_s')} for f in (frames[:1]+frames[-1:] if len(frames)>1 else frames)],
            note='Full per-frame manifest remains at manifest_path and full_result_path.')
    exported = view.get('world_export')
    if isinstance(exported, dict):
        compact = {key: value for key, value in exported.items()
                   if key not in {'timestamps_s', 'artifact_sha256', 'camera', 'object_bindings'}}
        times = exported.get('timestamps_s')
        if isinstance(times, list):
            compact.update(timestamp_count=len(times), timestamp_range_s=[times[0], times[-1]] if times else [])
        artifacts = exported.get('artifact_sha256')
        if isinstance(artifacts, dict):
            compact['artifact_count'] = len(artifacts)
        unsupported = exported.get('unsupported_material_trajectories')
        if isinstance(unsupported, list):
            compact['unsupported_material_trajectories'] = {'count':len(unsupported), 'examples':unsupported[:3]}
        bindings = exported.get('object_bindings')
        if isinstance(bindings, dict):
            compact['object_bindings'] = {'count': len(bindings), 'examples': dict(list(bindings.items())[:3])}
        camera = exported.get('camera')
        if isinstance(camera, dict):
            compact['camera'] = {'fields': list(camera), 'calibration': 'Complete calibration is in world/camera.json'}
        solution = exported.get('solution')
        if isinstance(solution, dict):
            compact['solution'] = {key:value for key,value in solution.items() if key not in {'files','workers'}}
            files=solution.get('files')
            if isinstance(files,list):compact['solution'].update(file_count=len(files),file_examples=files[:3])
            workers=solution.get('workers')
            if isinstance(workers,dict):compact['solution']['worker_count']=len(workers)
            compact['solution']['manifest']='solution/bundle.py contains the complete file and hash manifest.'
        compact['note'] = 'Complete export receipt, timestamps and artifact hashes remain in full_result_path and the render archive.'
        view['world_export'] = compact
    return view


def execution_failure(result):
    if result.get('exit_code') in (141, -13):
        message = 'Command received SIGPIPE: a downstream pipe may have closed early (for example head). Output may be partial; inspect the command and archived logs.'
        return {'code':'tool_execution_failed','message':message,
                'signature':{'code':'tool_execution_failed','exit_code':result['exit_code'],'signal':'SIGPIPE'}}
    stderr = re.sub(r'\x1b\[[0-9;]*m','',result.get('stderr',''))
    stdout = re.sub(r'\x1b\[[0-9;]*m','',result.get('stdout',''))
    # Tracebacks may use application exception names such as RuntimeFailure or
    # StopIteration. Prefer the terminal Python exception over stdout kernel logs.
    lines=[line.strip() for line in stderr.splitlines() if line.strip()]
    trace=stderr if 'Traceback (most recent call last):' in stderr else stdout if 'Traceback (most recent call last):' in stdout else ''
    exception=next((line.strip() for line in reversed(trace.splitlines())
                    if re.match(r'^[A-Za-z_]\w*(?:\.\w+)*:',line.strip())),None)
    if exception is None:
        exception=next((line for line in reversed(lines)
                        if re.match(r'^[\w.]*(?:Error|Exception|Failure):',line)),None)
    if not lines:lines=[line.strip() for line in stdout.splitlines() if line.strip()]
    message = exception or ('Tool timed out' if result.get('exit_code')==124 else
                            (lines[-1][:800] if lines else f'Tool exited with code {result.get("exit_code")}'))
    return {'code':'tool_execution_failed','message':message,
            'signature':{'code':'tool_execution_failed','exit_code':result.get('exit_code'),'cause':message}}


def presentation_metadata(raw):
    with Image.open(io.BytesIO(raw)) as original:
        orientation=int(original.getexif().get(274,1))
        oriented=ImageOps.exif_transpose(original)
        view=oriented.copy();view.thumbnail((1280,1280))
        return {'original_size_px':list(original.size),'oriented_size_px':list(oriented.size),
                'presented_size_px':list(view.size),'exif_orientation':orientation,
                'source_to_presentation':[{'operation':'exif_orientation','orientation':orientation},
                    {'operation':'resize','from_size_px':list(oriented.size),'to_size_px':list(view.size)}]}


def image_item(raw):
    # Presentation resizing affects model context only. Metrics use original RGB.
    with Image.open(io.BytesIO(raw)) as original:
        view = ImageOps.exif_transpose(original).convert('RGB')
        view.thumbnail((1280, 1280))
        out = io.BytesIO()
        view.save(out, format='JPEG', quality=90)
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' +
            base64.b64encode(out.getvalue()).decode()}

def tool(name, description, properties, required):
    return {'type': 'function', 'name': name, 'description': description, 'strict': False,
            'parameters': {'type': 'object', 'properties': properties, 'required': required,
                           'additionalProperties': False}}
