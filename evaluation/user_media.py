"""Official reference-video and geometry metrics for user-owned media."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import uuid

from harness.evidence.archive import inside, sha

GEOMETRY=('mesh','interpenetration')
IMAGE=('semantic','depth','uni3d',*GEOMETRY)
VIDEO=('semantic','geophys','flow','track2d','depth','uni3d',*GEOMETRY)
GROUPS={'semantic':'semantic','geophys':'geophys','flow':'flow','track2d':'tracks','depth':'depth','uni3d':'moge'}


def initialize_result(task,result,*,render_only=False):
    """Keep applicability honest even if scoring cannot start."""
    from evaluation.evaluator import OFFICIAL_KEYS,TYPE_KEYS,PROTOCOL
    media=task.get('media_reference')
    temporal=media and media['kind']=='video' and media.get('video',{}).get('frame_count',2)>1
    types=(('semantic','geophys') if temporal else ('semantic',) if media else ()) if render_only else VIDEO if temporal else IMAGE if media else GEOMETRY
    expected=['4dcodebench/'+key for name in types for key in TYPE_KEYS[name] if key!='uni3d_point_scene']
    result.update(full_profile=False,evaluation_profile='user_media',requested_types=list(types),
        reference_scope='uploaded_media' if media else 'generated_geometry_only',
        profile_note='Official metric implementations on user media; no official case ground truth, annotation mask or VQA questions.')
    for key in OFFICIAL_KEYS:
        name='4dcodebench/'+key
        if name in expected:continue
        reason=('Native camera/material profile is outside the fixed-camera official geometry contract' if render_only and key not in {'semantic_dinov3','semantic_tips','geophys_dinov2','geophys_dinov3'}
                else 'No reference 3D world' if key in {'scene_3d','trajectory_dtw','emd_step','occupancy_dtw','uni3d_point_scene'}
                else 'No annotated reference dynamic mask' if key=='dynamic_iou'
                else 'No temporal video reference' if media else 'No selected reference media')
        result['metrics'][name]={'status':'not_applicable','value':None,'reason':reason,'protocol':PROTOCOL,'source':'user_media_profile'}
    for key in ('vqa_accuracy','vqa_balanced_accuracy'):
        result['metrics']['4dcodebench/'+key]={'status':'not_applicable','value':None,
            'reason':'User media has no official VQA question bank','source':'user_media_profile'}
    return types,expected


def evaluate_media(evaluator,task_dir,task,result,artifact_dir,artifact_source,execution,checkpoint):
    from evaluation.evaluator import FourDCodeBench,TYPE_KEYS
    from evaluation.groups import metric_groups
    task_dir=Path(task_dir)
    media=task.get('media_reference')
    types,expected=initialize_result(task,result)
    def finish(status,reason=None):
        missing=[name for name in expected if name not in result['metrics']]
        for name in missing:result['metrics'][name]={'status':status if status!='ok' else 'error','value':None,
                                                  'reason':reason or 'Official scorer omitted an applicable metric','source':'task_evaluator'}
        failed=any(result['metrics'][name]['status'] in {'error','not_ready','unavailable'} for name in expected)
        result.update(quality_status='error' if status=='ok' and (missing or failed) else status,
            metric_coverage={'expected_count':len(expected),'reported_count':len(expected)-len(missing),
                             'missing':missing,'complete':not missing},metric_groups=metric_groups(result['metrics']))
        if reason:result['reason']=reason
        return result
    runs=task.get('runs',[]);last=runs[-1] if runs else {}
    if artifact_dir is None and last and (last.get('failure') or last.get('status') not in {None,'completed'}):
        return finish('unavailable','The final reconstruction run failed; no validated archive was explicitly selected')
    if artifact_dir is None:
        artifact_dir=inside(task_dir,last['render_dir']) if last.get('render_dir') else None
    if artifact_dir is None:return finish('unavailable','No native artifact available for evaluation')
    artifact=inside(task_dir,artifact_dir)
    if not (artifact/'world/render.mp4').is_file():return finish('unavailable','Native world export has no render.mp4')
    export_path=artifact/'world_export.json'
    native_export=json.loads(export_path.read_text()) if export_path.is_file() else {}
    if native_export.get('protocol') in {'newton-camera-trajectory-v1','newton-native-render-v1'} or native_export.get('status')=='render_only':
        types,expected=initialize_result(task,result,render_only=True)
        result.update(native_profile=native_export.get('protocol'),camera_profile=native_export.get('camera_profile'),
            profile_note='Native moving-camera/material profile. Official semantic/geophys use render.mp4 only; fixed-camera geometry metrics are explicitly not applicable.')
        if not types:return finish('not_applicable','This native profile has no reference video and no compatible official geometry metrics')
    relative=str(artifact.relative_to(task_dir))
    if artifact_source and artifact_source.get('render_dir',relative)!=relative:raise ValueError('Artifact source does not match selected user-media artifact')
    selected=next((run for run in reversed(runs) if run.get('render_dir')==relative),{})
    if selected and artifact_source and artifact_source.get('run_id',selected.get('id'))!=selected.get('id'):
        raise ValueError('Artifact source run does not own the selected render')
    current=selected==last and last.get('status')=='completed' and not last.get('failure')
    result['artifact_source']={'run_id':selected.get('id'),'is_current':current,'fallback':not current,
                               **(artifact_source or {}),'render_dir':relative}
    result['task_outcome']={'status':task.get('status'),'run_id':last.get('id'),'run_status':last.get('status'),'failure':last.get('failure')}
    root=task_dir/'evaluation/media-reference';case_root=root/'case';data_root=root/'estimates-cache'
    case_root.mkdir(parents=True,exist_ok=True);data_root.mkdir(parents=True,exist_ok=True)
    case='real/'+task['id']
    reference={'case_root':str(case_root),'data_root':str(data_root)}
    try:
        if checkpoint:checkpoint()
        if media:
            path=inside(task_dir,media['reference_path'])
            if sha(path)!=media['reference_sha256']:raise ValueError('Frozen user reference changed')
            observation=next(o for o in task['inputs'] if o['path']==media['source_path'])
            if sha(inside(task_dir,observation['local_path']))!=media['source_sha256']:raise ValueError('Archived user input changed')
        else:
            # Geometry-only metrics never compare with this video; it supplies
            # the native dimensions to the official format gate.
            shutil.copyfile(artifact/'world/render.mp4',case_root/'reference.mp4')
        result['reference_identity']={'source_sha256':media['source_sha256'] if media else None,
                                      'evaluation_video_sha256':sha(case_root/'reference.mp4')}
        result['protocol_sha256']=hashlib.sha256(json.dumps({'config':evaluator.config,'types':types,
            'reference':result['reference_identity'],'source':evaluator.identity()['source']},sort_keys=True).encode()).hexdigest()
        scorer=FourDCodeBench(execution,evaluator.resources)
        active=[];checks=[];preparation=[]
        for name in types:
            if checkpoint:checkpoint()
            ready=evaluator.readiness(case,execution_config=execution,reference=reference,types=[name],estimates_required=False)
            checks.append(ready)
            failure=None
            if ready['status']!='ready':failure=('not_ready','Required scorer runtime or model resources are unavailable')
            elif name in GROUPS:
                destination=root/('prepare_'+uuid.uuid4().hex)
                prepared=scorer.job(case,artifact,destination,stage='prepare',types=[GROUPS[name]],reference=reference,checkpoint=checkpoint)
                preparation.append({**prepared['protocol'],'evidence_path':str(destination.relative_to(task_dir))})
                if prepared['protocol']['status']!='ok':failure=('error','Reference feature preparation failed; inspect '+str(destination.relative_to(task_dir)))
                else:
                    ready=evaluator.readiness(case,execution_config=execution,reference=reference,types=[name])
                    checks.append(ready)
                    if ready['status']!='ready':failure=('not_ready','Reference features are incomplete')
            if failure:
                for key in TYPE_KEYS[name]:
                    if '4dcodebench/'+key in expected:
                        result['metrics']['4dcodebench/'+key]={'status':failure[0],'value':None,'reason':failure[1],'source':'task_evaluator'}
            else:active.append(name)
        result['reference_preparation']=preparation
        result['readiness']={'status':'ready' if all(c['status']=='ready' for c in checks) else 'not_ready',
                             'missing':[m for c in checks for m in c.get('missing',[])],'profiles':checks}
        if not active:return finish('not_ready','No applicable metric has a working scoring runtime')
        destination=task_dir/'evaluation'/('score_'+uuid.uuid4().hex)
        scored=scorer.job(case,artifact,destination,types=active,reference=reference,checkpoint=checkpoint)
        result['evidence_path']=str(destination.relative_to(task_dir));result['fourd']=scored
        result['metrics'].update({name:reading for name,reading in scored['metrics'].items() if name in expected})
        return finish(scored['protocol']['status'])
    except InterruptedError:raise
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return finish('error',str(exc))
