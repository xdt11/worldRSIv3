"""Read-only, task-local experiment evidence for the safeRSI task critic."""
from __future__ import annotations

from pathlib import PurePosixPath
import subprocess

from harness.evidence.archive import inside, load, sha
from harness.evidence.context import image_item
from harness.evidence.review import bounded_summary, decode_reference_frame, reference_timestamps


def _ordinary(task_directory, root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Experiment evidence must have a task-relative path')
    original = task_directory/str(relative)
    target = inside(root,str(original.relative_to(root)))
    if not target.is_file() or original.is_symlink():
        raise ValueError('Experiment evidence must be an ordinary file')
    if any(parent.is_symlink() for parent in original.parents if parent == root or root in parent.parents):
        raise ValueError('Experiment evidence cannot contain a symlink')
    return target


def critic_rollouts(context, rollout_id=None):
    """Index genuine reports; give initial pixels only for the latest complete run.

    All report/log/setup paths stay inside the recorded experiment directory.
    Video frames are decoded from actual archives and retain their own PTS;
    simulation time is added only when the controller declares its origin.
    """
    directory = context.directory
    summaries, allowed, images, visual = [], set(), [], []
    candidates = []
    suffixes = {'.json','.yaml','.yml','.md','.txt','.py','.csv',
                '.png','.jpg','.jpeg','.webp','.mp4','.mov','.mkv','.webm'}
    for row in context.record.get('rollouts',[]):
        identity = row.get('id')
        if rollout_id is not None and identity != rollout_id:
            continue
        if not isinstance(identity,str) or PurePosixPath(identity).name != identity or identity in {'.','..'}:
            continue
        root = directory/'evaluation/rollouts'/identity
        summary = {key:row[key] for key in ('id','status','report_path') if key in row}
        try:
            report_path = row['report_path']
            report_file = _ordinary(directory,root,report_path)
            report = load(report_file)
            summary['report_sha256'] = sha(report_file)
            summary['result'] = bounded_summary({key:report[key] for key in
                ('status','error','agent_error','agent','reason','metrics','task_spec','scene_source','scene_identity',
                 'source_unchanged','policies','renderer','timeline','duration_s') if key in report})
            episode=report.get('result') or {}
            summary['episode']={key:episode[key] for key in ('simulated_seconds','termination_reason',
                'goal_reached','fallen','goal_distance_m','travel_distance_m','frame_count','render_fps') if key in episode}
            if 'duration_s' in (report.get('setup') or {}):
                summary['episode']['configured_duration_s']=report['setup']['duration_s']
            allowed.add(report_path)
            artifacts=report.get('artifacts',[])
            # Reserve the canonical archive before indexing optional per-frame files.
            # Long recordings can exhaust the limit before the trajectory or videos.
            for artifact in artifacts:
                path=artifact.get('path') if isinstance(artifact,dict) else None
                if not isinstance(path,str): continue
                try: _ordinary(directory,root,path)
                except (OSError,ValueError): continue
                allowed.add(path)
            for target in sorted(root.rglob('*')):
                if target.suffix.lower() not in suffixes or len(allowed) >= 1024:
                    continue
                path = str(target.relative_to(directory))
                try: _ordinary(directory,root,path)
                except (OSError,ValueError): continue
                allowed.add(path)
            process=[]
            for invocation in sorted((root/'invocations').glob('*/invocation.json')):
                metadata=load(invocation,{})
                entry={key:metadata[key] for key in ('id','role','status','error','usage') if key in metadata}
                entry['evidence_paths']=[str(path.relative_to(directory)) for path in
                    (invocation,invocation.parent/'context.json',invocation.parent/'tools.json',invocation.parent/'result.json')
                    if str(path.relative_to(directory)) in allowed]
                process.append(entry)
            summary['evaluator_process']=process
            summary['artifacts'] = artifacts
            if report.get('receipt_path'):
                receipt_path=str((root/report['receipt_path']).relative_to(directory))
                receipt=_ordinary(directory,root,receipt_path)
                summary.update(receipt_path=receipt_path,receipt_sha256=sha(receipt))
                allowed.add(receipt_path)
            if row.get('status') == report.get('status') == 'completed':
                candidates.append((root,artifacts))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            summary['evidence_error'] = str(exc)
        summaries.append(summary)
    if candidates:
        root, artifacts = candidates[-1]
        roles=['first_person_neural','third_person_proxy']
        if any(isinstance(item,dict) and item.get('role')=='first_person_proxy' for item in artifacts):
            roles.insert(1,'first_person_proxy')
        for role in roles:
            artifact = next((item for item in artifacts
                             if isinstance(item,dict) and item.get('role')==role),None)
            if artifact is None:
                visual.append({'role':role,'status':'missing','reason':'No archived video for this perspective'})
                continue
            path = artifact.get('path')
            try:
                target = _ordinary(directory,root,path)
                times = reference_timestamps(target)
                for timestamp in sorted({times[0],times[-1]}):
                    raw, receipt = decode_reference_frame(target,timestamp)
                    label = {'role':role,'path':path,'decoded_time_s':receipt['decoded_time_s'],
                             'frame_index':receipt['frame_index'],'status':'provided'}
                    if type(artifact.get('simulation_start_s')) in (int,float):
                        label['simulation_time_s'] = artifact['simulation_start_s']+receipt['decoded_time_s']
                    visual.append(label)
                    images.append({'role':'user','content':[
                        {'type':'input_text','text':f'Actual rollout evidence: {role}; {path}; video PTS {receipt["decoded_time_s"]} s. Neural appearance is generated; geometry/state remain native.'},
                        image_item(raw)]})
                allowed.add(path)
            except (OSError,ValueError,KeyError,TypeError,IndexError,subprocess.SubprocessError) as exc:
                visual.append({'role':role,'path':path,'status':'error','reason':str(exc)})
    return {'reports':summaries,'initial_video_frames':visual,
            'scope':'Independent execution evidence. Policy outcomes, native physics and neural-renderer failures must be attributed separately.'}, images, allowed
