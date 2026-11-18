"""Decode explicitly requested reference frames and return their image evidence."""
from __future__ import annotations

import subprocess
import bisect
import hashlib
import json

if __package__:
    from harness.tools._media import PATH_SCHEMA,OUTPUT_PROPERTY,checked,output_directory,run_worker,validate_arguments
    from harness.tools.probe_media import video_timeline
else:
    from _media import PATH_SCHEMA,OUTPUT_PROPERTY,checked,output_directory,run_worker,validate_arguments
    from probe_media import video_timeline

NAME='extract_frames'
DESCRIPTION=('Decode requested video frames at explicit timestamps in seconds and return their actual pixels. '
    'Use the valid timestamp range from probe_media or the frozen input timeline; duration itself may be past the last frame. '
    'Returns measured decoded PTS, actual time and request error. Files under /work are input observations, not simulation outputs.')
ARGUMENTS={'type':'object','properties':{'path':PATH_SCHEMA,'output_dir':OUTPUT_PROPERTY,
    'timestamps':{'type':'array','items':{'type':'number','minimum':0},'minItems':1}},
    'required':['path','timestamps'],'additionalProperties':False}
SCHEMA={'type':'function','name':NAME,'description':DESCRIPTION,'strict':False,'parameters':ARGUMENTS}


def execute(ctx,box,path,timestamps,output_dir=None):
    from harness.tools.evidence import view_image,frame_identity,visual_groups
    args={'path':path,'timestamps':timestamps}
    if output_dir is not None:args['output_dir']=output_dir
    result=box.run_media_tool(NAME,args,ARGUMENTS)
    if result.get('exit_code',0) or result.get('error'):return result,None
    frames=result.get('result',{}).get('frames',[])
    if not frames:return result,None
    receipt,visual=view_image(ctx,box,paths=[frame['path'] for frame in frames])
    pixels=[item for item in visual if item['type']=='input_image']
    for image,frame in zip(receipt['images'],frames):
        identity={key:frame[key] for key in ('frame_id','frame_index','requested_time_s','actual_pts','time_base',
                 'decoded_time_s','timestamp_error_s','derivation','source_sha256') if key in frame}
        image.update({**frame_identity(ctx,path,identity),**identity,'source_path':path,'kind':'reference_frame',
                      'time_alignment':'measured_decoded_pts' if 'decoded_time_s' in frame else 'unavailable'})
        image['anchor_role']='temporal_reference'
        if 'requested_time_s' not in image and 'timestamp_s' in frame:
            image['requested_time_s']=frame['timestamp_s']
        _store_provenance(ctx,frame['path'],image)
    return {**result,'images':receipt['images'],'image_count':len(frames)},visual_groups(receipt['images'],pixels)


def _store_provenance(ctx,path,record):
    """Keep source identity recoverable across crop, reopen and later runs."""
    target=ctx.task.directory/'work/evidence-index.json';target.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(target.read_text()) if target.is_file() else {}
    previous[path]=record
    target.write_text(json.dumps(previous,ensure_ascii=False,indent=2))


def worker(args):
    validate_arguments(NAME,args,ARGUMENTS)
    source=checked(args['path']);output=output_directory(args,NAME);frames=[]
    _,stream,timeline=video_timeline(source)
    times=[frame['decoded_time_s'] for frame in timeline]
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    for index,timestamp in enumerate(args['timestamps']):
        if timestamp<times[0]-1e-9 or timestamp>times[-1]+1e-9:
            raise ValueError(f'No decoded frame at {timestamp}s; valid source timestamp range is {times[0]}..{times[-1]}')
        insertion=bisect.bisect_left(times,timestamp)
        candidates=[value for value in (insertion-1,insertion) if 0<=value<len(times)]
        selected=min(candidates,key=lambda value:(abs(times[value]-timestamp),value))
        actual=timeline[selected]
        path=output/f'frame_{index:04d}.png'
        # Index-select the exact decoded frame whose PTS was probed. Accurate
        # seek alone does not identify the timestamp of the pixels returned.
        subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(source),
            '-vf',f'select=eq(n\\,{actual["frame_index"]})','-vsync','0','-frames:v','1',
            '-threads','1','-filter_threads','1','-y',str(path)],capture_output=True,check=True)
        if not path.exists():
            raise ValueError(f'No frame exists at {timestamp}s; choose a timestamp in probe_media valid_timestamp_range_s, not the video duration endpoint')
        from PIL import Image
        with Image.open(path) as decoded_image:
            decoded_size=list(decoded_image.size)
        frame={**actual,'frame_id':f'{source_sha[:16]}:{actual["frame_index"]}',
               'requested_time_s':timestamp,'timestamp_error_s':actual['decoded_time_s']-timestamp,
               'source_sha256':source_sha,'path':str(path),
               'derivation':[{'operation':'decode_frame','source':str(source),'frame_index':actual['frame_index'],
                              'width':stream.get('width'),'height':stream.get('height'),'autorotate':True},
                             {'operation':'video_display_rotation','degrees':stream.get('display_rotation_degrees',0),
                              'source_size_px':[stream.get('width'),stream.get('height')],
                              'decoded_size_px':decoded_size}]}
        frames.append(frame)
    return {'frames':frames,'source':str(source),'sampling':'nearest_measured_decoded_frame'}


def main():run_worker(NAME,ARGUMENTS,worker)

if __name__=='__main__':main()
