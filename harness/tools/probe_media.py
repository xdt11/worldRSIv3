"""Inspect image dimensions or the actual decoded video timestamp range."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
import subprocess

if __package__:
    from harness.tools._media import PATH_SCHEMA,checked,run_worker,validate_arguments
else:
    from _media import PATH_SCHEMA,checked,run_worker,validate_arguments

NAME='probe_media'
DESCRIPTION=('Read image dimensions or video size, frame count, fps and valid decoded timestamp range. '
    'Returns metadata, not image pixels. Paths must be under /input or /work.')
ARGUMENTS={'type':'object','properties':{'path':PATH_SCHEMA},'required':['path'],'additionalProperties':False}
SCHEMA={'type':'function','name':NAME,'description':DESCRIPTION,'strict':False,'parameters':ARGUMENTS}


def execute(ctx,box,path):
    return box.run_media_tool(NAME,{'path':path},ARGUMENTS),None


def video_timeline(source):
    """Return actual decoded frame identities, never an inferred CFR clock."""
    # Keep stream side-data separate from frame selection. Older ffprobe builds
    # cannot address stream_side_data and some emit invalid JSON when the generic
    # side_data selector is combined with frames.
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-show_format','-of','json',str(source)],capture_output=True,text=True,check=True)
    metadata=json.loads(result.stdout);streams=metadata.get('streams',[])
    if not streams:raise ValueError('Media has no video stream')
    stream=streams[0];frames=[]
    rotation=next((entry['rotation'] for entry in stream.get('side_data_list',[]) if 'rotation' in entry),
                  stream.get('tags',{}).get('rotate',0))
    stream['display_rotation_degrees']=float(rotation)
    decoded=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp,best_effort_timestamp_time','-show_frames',
        '-of','json',str(source)],capture_output=True,text=True,check=True)
    for index,frame in enumerate(json.loads(decoded.stdout).get('frames',[])):
        value=frame.get('best_effort_timestamp_time')
        raw=frame.get('best_effort_timestamp')
        if raw is not None and stream.get('time_base'):
            timestamp=float(int(raw)*Fraction(stream['time_base']))
        elif value is not None:
            timestamp=float(value)
        else:
            raise ValueError(f'Decoded frame {index} has no measured timestamp; cannot infer motion timing')
        frames.append({'frame_index':index,'actual_pts':int(raw) if raw is not None else None,
                       'time_base':stream.get('time_base'),'decoded_time_s':timestamp})
    if not frames:raise ValueError('Media contains no decoded video frames')
    if any(a['decoded_time_s']>b['decoded_time_s'] for a,b in zip(frames,frames[1:])):
        raise ValueError('Decoded timestamps are not ordered; explicit segment handling is required')
    return metadata,stream,frames


def worker(args):
    validate_arguments(NAME,args,ARGUMENTS)
    from PIL import Image
    source=checked(args['path'])
    if source.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:
        with Image.open(source) as image:
            return {'kind':'image','width':image.width,'height':image.height,'mode':image.mode,
                    'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    metadata,stream,frames=video_timeline(source)
    times=[frame['decoded_time_s'] for frame in frames]
    try:fps=float(Fraction(stream.get('avg_frame_rate','0/1')))
    except (ValueError,ZeroDivisionError):fps=None
    try:duration=float(stream.get('duration') or metadata.get('format',{}).get('duration'))
    except (ValueError,TypeError):duration=None
    examples=[times[i] for i in sorted({0,len(times)//2,len(times)-1})] if times else []
    return {'kind':'video','path':str(source),'width':stream.get('width'),'height':stream.get('height'),
        'fps':fps,'frame_count':len(times),'duration_s':duration,
        'time_base':stream.get('time_base'),'source_start_time_s':stream.get('start_time'),
        'display_rotation_degrees':stream.get('display_rotation_degrees',0),
        'valid_timestamp_range_s':[min(times),max(times)] if times else None,'frame_timestamp_examples_s':examples,
        'timestamp_note':'Range/examples use measured decoded PTS in the source clock. Duration may extend past the final frame. Source time is not automatically simulation time.'}


def main():run_worker(NAME,ARGUMENTS,worker)

if __name__=='__main__':main()
