"""Freeze a user's public media as the reference; no benchmark ground truth."""
from fractions import Fraction
from pathlib import Path
import subprocess

from PIL import Image, ImageOps

from harness.evidence.archive import inside, save, sha


def prepare_reference(directory, observation):
    from evaluation.evaluator import video_contract
    directory=Path(directory)
    source=inside(directory,observation['local_path'])
    case=directory/'evaluation/media-reference/case'
    case.mkdir(parents=True,exist_ok=True)
    video=case/'reference.mp4'
    if observation['kind']=='image':
        with Image.open(source) as image:
            image=ImageOps.exif_transpose(image).convert('RGB')
            width,height=image.size
            image.save(case/'reference.png')
        command=['ffmpeg','-v','error','-nostdin','-y','-i',str(case/'reference.png'),'-frames:v','1','-r','1']
        fps=Fraction(1)
    else:
        original=video_contract(source)
        fps=Fraction(original['fps'])
        if fps<=0 or fps>240:raise ValueError('Reference video frame rate must be within 0..240 fps')
        # A frozen CFR reference gives the renderer and scorer the same timeline,
        # including for phone VFR videos. The original remains unchanged.
        command=['ffmpeg','-v','error','-nostdin','-y','-i',str(source),'-map','0:v:0','-an',
                 '-vf',f'fps={fps}', '-vsync','cfr']
        width,height=original['width'],original['height']
    command+=['-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-pix_fmt',
              'yuv420p' if width%2==height%2==0 else 'yuv444p',str(video)]
    try:
        subprocess.run(command,check=True,capture_output=True,timeout=180)
        contract=video_contract(video)
    except (OSError,subprocess.SubprocessError) as exc:
        raise ValueError('Cannot decode or normalize the selected reference media') from exc
    fps=Fraction(contract['fps'])
    times=[float(Fraction(i)/fps) for i in range(contract['frame_count'])]
    result={'kind':observation['kind'],'source_path':observation['path'],'source_sha256':sha(source),
            'reference_path':str(video.relative_to(directory)),'reference_sha256':sha(video),
            'video':contract,'timestamps_s':times,
            'scope':'User media; CFR/H.264 evaluation copy, original input retained. No official case ground truth or question bank.'}
    save(case.parent/'identity.json',result)
    return result
