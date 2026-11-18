"""Cached downloadable video from immutable native renderer frames."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
import warnings

from PIL import Image, ImageOps
from imageio_ffmpeg import get_ffmpeg_exe

from harness.evidence.archive import load
from harness.evidence.archive import save
from harness.evidence.archive import sha

_ENCODE_LOCK = threading.Lock()
FFMPEG = get_ffmpeg_exe()  # Independently copied Genesis runtime's bundled encoder.
THUMBNAIL_VERSION = '320x180-v1'
_THUMBNAIL_SLOTS = threading.BoundedSemaphore(2)
_THUMBNAIL_LOCK_GUARD = threading.Lock()


def public_input_path(input_root: Path, relative: str) -> Path:
    """Resolve only regular public inputs; never follow a link into GT/resources."""
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or '..' in name.parts:
        raise ValueError('Select a public input path')
    root = Path(input_root).resolve()
    path = root.joinpath(*name.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError('Public input paths cannot contain symlinks')
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Input must remain inside the public input directory')
    if path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.mp4','.mov','.webm','.mkv'}:
        raise ValueError('Select an image or video input')
    return path


def _file_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=2048)
def _content_hash(path, signature):
    result = sha(path)
    if _file_signature(path) != signature:
        raise ValueError('Input changed while its preview was being identified; retry')
    return result


def thumbnail_revision(path: Path) -> str:
    return _content_hash(path, _file_signature(path)) + '-' + THUMBNAIL_VERSION


@lru_cache(maxsize=512)
def _thumbnail_lock(revision):
    return threading.Lock()


def input_thumbnail(path: Path, cache_dir: Path, revision: str) -> Path:
    """Decode one public frame on demand, bounded to two processes and 12 seconds."""
    if revision != thumbnail_revision(path):
        raise ValueError('Input changed before its thumbnail was generated')
    cache = Path(cache_dir)
    target, failure = cache/(revision+'.jpg'), cache/(revision+'.failed.json')
    with _THUMBNAIL_LOCK_GUARD:
        lock = _thumbnail_lock(revision)
    with lock:
        if target.is_file():
            return target
        previous = load(failure, {})
        if previous and time.time()-previous.get('timestamp', 0)<60:
            raise ValueError(previous['error'])
        if not _THUMBNAIL_SLOTS.acquire(timeout=15):
            raise TimeoutError('Thumbnail workers are busy; retry shortly')
        temporary = None
        try:
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix='.thumbnail-', suffix='.jpg', dir=cache, delete=False) as file:
                temporary = Path(file.name)
            if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:
                with warnings.catch_warnings(), Image.open(path) as original:
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    if original.width*original.height > 40_000_000:
                        raise ValueError('Input image is too large for a thumbnail')
                    image = ImageOps.exif_transpose(original).convert('RGBA')
                    image.thumbnail((320,180))
                    canvas = Image.new('RGB',(320,180),'#eef2f8')
                    canvas.paste(image,((320-image.width)//2,(180-image.height)//2),image)
                    canvas.save(temporary,format='JPEG',quality=82)
            else:
                demuxer = 'mov' if path.suffix.lower() in {'.mp4','.mov'} else 'matroska'
                command = [FFMPEG,'-v','error','-nostdin','-y','-max_alloc','134217728',
                    '-protocol_whitelist','file','-f',demuxer,'-threads','1','-i',str(path),'-map','0:v:0','-an','-sn','-dn',
                    '-frames:v','1','-vf','scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2:color=0xeef2f8',
                    '-threads','1','-filter_threads','1','-q:v','3',str(temporary)]
                result = subprocess.run(command,capture_output=True,timeout=12)
                if result.returncode:
                    raise ValueError('Public input thumbnail could not be decoded')
            with Image.open(temporary) as image:
                if image.size != (320,180) or image.format != 'JPEG':
                    raise ValueError('Thumbnail decoder produced an invalid image')
                image.verify()
            if revision != thumbnail_revision(path):
                raise ValueError('Input changed during thumbnail generation; retry')
            os.replace(temporary,target)
            failure.unlink(missing_ok=True)
            return target
        except (OSError, ValueError, subprocess.SubprocessError,
                Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            message = 'Public input thumbnail is unavailable'
            save(failure,{'timestamp':time.time(),'error':message})
            raise ValueError(message) from exc
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
            _THUMBNAIL_SLOTS.release()


def encode_video(render_dir: Path, cache_dir: Path, *, duration_s: float | None = None,
                 video_format: str = 'mp4') -> tuple[Path, dict]:
    """Encode original-view RGB without a Genesis replay or evaluator mutation.

    Timestamps become a variable-frame-rate MP4 (microseconds) or WebM (milliseconds).
    A repeated last image anchors the requested final hold; its timestamp is
    recorded, and it never enters the quantitative evaluation frame manifest.
    """
    if video_format not in {'mp4', 'webm'}:
        raise ValueError('Archive video format must be mp4 or webm')
    root = Path(render_dir).resolve()
    manifest = load(root / 'render_manifest.json', {})
    frames = manifest.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('No native renderer frames are available for video download')
    paths, times, identities, size = [], [], [], None
    for frame in frames:
        name, timestamp = frame.get('path'), frame.get('timestamp_s')
        if (not isinstance(name, str) or PurePosixPath(name).is_absolute()
                or '..' in PurePosixPath(name).parts):
            raise ValueError('Video frame path must remain inside its renderer archive')
        path = root / name
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            raise ValueError('Video frame paths may not use symlinks')
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Video frame path escapes its renderer archive')
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0
                or (times and timestamp <= times[-1])):
            raise ValueError('Video timestamps must be finite, nonnegative and strictly increasing')
        with Image.open(path) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('Video frames must be PNG, JPEG or WebP images')
            current_size = image.size
            image.verify()
        if size is not None and current_size != size:
            raise ValueError('Video frames must have identical dimensions')
        size = current_size
        paths.append(path)
        times.append(float(timestamp))
        identities.append({'path': name, 'timestamp_s': timestamp, 'sha256': sha(path)})
    origin = times[0]
    if duration_s is None:
        if len(times) == 1:
            raise ValueError('A single-frame video requires an explicit duration_s')
        duration_s = times[-1] - origin + times[-1] - times[-2]
        duration_policy = 'Last frame held for the preceding frame interval'
    else:
        duration_policy = 'Explicit requested duration from first render timestamp'
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= times[-1] - origin:
        raise ValueError('Video duration must be finite and extend beyond the final frame timestamp')
    identity = {'protocol': 'native-rgb-vfr-mp4-v1', 'frames': identities,
                'encoder_sha256': sha(FFMPEG),
                'timeline_origin_s': origin, 'duration_s': duration_s,
                'duration_policy': duration_policy, 'endpoint_frame_repeated': True,
                'encoding': {'codec': 'libx264', 'pixel_format': 'yuv420p', 'crf': 18,
                             'threads': 2, 'timestamp_resolution_s': .000001,
                             'padding': 'At most one black row/column for even H.264 dimensions'}}
    if video_format == 'webm':
        identity['protocol'] = 'native-rgb-vfr-webm-v1'
        identity['encoding'].update(codec='libvpx-vp9', pixel_format='yuv420p', crf=28,
                                    timestamp_resolution_s=.001,
                                    padding='At most one black row/column for even preview dimensions')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache = Path(cache_dir)
    target, receipt = cache / (key + '.' + video_format), cache / (key + '.json')
    with _ENCODE_LOCK:
        if target.is_file() and receipt.is_file():
            return target, load(receipt)
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.video-', dir=cache) as temporary:
            staging = Path(temporary)
            lines = ['ffconcat version 1.0']
            for index, path in enumerate(paths):
                name = f'{index:08d}{path.suffix.lower()}'
                shutil.copyfile(path, staging / name)
                end = times[index + 1] - origin if index + 1 < len(times) else duration_s
                lines.extend([f"file '{name}'", 'option framerate 1000000',
                              f'duration {end - (times[index] - origin):.6f}'])
            lines.extend([f"file '{len(paths) - 1:08d}{paths[-1].suffix.lower()}'", 'option framerate 1000000'])
            source = staging / 'frames.ffconcat'
            source.write_text('\n'.join(lines) + '\n')
            output = staging / ('scene.' + video_format)
            codec = (['-c:v', 'libx264', '-preset', 'fast', '-crf', '18'] if video_format == 'mp4'
                     else ['-c:v', 'libvpx-vp9', '-deadline', 'realtime', '-cpu-used', '6',
                           '-crf', '28', '-b:v', '0'])
            container = (['-video_track_timescale', '1000000', '-movie_timescale', '1000000',
                          '-movflags', '+faststart'] if video_format == 'mp4' else [])
            command = [FFMPEG, '-v', 'error', '-nostdin', '-y', '-safe', '0', '-f', 'concat',
                       '-i', str(source), '-an', '-vsync', 'vfr', *codec,
                       '-threads', '2', '-pix_fmt', 'yuv420p',
                       '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', *container, str(output)]
            process = subprocess.run(command, capture_output=True, timeout=300)
            if process.returncode:
                raise ValueError('Video export failed: ' + process.stderr.decode(errors='replace')[-3000:])
            if not output.is_file() or not output.stat().st_size:
                raise ValueError('Video encoder produced no download')
            os.replace(output, target)
            save(receipt, {**identity, 'video_sha256': sha(target),
                           'width': size[0] + size[0] % 2, 'height': size[1] + size[1] % 2,
                           'source_frame_count': len(paths), 'encoded_frame_count': len(paths) + 1})
    return target, load(receipt)
