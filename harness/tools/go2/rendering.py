"""Host-only image-edit and AgentGarten clients; no credentials enter the sandbox."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
import httpx
import numpy as np
from PIL import Image
from harness.evidence.archive import inside, save, sha, now


REFERENCE_PROMPT = '''Edit the FIRST attached image (proxy.png). Return one photorealistic rendering of that exact frame from the robot's onboard camera.

Use proxy.png as the canvas. Keep its framing, horizon, field of view, ground boundaries, silhouettes, occlusions and visible free space in the same image positions. Replace the proxy surfaces with realistic materials and lighting.

The SECOND attachment (original.png) is a complete real photograph supplied for appearance sampling only. Use it to choose surface textures, colors, light and photographic detail. Do not copy Image 2's composition or render a variation of that photograph. For example, if a bench is visible at the right edge of original.png but absent there in proxy.png, keep that part of proxy.png clear.

The task text below describes the experiment. Apply its requested visual conditions, such as fog, to the first image. A start landmark, destination or robot mentioned in the text may be outside the camera view. Show it only if it is already visible in proxy.png.

Experiment context:
'''


def reference_image(proxy, original, output, *, base_url, api_key, instruction, checkpoint=lambda: None, client=None):
    """Generate the initial Go2 observation and first AgentGarten window anchor."""
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise ValueError('A rollout reference is immutable; do not regenerate it')
    checkpoint()
    prompt = REFERENCE_PROMPT + instruction + '\n\nFinal output: render attachment 1 (proxy.png) from its existing camera pose.'
    request = {'model': 'gpt-image-2', 'prompt': prompt, 'size': 'auto', 'n': '1'}
    save(output.with_suffix('.request.json'), {**request, 'created_at': now(),
        'images': [{'role': 'geometry', 'sha256': sha(proxy)}, {'role': 'appearance', 'sha256': sha(original)}]})
    own = client is None
    client = client or httpx.Client(timeout=600, trust_env=False)
    try:
        files = [('image[]', ('proxy.png', Path(proxy).read_bytes(), 'image/png')),
                 ('image[]', ('original.png', Path(original).read_bytes(), 'image/png'))]
        response = client.post(base_url.rstrip('/') + '/images/edits', data=request, files=files,
                               headers={'Authorization': 'Bearer ' + api_key})
        if response.status_code >= 400:
            raise RuntimeError('gpt-image-2 image edit failed with HTTP ' + str(response.status_code))
        body = response.json(); item = body['data'][0]
        if item.get('b64_json'):
            raw = base64.b64decode(item['b64_json'], validate=True)
        else:
            # Provider-generated URLs are fetched only by the trusted controller.
            url = httpx.URL(item['url'])
            if url.scheme != 'https': raise ValueError('Image response URL must use HTTPS')
            result = client.get(str(url)); result.raise_for_status(); raw = result.content
        with Image.open(io.BytesIO(raw)) as image:
            image.convert('RGB').save(output)
        save(output.with_suffix('.receipt.json'), {'status': 'completed', 'model': 'gpt-image-2',
            'reference_sha256': sha(output), 'usage': body.get('usage'), 'created_at': now(),
            'request_id': response.headers.get('x-request-id'), 'scope': 'Appearance generation, not physical ground truth'})
        return output
    finally:
        if own: client.close()


class AgentGarten:
    def __init__(self, url, *, timeout_s=1800, checkpoint=lambda: None, client=None):
        parsed = httpx.URL(url)
        if parsed.host not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Use the existing local renderer through loopback')
        self.url, self.timeout, self.checkpoint = url.rstrip('/'), timeout_s, checkpoint
        self.client = client or httpx.Client(timeout=60, trust_env=False)
        self.owned = client is None

    def close(self):
        if self.owned: self.client.close()

    def render(self, episode, destination, *, fps, width, height, prompt, seed, gpu='auto'):
        episode, destination = Path(episode), Path(destination)
        # Never allow this adapter to fall back to monocular depth or derived normals.
        depth = np.load(episode / 'depth.npy', mmap_mode='r', allow_pickle=False)
        normal = np.load(episode / 'normal.npy', mmap_mode='r', allow_pickle=False)
        if depth.ndim != 3 or normal.shape != (*depth.shape, 3): raise ValueError('Native depth/normal arrays disagree')
        if not 2 <= len(depth) <= 1024: raise ValueError('AgentGarten accepts 2..1024 geometry frames per request')
        if any(not np.isfinite(d).all() or not np.isfinite(n).all() for d, n in zip(depth, normal)):
            raise ValueError('Native geometry is nonfinite')
        self.checkpoint()
        response = self.client.post(self.url + '/api/inputs', json={'path': str(episode.resolve())})
        response.raise_for_status(); input_id = response.json()['id']
        payload = {'input_id': input_id, 'prompt': prompt, 'modality': 'both', 'frames': len(depth),
                   'fps': fps, 'resolution': f'{width}x{height}', 'length_mode': 'custom',
                   'seed': seed, 'decoder': 'tae', 'engine': 'triton', 'gpu': gpu,
                   'start_frame': 0, 'depth_scale': 1}
        response = self.client.post(self.url + '/api/jobs', json=payload)
        response.raise_for_status(); job = response.json(); identity = job['id']
        save(destination.with_suffix('.request.json'), {**payload, 'job_id': identity,
            'geometry': {name: sha(episode / name) for name in ('depth.npy', 'normal.npy', 'reference.png')},
            'submitted_at': now()})
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                self.checkpoint()
                response = self.client.get(self.url + '/api/jobs'); response.raise_for_status()
                listing = response.json(); rows = listing if isinstance(listing, list) else listing.get('jobs', [])
                job = next((row for row in rows if row['id'] == identity), None)
                if job is None: raise RuntimeError('Renderer job disappeared from its queue')
                if job['status'] == 'completed': break
                if job['status'] in {'failed', 'cancelled'}: raise RuntimeError('AgentGarten: ' + str(job.get('error') or job['status']))
                if time.monotonic() >= deadline: raise TimeoutError('AgentGarten timed out')
                time.sleep(.5)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self.client.stream('GET', self.url + f'/files/jobs/{identity}/rendered.mp4') as stream:
                stream.raise_for_status()
                with destination.open('wb') as target:
                    for chunk in stream.iter_bytes(): target.write(chunk)
            response = self.client.get(self.url + f'/files/jobs/{identity}/result.json'); response.raise_for_status()
            receipt = response.json()
            if receipt.get('normal_source') != 'user-provided' or receipt.get('input_kind') != 'depth':
                raise ValueError('Renderer did not use supplied native depth and normals')
            if int(receipt['frames']) != len(depth): raise ValueError('Renderer truncated physical timestamps')
            receipt.update(job_id=identity, video_sha256=sha(destination), geometry_source='warp_ground_truth')
            save(destination.with_suffix('.receipt.json'), receipt)
            return receipt
        except BaseException:
            try: self.client.post(self.url + f'/api/jobs/{identity}/cancel')
            except Exception: pass
            raise


def decode_frame(video, index, output):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-vf', f'select=eq(n\\,{int(index)})',
                    '-frames:v', '1', str(output)], check=True, capture_output=True, timeout=120)
    if not output.is_file(): raise ValueError('The requested rendered observation frame does not exist')
    return output


def decode_observations(video, first, stop, directory, *, frame_offset=0):
    """Decode the newly observed [first, stop) interval in physical frame order."""
    if first < 0 or stop <= first:
        raise ValueError('Invalid rendered observation interval')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = [directory / f'{index + frame_offset:06d}.png' for index in range(first, stop)]
    if any(path.exists() for path in expected):
        raise ValueError('A recorded observation already exists in the requested interval')
    with TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary)
        subprocess.run([
            'ffmpeg', '-v', 'error', '-n', '-i', str(video),
            '-vf', f'trim=start_frame={first}:end_frame={stop}', '-vsync', '0',
            '-frames:v', str(stop - first), '-start_number', str(first + frame_offset),
            str(staging / '%06d.png')
        ], check=True, capture_output=True, timeout=120)
        if not all((staging / path.name).is_file() for path in expected):
            raise ValueError('The renderer video ended before the requested observation interval')
        for path in expected:
            (staging / path.name).rename(path)


def render_observation_chunk(renderer, directory, rows, *, first_new, number, fps, prompt, seed, gpu):
    """Render within one window and append only newly executed frames to the episode."""
    work = directory / 'work'
    start, end = rows[0]['index'], rows[-1]['index']
    anchor = work/'reference.png' if start == 0 else work/'neural_frames'/f'{start:06d}.png'
    episode = work / f'geometry_{number:04}'
    episode.mkdir()
    camera = rows[0]['camera']
    for channel in ('depth', 'normal'):
        shape = (len(rows), camera['height'], camera['width'])
        if channel == 'normal':
            shape += (3,)
        array = np.lib.format.open_memmap(episode/f'{channel}.npy', mode='w+', dtype=np.float32, shape=shape)
        for index, row in enumerate(rows):
            relative = Path(row[channel]).relative_to('/work')
            sample = np.load(inside(work, relative), mmap_mode='r', allow_pickle=False)
            if sample.shape != shape[1:] or sample.dtype != array.dtype:
                raise ValueError('Native ' + channel + ' differs from the camera raster or float32 format')
            array[index] = sample
        del array
    (episode/'reference.png').write_bytes(anchor.read_bytes())
    intrinsics = camera['intrinsics']
    save(episode/'render.json', {'fps': fps, 'camera_intrinsics': {
        'fx':intrinsics[0][0], 'fy':intrinsics[1][1], 'cx':intrinsics[0][2], 'cy':intrinsics[1][2]},
        'geometry_source':'warp_ground_truth', 'timestamps_s':[row['time_s'] for row in rows],
        'normal_convention':'OpenGL camera: X right, Y up, Z back'})
    video = work / f'rendered_{number:04}.mp4'
    receipt = renderer.render(episode, video, fps=fps, width=camera['width'], height=camera['height'],
                              prompt=prompt, seed=seed, gpu=gpu)
    decode_observations(video, first_new-start, len(rows), work/'neural_frames', frame_offset=start)
    return {'window_start_frame':start, 'window_start_s':rows[0]['time_s'], 'last_frame':end,
            'last_time_s':rows[-1]['time_s'], 'physical_frames':len(rows), 'emitted_frame_start':first_new,
            'reference_path':str(anchor.relative_to(directory)), 'reference_sha256':sha(anchor),
            'geometry_only_through_s':rows[-1]['time_s'], 'renderer_job_id':receipt['job_id'],
            'video_path':str(video.relative_to(directory))}
