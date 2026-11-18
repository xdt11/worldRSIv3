"""Causal Go2 experiment: native dynamics -> Warp geometry -> neural video -> navigation."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import select
import subprocess
from tempfile import TemporaryDirectory
import time
import numpy as np
from PIL import Image
from harness.evidence.archive import inside, save, sha
from harness.tools.go2.rendering import AgentGarten, reference_image, decode_frame, render_observation_chunk


RENDER_WINDOW_S = 16
VIDEO_ROLES = {'first_person':'first_person_neural', 'first_person_proxy':'first_person_proxy',
               'third_person':'third_person_proxy'}


def validate_setup(value, limits):
    if not isinstance(value, dict): raise ValueError('The experiment setup must be a JSON object')
    for key in ('start_position_m', 'goal_position_m'):
        vector = value.get(key)
        if not isinstance(vector, list) or len(vector) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError(key + ' must be an explicit finite world-coordinate position')
    if float(value.get('navigation_hz', .5)) != .5: raise ValueError('The Go2 prototype uses xNavDP at exactly 0.5 Hz')
    fps = value.get('render_fps', 10)
    if type(fps) is not int or not 8 <= fps <= 30: raise ValueError('render_fps must be an integer in 8..30')
    maximum = limits.get('max_duration_s', 160)
    duration = value.get('duration_s', min(limits.get('duration_s', maximum), maximum))
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= maximum:
        raise ValueError('The duration exceeds the configured experiment budget')
    if round(duration*fps)<1 or abs(duration*fps-round(duration*fps))>1e-6:
        raise ValueError('Duration must end on a positive geometry sampling timestamp')
    camera = {**{'width': 320, 'height': 256, 'horizontal_fov_deg': 70}, **value.get('camera', {})}
    for key in ('width', 'height'):
        if type(camera[key]) is not int or not 128 <= camera[key] <= 832 or camera[key] % 32:
            raise ValueError('Neural image dimensions must be multiples of 32 within 128..832')
    if not 10 <= camera['horizontal_fov_deg'] <= 150: raise ValueError('Invalid robot camera field of view')
    if len(str(value.get('task_description', ''))) > 12000: raise ValueError('Task description is too long')
    return {**value, 'duration_s': float(duration), 'render_fps': fps, 'navigation_hz': .5,
            'start_yaw_rad': float(value.get('start_yaw_rad', 0)), 'camera': camera}


class JsonWorker:
    """One concrete persistent worker, with cancellation while waiting for stdout."""
    def __init__(self, process, log, checkpoint, timeout=600):
        self.process, self.log, self.checkpoint, self.timeout = process, log, checkpoint, timeout
        self.buffer = b''

    def read(self):
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            self.checkpoint()
            if time.monotonic() > deadline: raise TimeoutError('Go2 worker timed out; see its stderr archive')
            readable, _, _ = select.select([self.process.stdout], [], [], .2)
            if not readable: continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk: raise RuntimeError('Go2 worker exited; see its stderr archive')
            self.buffer += chunk
            if len(self.buffer) > 32 * 1024 * 1024: raise ValueError('Go2 worker message exceeds its limit')
        line, self.buffer = self.buffer.split(b'\n', 1)
        value = json.loads(line)
        if value.get('status') == 'error' or value.get('error'): raise RuntimeError(str(value.get('error')))
        return value

    def request(self, value):
        self.checkpoint()
        self.process.stdin.write((json.dumps(value) + '\n').encode()); self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            try: self.process.stdin.close()
            except BrokenPipeError: pass
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()


def world_to_body(goal, position, quaternion):
    x, y, z, w = quaternion
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return (rotation.T @ (np.array(goal)-np.array(position))).tolist()


def encode_images(paths, fps, output):
    """Encode the recorded frames in order and publish only a complete video."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = (['-c:v','libvpx-vp9','-deadline','realtime','-cpu-used','5','-b:v','0','-crf','28']
             if output.suffix=='.webm' else ['-c:v','libx264','-movflags','+faststart'])
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary)
        for index, source in enumerate(paths):
            (staging/f'{index:06}.png').symlink_to(Path(source).resolve(strict=True))
        encoded = staging/output.name
        subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','2','-framerate',str(fps),
                        '-i',str(staging/'%06d.png'), *codec,'-threads','2','-pix_fmt','yuv420p',str(encoded)],
                       check=True, capture_output=True, timeout=max(120, len(paths)))
        encoded.replace(output)


def execute(box, directory, setup, *, task, resources, config, api_key, checkpoint=lambda: None, update=lambda **_: None):
    """No policy or RGB/depth approximation fallback is allowed on failure."""
    directory = Path(directory); work = directory / 'work'
    setup = validate_setup(setup, config)
    fps = setup['render_fps']
    last_frame = round(setup['duration_s'] * fps)
    decision_frames = round(fps / setup['navigation_hz'])
    window_frames = RENDER_WINDOW_S * fps
    save(directory/'setup.json', setup)
    seed = task['config'].get('seed')
    if seed is None:
        seed = 7
    native = navigation = renderer = None
    frames, observations, decisions, chunks = [], [], [], []
    def local(path):
        if not str(path).startswith('/work/'): raise ValueError('Expected a native work artifact')
        return inside(work, str(path)[6:])
    def merge(rows):
        for row in rows:
            if frames and row['index'] == frames[-1]['index'] and abs(row['time_s'] - frames[-1]['time_s']) < 1e-6:
                continue
            if row['index'] != len(frames) or not math.isclose(row['time_s'], len(frames)/fps, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Native frame index and time must match the geometry sampling clock')
            frames.append(row)
        save(directory/'trajectory.json', frames)
    def start_native(log_name):
        log = (directory/log_name).open('wb')
        process = box.open_process(['python','-u','-m','harness.tools.newton.workers.go2_entry'], stderr=log)
        return JsonWorker(process, log, checkpoint)
    try:
        scene = {'code': '/source/scene/scene.py', 'resources': '/opt/resources/go2/manifest.json',
                 'config': setup, 'device': 'cuda:0', 'seed': seed, 'export_segments': False}
        update(phase='physics_preflight')
        trial = start_native('preflight.stderr')
        try:
            checked = trial.request({**scene, 'action':'preflight', 'output':'/work/native_preflight'})
        finally:
            trial.close()
        save(directory/'physics_preflight.json', checked['preflight'])
        update(phase='native_initialization')
        native = start_native('native.stderr')
        initial = native.request({**scene, 'action':'init', 'output':'/work/native'})
        merge(initial['frames'])
        if not initial.get('source_unchanged', False): raise ValueError('Native environment changed the source scene')
        source_video = next((row for row in task.get('inputs', []) if row['kind'] == 'video'), None)
        source_image = next((row for row in task.get('inputs', []) if row['kind'] == 'image'), None)
        source = source_video or source_image
        if source is None: raise ValueError('The reconstruction has no original video/image for the requested appearance anchor')
        source_path = inside(Path(config['task_directory']), source['local_path'])
        if sha(source_path) != source['sha256']: raise ValueError('Original source input identity changed')
        appearance = directory/'original_reference.png'
        if source_video:
            index = int(setup.get('reference_frame_index', 0))
            if index < 0: raise ValueError('Reference frame index must be nonnegative')
            decode_frame(source_path, index, appearance)
        else:
            index = None
            with Image.open(source_path) as image: image.convert('RGB').save(appearance)
        save(directory/'appearance_source.json', {'source_path': source['local_path'], 'source_sha256': source['sha256'],
            'frame_index': index, 'frame_sha256': sha(appearance), 'role': 'original reconstruction appearance, not geometry'})
        update(phase='reference_image')
        reference = reference_image(local(frames[0]['first_proxy']), appearance, directory/'reference.png',
            base_url=task['config']['base_url'], api_key=api_key, instruction=task['instruction'], checkpoint=checkpoint)
        # Match raster size without changing camera calibration or generating another anchor.
        with Image.open(reference) as image:
            image.resize((setup['camera']['width'], setup['camera']['height']), Image.Resampling.LANCZOS).save(work/'reference.png')
        observations.append({'time_s': frames[0]['time_s'], 'path': 'work/reference.png', 'source': 'gpt-image-2_anchor', 'frame_index': 0})
        save(directory/'observations.json', observations)
        paths = resources['host_paths']
        env = dict(os.environ)
        gpu = str(box.config.get('gpus') or '').removeprefix('device=').strip('"')
        if gpu: env['CUDA_VISIBLE_DEVICES'] = gpu
        log = (directory/'navigation.stderr').open('wb')
        command = [paths['navigation_python'], '-u', str(Path(__file__).with_name('navigation.py')),
            '--repo', paths['navigation_repo'], '--checkpoint', paths['navigation_checkpoint'],
            '--device', 'cuda:0', '--seed', str(seed)]
        navigation = JsonWorker(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0), log, checkpoint)
        navigation_identity = navigation.read()
        renderer = AgentGarten(config['renderer_url'], timeout_s=config.get('render_timeout_s', 1800), checkpoint=checkpoint)
        while len(frames)-1 < last_frame:
            current = len(frames)-1
            elapsed = frames[-1]['time_s']
            checkpoint(); update(phase='navigation', sim_time_s=elapsed)
            state = frames[-1]['go2']
            observation = inside(directory, observations[-1]['path'])
            prediction = navigation.request({'rgb_path': str(observation), 'depth_path': str(local(frames[-1]['depth'])),
                'goal_local': world_to_body(setup['goal_position_m'], state['position_m'], state['quaternion_xyzw']),
                'position': state['position_m'], 'quaternion_xyzw': state['quaternion_xyzw'], 'time_s': elapsed})
            command_body = prediction['command_body']
            end = min(current + decision_frames, last_frame)
            duration = (end - current) / fps
            decisions.append({'time_s': elapsed, 'observation': observations[-1], 'prediction': prediction,
                'command_body': command_body, 'duration_s': duration})
            save(directory/'decisions.json', decisions)
            update(phase='physics', sim_time_s=elapsed)
            result = native.request({'action':'advance', 'command_body':command_body, 'duration_s':duration})
            if not result.get('source_unchanged'): raise ValueError('Source-scene identity was lost during execution')
            merge(result['frames'])
            if len(frames) != end+1 or not math.isclose(result['time_s'], end/fps, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Native worker did not reach the requested experiment time')
            # Eight two-second actions fill a window. Its final observation is
            # already decoded when the next action starts from that same frame.
            start = current - current % window_frames
            update(phase='neural_render', sim_time_s=frames[-1]['time_s'])
            chunks.append(render_observation_chunk(renderer, directory, frames[start:end+1],
                first_new=current+1, number=len(chunks), fps=fps, prompt=task['instruction'],
                seed=seed, gpu=config.get('assigned_renderer_gpu','auto')))
            save(directory/'render_chunks.json', chunks)
            observations.append({'time_s':frames[-1]['time_s'], 'path':f'work/neural_frames/{end:06}.png',
                'frame_index':end, 'source':'agentgarten_video', 'renderer_job_id':chunks[-1]['renderer_job_id']})
            save(directory/'observations.json', observations)
            goal_distance = float(np.linalg.norm(np.array(frames[-1]['go2']['position_m'][:2])-np.array(setup['goal_position_m'][:2])))
            if frames[-1]['go2'].get('fallen') or goal_distance <= float(setup.get('goal_tolerance_m', .4)): break
        neural = [work/'reference.png'] + [work/'neural_frames'/f'{index:06}.png' for index in range(1,len(frames))]
        tracks = {'first_person':neural, 'first_person_proxy':[local(row['first_proxy']) for row in frames],
                  'third_person':[local(row['third_proxy']) for row in frames]}
        for name, images in tracks.items():
            for extension in ('mp4','webm'):
                encode_images(images, fps, directory/'videos'/f'{name}.{extension}')
        positions = np.array([row['go2']['position_m'] for row in frames])
        distance = float(np.linalg.norm(positions[-1,:2]-np.array(setup['goal_position_m'][:2])))
        reached = distance <= float(setup.get('goal_tolerance_m', .4))
        fallen = any(row['go2'].get('fallen', False) for row in frames)
        result = {'status': 'completed', 'simulated_seconds': frames[-1]['time_s'], 'frame_count': len(frames),
            'navigation_hz': .5, 'render_fps': setup['render_fps'], 'goal_distance_m': distance,
            'goal_reached': reached, 'fallen': fallen, 'duration_s':setup['duration_s'],
            'termination_reason':'fallen' if frames[-1]['go2'].get('fallen') else 'goal_reached' if reached else 'duration_limit',
            'travel_distance_m': float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
            'source_unchanged': True, 'original_solver': initial.get('original_solver'),
            'timing': result['timing'], 'physics_preflight_path': 'physics_preflight.json',
            'navigation': navigation_identity, 'locomotion': 'MoE-CTS pretrained 45D/12-joint policy',
            'observations': 'Initial gpt-image-2 anchor, thereafter only AgentGarten video frames; native Warp depth',
            'geometry': 'Native Warp ray-intersection depth and camera-space surface normals',
            'renderer_mode': 'windowed_tail_frame', 'render_window_s':RENDER_WINDOW_S,
            'scope': 'One actual simulated episode; no aggregate safety capability claim'}
        save(directory/'result.json', result)
        return result
    finally:
        if renderer: renderer.close()
        if navigation: navigation.close()
        if native: native.close()
