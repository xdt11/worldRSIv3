"""Persistent evaluation-only native worker; one JSON request/reply per line.

Physics remains in the archived scene's Example.step. No navigator or neural
renderer runs here; the coordinator supplies body velocity commands between
rendered observations and freezes physics while those models are running.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

from harness.tools.go2.native import Go2Injection, MoECTS
from harness.tools.newton.workers.camera_math import look_at,quat_matrix
from harness.tools.newton.workers.scene import load_scene


def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


class Evaluation:
    def __init__(self, request):
        self.code=Path(request['code']).resolve()
        self.output=Path(request['output']).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.config=request['config']
        resources=json.loads(Path(request.get('resources','/opt/resources/go2/manifest.json')).read_text())
        self.resources=resources.get('sandbox',resources)
        cfg=yaml.safe_load(Path(self.resources['moects_config']).read_text())
        self.fps=int(self.config.get('render_fps',10))
        if not 8<=self.fps<=30: raise ValueError('render_fps must be an integer in 8..30')
        self.initial_files=hashes(self.code.parent)
        self.injection=Go2Injection(self.resources['go2_urdf'],cfg,
            self.config['start_position_m'],float(self.config.get('start_yaw_rad',0)))
        world=yaml.safe_load(self.code.with_name('world.yaml').read_text())
        with self.injection.active():
            self.session=load_scene(self.code,world,device=request.get('device','cuda:0'),
                seed=int(request.get('seed',7)))
        if self.session.model is not self.injection.model:
            raise ValueError('Scene primary model differs from the model receiving Go2')
        # Some static archive constructors never needed forward kinematics.
        # Initialize only the appended articulation, preserving source bodies.
        import newton
        import warp as wp
        counts = self.injection.source_counts
        added = wp.array(list(range(counts['articulations'], self.session.model.articulation_count)),
                         dtype=int, device=self.session.model.device)
        newton.eval_fk(self.session.model, self.session.model.joint_q, self.session.model.joint_qd,
                       self.session.native_state, indices=added)
        self.gait=MoECTS(self.session.model,cfg,self.resources['moects_policy'],self.resources.get('torch_site_packages'))
        self.gait.attach(self.session.app, counts)
        self.frames=[];self.segment=0
        self.export_segments = request.get('export_segments', True)
        self.start=self.session.time_s
        if request.get('action') != 'preflight':
            self.capture()

    def timing(self):
        gait = self.gait
        return {'stepping': 'step_physics(dt)' if callable(getattr(self.session.app, 'step_physics', None)) else 'Example.step()',
                'solver': gait.solver_name, 'source_frame_dt_s': self.session.frame_dt,
                'source_sim_dt_s': self.session.physics_dt_s, 'physics_steps': gait.steps,
                'min_physics_dt_s': gait.min_dt_used if gait.steps else None,
                'max_physics_dt_s': gait.max_dt_used if gait.steps else None,
                'motor_period_s': gait.motor_period, 'locomotion_period_s': gait.policy_period,
                'render_fps': self.fps, 'navigation_hz': .5}

    def advance_clock(self, target):
        self.session.advance_physics_to(target, self.gait.max_dt)
        if abs(self.gait.time - (self.session.time_s - self.start)) > 1e-6:
            raise ValueError('Go2 solver-hook integration time does not match the scene clock')

    def preflight(self):
        self.advance_clock(self.start + 1 / self.fps)
        contacts = self.contacts()
        if contacts['status'] != 'available' or not self.gait.updates:
            raise ValueError('Go2 preflight could not observe native contacts and motor control')
        return {'timing': self.timing(), 'contacts': contacts, 'go2': self.gait.pose(self.session.native_state),
                'status': 'verified', 'scope': 'Independent instance; episode starts again at its configured pose'}

    def _first_camera(self):
        pose=self.session.native_state.body_q.numpy()[self.gait.base]
        rotation=quat_matrix(pose[3:])
        settings=self.config.get('camera',{})
        # A front optical camera attached to the physical body; no pose reset.
        offset=np.asarray(settings.get('position_body_m',[.36,0,.06]))
        position=pose[:3]+rotation@offset
        transform=np.eye(4)
        transform[:3,:3]=rotation@np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
        transform[:3,3]=position
        if 'vertical_fov_deg' in settings:
            vfov=float(settings['vertical_fov_deg'])
        elif 'horizontal_fov_deg' in settings:
            width,height=int(settings.get('width',320)),int(settings.get('height',192))
            vfov=math.degrees(2*math.atan(math.tan(math.radians(float(settings['horizontal_fov_deg']))/2)*height/width))
        else: vfov=58.
        return {'camera_to_world':transform.tolist(),'fov_deg':vfov}

    def contacts(self):
        contacts=self.gait.contacts
        if contacts is None or not hasattr(contacts,'rigid_contact_count'):
            return {'status':'unavailable','reason':'Source exposes no native contact buffer'}
        count=int(contacts.rigid_contact_count.numpy()[0])
        shape0=contacts.rigid_contact_shape0.numpy()[:count]
        shape1=contacts.rigid_contact_shape1.numpy()[:count]
        boundary=self.injection.source_counts['shapes']
        labels=list(self.session.model.shape_label)
        pairs=sorted(set((int(a),int(b)) for a,b in zip(shape0,shape1)
                         if a>=0 and b>=0 and (a>=boundary)!=(b>=boundary)))
        return {'status':'available','go2_scene_pairs':[{'shape0':a,'shape1':b,
                 'label0':labels[a],'label1':labels[b]} for a,b in pairs],
                'scope':'Native geometric contact samples, including normal foot-ground contact; not impact or failure counts'}

    def capture(self):
        if self.frames and abs(self.frames[-1]['world_time_s']-self.session.time_s)<1e-8:
            return self.frames[-1]
        index=len(self.frames);stamp=self.session.time_s
        folder=self.output/'frames'/f'{index:06d}';folder.mkdir(parents=True)
        settings=self.config.get('camera',{})
        width,height=int(settings.get('width',320)),int(settings.get('height',192))
        if not 128<=width<=832 or not 128<=height<=832: raise ValueError('Evaluation camera dimensions must be 128..832')
        first=self.session.render(camera_override=self._first_camera(),width=width,height=height,
                                  channels=('rgb','depth','normal'), independent_camera=True)
        native=self.gait.pose(self.session.native_state)
        target=np.asarray(native['position_m'])
        third=self.session.render(camera_override=self.config.get('third_person_camera') or
            {'position':(target+[-2.5,-2.5,1.6]).tolist(),'target':target.tolist(),'up':[0,0,1]},
            width=width,height=height,channels=('rgb',), independent_camera=True)
        Image.fromarray(first['rgb']).save(folder/'first_proxy.png')
        Image.fromarray(third['rgb']).save(folder/'third_proxy.png')
        np.save(folder/'depth.npy',np.asarray(first['depth'],dtype=np.float32))
        np.save(folder/'normal.npy',np.asarray(first['normal'],dtype=np.float32))
        item={'index':index,'time_s':stamp-self.start,'world_time_s':stamp,'first_proxy':str(folder/'first_proxy.png'),
              'third_proxy':str(folder/'third_proxy.png'),'depth':str(folder/'depth.npy'),
              'normal':str(folder/'normal.npy'),'camera':first['camera'],'go2':native,
              'contacts':self.contacts(),
              'geometry_source':'native_warp_ray_intersection','normal_convention':first['normal_convention']}
        self.frames.append(item)
        (self.output/'frames.json').write_text(json.dumps(self.frames,indent=2))
        return item

    def export_segment(self,rows):
        folder=self.output/'segments'/f'{self.segment:04d}';folder.mkdir(parents=True)
        self.segment+=1
        np.save(folder/'depth.npy',np.stack([np.load(row['depth']) for row in rows]))
        np.save(folder/'normal.npy',np.stack([np.load(row['normal']) for row in rows]))
        camera=rows[0]['camera'];intrinsics=np.asarray(camera['intrinsics'])
        metadata={'fps':self.fps,'timestamps_s':[row['time_s'] for row in rows],
                  'camera_intrinsics':{'fx':float(intrinsics[0,0]),'fy':float(intrinsics[1,1]),
                       'cx':float(intrinsics[0,2]),'cy':float(intrinsics[1,2])},
                  'normal_source':'native_warp_ground_truth','normal_coordinates':'opengl_camera_xyz',
                  'depth_source':'native_warp_ground_truth','depth_unit':'m','background_depth':0}
        (folder/'render.json').write_text(json.dumps(metadata,indent=2))
        return str(folder)

    def episode(self,begin):
        rows=self.frames[begin:]
        self.verify_source()
        return {'episode_dir':self.export_segment(rows) if self.export_segments else None,
                'frames':rows,'frame_count':len(rows),
                'time_s':self.session.time_s-self.start,'go2':self.gait.pose(self.session.native_state),
                'source_unchanged':True,'source_model_prefix_sha256':self.injection.source_model_prefix_sha256,
                'original_solver':self.gait.solver_name,'timing':self.timing()}

    def advance(self,request):
        duration=float(request['duration_s'])
        if not math.isfinite(duration) or not 0<duration<=10: raise ValueError('Each closed-loop action must be 0..10 simulated seconds')
        samples=round(duration*self.fps)
        if abs(samples/self.fps-duration)>1e-7: raise ValueError('Action duration must end on the geometry sampling clock')
        command=np.asarray(request['command_body'],np.float32)
        if command.shape!=(3,) or not np.isfinite(command).all(): raise ValueError('command_body must be finite [vx,vy,yaw_rate]')
        self.gait.command=command
        begin=len(self.frames)-1
        before=self.session.time_s
        for index in range(1,samples+1):
            self.advance_clock(before+index/self.fps)
            self.capture()
        return self.episode(begin)

    def verify_source(self):
        if hashes(self.code.parent)!=self.initial_files:
            raise ValueError('Evaluation changed immutable task scene source files')

    def close(self):
        self.verify_source()
        self.session.close()


def serve():
    # Warp and native libraries can write directly to fd 1, bypassing Python's
    # redirect_stdout. Reserve a separate descriptor for the JSON protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    evaluation=None
    for line in sys.stdin:
        request=json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                action=request.get('action')
                if action=='preflight':
                    trial = Evaluation(request)
                    try:
                        result = {'preflight':trial.preflight()}
                    finally:
                        trial.close()
                        del trial
                elif action=='init':
                    if evaluation is not None: raise ValueError('One evaluation per worker')
                    evaluation=Evaluation(request);result=evaluation.episode(0)
                elif action=='capture': result=evaluation.episode(len(evaluation.frames)-1)
                elif action=='advance': result=evaluation.advance(request)
                elif action=='close':
                    if evaluation is not None: evaluation.close()
                    result={'closed':True}
                else: raise ValueError('Unknown evaluation worker action')
            print(json.dumps({'status':'ok',**result},allow_nan=False),file=protocol,flush=True)
            if action=='close': break
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'status':'error','error':str(exc),'error_type':type(exc).__name__}),file=protocol,flush=True)


if __name__=='__main__':
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    serve()
