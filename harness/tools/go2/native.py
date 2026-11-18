"""Append a Go2 to a scene's builder, retaining its source step and solver.

Only the evaluation process uses this module. Source scene files are never
rewritten. Unsupported scenes fail explicitly rather than becoming static maps.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp
import yaml

from harness.tools.newton.workers.camera_math import quat_matrix


def digest(value):
    def convert(item):
        if isinstance(item, dict): return {key: convert(v) for key,v in item.items()}
        if isinstance(item, (tuple,list)): return [convert(v) for v in item]
        if isinstance(item, (str,int,float,bool)) or item is None: return item
        return np.asarray(item).tolist()
    return hashlib.sha256(json.dumps(convert(value),sort_keys=True).encode()).hexdigest()


class Go2Injection:
    """A scoped constructor edit: append one articulation before finalization."""
    fields = ('body_label','body_mass','body_inertia','body_com','body_q',
              'shape_label','shape_body','shape_transform','shape_type','shape_scale',
              'shape_material_mu','shape_material_ke','shape_material_kd',
              'joint_label','joint_q','joint_qd')

    def __init__(self, urdf, config, position, yaw):
        self.urdf, self.config = Path(urdf), config
        self.position, self.yaw = position, yaw
        self.count = 0

    @contextlib.contextmanager
    def active(self):
        original = newton.ModelBuilder.finalize
        def finalize(builder, *args, **kwargs):
            self.count += 1
            if self.count != 1:
                raise ValueError('Go2 evaluation supports exactly one finalized scene model; preserve multi-model custom coupling with an explicit evaluation assembly')
            before = {key:copy.deepcopy(getattr(builder,key)) for key in self.fields if hasattr(builder,key)}
            self.source_counts = {'bodies':builder.body_count,'shapes':builder.shape_count,'joints':builder.joint_count,
                                  'particles':builder.particle_count, 'articulations':builder.articulation_count}
            gravity = copy.deepcopy(builder._gravity)
            robot = newton.ModelBuilder(gravity=builder._gravity)
            robot.default_joint_cfg.target_ke = 0.0
            robot.default_joint_cfg.target_kd = 0.0
            robot.default_joint_cfg.armature = 0.01
            # Resolve the already downloaded official URDF's package meshes
            # explicitly; importing XML avoids fragile ROS package lookup.
            tree = ET.fromstring(self.urdf.read_text())
            for mesh in tree.iter('mesh'):
                filename=mesh.attrib['filename'].replace('package://go2/meshes/','meshes/')
                if not Path(filename).is_absolute(): filename=str((self.urdf.parent/filename).resolve())
                mesh.attrib['filename']=filename
            xml=ET.tostring(tree,encoding='unicode')
            robot.add_urdf(xml, xform=wp.transform(self.position,wp.quat_from_axis_angle(wp.vec3(0,0,1),self.yaw)),
                           floating=True, enable_self_collisions=False, collapse_fixed_joints=True)
            names = self.config['policy_joint_names']
            for name, angle in zip(names,self.config['default_angles']):
                index = next(index for index,label in enumerate(robot.joint_label) if label.rsplit('/',1)[-1]==name)
                robot.joint_q[robot.joint_q_start[index]] = float(angle)
            if any(label.startswith('go2/') for label in builder.body_label):
                raise ValueError('Source already contains a Go2; evaluation must add exactly one embodiment')
            builder.add_builder(robot)
            # add_builder normally inherits the appended builder gravity.
            builder._gravity = gravity
            after = {key:getattr(builder,key)[:len(values)] for key,values in before.items()}
            if digest(before) != digest(after):
                raise ValueError('Appending Go2 altered original scene model parameters')
            self.source_model_prefix_sha256 = digest(before)
            self.model = original(builder,*args,**kwargs)
            return self.model
        newton.ModelBuilder.finalize = finalize
        try:
            yield self
            if self.count != 1: raise ValueError('Scene must finalize one Newton model to add Go2')
        finally:
            newton.ModelBuilder.finalize = original


class MoECTS:
    """Released 45D stateful deployment student, native body/joint readback."""
    def __init__(self, model, cfg, checkpoint, torch_site_packages=None):
        if torch_site_packages and str(torch_site_packages) not in sys.path:
            sys.path.append(str(torch_site_packages))
        import torch
        self.torch, self.model, self.cfg = torch,model,cfg
        torch.set_num_threads(1)
        self.policy = torch.jit.load(str(checkpoint),map_location='cpu').eval()
        if not hasattr(self.policy,'reset'): raise ValueError('Expected stateful MoE-CTS 45D student')
        self.policy.reset()
        labels = list(model.joint_label)
        self.joints = [labels.index('go2/'+name) for name in cfg['policy_joint_names']]
        self.qadr = model.joint_q_start.numpy()[self.joints]
        self.vadr = model.joint_qd_start.numpy()[self.joints]
        self.base = list(model.body_label).index('go2/base')
        self.q = wp.zeros(model.joint_coord_count,dtype=float,device=model.device)
        self.qd = wp.zeros(model.joint_dof_count,dtype=float,device=model.device)
        self.default = np.asarray(cfg['default_angles'],np.float32)
        self.previous = np.zeros(12,np.float32)
        self.target = self.default.copy()
        self.command = np.zeros(3,np.float32)
        self.motor_period = float(cfg['simulation_dt'])
        self.policy_period = self.motor_period * int(cfg['control_decimation'])
        self.time = 0.0;self.next_update = 0.0;self.updates=0
        self.control = model.control()

    def update(self, state):
        newton.eval_ik(self.model,state,self.q,self.qd)
        q,qd = self.q.numpy(),self.qd.numpy()
        pose = state.body_q.numpy()[self.base]
        twist = state.body_qd.numpy()[self.base]
        rotation = quat_matrix(pose[3:])
        command = np.clip(self.command,-np.asarray(self.cfg['max_velocity_command']),self.cfg['max_velocity_command'])
        if self.time+1e-9 >= self.next_update:
            obs = np.concatenate([rotation.T@twist[3:]*self.cfg['ang_vel_scale'],
                rotation.T@np.array([0.,0.,-1.]),command*np.asarray(self.cfg['command_scale']),
                (q[self.qadr]-self.default)*self.cfg['dof_pos_scale'],
                qd[self.vadr]*self.cfg['dof_vel_scale'],self.previous]).astype(np.float32)
            if obs.shape != (45,) or not np.isfinite(obs).all(): raise ValueError('Invalid native Go2 proprioception')
            with self.torch.no_grad():
                action=self.policy(self.torch.from_numpy(obs[None]))
            if isinstance(action,tuple): action=action[0]
            action=action.detach().cpu().numpy().reshape(12)
            if not np.isfinite(action).all(): raise ValueError('Nonfinite MoE-CTS action')
            self.previous=np.clip(action,-self.cfg['action_clip'],self.cfg['action_clip'])
            self.target=self.default+self.cfg['action_scale']*self.previous
            self.next_update += self.policy_period;self.updates+=1
        velocity = qd[self.vadr]
        effort=np.asarray(self.cfg['kps'])*(self.target-q[self.qadr])-np.asarray(self.cfg['kds'])*velocity
        motor=self.cfg['motor']
        if motor['use_torque_speed_curve']:
            maximum=np.where(velocity*effort>0,motor['Y1'],motor['Y2'])
            factor=np.where(np.abs(velocity)<motor['X1'],1.,np.maximum(1-(np.abs(velocity)-motor['X1'])/(motor['X2']-motor['X1']),0.))
            effort=np.clip(effort,-maximum*factor,maximum*factor)
        return effort

    def attach(self, scene, source_counts):
        if getattr(scene,'graph',None) is not None:
            raise ValueError('Captured CUDA graph bypasses per-substep Go2 control; evaluation needs an uncaptured source execution path')
        solver=getattr(scene,'solver',None)
        if type(solver) not in (newton.solvers.SolverXPBD, newton.solvers.SolverSemiImplicit):
            actual=type(solver).__module__+'.'+type(solver).__name__
            raise ValueError('Unsupported Go2 joint-effort/contact adapter for source solver: '+actual)
        # The spring-based joint attachments of SemiImplicit need shorter
        # integration steps than the motor controller period for this URDF.
        self.max_dt = min(self.motor_period, .00005) if type(solver) is newton.solvers.SolverSemiImplicit else self.motor_period
        static_source = not any(source_counts[key] for key in ('bodies', 'joints', 'particles'))
        self.contacts = None
        self.contact_pipeline = None
        self.steps = 0
        self.min_dt_used, self.max_dt_used = math.inf, 0.
        self.next_motor_update = 0.
        scratch = None
        original=solver.step
        def step(state_0,state_1,control,contacts,dt):
            nonlocal scratch
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError('Source solver needs a finite positive dt')
            count = max(1, math.ceil(dt / self.max_dt - 1e-8))
            if count > 1 and not static_source:
                raise ValueError('Dynamic source needs step_physics(dt) or a source solver substep '
                                 f'<= {self.max_dt:g}s to retain custom forces and contacts')
            active = self.control if control is None else control
            if active.joint_f is None: raise ValueError('Source solver control has no joint effort channel')
            if count > 1 and scratch is None:
                scratch = self.model.state()
            current, h = state_0, dt / count
            for index in range(count):
                destination = state_1 if (count - index) % 2 else scratch
                if index:
                    current.clear_forces()
                if contacts is None or count > 1:
                    if self.contact_pipeline is None:
                        self.contact_pipeline = newton.CollisionPipeline(self.model)
                        self.robot_contacts = self.contact_pipeline.contacts()
                    self.contact_pipeline.collide(current, self.robot_contacts)
                    self.contacts = self.robot_contacts
                else:
                    self.contacts = contacts
                update_motor = self.time + 1e-9 >= self.next_motor_update
                if update_motor:
                    self.effort = self.update(current)
                    self.next_motor_update += self.motor_period
                # The caller may clear or replace Control on every source step.
                if index == 0 or update_motor:
                    values = active.joint_f.numpy()
                    values[self.vadr] = self.effort
                    active.joint_f.assign(values)
                original(current, destination, active, self.contacts, h)
                current = destination
                self.time += h
                self.steps += 1
            self.min_dt_used = min(self.min_dt_used, h)
            self.max_dt_used = max(self.max_dt_used, h)
        solver.step=step
        self.solver_name=type(solver).__module__+'.'+type(solver).__name__

    def pose(self,state):
        pose=state.body_q.numpy()[self.base]
        up=quat_matrix(pose[3:])[:,2]
        return {'position_m':pose[:3].tolist(),'quaternion_xyzw':pose[3:].tolist(),
                'up_z':float(up[2]),'fallen':bool(up[2]<.5 or pose[2]<.12),
                'policy_updates':self.updates}
