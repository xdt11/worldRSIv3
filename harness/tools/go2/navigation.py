"""Persistent X-NavDP point-goal inference using the released Go2 checkpoint.

Run this worker in the existing navigation Python environment. Physics and
neural rendering belong to the rollout controller; this process only infers
from its supplied, timestamped RGB-D observation and explicit point goal.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def official_modules(repo):
    source=Path(repo).resolve()/'baselines/x-navdp/eval/src'
    package='_worldrsi_xnavdp_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if package not in sys.modules:
        module=types.ModuleType(package)
        module.__path__=[str(source)]
        sys.modules[package]=module
    return source,importlib.import_module(package+'.policy_network_embodiment'),importlib.import_module(package+'.policy_agent')


def trajectory_command(trajectory):
    """Translate the chosen learned local path into a bounded body velocity.

    This is a transparent execution controller, not a second path planner.
    No scene geometry, ground-truth route or goal enters this conversion.
    """
    path=np.asarray(trajectory,np.float64)
    if path.ndim!=2 or path.shape[1]<2 or not len(path) or not np.isfinite(path).all():
        raise ValueError('The learned trajectory must be a finite [T,>=2] local path')
    distances=np.linalg.norm(path[:,:2],axis=1)
    eligible=np.flatnonzero(distances>=.5)
    index=int(eligible[0]) if len(eligible) else len(path)-1
    x,y=path[index,:2]
    heading=math.atan2(y,x) if distances[index]>1e-6 else 0.
    limits=[.6,.25,.8]
    command=[np.clip(.8*x,0,limits[0])*max(0.,math.cos(heading)),
             np.clip(.8*y,-limits[1],limits[1]),np.clip(1.5*heading,-limits[2],limits[2])]
    return {'command_body':[float(value) for value in command],
            'execution_controller':{'kind':'body-frame trajectory lookahead proportional control',
                'waypoint_index':index,'waypoint_xy_m':[float(x),float(y)],
                'lookahead_distance_m':.5,'linear_gain_s_inv':.8,'yaw_gain_s_inv':1.5,
                'velocity_limits_m_s_m_s_rad_s':limits,'heading_error_rad':heading,
                'inputs':'Only the X-NavDP selected local trajectory; no geometry-based override'}}


class XNavDPPointGoal:
    def __init__(self,repo,checkpoint,device='cuda:0',seed=7):
        import torch
        self.torch=torch
        source,network,agents=official_modules(repo)
        torch.manual_seed(seed)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        layers=len({key.split('.')[2] for key in state if key.startswith('decoder.layers.')})
        self.net=network.NavDP_Policy_Embodiment(temporal_depth=layers,device=device)
        extras=sorted(set(state)-set(self.net.state_dict()))
        allowed=('pixel_encoder.','image_encoder.','critic_head.')
        if any(key!='log_alpha' and not key.startswith(allowed) for key in extras):
            raise ValueError('Unexpected X-NavDP checkpoint tensors')
        # The released eval class omits these training-only containers. Retain
        # every tensor so that the complete released checkpoint loads strictly.
        for key in extras:
            parent=self.net
            parts=key.split('.')
            for name in parts[:-1]:
                if not hasattr(parent,name):parent.add_module(name,torch.nn.Module())
                parent=getattr(parent,name)
            parent.register_buffer(parts[-1],torch.empty_like(state[key]))
        self.net.load_state_dict(state,strict=True)
        self.net=self.net.to(device).eval()
        self.agent=agents.NavDP_Agent.__new__(agents.NavDP_Agent)
        for name,value in {'image_size':224,'memory_size':8,'predict_size':24,
                'embodiment':2,'is_real':False,'device':device,'navi_former':self.net}.items():
            setattr(self.agent,name,value)
        self.reset()
        self.loading_info={'model':'X-NavDP RL post-trained pointgoal','embodiment':'Unitree Go2',
            'checkpoint_sha256':sha256(checkpoint),'strict_state_dict_load':True,
            'checkpoint_tensors':len(state),'retained_unused_training_only_tensors':len(extras),
            'unused_containers_executed':False,'temporal_layers':layers,'seed':seed,
            'samples':8,'embodiment_id':2,'quaternion_order':'xyzw',
            'rgb_input':'RGB uint8; converted to BGR according to released evaluation',
            'depth_input':'native optical-Z metres, released clipping to [0.1,5]',
            'history':'Released quadruped 40-frame queue, 7 past samples plus current',
            'selection':'argmax released twin-Q mean; no geometric filtering or random override',
            'source_sha256':{name:sha256(source/name) for name in
                ('policy_network_embodiment.py','policy_backbone.py','policy_agent.py','policy_server.py')}}

    def reset(self):
        self.agent.reset(1)

    def predict(self,rgb,depth,goal_local,position,quaternion_xyzw):
        rgb=np.asarray(rgb)
        depth=np.asarray(depth,np.float32)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or depth.shape!=rgb.shape[:2]:
            raise ValueError('RGB must be uint8 [H,W,3], aligned with native depth [H,W]')
        if not np.isfinite(depth).all() or (depth<0).any():
            raise ValueError('Native depth must be finite, nonnegative optical-Z metres')
        goal=np.asarray(goal_local,np.float32)
        goal_z_provided=goal.shape==(3,)
        if goal.shape==(2,):
            goal=np.concatenate([goal,np.zeros(1,np.float32)])
        position=np.asarray(position,np.float64)
        quaternion=np.asarray(quaternion_xyzw,np.float64)
        if goal.shape!=(3,) or position.shape!=(3,) or quaternion.shape!=(4,):
            raise ValueError('Expected goal_local[3] (or XY with zero-Z), position[3], quaternion_xyzw[4]')
        if not all(np.isfinite(v).all() for v in (goal,position,quaternion)) or not np.isclose(np.linalg.norm(quaternion),1,atol=1e-3):
            raise ValueError('Navigation pose and point goal must be finite with a unit quaternion')
        a=self.agent
        bgr=np.ascontiguousarray(rgb[...,::-1])[None]
        image=a.process_image(bgr)
        distance=a.process_depth(depth[None,:,:,None].copy())
        history=a._update_and_sample_history(image,num_samples=8)
        actual_goal=a.process_pointgoal(goal[None])
        position=position[None]
        quaternion=quaternion[None]
        previous,valid,_=a.get_guidance_trajectory(position,quaternion)
        factors=np.array([[.5,.5,.5,.5,.5,.5,.05,.05]],np.float32)
        start=time.perf_counter()
        # Released temporal guidance differentiates its correction wrt sampled
        # trajectories. A blanket no_grad/inference_mode breaks that solver.
        trajectories,values,_,_=self.net.predict_pointgoal_action_with_guidance(
            actual_goal,history,distance,sample_num=8,valid_segment_len=valid,prev_action=previous,
            start_index=0,end_index=23,guidance_factor=factors,guidance_step=5,
            prefix_attention_schedule='exp',embodiment=2)
        trajectories,values=trajectories[0],values[0]
        if not np.isfinite(trajectories).all() or not np.isfinite(values).all():
            raise ValueError('Nonfinite X-NavDP prediction')
        chosen=int(values.argmax())
        a._update_last_trajectory_state(trajectories[chosen:chosen+1],position,quaternion)
        return {'trajectory_body':trajectories[chosen].tolist(),'chosen':chosen,
            'values':values.tolist(),'goal_body_policy_input':actual_goal[0].tolist(),
            'goal_z_provided':goal_z_provided,
            'guidance_valid_segment_length':int(valid[0]),'latency_s':time.perf_counter()-start,
            **trajectory_command(trajectories[chosen])}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy=XNavDPPointGoal(args.repo,args.checkpoint,args.device,args.seed)
    def emit(value):print(json.dumps(value,allow_nan=False),flush=True)
    emit({'status':'ready','loading_info':policy.loading_info})
    for line in sys.stdin:
        try:
            request=json.loads(line)
            operation=request.get('operation','predict')
            if operation=='close':break
            if operation=='reset':
                policy.reset()
                emit({'status':'reset'})
                continue
            if operation!='predict':raise ValueError('Unknown navigation operation')
            from PIL import Image
            rgb=np.asarray(Image.open(request['rgb_path']).convert('RGB'))
            depth=np.load(request['depth_path'],allow_pickle=False)
            with contextlib.redirect_stdout(sys.stderr):
                result=policy.predict(rgb,depth,request['goal_local'],request['position'],request['quaternion_xyzw'])
            emit({'status':'ok','time_s':request.get('time_s'),**result})
        except Exception as exc:
            emit({'status':'error','error':str(exc)})


if __name__=='__main__':main()
