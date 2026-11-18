"""Optical cameras and inspection poses for the Newton worker."""
from __future__ import annotations
import math
import numpy as np


class CameraError(ValueError):
    def __init__(self, code, message, *, status="error", **details):
        self.code, self.status = code, status
        self.diagnostic = {"code": code, "message": message, "status": status, **details}
        super().__init__(message)


def _claim_value(claim, default=None):
    if isinstance(claim, dict) and "value" in claim:
        return claim["value"] if claim.get("status") != "unknown" else default
    return default if claim is None else claim


def _quat_from_matrix(r):
    # Symmetric eigensolve is stable near rotations by pi and returns xyzw.
    q = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r[0,0]+r[1,1]+r[2,2]]
    ]) / 3.0
    return np.linalg.eigh(q)[1][:, -1]


def quat_matrix(q):
    x,y,z,w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def transform_matrix(position, quaternion):
    result=np.eye(4)
    result[:3,:3]=quat_matrix(quaternion)
    result[:3,3]=position
    return result


def look_at(position, target, up=(0,0,1)):
    position,target,up = (np.asarray(v,dtype=float) for v in (position,target,up))
    forward=target-position
    if np.linalg.norm(forward)<1e-10:
        raise CameraError("invalid_camera", "Camera position and target coincide")
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-10:
        raise CameraError("invalid_camera", "Camera up is parallel to view direction")
    right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    result=np.eye(4)
    result[:3,:3]=np.stack([right,down,forward],axis=1)
    result[:3,3]=position
    return result


def _interpolate(samples, time_s, field, interpolation, *, transform=False):
    if not samples:
        raise CameraError("unknown_camera", "Camera trajectory has no known samples", status="unsupported")
    times=np.array([sample["time_s"] for sample in samples],dtype=float)
    if np.any(np.diff(times)<=0):
        raise CameraError("invalid_camera", "Camera sample times must increase")
    index=int(np.searchsorted(times,time_s,side="right")-1)
    index=max(0,min(index,len(samples)-1))
    left=_claim_value(samples[index][field])
    if left is None:
        raise CameraError("unknown_camera", "Camera sample is unresolved", status="unsupported")
    left=np.array(left,dtype=float)
    if index==len(samples)-1 or interpolation in ("none","hold") or time_s<=times[0]:
        return left
    right=_claim_value(samples[index+1][field])
    if right is None:
        raise CameraError("unknown_camera", "Camera sample is unresolved", status="unsupported")
    right=np.array(right,dtype=float)
    alpha=(time_s-times[index])/(times[index+1]-times[index])
    if not transform:
        return left*(1-alpha)+right*alpha
    qa,qb=_quat_from_matrix(left[:3,:3]),_quat_from_matrix(right[:3,:3])
    cosine=float(np.dot(qa,qb))
    if cosine<0: qb=-qb; cosine=-cosine
    if cosine>.9995:
        q=(1-alpha)*qa+alpha*qb
        q/=np.linalg.norm(q)
    else:
        angle=math.acos(max(-1,min(1,cosine)))
        q=(math.sin((1-alpha)*angle)*qa+math.sin(alpha*angle)*qb)/math.sin(angle)
    return transform_matrix((1-alpha)*left[:3,3]+alpha*right[:3,3],q)


def _mapped_source_time(world,source,clock,world_time):
    if clock=="world": return world_time
    candidates=[]
    for mapping in world.get("timeline",{}).get("source_mappings",[]):
        if mapping.get("source")!=source or mapping.get("source_clock")!=clock or mapping.get("status")=="unknown": continue
        a,b=mapping["world_interval_s"]
        if a-1e-9<=world_time<=b+1e-9:
            c,d=mapping["source_interval_s"]
            candidates.append(c+(world_time-a)*(d-c)/(b-a))
    if not candidates or max(candidates)-min(candidates)>1e-7:
        raise CameraError("unknown_camera_clock", "No unique source/world time mapping for camera",status="unsupported",clock=clock,time_s=world_time)
    return candidates[0]


def source_time(world, camera, world_time):
    return _mapped_source_time(world,camera.get("source"),camera.get("pose",{}).get("clock","world"),world_time)


def choose_camera(world, time_s, camera_id=None, source_id=None):
    cameras=world.get("cameras",{})
    if camera_id is not None:
        if camera_id not in cameras: raise CameraError("unknown_camera",f"Unknown camera {camera_id}")
        if source_id is not None and cameras[camera_id].get("source")!=source_id:
            raise CameraError("reference_camera_mismatch","Requested camera belongs to another reference source")
        return camera_id,cameras[camera_id]
    sources=world.get("sources",{})
    if source_id is not None and source_id not in sources:
        raise CameraError("reference_camera_unresolved",f"Reference source {source_id} is absent from world.sources",status="unsupported")
    selected=list(sources.items()) if source_id is None else [(source_id,sources[source_id])]
    for selected_id,source in selected:
        shots=source.get("shots",[])
        if not shots:continue
        clock=source.get("clock")
        if clock is None:
            mappings=[m for m in world.get("timeline",{}).get("source_mappings",[]) if m.get("source")==selected_id and m.get("status")!="unknown"]
            clock=mappings[0]["source_clock"] if mappings else None
        if clock is None:raise CameraError("unknown_shot_clock","Source shots require a known source clock",status="unsupported")
        t=_mapped_source_time(world,selected_id,clock,time_s)
        for index,shot in enumerate(shots):
            camera=cameras[shot["camera"]]
            a,b=shot["interval_s"]
            if a<=t<b or index==len(shots)-1 and abs(t-b)<=1e-8:
                return shot["camera"],camera
        if source_id is not None:raise CameraError("reference_camera_unresolved","No shot of the selected reference covers this sample time",status="unsupported")
    if source_id is not None:
        source=sources[source_id]
        if source.get("camera") in cameras:
            selected_camera=source["camera"]
            if cameras[selected_camera].get("source")!=source_id:
                raise CameraError("reference_camera_mismatch","Reference camera points back to a different source")
            return selected_camera,cameras[selected_camera]
        options=[(identity,camera) for identity,camera in cameras.items() if camera.get("source")==source_id and camera.get("role")=="source"]
        if len(options)!=1:raise CameraError("reference_camera_unresolved","Selected reference needs one source camera or explicit shots",status="unsupported",source_id=source_id)
        return options[0]
    for role in ("source","evaluation","inspection","robot_sensor"):
        for identity,camera in cameras.items():
            if camera.get("role")==role: return identity,camera
    return "default",{"role":"inspection","model":"pinhole","resolution_px":[640,360],
                       "intrinsics":{"mode":"constant","value":{"value":[[400,0,320],[0,400,180],[0,0,1]],"status":"implementation_assumption"}},
                       "pose":{"mode":"static","clock":"world","transform":{"value":look_at((3,-5,3),(0,0,.5)).tolist(),"status":"implementation_assumption"}}}


def calibration(world,time_s,camera_id=None,override=None,width=None,height=None,body_transforms=None,source_id=None):
    if override:
        if camera_id is not None and camera_id not in world.get("cameras",{}):
            raise CameraError("unknown_camera",f"Unknown camera {camera_id}")
        try:
            base=calibration(world,time_s,camera_id,width=width,height=height,body_transforms=body_transforms,source_id=source_id)
        except CameraError as exc:
            if exc.code not in {"unknown_intrinsics","unknown_camera_pose","unknown_camera_attachment","unknown_camera_clock","unknown_shot_clock","unsupported_camera_model","reference_camera_unresolved"}:raise
            base=calibration({},time_s,width=width,height=height)
            base["inspection_base_assumption"]=str(exc)
        transform=np.asarray(base["camera_to_world"],dtype=float)
        k=np.asarray(base["intrinsics"],dtype=float)
        if "camera_to_world" in override:transform=np.asarray(override["camera_to_world"],dtype=float)
        elif any(key in override for key in ("position","target","up")):
            transform=look_at(override.get("position",transform[:3,3]),override.get("target",transform[:3,3]+transform[:3,2]),override.get("up",-transform[:3,1]))
        if "fov" in override or "fov_deg" in override:
            f=float(override.get("fov_deg",override.get("fov")))
            if not 1<f<179:raise CameraError("invalid_camera","fov must be between 1 and 179 degrees")
            k[0,0]=k[1,1]=base["height"]/(2*math.tan(math.radians(f)/2))
        if transform.shape!=(4,4) or not np.all(np.isfinite(transform)) or not np.allclose(transform[3],[0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(transform[:3,:3])<.99999:
            raise CameraError("invalid_camera","Inspection camera must be a finite right-handed rigid transform")
        return {**base,"base_camera_id":base["id"],"id":"inspection_override","role":"inspection","intrinsics":k.tolist(),"camera_to_world":transform.tolist(),"extrinsic":np.linalg.inv(transform).tolist(),
                "position":transform[:3,3].tolist(),"target":(transform[:3,3]+transform[:3,2]).tolist(),"up":(-transform[:3,1]).tolist(),
                "fov":math.degrees(2*math.atan(base["height"]/(2*k[1,1]))),"fov_deg":math.degrees(2*math.atan(base["height"]/(2*k[1,1])))}
    identity,camera=choose_camera(world,time_s,camera_id,source_id=source_id)
    if camera.get("model","pinhole") not in ("pinhole",):
        raise CameraError("unsupported_camera_model","Warp renderer currently supports pinhole cameras",status="unsupported")
    stamp=source_time(world,camera,time_s)
    intrinsic=camera.get("intrinsics",{})
    if intrinsic.get("mode")=="constant": k=_claim_value(intrinsic.get("value"))
    elif intrinsic.get("mode")=="keyframes": k=_interpolate(intrinsic.get("samples",[]),stamp,"value",intrinsic.get("interpolation","hold"))
    else: k=None
    if k is None: raise CameraError("unknown_intrinsics",f"Camera {identity} intrinsics unresolved",status="unsupported")
    k=np.asarray(k,dtype=float).copy()
    if camera.get("attachment"):
        attachment=camera["attachment"]
        parent=(body_transforms or {}).get(attachment["object"])
        local=_claim_value(attachment.get("camera_to_parent"))
        if parent is None or local is None:
            raise CameraError("unknown_camera_attachment","Robot camera attachment is unresolved",status="unsupported")
        transform=parent@np.asarray(local,dtype=float)
    else:
        pose=camera.get("pose",{})
        if pose.get("mode")=="static": transform=_claim_value(pose.get("transform"))
        elif pose.get("mode")=="trajectory": transform=_interpolate(pose.get("samples",[]),stamp,"transform",pose.get("interpolation","hold"),transform=True)
        else: transform=None
        if transform is None: raise CameraError("unknown_camera_pose",f"Camera {identity} pose unresolved",status="unsupported")
        transform=np.asarray(transform,dtype=float)
    native_width,native_height=camera.get("resolution_px",[640,360])
    requested_width=int(width or native_width);requested_height=int(height or native_height)
    if not (1<=requested_width<=4096 and 1<=requested_height<=4096):
        raise CameraError("invalid_resolution","Render dimensions must be 1..4096")
    k[0,:]*=requested_width/native_width;k[1,:]*=requested_height/native_height
    if transform.shape!=(4,4) or k.shape!=(3,3) or not np.all(np.isfinite(transform)) or not np.all(np.isfinite(k)):
        raise CameraError("invalid_camera","Camera calibration must be finite 4x4 / 3x3 matrices")
    return {"id":identity,"role":"inspection" if override else camera.get("role","inspection"),"intrinsics":k.tolist(),
            "camera_to_world":transform.tolist(),"extrinsic":np.linalg.inv(transform).tolist(),"width":requested_width,"height":requested_height,
            "position":transform[:3,3].tolist(),"target":(transform[:3,3]+transform[:3,2]).tolist(),"up":(-transform[:3,1]).tolist(),
            "fov":math.degrees(2*math.atan(requested_height/(2*k[1,1]))),"fov_deg":math.degrees(2*math.atan(requested_height/(2*k[1,1]))),"time_s":time_s,"camera_time_s":stamp}


def normalize_camera_pose(value):
    if not isinstance(value,dict):
        raise ValueError('camera must contain position, target and up vectors')
    result={}
    for name in ('position','target','up'):
        vector=value.get(name)
        if (not isinstance(vector,list) or len(vector)!=3 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in vector)):
            raise ValueError(f'camera.{name} must contain three finite numbers')
        result[name]=list(vector)
    direction=[b-a for a,b in zip(result['position'],result['target'])]
    distance=math.hypot(*direction)
    if not math.isfinite(distance) or distance==0:
        raise ValueError('Camera position and target must differ by a finite distance')
    forward=[x/distance for x in direction]
    up_length=math.hypot(*result['up'])
    up=[x/up_length for x in result['up']] if up_length else [0.,0.,0.]
    projection=sum(a*b for a,b in zip(up,forward))
    orthogonal=[a-projection*b for a,b in zip(up,forward)]
    norm=math.hypot(*orthogonal)
    if norm<1e-6:
        # Choose a basis by angle, independent of scene scale or up magnitude.
        axis=min(range(3),key=lambda i:abs(forward[i]))
        up=[float(i==axis) for i in range(3)]
        projection=forward[axis]
        orthogonal=[a-projection*b for a,b in zip(up,forward)]
        norm=math.hypot(*orthogonal)
    result['up']=[x/norm for x in orthogonal]
    return result
