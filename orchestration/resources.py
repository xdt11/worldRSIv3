"""One controller-wide CPU, RAM and exclusive GPU admission queue."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from collections import deque
import math
import os
import re
import subprocess
import threading
import time
import uuid


def memory_bytes(value):
    if type(value) is int and value>0:return value
    match=re.fullmatch(r'(\d+(?:\.\d+)?)\s*([bkmgt]?)b?',str(value).lower())
    if not match:raise ValueError('Memory must be a positive byte count or use k/m/g/t units')
    result=int(float(match[1])*1024**{'':0,'b':0,'k':1,'m':2,'g':3,'t':4}[match[2]])
    if result<=0:raise ValueError('Memory must be positive')
    return result


def host_resources():
    with open('/proc/meminfo') as source:
        available=int(next(line.split()[1] for line in source if line.startswith('MemAvailable:')))*1024
    gpus=[]
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
            text=True,capture_output=True,timeout=5)
        for line in result.stdout.splitlines() if result.returncode==0 else []:
            device,index,free,utilization=(part.strip() for part in line.split(','))
            # Some devices expose memory but report N/A for compute utilization.
            gpus.append({'id':device,'index':int(index),'free_memory_bytes':int(float(free)*1024**2),
                         'utilization_percent':int(utilization) if utilization.isdigit() else None})
    except (OSError,ValueError,subprocess.SubprocessError):pass
    return {'cpus':len(os.sched_getaffinity(0)),'memory_bytes':available,'gpus':gpus}


class ResourceManager:
    def __init__(self,options=None,*,capacity=None):
        self.options={'max_cpus':128,'max_memory':None,'memory_fraction':.85,'max_gpus':8,
                      'min_gpu_free_memory':'8g',**(options or {})}
        self.condition=threading.Condition();self.probe_lock=threading.Lock()
        self.closed=False;self.queue=deque();self.active={};self.history=deque(maxlen=128)
        self.injected=capacity is not None
        host=copy.deepcopy(capacity) if capacity is not None else host_resources()
        if 'gpu_devices'in host:
            host['gpus']=[{'id':device,'index':index,'free_memory_bytes':2**60} for index,device in enumerate(host['gpu_devices'])]
        host.setdefault('gpus',[])
        self.host=host;self.last_probe=time.monotonic()
        ram=int(host['memory_bytes']*self.options['memory_fraction'])
        if self.options['max_memory'] is not None:ram=min(ram,memory_bytes(self.options['max_memory']))
        self.capacity={'cpus':min(float(self.options['max_cpus']),host['cpus']),'memory_bytes':ram,
                       'gpu_devices':[row['id'] for row in host['gpus'][:self.options['max_gpus']]]}
        self.minimum_gpu_memory=memory_bytes(self.options['min_gpu_free_memory'])

    def _refresh(self):
        # Device probes and /proc reads never hold the resource queue lock.
        if self.injected or time.monotonic()-self.last_probe<1:return
        if not self.probe_lock.acquire(blocking=False):return
        try:
            host=host_resources()
            with self.condition:self.host=host;self.last_probe=time.monotonic()
        finally:self.probe_lock.release()

    def _request(self,cfg,phase,owner,priority):
        if phase not in {'sandbox','evaluation'}:raise ValueError('Unknown resource phase')
        section=cfg[phase];cpus=section.get('cpus',4);ram=memory_bytes(section.get('memory','12g'))
        if type(cpus) not in (int,float) or not math.isfinite(cpus) or cpus<=0:raise ValueError('Requested CPUs must be positive and finite')
        if cpus>self.capacity['cpus'] or ram>self.capacity['memory_bytes']:
            raise ValueError('Requested CPU/RAM exceeds controller resource capacity')
        raw=section.get('gpus');devices=[];count=0
        if raw is not None:
            if raw=='auto':count=1
            elif raw=='all':count=len(self.capacity['gpu_devices']) or 1
            else:
                if not isinstance(raw,str) or not raw.startswith('device='):raise ValueError('GPU policy must be null, auto, all, or device=<indices/UUIDs>')
                selected=raw[7:].strip('"').split(',')
                by_index={str(row['index']):row['id'] for row in self.host['gpus']}
                devices=[by_index.get(device,device) for device in selected]
                if len(devices)!=len(set(devices)) or any(device not in self.capacity['gpu_devices'] for device in devices):
                    raise ValueError('Requested GPU is outside the controller device pool')
                count=len(devices)
            if count>len(self.capacity['gpu_devices']):raise ValueError('Requested GPU count exceeds controller capacity; use gpus=null for CPU')
        return {'id':uuid.uuid4().hex,'owner':owner,'phase':phase,'thread_id':threading.get_ident(),'cpus':float(cpus),'memory_bytes':ram,
                'gpu_count':count,'pinned_gpu_devices':devices,'gpu_devices':[],'status':'queued','priority':priority,
                'waiting_for':None,'cancel_requested':False,'admission_ready':False,'queued_at':time.time()}

    def _available(self,request):
        used_cpu=sum(row['cpus'] for row in self.active.values());used_ram=sum(row['memory_bytes'] for row in self.active.values())
        if used_cpu+request['cpus']>self.capacity['cpus']:return None,'cpu'
        if used_ram+request['memory_bytes']>self.capacity['memory_bytes']:return None,'memory'
        if request['memory_bytes']>int(self.host['memory_bytes']*self.options['memory_fraction']):return None,'host_memory'
        occupied={device for row in self.active.values() for device in row['gpu_devices']}
        free={row['id']:row for row in self.host['gpus'] if row['free_memory_bytes']>=self.minimum_gpu_memory}
        eligible=[device for device in self.capacity['gpu_devices'] if device not in occupied and device in free]
        if request['gpu_count']==1 and not request['pinned_gpu_devices']:
            def load(device):
                utilization=free[device].get('utilization_percent')
                return (utilization if utilization is not None else 101, -free[device]['free_memory_bytes'])
            eligible.sort(key=load)
        chosen=request['pinned_gpu_devices'] or eligible[:request['gpu_count']]
        if len(chosen)<request['gpu_count'] or any(device not in eligible for device in chosen):return None,'gpu'
        return chosen,None

    @contextmanager
    def lease(self,cfg,*,phase,owner,checkpoint=None,priority=0):
        request=self._request(cfg,phase,owner,priority);granted=False
        with self.condition:
            if self.closed:raise InterruptedError('Resource scheduler is shutting down')
            self.queue.append(request);self.condition.notify_all()
        try:
            while not granted:
                with self.condition:request['admission_ready']=False
                if checkpoint:checkpoint()
                self._refresh()
                with self.condition:
                    request['admission_ready']=True
                    if self.closed or request['cancel_requested']:raise InterruptedError('Resource request cancelled')
                    # Evolution advances the skill lineage, so give its next
                    # phase a free slot before queued tests. Never preempt an
                    # active container. FIFO remains stable within each priority,
                    # and a pinned busy GPU cannot block unrelated CPU work.
                    devices,reason=self._available(request)
                    earlier=False;ahead=True
                    for row in self.queue:
                        if row is request:ahead=False;continue
                        if (row['priority']>request['priority'] or
                                (row['priority']==request['priority'] and ahead)) and \
                                row['admission_ready'] and self._available(row)[0] is not None:
                            earlier=True;break
                    if devices is not None and not earlier:
                        self.queue.remove(request);request.update(gpu_devices=devices,status='active',waiting_for=None,started_at=time.time())
                        self.active[request['id']]=request;granted=True;self.condition.notify_all()
                    else:
                        request['waiting_for']=reason or 'queue_order';self.condition.wait(.1)
            result=copy.deepcopy(cfg)
            devices=request['gpu_devices'];result[phase]['gpus']='device='+','.join(devices) if devices else None
            yield result
        finally:
            with self.condition:
                if request in self.queue:self.queue.remove(request)
                self.active.pop(request['id'],None)
                request.update(status='released' if granted else 'cancelled',finished_at=time.time())
                self.history.append(copy.deepcopy(request));self.condition.notify_all()

    def cancel(self,owner):
        with self.condition:
            matches=[row for row in [*self.queue,*self.active.values()] if row['owner']==owner]
            for row in matches:row['cancel_requested']=True
            self.condition.notify_all()
            return len(matches)

    def cancelled_for_thread(self,identity):
        with self.condition:
            return any(row['thread_id']==identity and row['cancel_requested'] for row in [*self.queue,*self.active.values()])

    def shutdown(self):
        with self.condition:self.closed=True;self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return copy.deepcopy({'capacity':self.capacity,'host_available_memory_bytes':self.host['memory_bytes'],
                'gpu_free_memory_bytes':{row['id']:row['free_memory_bytes'] for row in self.host['gpus']},
                'used':{'cpus':sum(row['cpus'] for row in self.active.values()),
                        'memory_bytes':sum(row['memory_bytes'] for row in self.active.values()),
                        'gpu_devices':[device for row in self.active.values() for device in row['gpu_devices']]},
                'active':list(self.active.values()),'queued':list(self.queue),'recent':list(self.history)})
