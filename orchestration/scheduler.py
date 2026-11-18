"""Concurrent job ownership, per-target guards and cooperative campaign controls."""
import threading
import traceback
from contextlib import contextmanager


class Control:
    """Cooperative pause/stop at paid-call and tool boundaries, shared by all workers."""
    def __init__(self, changed=lambda: None):
        self.condition = threading.Condition()
        self.changed = changed
        self.paused = self.stopped = self.waiting = self.closed = False
        self.workers, self.parked = set(), set()

    @property
    def status(self):
        with self.condition:
            if self.stopped: return 'stopping'
            if self.paused: return 'paused' if self.workers <= self.parked else 'pausing'
            return 'awaiting_confirmation' if self.waiting else 'running'

    def request(self, action):
        with self.condition:
            if self.closed or self.stopped: raise ValueError('Experiment is no longer controllable')
            if action == 'pause': self.paused = True
            elif action == 'resume': self.paused = False
            elif action == 'stop': self.stopped = True; self.paused = False
            else: raise ValueError('Unknown control action')
            self.condition.notify_all()
        self.changed()

    def checkpoint(self,cancelled=None):
        identity = threading.get_ident()
        with self.condition:
            if self.paused: self.parked.add(identity)
        self.changed()
        try:
            while True:
                if cancelled and cancelled():raise InterruptedError('Work cancellation requested')
                with self.condition:
                    if self.stopped:raise InterruptedError('Experiment stop requested')
                    if not self.paused:break
                    self.condition.wait(.1)
        finally:
            with self.condition: self.parked.discard(identity)

    @contextmanager
    def worker(self):
        identity = threading.get_ident()
        with self.condition: self.workers.add(identity)
        try:
            self.checkpoint()
            yield
        finally:
            with self.condition:
                self.workers.discard(identity)
                self.parked.discard(identity)
            self.changed()


class Scheduler:
    """Concurrent jobs with short registry locks and per-task execution guards."""
    def __init__(self,store):
        from orchestration.resources import ResourceManager
        self.store=store;self.condition=threading.Condition();self.stop=threading.Event();self.controls={}
        self.jobs={};self.cancellations={};self.threads={};self.owners={};self.local=threading.local()
        self.shutdown_errors=[]
        cfg=store.config() if callable(getattr(store,'config',None)) else {}
        self.options=cfg.get('scheduler',{});self.resources=ResourceManager(self.options)

    def checkpoint(self,record=None):
        record=record or {}
        def task_stopped():
            if record.get('id') and callable(getattr(self.store,'task',None)):
                return bool(self.store.task(record['id']).get('stop_requested'))
            return bool(record.get('stop_requested'))
        if self.stop.is_set():raise InterruptedError('Controller is shutting down')
        identity=getattr(self.local,'job_id',None)
        if identity and self.cancellations[identity].is_set():raise InterruptedError('Job cancellation requested')
        if task_stopped():raise InterruptedError('Task stop requested')
        control=self.controls.get(record.get('campaign_id'))
        if control:control.checkpoint(cancelled=lambda:self.stop.is_set() or task_stopped() or
            bool(identity and self.cancellations[identity].is_set()) or self.resources.cancelled_for_thread(threading.get_ident()))

    @property
    def busy(self):
        return self.is_busy()

    def is_busy(self,*,task_ids=None,key=None):
        with self.condition:
            jobs=[job for job in self.jobs.values() if job['finished_at'] is None]
            if key is not None:
                return key in self.owners or any(key==job['key'] or key in job['conflict_keys'] for job in jobs)
            if task_ids is not None:
                tasks=set(task_ids)
                return any(tasks.intersection(job['task_ids']) for job in jobs) or any('task:'+task in self.owners for task in tasks)
            active=bool(jobs or self.owners)
        resources=self.resources.snapshot()
        return active or bool(resources['active'] or resources['queued'])

    def snapshot(self):
        import copy
        with self.condition:jobs=copy.deepcopy(list(self.jobs.values()))
        return {'busy':self.busy,'jobs':jobs,'resources':self.resources.snapshot(),'shutdown_errors':list(self.shutdown_errors)}

    @contextmanager
    def _guard(self,keys,record=None):
        keys=tuple(sorted(set(keys)));identity=threading.get_ident();job_id=getattr(self.local,'job_id',None)
        acquired=False
        try:
            while not acquired:
                self.checkpoint(record)
                with self.condition:
                    conflicts=[key for key in keys if key in self.owners and self.owners[key][0]!=identity]
                    if not conflicts:
                        for key in keys:
                            previous=self.owners.get(key,(identity,0));self.owners[key]=(identity,previous[1]+1)
                        acquired=True
                        if job_id:self.jobs[job_id]['waiting_for']=None
                    else:
                        if job_id:self.jobs[job_id]['waiting_for']=conflicts
                        self.condition.wait(.1)
            yield
        finally:
            if acquired:
                with self.condition:
                    for key in keys:
                        depth=self.owners[key][1]-1
                        if depth:self.owners[key]=(identity,depth)
                        else:del self.owners[key]
                    self.condition.notify_all()

    @contextmanager
    def execute_guard(self,task_id,record=None):
        with self._guard(('task:'+task_id,),record):yield

    @contextmanager
    def mutation_guard(self,*,task_ids=(),conflict_keys=()):
        """Atomically reject target mutations while work is queued or executing."""
        keys=set('task:'+task for task in task_ids)|set(conflict_keys);identity=threading.get_ident()
        with self.condition:
            jobs=[job for job in self.jobs.values() if job['finished_at'] is None]
            if any(keys.intersection([*('task:'+task for task in job['task_ids']),*job['conflict_keys']]) for job in jobs):
                raise ValueError('The selected target has queued or running work')
            if any(key in self.owners and self.owners[key][0]!=identity for key in keys):
                raise ValueError('The selected target is executing')
            for key in keys:self.owners[key]=(identity,self.owners.get(key,(identity,0))[1]+1)
        try:yield
        finally:
            with self.condition:
                for key in keys:
                    depth=self.owners[key][1]-1
                    if depth:self.owners[key]=(identity,depth)
                    else:del self.owners[key]
                self.condition.notify_all()

    def schedule(self,action,*args,key=None,kind='task',task_ids=(),conflict_keys=(),handoff=False):
        import time
        import uuid
        identity='job_'+uuid.uuid4().hex[:16];key=key or identity
        if not isinstance(key,str) or not key:raise ValueError('Job key must be a nonempty string')
        task_ids=tuple(dict.fromkeys(task_ids));conflict_keys=tuple(dict.fromkeys(conflict_keys))
        if any(not isinstance(value,str) or not value for value in (*task_ids,*conflict_keys)):
            raise ValueError('Task IDs and conflict keys must be nonempty strings')
        with self.condition:
            if self.stop.is_set():raise ValueError('Controller is shutting down')
            active=[job for job in self.jobs.values() if job['finished_at'] is None]
            if any(job['key']==key for job in active):raise ValueError('This job is already scheduled')
            # A follow-on for the same task waits for its parent's guard and takes over its slot.
            parent=self.jobs.get(getattr(self.local,'job_id',None))
            transfer=bool(handoff and parent and parent['status']=='running' and task_ids
                          and tuple(parent['task_ids'])==task_ids)
            if len(active)-int(transfer)>=self.options.get('max_jobs',128):raise ValueError('Controller job queue is full')
            job={'id':identity,'key':key,'kind':kind,'task_ids':list(task_ids),'conflict_keys':list(conflict_keys),
                 'status':'queued','waiting_for':None,'created_at':time.time(),'started_at':None,
                 'finished_at':None,'error':None,'cancel_requested':False}
            self.jobs[identity]=job;self.cancellations[identity]=threading.Event()
            # Keep a bounded completed history without removing active controls.
            finished=[item for item in self.jobs if self.jobs[item]['finished_at'] is not None]
            for old in finished[:-128]:
                self.jobs.pop(old,None);self.cancellations.pop(old,None);self.threads.pop(old,None)
        def work():
            self.local.job_id=identity
            try:
                with self._guard((*('task:'+task for task in task_ids),*conflict_keys)):
                    self.checkpoint()
                    with self.condition:
                        if self.cancellations[identity].is_set():raise InterruptedError('Job cancellation requested')
                        job.update(status='running',started_at=time.time())
                    action(*args)
                with self.condition:job['status']='cancelled' if self.cancellations[identity].is_set() else 'completed'
            except InterruptedError as exc:
                with self.condition:job.update(status='cancelled',error=str(exc))
            except Exception as exc:
                import pathlib
                evidence=traceback.format_exc()
                with self.condition:job.update(status='failed',error=str(exc))
                try:
                    path=pathlib.Path(self.store.data)/'job_errors'/f'{identity}.log';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(evidence)
                except OSError as log_error:
                    with self.condition:job['log_error']=str(log_error)
                else:
                    with self.condition:job['error_path']=str(path.relative_to(self.store.data))
            finally:
                with self.condition:job.update(finished_at=time.time(),waiting_for=None);self.condition.notify_all()
                self.local.job_id=None
        worker=threading.Thread(target=work,daemon=True,name=identity)
        with self.condition:
            if self.stop.is_set():
                job.update(status='cancelled',cancel_requested=True,finished_at=time.time())
                raise ValueError('Controller is shutting down')
            self.threads[identity]=worker
            try:worker.start()
            except Exception as exc:
                job.update(status='failed',error=str(exc),finished_at=time.time())
                raise
            return dict(job)

    def cancel(self,key):
        with self.condition:
            job=next((row for row in self.jobs.values() if key in (row['id'],row['key']) and row['status'] in {'queued','running'}),None)
            if job is None:raise ValueError('No active job matches this key')
            self.cancellations[job['id']].set();job['cancel_requested']=True;self.condition.notify_all()
            task_ids=list(job['task_ids']);result=dict(job)
        for task_id in task_ids:self.resources.cancel(task_id)
        return result

    def shutdown(self,wait=False):
        """Cancel first; optionally drain writers before releasing controller ownership."""
        import time
        self.stop.set();self.resources.shutdown()
        with self.condition:
            for job in self.jobs.values():
                if job['status'] in {'queued','running'}:
                    self.cancellations[job['id']].set();job['cancel_requested']=True
                    if job['id'] not in self.threads:job.update(status='cancelled',finished_at=time.time())
            self.condition.notify_all()
            threads=list(self.threads.values())
        for control in list(self.controls.values()):
            try:
                if not control.closed and not control.stopped:control.request('stop')
            except Exception as exc:
                # Control notifications may write evidence. Their failures must
                # not leave other controls live or skip joining existing writers.
                self.shutdown_errors.append(str(exc))
        if wait:
            current=threading.current_thread()
            for worker in threads:
                if worker is not current and worker.is_alive():worker.join()
