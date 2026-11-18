"""Lifecycle and resource management for frontend observation sessions."""
import threading
import time
from harness.tools.newton.session import NativeSession
from harness.evidence.artifacts import archived_render

class LivePool:
    def __init__(self, store, sandbox_factory, *, session_factory=None):
        self.store, self.sandbox_factory = store, sandbox_factory
        self.session_factory = session_factory or NativeSession
        self.lock = threading.Lock()
        self.reservation_lock = threading.Lock()
        self.sessions = {}
        self.stop = threading.Event()
        self.reaper = None

    def start(self):
        self.stop.clear()
        self.reaper = threading.Thread(target=self._reap, daemon=True, name='newton-session-reaper')
        self.reaper.start()

    def reserve(self, task_id, run_id):
        settings = self.store.config()['live']
        revision = None
        if self.session_factory is NativeSession:
            value = self.store.task(task_id)
            run = next((row for row in value['runs'] if row['id'] == run_id), None)
            archive = archived_render(self.store.task_path(task_id), run)
            if not archive: raise ValueError('This run has no native output')
            revision = archive['render_revision']
        key = (task_id, run_id, revision)
        # Reserve operations serialize, but never hold the observable pool lock
        # during a slow container shutdown or a cold Newton compilation.
        with self.reservation_lock:
            with self.lock:
                session = self.sessions.get(key)
                victim = None
                if session and (session.closed or getattr(session, 'cancel', threading.Event()).is_set()):
                    victim = session
                    session = None
                if not session and not victim and len(self.sessions) >= settings['max_sessions']:
                    idle = [s for s in self.sessions.values() if not s.leases]
                    if not idle:
                        raise ValueError(f'All {settings["max_sessions"]} live rendering slots are in use')
                    victim = min(idle, key=lambda s: s.last_used)
            if victim:
                victim.close()
                with self.lock:
                    self.sessions.pop(victim.pool_key, None)
            with self.lock:
                if not session:
                    session = self.session_factory(task_id, run_id, settings, store=self.store,
                                                   sandbox_factory=self.sandbox_factory)
                    session.pool_key = key
                    self.sessions[key] = session
                session.leases += 1
            return session

    def acquire(self, task_id, run_id):
        session = self.reserve(task_id, run_id)
        try:
            session.initialize()
            return session
        except Exception:
            self.release(session)
            raise

    def release(self, session):
        with self.lock:
            session.leases = max(0, session.leases - 1)
            session.last_used = time.monotonic()
            # A short page/tab interruption must not repeatedly cancel cold
            # compilation. The pool retains one bounded startup for reconnects.

    def snapshot(self):
        with self.lock:
            return [{'task_id': s.key[0], 'run_id': s.key[1], 'status': 'closed' if s.closed else
                     'ready' if s.ready else 'starting', 'viewers': s.leases,
                     'phase': getattr(s,'phase','ready' if s.ready else 'starting'),
                     'render_revision':getattr(s,'archive_revision',None),
                     'idle_s': round(time.monotonic() - s.last_used, 1)} for s in self.sessions.values()]

    def _reap(self):
        while not self.stop.wait(5):
            ttl = self.store.config()['live']['idle_timeout_s']
            with self.reservation_lock:
                with self.lock:
                    expired = [s for s in self.sessions.values() if not s.leases and
                               (s.closed or (not getattr(s,'initializing',False) and time.monotonic() - s.last_used > ttl))]
                for session in expired:
                    session.close()
                    with self.lock:
                        self.sessions.pop(session.pool_key, None)

    def close(self):
        self.stop.set()
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.close()
        if self.reaper:
            self.reaper.join(timeout=6)
