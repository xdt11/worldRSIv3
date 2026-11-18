"""Filesystem adapter; paths and locks belong to an application instance."""
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path





def now():
    return datetime.now(timezone.utc).isoformat()

def uid(prefix):
    return f'{prefix}_{uuid.uuid4().hex[:12]}'

def inside(root, path):
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Path escapes its scope')
    return target

def identifier(value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,90}', value):
        raise ValueError('Invalid identifier')
    return value

def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temp.open('w') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class InvocationArchive:
    """One invocation's immutable request/tool receipts, independent of business state."""
    def __init__(self, task_directory, *, role, task_id=None, run_id=None, invocation_id=None):
        self.id = invocation_id or uid('invocation')
        self.role, self.task_id, self.run_id = role, task_id, run_id
        self.task_directory = Path(task_directory)
        self.directory = self.task_directory / 'invocations' / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.record = {'id': self.id, 'role': role, 'stage': role, 'task_id': task_id,
                       'run_id': run_id, 'status': 'prepared', 'created_at': now()}
        save(self.directory / 'invocation.json', self.record)

    def update(self, **changes):
        self.record.update(changes, updated_at=now())
        save(self.directory / 'invocation.json', self.record)

    def usage(self):
        from .usage import summarize_usage
        return summarize_usage(self.calls)

    def event(self, event):
        path = self.directory / 'tools.json'
        save(path, [*load(path, []), event])

    def relative(self, path):
        return str(Path(path).relative_to(self.task_directory))
