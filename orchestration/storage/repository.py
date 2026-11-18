"""Filesystem adapter; paths and locks belong to an application instance."""
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from orchestration.config import normalize_config
from harness.evidence.usage import call_usage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from harness.evidence.archive import now, uid, inside, load, save, sha


def identifier(value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,90}',value):raise ValueError('Invalid identifier')
    return value


class ArtifactStore:
    _lock_registry = {}
    _registry_guard = threading.Lock()

    def __init__(self, root: Path = PROJECT_ROOT, data: Path | None = None):
        self.root = Path(root).resolve()
        self.data = (self.root / data).resolve() if data is not None else self.root / 'data'
        with self._registry_guard:
            self.lock = self._lock_registry.setdefault(self.data, threading.RLock())
        self._calls_cache = {}

    def config(self):
        value=normalize_config(load(self.data / 'config.json', {}))
        # Legacy defaults may contain false; new campaigns always auto-adopt.
        value['campaign']['auto_adopt']=True
        return value

    def api_key(self):
        key = os.environ.get('WORLD_RSI_API_KEY')
        if key:
            return key
        path = self.root / '.env'
        for line in path.read_text().splitlines() if path.exists() else []:
            if line.startswith('WORLD_RSI_API_KEY='):
                return line.split('=', 1)[1].strip()
        return ''

    def task_path(self, task_id):
        identity = identifier(task_id)
        root = self.data / 'safe/tasks' if identity.startswith('safe_task_') else self.data / 'tasks'
        return root / identity

    def task_records(self, kind=None):
        """Separate scene-edit archives while sharing scheduling and skill versions."""
        if kind not in (None, 'reconstruction', 'threat'):
            raise ValueError('Unknown task kind')
        roots = ([self.data / 'tasks'] if kind == 'reconstruction' else
                 [self.data / 'safe/tasks'] if kind == 'threat' else
                 [self.data / 'tasks', self.data / 'safe/tasks'])
        return sorted(path for root in roots for path in root.glob('*/task.json'))

    def task(self, task_id):
        result = load(self.task_path(task_id) / 'task.json')
        if result is None:
            raise FileNotFoundError(task_id)
        return result

    def calls(self, task_id):
        """Small derived receipts, cached by file identity; raw history is never rewritten."""
        records = []
        for path in (self.task_path(task_id)/'invocations').glob('*/calls/*/meta.json'):
            response = path.with_name('response.json')
            signature = tuple((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
                              if p.exists() else None for p in (path,response))
            with self.lock:
                cached = self._calls_cache.get(path)
                if cached and cached[0]==signature:
                    records.append(cached[1]); continue
            record = load(path)
            if 'llmgate_usage' not in record and 'charged_amount' not in record:
                body = load(response,{})
                record = {**record, **{key:body.get(key) for key in ('llmgate_usage','charged_amount','currency')}}
            record = {**record,'usage':call_usage(record)}
            with self.lock:
                if len(self._calls_cache)>=4096: self._calls_cache.pop(next(iter(self._calls_cache)))
                self._calls_cache[path] = (signature,record)
            records.append(record)
        return records

    def update_task(self, task_id, **changes):
        return self.update_json(self.task_path(task_id) / 'task.json',
            lambda value: value.update(changes, updated_at=now()))

    def update_json(self, path, change):
        """Read the latest record and atomically replace it under the store lock."""
        path = inside(self.data, path)
        with self.lock:
            value = load(path)
            if value is None:
                raise FileNotFoundError(path)
            change(value)
            save(path, value)
            return value

    def run_update(self, task_id, run_id, **changes):
        with self.lock:
            value = self.task(task_id)
            record = next(r for r in value['runs'] if r['id'] == run_id)
            record.update(changes, updated_at=now())
            return self.update_task(task_id, runs=value['runs'])

    def inputs(self):
        result = []
        benchmark = {r['path']: r for r in load(self.data / 'benchmarks/registry.json', {}).get('cases', [])}
        for path in sorted((self.data / 'inputs').rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            suffix = path.suffix.lower()
            kind = 'video' if suffix in {'.mp4', '.mov', '.webm', '.mkv'} else 'image' if suffix in {'.png', '.jpg', '.jpeg', '.webp'} else 'file'
            relative = path.relative_to(self.data / 'inputs')
            result.append({'path': str(relative), 'kind': kind, 'size': path.stat().st_size,
                           'split': relative.parts[0],
                           **({'benchmark': benchmark[str(relative)]} if str(relative) in benchmark else {})})
        return result
