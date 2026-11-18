"""Task and run scopes carry dependencies; importing a module starts no work."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskContext:
    record: dict
    directory: Path
    services: Any

    @property
    def id(self): return self.record['id']

    @property
    def config(self): return self.record['config']

    def refresh(self):
        self.record = self.services.store.task(self.id)
        return self.record

    def stopped(self):
        try: self.services.checkpoint(self.record)
        except InterruptedError: return True
        return self.services.stop_requested() or self.services.store.task(self.id).get('stop_requested', False)

    def run(self, run_id):
        record = next(row for row in self.record['runs'] if row['id'] == run_id)
        return RunContext(self, record, self.directory / 'runs' / run_id)


@dataclass
class RunContext:
    task: TaskContext
    record: dict
    directory: Path
    state: dict = field(default_factory=dict)

    @property
    def id(self): return self.record['id']

    @property
    def services(self): return self.task.services
