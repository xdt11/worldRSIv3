"""Immutable skill-only Git snapshots, without historical task knowledge.

The active ref advances after explicit manual or experiment adoption. Candidate
commits never alter the active version or a running task's pinned snapshot.
"""
from __future__ import annotations

import os
import json
import hashlib

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import fcntl
import threading
from contextlib import contextmanager


from harness.skills.validation import ALLOWED, validate_skills

COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class VersionStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self._depth = 0

    @contextmanager
    def mutation(self):
        """Coordinate repository writes across threads and VersionStore instances."""
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            outer = self._depth == 0
            if outer:
                stream = (self.root / 'worldrsi.lock').open('a')
                fcntl.flock(stream, fcntl.LOCK_EX)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if outer:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()

    @staticmethod
    def active_ref(lineage=None):
        if lineage is None:
            return 'refs/worldrsi/active'
        if not isinstance(lineage, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,90}', lineage):
            raise ValueError('Invalid campaign version lineage')
        return 'refs/worldrsi/campaigns/' + lineage + '/active'

    def _git(self, *args: str, data: str | None = None, strip: bool = True) -> str:
        env = {**os.environ, "GIT_AUTHOR_NAME": "worldRSI", "GIT_AUTHOR_EMAIL": "worldrsi@localhost",
               "GIT_COMMITTER_NAME": "worldRSI", "GIT_COMMITTER_EMAIL": "worldrsi@localhost"}
        result = subprocess.run(["git", "-C", str(self.root), *args], input=data, text=True,
                                capture_output=True, env=env)
        if result.returncode:
            raise ValueError(result.stderr.strip() or "Git operation failed")
        return result.stdout.rstrip("\n") if strip else result.stdout

    def _commit(self, commit: str) -> str:
        if not COMMIT.fullmatch(commit):
            raise ValueError("Version must be a full Git commit SHA")
        self._git("cat-file", "-e", commit + "^{commit}")
        return commit

    def initialize(self) -> str:
        with self.mutation():
            return self._initialize()

    def _initialize(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists() and not (self.root / "HEAD").exists():
            self._git("init", "--bare", "--quiet")
        try:
            return self.active()
        except ValueError:
            pass
        # Prompts belong to the fixed harness; the initial skill tree is empty.
        tree = self._tree({})
        commit = self._git("commit-tree", tree, data="Initial empty skills (v3)\n")
        self._git("update-ref", "refs/worldrsi/active", commit, "0" * 40)
        self._git("update-ref", "refs/heads/main", commit)
        return commit

    def active(self, lineage=None) -> str:
        return self._git('rev-parse', '--verify', self.active_ref(lineage) + '^{commit}')

    def fork_lineage(self, lineage, parent=None):
        """Create a campaign head once; never repoint an existing experiment."""
        with self.mutation():
            ref = self.active_ref(lineage)
            try:
                return self.active(lineage)
            except ValueError:
                parent = self._commit(parent or self.active())
                self._git('update-ref', '--create-reflog', ref, parent, '0' * 40)
                return parent

    def snapshot(self, commit: str | None = None) -> dict[str, str]:
        commit = self._commit(commit or self.active())
        rows = self._git("ls-tree", "-r", commit).splitlines()
        names = []
        for row in rows:
            header, name = row.split("\t", 1)
            mode, kind, _ = header.split()
            if mode not in {'100644', '100755'} or kind != 'blob' or not (ALLOWED.fullmatch(name)):
                raise ValueError("Version contains unsafe or non-skill files")
            names.append(name)
        return {name: self._git("show", f"{commit}:{name}", strip=False) for name in names}

    def skill_snapshot(self, commit: str | None = None) -> dict[str, str]:
        return validate_skills({name: value for name, value in self.snapshot(commit).items()
                                if ALLOWED.fullmatch(name)})

    def materialize(self, commit: str, path: Path) -> dict[str, str]:
        """Write/verify the pinned read-only directory mounted at /skills."""
        files = self.skill_snapshot(commit)
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Skill snapshot root cannot be a symlink')
        path.mkdir(parents=True, exist_ok=True)
        expected = {name.removeprefix('skills/'): value for name, value in files.items()}
        actual = {str(item.relative_to(path)): item for item in path.rglob('*') if item.is_file() or item.is_symlink()}
        if actual:
            if set(actual) != set(expected) or any(item.is_symlink() or item.read_text() != expected[name] for name, item in actual.items()):
                raise ValueError('Task skill files differ from the pinned Git snapshot')
        else:
            for name, value in expected.items():
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(value)
                target.chmod(0o444)
        return {name: hashlib.sha256(value.encode()).hexdigest() for name, value in expected.items()}

    def _tree(self, files: dict[str, str]) -> str:
        validate_skills(files)
        root = {}
        for name, value in sorted(files.items()):
            node = root
            parts = name.split('/')
            for folder in parts[:-1]:
                node = node.setdefault(folder, {})
            node[parts[-1]] = self._git("hash-object", "-w", "--stdin", data=value)
        def tree(node):
            rows = []
            for name, value in sorted(node.items()):
                kind, mode, digest = ('tree', '040000', tree(value)) if isinstance(value, dict) else ('blob', '100644', value)
                rows.append(f'{mode} {kind} {digest}\t{name}\n')
            return self._git('mktree', data=''.join(rows))
        return tree(root)

    def validate_changes(self, parent: str, changes: dict[str, str | None]) -> dict[str, str]:
        if not isinstance(changes, dict):
            raise ValueError('Skill changes must be a mapping')
        files = self.snapshot(parent)
        if any(not ALLOWED.fullmatch(name) for name in files):
            raise ValueError('Only skill content may exist in a skill version')
        for name, value in changes.items():
            if not isinstance(name, str) or not ALLOWED.fullmatch(name):
                raise ValueError(f'Evolution path not allowed: {name}; only skills/<name>/... may evolve')
            if value is None:
                files.pop(name, None)
            else:
                files[name] = value
        return validate_skills(files)

    def propose(self, parent: str, changes: dict[str, str | None], summary: str, *, provenance=None) -> str:
        with self.mutation():
            return self._propose(parent, changes, summary, provenance=provenance)

    def _propose(self, parent, changes, summary, *, provenance=None):
        parent = self._commit(parent)
        files = self.validate_changes(parent, changes)
        tree = self._tree(files)
        if tree == self._git('rev-parse', parent + '^{tree}'):
            return parent
        message = (summary.strip() or 'Skill candidate') + '\n'
        if provenance:
            message += '\nEvidence provenance: ' + json.dumps(provenance, ensure_ascii=False, sort_keys=True) + '\n'
        commit = self._git("commit-tree", tree, "-p", parent, data=message)
        self._git("update-ref", f"refs/worldrsi/candidates/{commit}", commit)
        return commit

    def diff(self, base: str, candidate: str) -> str:
        return self._git("diff", "--no-ext-diff", self._commit(base), self._commit(candidate), "--")

    def adopt(self, candidate: str, decision: dict, *, lineage=None) -> None:
        with self.mutation():
            self._adopt(candidate, decision, lineage=lineage)

    def _adopt(self, candidate, decision, *, lineage=None):
        if lineage is None and decision.get('mode') != 'manual':
            raise ValueError('The global active version requires explicit manual adoption')
        candidate = self._commit(candidate)
        active = self.active(lineage)
        parents = self._git("rev-list", "--parents", "-n", "1", candidate).split()[1:]
        if parents != [active]:
            raise ValueError('Only a direct child of this active lineage can be adopted')
        if (decision.get("decision") != "accepted" or decision.get("baseline_version") != active
                or decision.get("candidate_version") != candidate):
            raise ValueError("Adoption requires an accepted decision for these exact versions")
        if decision.get('scope') and decision['scope'].get('ref') != self.active_ref(lineage):
            raise ValueError('Adoption decision belongs to another version lineage')
        changed = self._git('diff', '--name-only', active, candidate, '--').splitlines()
        if any(not ALLOWED.fullmatch(name) for name in changed):
            raise ValueError('Only skill-only candidates may be adopted by this harness')
        validate_skills(self.snapshot(candidate))
        self._git("update-ref", "--create-reflog", "-m", "worldRSI " + decision.get("mode", "policy"),
                  self.active_ref(lineage), candidate, active)

    def adopt_manual(self, candidate: str, reason: str = 'User confirmed', *, lineage=None) -> dict:
        return self._adopt_by_mode(candidate, 'manual', reason, lineage=lineage)

    def adopt_auto(self, candidate: str, reason: str = 'Experiment automatic adoption enabled', *, lineage=None) -> dict:
        if lineage is None:
            raise ValueError('Experiment adoption requires its campaign lineage')
        return self._adopt_by_mode(candidate, 'experiment_auto', reason, lineage=lineage)

    def _adopt_by_mode(self, candidate: str, mode: str, reason: str, *, lineage=None) -> dict:
        with self.mutation():
            decision = {'decision': 'accepted', 'mode': mode, 'reason': reason,
                'baseline_version': self.active(lineage), 'candidate_version': candidate,
                'confirmed_at': datetime.now(timezone.utc).isoformat(),
                'scope': {'kind': 'campaign' if lineage else 'global', 'id': lineage,
                          'ref': self.active_ref(lineage)}}
            self.adopt(candidate, decision, lineage=lineage)
            return decision

    def history(self) -> list[dict]:
        active = self.active()
        rows = self._git("log", "--all", "--format=%H%x09%P%x09%aI%x09%s").splitlines()
        result = []
        for row in rows:
            commit, parents, created, summary = row.split("\t", 3)
            result.append({"commit": commit, "parents": parents.split(), "created_at": created,
                           "summary": summary, "active": commit == active,
                           "schema_version": 3, "evolution_scope": "skills"})
        return result
