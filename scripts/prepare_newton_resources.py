#!/usr/bin/env python3
"""Provision or fingerprint the workspace-owned, relocatable Newton prefix.

The simulation prefix and manifest are independent of the historical Genesis
and scorer resource identity. Recording does not compile or execute scene code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_inventory(resources, prefix, *, allow_owned_links=False):
    artifacts = []
    for path in sorted(prefix.rglob("*")):
        relative = path.relative_to(prefix)
        if {".git", ".cache", "__pycache__"}.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(prefix) or (path.is_symlink() and not allow_owned_links):
            raise ValueError("Resource must resolve inside its owned prefix: " + str(path))
        artifacts.append({"path": str(path.relative_to(resources)), "bytes": path.stat().st_size, "sha256": sha(path)})
    return artifacts


def record_perception(resources, previous):
    perception = resources / "perception"
    interpreter = perception / "python/bin/python3.12"
    if not interpreter.is_file():
        raise ValueError("Prepare the standalone perception Python prefix first")
    packages_dir = perception / ".venv/lib/python3.12/site-packages"
    command = (
        "import importlib.metadata as m,json,sys; "
        "print(json.dumps(sorted([{'name':d.metadata['Name'],'version':d.version} "
        "for d in m.distributions(path=[sys.argv[1]])],key=lambda d:d['name'].lower())))"
    )
    packages = json.loads(subprocess.check_output([str(interpreter), "-c", command, str(packages_dir)],
                                                   text=True, timeout=60))
    revisions = {key: value for key, value in previous.get("source_revisions", {}).items()
                 if not key.startswith("perception/")}
    for name in ("Depth-Anything-3", "sam3"):
        directory = perception / name
        if (directory / ".git").exists():
            revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                               text=True, timeout=10).strip()
            revisions["perception/" + name] = revision
    artifacts = [row for row in previous.get("artifacts", []) if not row["path"].startswith("perception/")]
    artifacts.extend(file_inventory(resources, perception, allow_owned_links=True))
    return {**previous, "packages": {**previous.get("packages", {}), "perception": packages},
            "source_revisions": revisions, "artifacts": artifacts,
            "verification_scope": "File identity of prepared resources. GPU probing is separate."}


def prepare(resources, *, install=False, python_source=None, wheelhouse=None, refresh_perception=False):
    resources = Path(resources).resolve()
    prefix = resources / "newton_env"
    if not prefix.exists():
        if not install:
            raise ValueError("Newton prefix is missing; use --install for explicit setup")
        source = Path(python_source or resources / "perception/python").resolve()
        if not (source / "bin/python").is_file() or (source / "pyvenv.cfg").exists():
            raise ValueError("Provide a standalone Python prefix, not a virtualenv")
        shutil.copytree(source, prefix, symlinks=False, ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "*.pyc", "EXTERNALLY-MANAGED"))
    interpreter = prefix / "bin/python"
    if install:
        # This marker is removed only from the new owned standalone copy.
        marker = prefix / "lib/python3.12/EXTERNALLY-MANAGED"
        if marker.exists():
            marker.unlink()
        subprocess.run([str(interpreter), "-m", "ensurepip", "--upgrade"], check=True)
        requirements = ROOT / "requirements-newton.lock"
        command = [str(interpreter), "-m", "pip", "install", "--no-cache-dir", "--timeout", "120",
                   "-r", str(requirements)]
        if wheelhouse:
            command += ["--no-index", "--find-links", str(Path(wheelhouse).resolve())]
        subprocess.run(command, check=True)
    code = ("import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,"
            "'prefix':sys.prefix,'packages':sorted([{'name':d.metadata['Name'],'version':d.version} "
            "for d in m.distributions()],key=lambda d:d['name'].lower())}))")
    result = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True, check=True, timeout=60)
    environment = json.loads(result.stdout)
    packages = {row["name"].lower(): row["version"] for row in environment["packages"]}
    if packages.get("newton") != "1.6.1" or packages.get("warp-lang") != "1.17.0":
        raise ValueError("This resource preparation expects Newton 1.6.1 and CUDA-12 Warp 1.17.0")
    artifacts = file_inventory(resources, prefix)
    legacy_path = resources / "identity.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.is_file() else {}
    if refresh_perception or (not legacy_path.exists() and (resources / "perception/python/bin/python3.12").is_file()):
        legacy = record_perception(resources, legacy)
        legacy_path.write_text(json.dumps(legacy, indent=2) + "\n")
    artifacts += [row for row in legacy.get("artifacts", []) if row["path"].startswith("perception/")]
    lock = ROOT / "requirements-newton.lock"
    identity = {
        "schema_version": 1, "backend": "newton-warp", "python": environment["python"],
        "packages": {"newton": environment["packages"], "perception": legacy.get("packages", {}).get("perception", [])},
        "source_revisions": legacy.get("source_revisions", {}), "artifacts": artifacts,
        "requirements_sha256": sha(lock) if lock.is_file() else None,
        "verification_scope": "SHA-256 of the complete owned Python/Newton/Warp prefix, including native libraries, "
                              "and existing independently owned perception resources. This records file identity, "
                              "not GPU compatibility, numerical correctness or adversarial host attestation.",
    }
    target = resources / "newton-identity.json"
    target.write_text(json.dumps(identity, indent=2) + "\n")
    return {"manifest": str(target), "manifest_sha256": sha(target), "artifacts": len(artifacts),
            "newton": packages["newton"], "warp": packages["warp-lang"], "runtime_validation": "not_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=ROOT / "data/resources")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--python-source", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--refresh-perception", action="store_true",
                        help="Update locally prepared perception files and source revisions in the resource manifests")
    args = parser.parse_args()
    print(json.dumps(prepare(args.resources, install=args.install, python_source=args.python_source,
                             wheelhouse=args.wheelhouse, refresh_perception=args.refresh_perception), indent=2))


if __name__ == "__main__":
    main()
