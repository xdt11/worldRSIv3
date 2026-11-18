"""Immutable scene/solver dependency closure and isolated Newton rebuild.

The official solution adapter uses literal Python data for non-code assets.
The adjacent delivery directory contains the same files in their ordinary form.
Neither packaging nor a manifest is evidence that a rebuild has succeeded.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
EXCLUDED = {"renders", "tool_results", "solution", "delivery", "snapshots", "__pycache__"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def worker_identity():
    harness = Path(__file__).resolve().parents[3]
    return {str(path.relative_to(harness)): digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py")) if path.is_file()}


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError("Dependency paths must be strings")
    path = PurePosixPath(value)
    if path.is_absolute():
        if not path.is_relative_to("/work"):
            raise ValueError("Dependency must be inside the task workspace")
        path = path.relative_to("/work")
    if not path.parts or any(part.startswith(".") or part in EXCLUDED for part in path.parts):
        raise ValueError("Dependency path must be a normal relative task file")
    return str(path)


def checked_file(root, value):
    root = Path(root).resolve()
    path = root / relative_path(value)
    if any(item.is_symlink() for item in [path, *path.parents] if item != root.parent):
        raise ValueError("Dependency closure cannot contain symlinks")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing dependency: " + str(value))
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Dependency exceeds 64 MiB: " + str(value))
    return path


def dependency_names(code, declared=None):
    code = Path(code).resolve()
    root = code.parent
    if declared is not None and (not isinstance(declared, (list, tuple)) or len(declared) > 512):
        raise ValueError("Declared dependencies must be a bounded list of relative files")
    names = {"scene.py", "world.yaml"}
    if (root / "output_contract.json").is_file():
        names.add("output_contract.json")
    for name in declared or []:
        names.add(relative_path(name))
    # Follow local imports from the scene and declared helpers. Diagnostic or
    # scratch Python files elsewhere in /work are not simulation dependencies.
    pending = [name for name in names if name.endswith(".py")]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = checked_file(root, name)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                base = root
            elif isinstance(node, ast.ImportFrom):
                base = path.parent if node.level else root
                for _ in range(max(0, node.level - 1)):
                    base = base.parent
                modules = [node.module or ""]
                modules += [".".join(filter(None, [node.module, alias.name]))
                            for alias in node.names if alias.name != "*"]
            else:
                continue
            for module in modules:
                target = base.joinpath(*module.split(".")) if module else base
                candidates = [target.with_suffix(".py"), target / "__init__.py"]
                candidates += [parent / "__init__.py" for parent in target.parents
                               if parent != root and parent.is_relative_to(root)]
                for candidate in candidates:
                    if candidate.is_file() and candidate.is_relative_to(root):
                        dependency = relative_path(str(candidate.relative_to(root)))
                        if dependency not in names:
                            names.add(dependency)
                            pending.append(dependency)
    if len(names) > 514:
        raise ValueError("Scene dependency closure exceeds 514 files")
    files = [(name, checked_file(root, name)) for name in sorted(names)]
    if sum(path.stat().st_size for _, path in files) > MAX_BUNDLE_BYTES:
        raise ValueError("Scene dependency closure exceeds 256 MiB")
    return files



def snapshot_inputs(code):
    """Record entry/import bytes before loading code, not the whole workspace."""
    return {name: digest(path) for name, path in dependency_names(code)}


def freeze_dependencies(code, declared, preimages, expected=None):
    """Reject build/step/close mutations instead of delivering different inputs."""
    files = []
    for name, path in dependency_names(code, declared):
        identity = digest(path)
        if name in preimages and preimages[name] != identity:
            raise ValueError("Scene dependencies changed during native execution: " + name +
                             ". Prepare all delivered code/assets before submit_scene; execution must not rewrite them.")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": identity})
    if expected is not None and files != expected:
        raise ValueError("Scene dependencies changed during native execution: the dependency closure changed")
    return files


def dependency_status(root, expected):
    """Inspect accepted inputs without importing or executing scene code."""
    changed, missing = [], []
    for item in expected:
        try:
            path = checked_file(root, item["path"])
        except (OSError, ValueError):
            missing.append(item["path"])
            continue
        if digest(path) != item["sha256"]:
            changed.append(item["path"])
    return {"current": not (changed or missing), "changed": changed[:20], "missing": missing[:20]}


def read_bundle(directory):
    path = Path(directory) / "bundle.py"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES * 2:
        raise ValueError("Invalid literal dependency bundle")
    tree = ast.parse(path.read_text())
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
            or len(tree.body[0].targets) != 1 or not isinstance(tree.body[0].targets[0], ast.Name)
            or tree.body[0].targets[0].id != "BUNDLE"):
        raise ValueError("Bundle must contain one literal BUNDLE assignment")
    return ast.literal_eval(tree.body[0].value)


def package_solution(output, code, manifest, declared=None, *, expected_files=None):
    output, code = Path(output), Path(code).resolve()
    solution, delivery = output / "solution", output / "delivery"
    solution.mkdir()
    delivery.mkdir()
    files, encoded = [], {}
    for name, source in dependency_names(code, declared):
        raw = source.read_bytes()
        target = delivery / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if name == "scene.py":
            (solution / name).write_bytes(raw)
        else:
            encoded[name] = base64.b64encode(raw).decode("ascii")
        files.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if expected_files is not None and files != expected_files:
        raise ValueError("Scene dependencies changed while packaging native execution inputs")
    record = {"schema_version": 2, "engine": "newton", "files": files,
              "encoded_files": encoded, "workers": worker_identity(), "python_version": sys.version,
              "engine_version": manifest["engine_version"], "warp_version": manifest["warp_version"],
              "effective_seed": manifest.get("effective_seed"),
              "output_contract_sha256": manifest.get("output_contract_sha256"),
              "reference_source_id": manifest.get("reference_source_id"),
              "expected_frames": [{key: frame[key] for key in ("timestamp_s", "pixel_sha256")} for frame in manifest["frames"]]}
    if manifest.get('inspection'):
        record['inspection'] = manifest['inspection']
    (solution / "bundle.py").write_text("BUNDLE = " + repr(record) + "\n")
    (solution / "rebuild.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "from harness.tools.newton.workers.solution import rebuild\n"
        "if __name__ == '__main__':\n"
        "    rebuild(Path(__file__).resolve().parent, sys.argv[1] if len(sys.argv)>1 else '/work/rebuilt')\n")
    (solution / "build.sh").write_text('#!/bin/sh\nset -eu\nexec python "$(dirname "$0")/rebuild.py" "${1:-/work/rebuilt}"\n')
    (solution / "build.sh").chmod(0o755)
    (delivery / "dependency_manifest.json").write_text(json.dumps({key: value for key, value in record.items() if key != "encoded_files"}, indent=2))
    return {"status": "packaged", "verification": "not_run", "files": files,
            "delivery": "delivery", "solution": "solution", "workers": record["workers"],
            "scope": "Workspace Python closure and explicitly declared assets; isolated rebuild verifies actual completeness"}


def rebuild(bundle, output, workspace=None, entrypoint=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    workspace = Path(workspace or "/work").resolve()
    record = read_bundle(bundle)
    if record.get("schema_version") != 2 or record.get("engine") != "newton":
        raise ValueError("Unsupported Newton solution format")
    if record["python_version"] != sys.version or record["workers"] != worker_identity():
        raise ValueError("Rebuild requires the recorded Newton worker and Python environment")
    allowed = {".cache"}
    if bundle.is_relative_to(workspace):
        if bundle.parent != workspace:
            raise ValueError("Mount solution directly under the empty workspace or outside it")
        allowed.add(bundle.name)
    if any(path.name not in allowed for path in workspace.iterdir()):
        raise ValueError("Rebuild requires an empty task workspace")
    if output.exists() or output.is_relative_to(bundle) or workspace.is_relative_to(output):
        raise ValueError("Rebuild output must be absent and outside the solution")
    files = record["files"]
    names = [relative_path(item["path"]) for item in files]
    if len(names) > 514 or len(set(names)) != len(names) or "scene.py" not in names or "world.yaml" not in names:
        raise ValueError("Invalid scene dependency closure")
    if set(record["encoded_files"]) != set(names) - {"scene.py"}:
        raise ValueError("Invalid encoded dependency manifest")
    decoded = {}
    for item in files:
        name = item["path"]
        raw = (bundle / name).read_bytes() if name == "scene.py" else base64.b64decode(record["encoded_files"][name], validate=True)
        if len(raw) != item["bytes"] or len(raw) > MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Dependency hash mismatch: " + name)
        decoded[name] = raw
    if sum(map(len, decoded.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("Dependency bundle exceeds total size limit")
    for name, raw in decoded.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    env = dict(os.environ)
    if record["effective_seed"] is not None:
        env["WORLD_RSI_SEED"] = str(record["effective_seed"])
    command = [sys.executable, "-m", "harness.tools.newton.workers.entry", str(workspace / "scene.py"), str(output)]
    if entrypoint is not None:
        command = [sys.executable, str(entrypoint), str(workspace / "scene.py"), str(output)]
    if record.get('inspection'):
        command.append(json.dumps(record['inspection']))
    result = subprocess.run(command, cwd=workspace, env=env)
    receipt = {"status": "failed", "exit_code": result.returncode, "empty_task_workspace": True,
               "effective_seed": record["effective_seed"], "bundle_sha256": digest(bundle / "bundle.py")}
    if result.returncode == 0:
        made = json.loads((output / "render_manifest.json").read_text())
        actual = [{key: frame[key] for key in ("timestamp_s", "pixel_sha256")} for frame in made["frames"]]
        matches = (actual == record["expected_frames"] and made["engine_version"] == record["engine_version"]
                   and made["warp_version"] == record["warp_version"]
                   and made.get("output_contract_sha256") == record.get("output_contract_sha256")
                   and made.get("reference_source_id") == record.get("reference_source_id"))
        receipt.update(status="verified" if matches else "mismatch", frames_match=matches)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rebuild_receipt.json").write_text(json.dumps(receipt, indent=2))
    if receipt["status"] != "verified":
        raise ValueError("Isolated Newton rebuild did not reproduce native pixels/times")
    return receipt


if __name__ == "__main__":
    rebuild(Path(__file__).resolve().parent, sys.argv[1] if len(sys.argv) > 1 else "/work/rebuilt")
