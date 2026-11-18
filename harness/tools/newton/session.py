"""Bounded isolated Newton observation sessions for agents and the viewer."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import threading
import time
import uuid

from PIL import Image

from harness.evidence.archive import identifier, inside, load, save, sha, now, uid
from harness.evidence.artifacts import archived_render
from harness.tools.newton.workers.camera_math import normalize_camera_pose
from harness.tools.newton.workers.solution import read_bundle, relative_path, worker_identity, MAX_FILE_BYTES, MAX_BUNDLE_BYTES


class FrameRequestError(ValueError):
    """An invalid camera/time request which does not invalidate native state."""


def _recorded_viewer_workers(task_dir, run_id, expected, startup_path=None):
    # The controller wrote sandbox receipts and snapshots outside /work. The
    # delivered scene may select a matching revision, never a host source path.
    packages = ("harness/__init__.py", "harness/tools/__init__.py", "harness/tools/newton/__init__.py")
    prefix = "tools/newton/workers/"
    if not isinstance(expected, dict) or prefix + "live_entry.py" not in expected or any(
            not isinstance(name, str) or not re.fullmatch(r"tools/newton/workers/[A-Za-z_]\w*\.py", name)
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for name, digest in expected.items()):
        raise ValueError("Invalid recorded Newton worker identities")
    root = task_dir / "runs" / identifier(run_id) / "sandboxes"
    if root.absolute() == root.resolve() and root.is_dir():
        if startup_path is None:
            candidates = sorted(root.iterdir())
        else:
            match = re.fullmatch(rf"runs/{re.escape(run_id)}/sandboxes/(worldrsi-[a-f0-9]{{32}})/startup\.json", startup_path)
            if not match:
                raise ValueError("Invalid recorded Newton runtime startup path")
            candidates = [root / match[1]]
        for box in candidates:
            startup = box / "startup.json"
            if (not re.fullmatch(r"worldrsi-[a-f0-9]{32}", box.name) or startup.resolve() != startup.absolute()
                    or not startup.is_file()):
                continue
            receipt = load(startup, {})
            if (not isinstance(receipt, dict) or receipt.get("run_id") != run_id or receipt.get("purpose") != "task"
                    or receipt.get("status") not in {"started", "recovered"} or receipt.get("container_name") != box.name
                    or receipt.get("labels", {}).get("org.worldrsi.task") != task_dir.name):
                continue
            hashes = receipt.get("worker_sha256", {})
            if not isinstance(hashes, dict):
                continue
            formal = {name.removeprefix("harness/"): digest for name, digest in hashes.items()
                      if name.startswith("harness/" + prefix)}
            if formal != expected:
                continue
            content = {}
            for name in (*packages, *("harness/" + worker for worker in formal)):
                path = box / "workers" / name
                if path.resolve() != path.absolute() or not path.is_file():
                    break
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != hashes.get(name):
                    break
                content[name] = raw
            else:
                return content
    raise ValueError("Recorded Newton runtime revision is unavailable; archived RGB remains available")


def render_device(requested):
    if requested == "cpu":
        return None, {"selection": "cpu", "reason": "Explicit Warp CPU device"}
    if requested != "auto":
        if not isinstance(requested, str) or not re.fullmatch(r"\d+|GPU-[a-fA-F0-9-]+", requested):
            raise ValueError("live.device must be auto, cpu, GPU index or UUID")
        return "device=" + requested, {"selection": "explicit", "device": requested}
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    candidates = []
    for line in result.stdout.splitlines() if result.returncode == 0 else []:
        identity, free, utilization = [part.strip() for part in line.split(",")]
        if re.fullmatch(r"GPU-[a-fA-F0-9-]+", identity):
            candidates.append((float(free), -float(utilization), identity))
    if not candidates:
        raise RuntimeError("No CUDA device is available for the requested Newton viewer")
    free, utilization, identity = max(candidates)
    return "device=" + identity, {"selection": "auto_most_free_memory", "device": identity,
                                  "free_memory_mb_at_start": free, "utilization_at_start": -utilization}


def render_command(value, settings):
    if not isinstance(value, dict) or value.get("type") != "render":
        raise ValueError("Expected a render request")
    timestamp = value.get("time_s", 0)
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("time_s must be finite and nonnegative")
    width = value.get("width", settings["width"])
    height = value.get("height", round(width * 9 / 16) if type(width) is int else 432)
    if type(width) is not int or type(height) is not int or not 128 <= width <= 1280 or not 72 <= height <= 720:
        raise ValueError("Live dimensions must be integers within 128..1280 × 72..720")
    camera = value.get("camera")
    if camera is not None:
        camera = normalize_camera_pose(camera)
    return {"op": "render", "request_id": value.get("request_id"), "time_s": timestamp,
            "width": width, "height": height, "camera": camera}


class _ProcessSession:
    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if self.cancel.is_set():
                raise InterruptedError("Newton observation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Newton worker timed out; inspect worker.stderr")
            readable, _, _ = select.select([self.process.stdout], [], [], min(remaining, .5))
            if not readable:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Newton worker exited; inspect worker.stderr")
            self.buffer += chunk
            if len(self.buffer) > self.settings["max_message_bytes"]:
                raise RuntimeError("Newton message exceeds max_message_bytes")
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("Newton worker returned a non-object response")
        return response

    def _start_worker(self, code="/work/scene.py"):
        config_path = f"/work/.observations/{self.directory.name}/config.json"
        self.box.write_file(config_path, json.dumps({"max_replay_steps": self.settings.get("max_replay_steps", 100000),
                                                     "session_id": self.directory.name, "inspection":self.inspection}))
        self.stderr = (self.directory / "worker.stderr").open("w+b")
        self.worker_args = ["python", "-u", "-m", "harness.tools.newton.workers.live_entry", code, config_path]
        self.process = self.box.open_process(self.worker_args, stderr=self.stderr, own_group=True)
        response = self._read(self.settings["startup_timeout_s"])
        if response.get("type") != "ready":
            raise RuntimeError(response.get("error", "Newton scene did not initialize"))
        self.dependency_files = response.pop("dependency_files", [])
        self.ready = response
        self.receipt.update(status="ready", ready=response, dependency_files=self.dependency_files)
        save(self.directory / "session.json", self.receipt)
        return response

    def request(self, command):
        with self.lock:
            if self.closed or not self.ready:
                raise RuntimeError("Newton observation session is unavailable")
            command = {**command, "request_id": command.get("request_id") or uuid.uuid4().hex}
            payload = json.dumps(command, allow_nan=False).encode() + b"\n"
            if len(payload) > 16000:
                raise ValueError("Observation control request exceeds 16000 bytes")
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                response = self._read(self.settings["render_timeout_s"])
                if response.get("request_id") != command["request_id"]:
                    raise RuntimeError("Newton worker returned another request response")
                if response.get("type") == "error":
                    if response.get("error_code") == "invalid_request" and not response.get("fatal", False):
                        raise FrameRequestError(response.get("error", "Invalid observation request"))
                    raise RuntimeError(response.get("error", "Native observation failed"))
                self.last_used = time.monotonic()
                return response
            except FrameRequestError:
                raise
            except Exception:
                self._close()
                raise

    def render(self, command):
        try:
            response = self.request(command)
            if response.get("type") != "frame":
                raise RuntimeError("Newton worker did not return a frame")
            jpeg = base64.b64decode(response.pop("jpeg_base64"), validate=True)
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.format != "JPEG" or image.size != (response["width"], response["height"]):
                    raise RuntimeError("Native frame dimensions or encoding differ from the receipt")
                image.load()
            return response, jpeg
        except FrameRequestError:
            raise
        except Exception:
            with self.lock:
                self._close()
            raise

    def _close(self):
        if self.closed:
            return
        self.closed = True
        if self.process:
            exit_code = self.process.poll()
            if exit_code is None:
                try:
                    self.process.stdin.write(b'{"op":"close"}\n')
                    self.process.stdin.flush()
                    exit_code = self.process.wait(timeout=3)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            if exit_code != 0 and not self.owns_box and not self.box._closing:
                killed = self.box.kill_worker_group(self.stderr, self.worker_args)
                self.receipt["worker_shutdown"] = "killed_process_group" if killed else "group_not_confirmed"
            if exit_code is None:
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        if self.owns_box and self.box:
            self.box.close()
        if self.stderr:
            self.stderr.close()
        self.receipt.update(status="closed", closed_at=now())
        save(self.directory / "session.json", self.receipt)
        if self.owns_box:
            for folder in ("work", "inputs"):
                shutil.rmtree(self.directory / folder, ignore_errors=True)

    def close(self):
        self.cancel.set()
        with self.lock:
            self._close()


class AgentSession(_ProcessSession):
    def __init__(self, ctx, box, *, code_hash, world_hash):
        self.ctx, self.box = ctx, box
        contract = ctx.state.get('output_contract', {})
        self.inspection = contract if contract.get('kind') == 'scene_edit' else None
        self.owns_box = False
        self.directory = ctx.directory / "observations" / uid("session")
        self.directory.mkdir(parents=True)
        self.settings = {**ctx.services.store.config()["live"],
                         "render_timeout_s": ctx.task.config["sandbox"].get("newton_timeout_s", 600)}
        self.lock, self.cancel = threading.Lock(), threading.Event()
        self.buffer, self.ready = b"", None
        self.closed, self.process, self.stderr = False, None, None
        self.code_hash, self.world_hash = code_hash, world_hash
        self.revision = "preview:" + self.directory.name
        self.last_used = time.monotonic()
        self.receipt = {"kind": "agent_newton_observation", "created_at": now(), "status": "starting",
                        "code_sha256": self.code_hash, "world_sha256": self.world_hash}

    def initialize(self):
        try:
            ready = self._start_worker()
            inputs = {item['path']: item['sha256'] for item in self.dependency_files}
            if inputs.get('scene.py') != self.code_hash or inputs.get('world.yaml') != self.world_hash:
                raise ValueError('Preview inputs changed while the worker was starting; retry the observation')
            return ready
        except Exception as exc:
            self.receipt.update(status="failed", error=str(exc))
            save(self.directory / "session.json", self.receipt)
            self.close()
            raise


class NativeSession(_ProcessSession):
    def __init__(self, task_id, run_id, settings, *, store, sandbox_factory):
        self.store, self.sandbox_factory = store, sandbox_factory
        self.key = (task_id, run_id)
        self.settings = settings
        self.owns_box = True
        self.lock, self.cancel = threading.Lock(), threading.Event()
        self.leases = 0
        self.last_used = time.monotonic()
        self.closed = self.initializing = False
        self.phase = "prepared"
        self.process = self.box = self.stderr = None
        self.buffer, self.ready = b"", None
        self.task_dir = store.task_path(task_id)
        self.value = store.task(task_id)
        run = next((row for row in self.value["runs"] if row["id"] == run_id), None)
        self.sandbox_startup_path = (run or {}).get("sandbox_startup_path")
        self.archive = archived_render(self.task_dir, run, for_replay=True)
        if not self.archive:
            raise ValueError("No archived native scene is available")
        self.render_dir = inside(self.task_dir, self.archive["render_dir"])
        manifest = load(self.render_dir / "render_manifest.json", {})
        self.inspection = manifest.get('inspection')
        if manifest.get("engine") != "newton":
            raise ValueError("Historical Genesis artifacts remain available as archived RGB; native replay requires a Newton artifact")
        self.archive_revision = self.archive["render_revision"]
        self.code_hash = manifest["code_sha256"]
        self.directory = self.task_dir / "runs" / run_id / "live" / uid("session")
        self.directory.mkdir(parents=True)
        self.receipt = {"kind": "newton_interactive_preview", "created_at": now(), "status": "starting",
                        "render_revision": self.archive_revision, "code_sha256": self.code_hash}

    def _stage_dependencies(self):
        bundle = self.render_dir / "solution"
        record = read_bundle(bundle)
        if record.get("schema_version") != 2 or record.get("engine") != "newton":
            raise ValueError("Newton replay requires its immutable delivered dependency closure")
        manifest = load(self.render_dir / "render_manifest.json", {})
        identity = hashlib.sha256(json.dumps(record["files"], sort_keys=True).encode()).hexdigest()
        if identity != manifest.get("code_manifest_sha256") or record["files"] != manifest.get("dependency_files"):
            raise ValueError("Replay dependency closure differs from the formal native manifest")
        self.replay_workers = None
        if record.get("workers") != worker_identity():
            self.replay_workers = _recorded_viewer_workers(self.task_dir, self.key[1], record.get("workers"), self.sandbox_startup_path)
        scene_record = next((item for item in record["files"] if item["path"] == "scene.py"), {})
        world_record = next((item for item in record["files"] if item["path"] == "world.yaml"), {})
        if scene_record.get("sha256") != manifest.get("code_sha256") or world_record.get("sha256") != manifest.get("world_sha256"):
            raise ValueError("Replay scene/world identity differs from the formal native manifest")
        contract_record = next((item for item in record["files"] if item["path"] == "output_contract.json"), {})
        if (contract_record.get("sha256") != manifest.get("output_contract_sha256")
                or record.get("output_contract_sha256") != manifest.get("output_contract_sha256")
                or record.get("reference_source_id") != manifest.get("reference_source_id")
                or record.get("inspection") != manifest.get("inspection")):
            raise ValueError("Replay output contract differs from the formal native manifest")
        if sum(item["bytes"] for item in record["files"]) > MAX_BUNDLE_BYTES:
            raise ValueError("Replay code closure exceeds its configured size")
        dependencies = []
        for item in record["files"]:
            name = relative_path(item["path"])
            raw = (bundle / name).read_bytes() if name == "scene.py" else base64.b64decode(record["encoded_files"][name], validate=True)
            if len(raw) > MAX_FILE_BYTES or len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Replay dependency does not match immutable identity: " + name)
            target = self.directory / "work" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            dependencies.append({"path": "work/" + name, "sha256": item["sha256"]})
        if contract_record and not manifest.get("inspection"):
            contract = load(self.directory / "work/output_contract.json", {})
            if contract.get("reference_source_id") != manifest.get("reference_source_id"):
                raise ValueError("Replay reference source differs from the formal native manifest")
        self.dependency_scope = "immutable delivered Newton scene/world/solver/assets/output contract"
        return dependencies

    def initialize(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Newton viewer session is closed")
            if self.ready:
                return self.ready
            self.initializing = True
            try:
                self.phase = "loading immutable Newton code and solver dependencies"
                dependencies = self._stage_dependencies()
                cfg = {**self.value["config"], "sandbox": {**self.value["config"]["sandbox"],
                       "image": self.store.config()["sandbox"]["image"], "startup_timeout_s": self.settings["startup_timeout_s"]}}
                gpu, device = render_device(self.settings["device"])
                cfg["sandbox"]["gpus"] = gpu
                cache = self.task_dir / "runs" / self.key[1] / "viewer_cache" / self.archive_revision
                cache.mkdir(parents=True, exist_ok=True)
                options = {"replay_workers": self.replay_workers} if self.replay_workers else {}
                self.box = self.sandbox_factory(self.directory, cfg, cache_dir=cache, purpose="viewer", **options)
                self.box.start()
                self.receipt.update(dependencies=dependencies, device=device)
                self.phase = "building native Newton/Warp state"
                return self._start_worker()
            except Exception as exc:
                self.receipt.update(status="failed", error=str(exc))
                self._close()
                raise
            finally:
                self.initializing = False
