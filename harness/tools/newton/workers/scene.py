"""Execute ordinary Newton examples inside the scene tool's worker process.

Scene code owns geometry, solvers and their composition. This adapter only
loads the entry, calls its step, observes native state and manages presentation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import random
import sys

import numpy as np

from .camera_math import calibration, transform_matrix


class RuntimeFailure(RuntimeError):
    def __init__(self, code, message, *, status="error", **details):
        self.code, self.status = code, status
        self.diagnostic = {"code": code, "message": message, "status": status, **details}
        super().__init__(message)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"type": type(value).__module__ + "." + type(value).__name__, "status": "not_serialized"}


class SceneSession:
    def __init__(self, example_class, world, context, identity):
        self.example_class, self.world = example_class, world
        self.output_contract = context.get("output_contract", {})
        self.device, self.identity = context["device"], identity
        self.tolerance = float(self.output_contract.get("timestamp_tolerance_s") or 1e-6)
        self.closed = False
        self._renderer = None
        self._meshes = None
        self._create(context.get("seed"))

    def _create(self, seed):
        import newton.examples
        import warp as wp
        from .render import NativeViewer
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed
        self._viewer = NativeViewer(self.device)
        with wp.ScopedDevice(self.device):
            parser = getattr(self.example_class, "create_parser", newton.examples.create_parser)()
            args = parser.parse_args([])
            args.device, args.viewer, args.headless = str(wp.get_device()), "null", True
            if self.output_contract.get("timestamps_s"):
                args.num_frames = len(self.output_contract["timestamps_s"])
            elif self.output_contract.get("kind") == "image":
                args.num_frames = 1
            self.app = self.example_class(self._viewer, args)
        if not callable(getattr(self.app, "step", None)) or not hasattr(self.app, "model"):
            raise ValueError("Example(viewer, args) must expose native model, state_0/state, sim_time and step()")
        self.frame_dt = float(getattr(self.app, "frame_dt", getattr(self.app, "sim_dt", 0.0)))
        self.physics_dt_s = float(getattr(self.app, "sim_dt", self.frame_dt))
        if not math.isfinite(self.frame_dt) or self.frame_dt <= 0 or not math.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0:
            raise ValueError("Example frame_dt/sim_dt must be positive finite seconds")
        self._start_time = self.time_s
        self.tick = 0
        if self._viewer.model is not self.model:
            self._viewer.set_model(self.model)
        self.bindings = self._bindings()
        self.output_times = getattr(self.app, "output_times", None)
        self.solution_assets = list(getattr(self.app, "solution_assets", []))
        self.code_dependencies = list(getattr(self.app, "code_dependencies", []))
        self._meshes = None
        self._finite_state()

    @property
    def model(self):
        return self.app.model

    @property
    def native_state(self):
        state = getattr(self.app, "state_0", None)
        if state is None:
            state = getattr(self.app, "state", None)
        state = state() if callable(state) else state
        if state is None:
            raise ValueError("Example must expose its current native state as state_0 or state")
        return state

    @property
    def time_s(self):
        if not hasattr(self.app, "sim_time"):
            raise ValueError("Example must expose actual sim_time; update it once per step()")
        actual = float(self.app.sim_time)
        if not math.isfinite(actual):
            raise ValueError("Example sim_time must be finite")
        return actual

    @property
    def control_decimation(self):
        ratio = self.frame_dt / self.physics_dt_s
        return round(ratio) if abs(ratio - round(ratio)) < 1e-6 else None

    def _bindings(self):
        semantics = {**self.world.get("objects", {}), **self.world.get("phenomena", {}), **self.world.get("cameras", {})}
        result = {}
        counts = {"body": self.model.body_count, "collider": self.model.shape_count, "joint": self.model.joint_count}
        def add(identity, kind, index):
            index = int(index)
            if not 0 <= index < counts[kind]:
                raise ValueError(f"Invalid native {kind} binding for {identity}: {index}")
            entry = {"kind": kind, "index": index, "id": f"{kind}_{index}"}
            if entry not in result.setdefault(identity, []):
                result[identity].append(entry)
        for kind, label_name in (("body", "body_label"), ("collider", "shape_label"), ("joint", "joint_label")):
            for index, label in enumerate(getattr(self.model, label_name, [])):
                if label in semantics:
                    add(label, kind, index)
        for identity, binding in getattr(self.app, "bindings", {}).items():
            if semantics and identity not in semantics:
                raise ValueError(f"Explicit binding targets an unknown world ID: {identity}")
            if isinstance(binding, (int, np.integer)):
                add(identity, "body", binding)
            elif isinstance(binding, dict):
                if binding.get("body") is not None:
                    add(identity, "body", binding["body"])
                for key, kind in (("shapes", "collider"), ("joints", "joint")):
                    for index in binding.get(key, []):
                        add(identity, kind, index)
            else:
                raise ValueError("bindings maps semantic IDs to body indices or {body, shapes, joints}")
        # A body's unlabelled decorative sub-shapes inherit its semantic parent.
        if self.model.shape_count:
            body_aliases = {entry["index"]: identity for identity, entries in result.items() for entry in entries if entry["kind"] == "body"}
            for index, body in enumerate(self.model.shape_body.numpy()):
                if body in body_aliases:
                    add(body_aliases[body], "collider", index)
        return result

    def _finite_state(self):
        for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
            value = getattr(self.native_state, name, None)
            if value is not None and not np.isfinite(value.numpy()).all():
                raise RuntimeError(f"Nonfinite native {name}")

    def step(self, action=None):
        if self.closed:
            raise RuntimeError("Scene session is closed")
        before = self.time_s
        if action is None:
            self.app.step()
        else:
            self.app.step(action)
        actual = self.time_s - before
        if actual <= 0 or abs(actual - self.frame_dt) > max(1e-8, self.frame_dt * 1e-6):
            raise ValueError(f"Example step() must advance actual sim_time by frame_dt={self.frame_dt}; advanced {actual}")
        self.tick += 1
        self._meshes = None
        self._finite_state()
        return {"time_s": self.time_s, "tick": self.tick}

    def advance_to(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Requested time must be finite and nonnegative")
        target = self._start_time + round((time_s - self._start_time) / self.frame_dt) * self.frame_dt
        if abs(target - time_s) > self.tolerance + 1e-10:
            raise RuntimeFailure("unreachable_time", "Requested time is not reachable by whole Example.step() calls. Align reconstruction frame steps with its samples, or expose step_physics(dt) for experiment and edit clocks.", requested_time_s=time_s,
                                 frame_dt_s=self.frame_dt, nearest_time_s=target, timestamp_tolerance_s=self.tolerance)
        if target < self.time_s - self.tolerance:
            raise RuntimeFailure("backward_time", "Rebuild/reset the Example before seeking backward")
        steps = round((target - self.time_s) / self.frame_dt)
        if steps > 1_000_000:
            raise ValueError("Requested advance exceeds one million scene step calls")
        for _ in range(max(0, steps)):
            # Avoid constructing/serializing a query for every physics tick.
            before = self.time_s
            self.app.step()
            if abs((self.time_s - before) - self.frame_dt) > max(1e-8, self.frame_dt * 1e-6):
                raise ValueError("Example step() sim_time increment disagrees with frame_dt")
            self.tick += 1
        if steps:
            self._meshes = None
        self._finite_state()
        return {"time_s": self.time_s, "tick": self.tick,
                "requested_time_s": time_s, "timestamp_error_s": self.time_s - time_s}

    def advance_physics_to(self, time_s, max_dt):
        """Use scene-owned variable substeps when declared; legacy step() owns its clock."""
        substep = getattr(self.app, "step_physics", None)
        if not callable(substep):
            return self.advance_to(time_s)
        duration = time_s - self.time_s
        if not math.isfinite(duration) or duration < -1e-9 or not math.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("Invalid physics advance or substep limit")
        if duration <= 1e-10:
            return
        count = max(1, math.ceil(duration / min(self.physics_dt_s, max_dt) - 1e-10))
        if count > 1_000_000:
            raise ValueError("Requested advance exceeds one million physics substeps")
        dt = duration / count
        for _ in range(count):
            before = self.time_s
            substep(dt)
            if not math.isclose(self.time_s - before, dt, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("step_physics(dt) must advance sim_time exactly once by dt")
            self.tick += 1
        self._meshes = None
        self._finite_state()

    def visual_meshes(self):
        if self._meshes is None:
            before = self.time_s
            self._viewer.log_state(self.native_state)
            if getattr(self.app, "viewer", None) is self._viewer and callable(getattr(self.app, "render", None)):
                self.app.render()
            if self.time_s != before:
                raise RuntimeError("Rendering must not advance Example sim_time")
            semantics = {entry["index"]: identity for identity, entries in self.bindings.items() for entry in entries if entry["kind"] == "collider"}
            meshes = self._viewer.visual_meshes(semantics)
            extra = getattr(self.app, "visual_meshes", None)
            if callable(extra):
                meshes.extend(extra())
            self._meshes = meshes
            self._geometry_snapshot = object()
        return self._meshes

    def render(self, camera_id=None, camera_override=None, width=None, height=None, source_id=None, *, channels=None,
               independent_camera=False):
        from .render import WarpRenderer
        if self._renderer is None:
            self._renderer = WarpRenderer(self.device)
        # Only attached cameras need a body-pose readback. World and free
        # cameras are calibrated entirely from their declared optical pose.
        optical_world = {} if independent_camera else self.world
        parents = {item["attachment"].get("object") for item in optical_world.get("cameras", {}).values()
                   if item.get("attachment")}
        transforms = {}
        if parents and self.model.body_count:
            poses = self.native_state.body_q.numpy()
            for parent in parents:
                for entry in self.bindings.get(parent, []):
                    if entry["kind"] == "body":
                        pose = poses[entry["index"]]
                        transforms[parent] = transform_matrix(pose[:3], pose[3:])
        camera = calibration(optical_world, self.time_s, camera_id, camera_override, width, height, transforms, source_id=source_id)
        meshes = self.visual_meshes()
        rendered = self._renderer.render(meshes, camera, channels=channels, snapshot=self._geometry_snapshot)
        return {**rendered, "camera": camera, "time_s": self.time_s, "tick": self.tick,
                "identity": self.identity, "native": True}

    def query(self):
        self._finite_state()
        entities, shapes = [], []
        aliases = {(entry["kind"], entry["index"]): [] for entries in self.bindings.values() for entry in entries}
        for identity, entries in self.bindings.items():
            for entry in entries:
                aliases[(entry["kind"], entry["index"])].append(identity)
        if self.model.body_count:
            q, qd, mass = self.native_state.body_q.numpy(), self.native_state.body_qd.numpy(), self.model.body_mass.numpy()
            for index in range(self.model.body_count):
                entities.append({"id": f"body_{index}", "world_ids": aliases.get(("body", index), []), "type": "NewtonRigidBody",
                    "position_m": {"status": "available", "value": q[index, :3].tolist()},
                    "quaternion_xyzw": {"status": "available", "value": q[index, 3:].tolist()},
                    "velocity_m_s": {"status": "available", "value": qd[index, :3].tolist()},
                    "angular_velocity_rad_s": {"status": "available", "value": qd[index, 3:].tolist()},
                    "mass_kg": {"status": "available", "value": float(mass[index])}})
        if self.model.shape_count:
            bodies, friction = self.model.shape_body.numpy(), self.model.shape_material_mu.numpy()
            for index in range(self.model.shape_count):
                shapes.append({"id": f"collider_{index}", "world_ids": aliases.get(("collider", index), []),
                               "body_id": int(bodies[index]), "friction": float(friction[index])})
        solver = getattr(self.app, "solver", None)
        config = {"physics_dt_s": self.physics_dt_s, "frame_dt_s": self.frame_dt, "control_decimation": self.control_decimation,
                  "solver": None if solver is None else type(solver).__module__ + "." + type(solver).__name__,
                  "solver_config": _jsonable(getattr(self.app, "solver_config", None)), "scene_class": type(self.app).__name__}
        metadata = getattr(self.app, "query_metadata", None)
        return {"type": "state", "engine": "newton", "time_s": self.time_s, "tick": self.tick,
                "dt_s": self.frame_dt, "physics_dt_s": self.physics_dt_s, "entities": entities, "shapes": shapes,
                "particle_count": self.model.particle_count, "bindings": self.bindings, "configuration": config,
                "capabilities": {"scene_owned_physics": True,
                                 "reset_via_rebuild": True, "world_count": self.model.world_count},
                "identity": self.identity, "metadata": _jsonable(metadata()) if callable(metadata) else {}}

    def reset(self, seed=None, initial_state=None):
        close = getattr(self.app, "close", None)
        if callable(close):
            close()
        self._create(self.seed if seed is None else seed)
        if initial_state:
            reset = getattr(self.app, "reset", None)
            if not callable(reset):
                raise ValueError("Explicit initial_state requires an Example.reset(initial_state) hook")
            reset(initial_state)
        return {"time_s": self.time_s, "tick": self.tick}

    def close(self):
        if not self.closed:
            close = getattr(self.app, "close", None)
            if callable(close):
                close()
            if self._renderer is not None:
                self._renderer.close()
            self.closed = True


def load_scene(code_path, world, *, device="cuda:0", seed=None, context=None):
    code_path = Path(code_path).resolve()
    root = code_path.parent
    import warp as wp
    wp.init()
    context = {**(context or {}), "device": device, "seed": seed}
    identity = {"code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
                "world_sha256": hashlib.sha256((root / "world.yaml").read_bytes()).hexdigest() if (root / "world.yaml").is_file() else None,
                "newton_version": importlib.metadata.version("newton"), "warp_version": importlib.metadata.version("warp-lang")}
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("reconstructed_scene", code_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        # Compile the entry directly so a same-second edit never reuses bytecode.
        with wp.ScopedDevice(device):
            exec(compile(code_path.read_bytes(), str(code_path), "exec"), module.__dict__)
        if not isinstance(getattr(module, "Example", None), type):
            raise ValueError("Expose the native Newton Example(viewer, args) class in the scene module")
        session = SceneSession(module.Example, world, context, identity)
        session.code_dependencies = sorted(set(session.code_dependencies) | {
            str(Path(loaded.__file__).resolve().relative_to(root)) for loaded in list(sys.modules.values())
            if getattr(loaded, "__file__", None) and Path(loaded.__file__).suffix == ".py" and Path(loaded.__file__).resolve().is_relative_to(root)})
        return session
    finally:
        sys.path[:] = old_path
