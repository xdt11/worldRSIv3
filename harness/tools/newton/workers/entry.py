"""Container-only formal scene execution through the fixed Newton runtime."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import yaml

from harness.tools.newton.workers.scene import load_scene
from harness.tools.newton.workers.camera_math import normalize_camera_pose
from harness.tools.newton.workers.protocol import edit_inspection, validate_output
from harness.tools.newton.workers.solution import package_solution, snapshot_inputs, freeze_dependencies
from harness.tools.newton.workers.world_export import WorldExport


def _progress(phase, completed=0, expected=0):
    if os.environ.get('WORLD_RSI_RENDER_PROGRESS'):
        print(f'WORLD_RSI_RENDER {phase} {completed} {expected}', file=sys.stderr, flush=True)


def _archive_rgb_frame(path, size, pixels, metadata, completed):
    Image.frombytes("RGB", size, pixels).save(path)
    _progress('render', completed)
    return {"path": path.name, **metadata, "pixel_sha256": hashlib.sha256(pixels).hexdigest()}


def _physics_digest(session):
    """Hash native parameters and state at a sampled world time."""
    physics = hashlib.sha256()
    for owner, names in (
        (session.model, ('gravity','body_mass','body_inertia','shape_type','shape_transform','shape_scale',
                         'shape_flags','shape_material_mu','shape_material_ke','shape_material_kd',
                         'shape_material_restitution','particle_mass','joint_limit_lower','joint_limit_upper')),
        (session.native_state, ('body_q','body_qd','particle_q','particle_qd'))):
        for name in names:
            array = getattr(owner, name, None)
            if array is not None:
                data = np.asarray(array.numpy() if hasattr(array, 'numpy') else array)
                physics.update(str((name, data.dtype, data.shape)).encode())
                physics.update(data.tobytes())
    return physics.hexdigest()


def _render_frames(session, contract, timestamps, output, exporter):
    frames = []
    pending = None
    # Only one PNG is in flight. Compression can run while Newton advances and
    # Warp draws the next frame, even when the renderer reuses its pixel buffer.
    with ThreadPoolExecutor(max_workers=1) as writer:
        for index, requested in enumerate(timestamps):
            camera = {}
            if contract.get('kind') == 'scene_edit':
                session.advance_physics_to(float(requested), session.physics_dt_s)
                pose = contract['views'][index].get('camera')
                if pose:
                    pose = {**pose, **normalize_camera_pose(pose)}
                camera = {'independent_camera': True, 'camera_override': pose}
            else:
                session.advance_to(float(requested))
            rendered = session.render(camera_id=contract.get("camera_id"), source_id=contract.get("reference_source_id"),
                                      width=contract.get("width"), height=contract.get("height"), channels=("rgb",), **camera)
            rgb = np.asarray(rendered["rgb"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("Native Warp renderer must return uint8 RGB pixels")
            actual = float(rendered["time_s"])
            if not np.isfinite(actual):
                raise ValueError("Native simulation returned nonfinite time")
            if pending is not None:
                frames.append(pending.result())
            metadata = {"timestamp_s": actual, "requested_time_s": float(requested),
                        "tick": int(rendered["tick"]), "camera": rendered["camera"]}
            if contract.get('kind') == 'scene_edit':
                metadata['physics_sha256'] = _physics_digest(session)
            pending = writer.submit(_archive_rgb_frame, output / f"frame_{index:04d}.png",
                                    (rgb.shape[1], rgb.shape[0]), rgb.tobytes(), metadata, index + 1)
            if exporter is not None:
                exporter.capture(session, rendered)
        if pending is not None:
            frames.append(pending.result())
    return frames


def execute(code, output, inspection=None):
    code, output = Path(code).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = code.parent
    preimages = snapshot_inputs(code)
    world_path = root / "world.yaml"
    world_bytes = world_path.read_bytes()
    world = yaml.safe_load(world_bytes)
    if world.get("schema_version") != 3:
        raise ValueError("Newton scene execution requires world schema 3")
    contract_path = root / "output_contract.json"
    contract_bytes = contract_path.read_bytes() if contract_path.is_file() else None
    contract = json.loads(contract_bytes) if contract_bytes is not None else {"schema_version": 1, "kind": "unconstrained"}
    if inspection is not None and inspection != edit_inspection(inspection['views']):
        raise ValueError('Invalid scene inspection plan')
    expected = len(contract.get('timestamps_s') or []) if contract['kind'] == 'video' else int(contract['kind'] == 'image')
    if inspection:
        expected = len(inspection['views'])
    _progress('initialize', expected=expected)
    seed = int(os.environ["WORLD_RSI_SEED"]) if "WORLD_RSI_SEED" in os.environ else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    sys.path.insert(0, str(root))
    session = load_scene(code, world, device=os.environ.get("WORLD_RSI_DEVICE", "cuda:0"), seed=seed,
                         context={"output_contract": contract})
    exporter = None if inspection else WorldExport(output)
    frozen = None
    def declared_dependencies():
        return list(dict.fromkeys([*(getattr(session, "solution_assets", None) or []),
                                   *(getattr(session, "code_dependencies", None) or [])]))
    try:
        frozen = freeze_dependencies(code, declared_dependencies(), preimages)
        if inspection:
            timestamps = [view['time_s'] for view in inspection['views']]
        elif contract["kind"] == "video":
            timestamps = contract.get("timestamps_s")
        elif contract["kind"] == "image":
            timestamps = [0.0]
        else:
            timestamps = getattr(session, "output_times", None) or [0.0]
        if not isinstance(timestamps, (list, tuple)) or not timestamps:
            raise ValueError("Formal timeline must contain at least one timestamp")
        _progress('render', expected=len(timestamps))
        frames = _render_frames(session, inspection or contract, timestamps, output, exporter)
        _progress('validate', len(frames), len(timestamps))
        freeze_dependencies(code, declared_dependencies(), preimages, frozen)
        manifest = {"engine": "newton", "engine_version": importlib.metadata.version("newton"),
                    "warp_version": importlib.metadata.version("warp-lang"), "renderer": "Python Warp Render",
                    "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                    "world_sha256": hashlib.sha256(world_bytes).hexdigest(),
                    "output_contract_sha256": hashlib.sha256(contract_bytes).hexdigest() if contract_bytes is not None else None,
                    "reference_source_id": None if inspection else contract.get("reference_source_id"),
                    "effective_seed": seed, "frames": frames,
                    "native_renders": [{key: frame[key] for key in ("pixel_sha256", "timestamp_s", "tick", "camera")} for frame in frames],
                    "provenance_scope": "Fixed runtime directly renders current native Newton/Warp state; physical/semantic agreement requires separate evidence"}
        runtime_state = session.query()
        if inspection:
            runtime_state['physics_sha256'] = frames[-1]['physics_sha256']
        (output / "runtime_state.json").write_text(json.dumps(runtime_state, indent=2, allow_nan=False))
        manifest.update(runtime_state_path="runtime_state.json", runtime_bindings=runtime_state.get("bindings", {}),
                        runtime_capabilities=runtime_state.get("capabilities", {}))
        if runtime_state.get("configuration"):
            manifest["runtime_configuration"] = runtime_state["configuration"]
        if inspection:
            manifest['inspection'] = inspection
        manifest["output_structure"] = validate_output(manifest, inspection or contract)
        (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "world.yaml").write_bytes(world_path.read_bytes())
        _progress('export', len(frames), len(timestamps))
        export = exporter.finish(manifest) if exporter else {
            'status':'ok', 'protocol':'newton-scene-edit-v1', 'frame_count':len(frames),
            'timestamps_s':timestamps, 'scope':'Executable scene delivery and selected inspection views'}
        _progress('package', len(frames), len(timestamps))
        solution = package_solution(output, code, manifest, declared_dependencies(), expected_files=frozen)
        manifest["code_manifest_sha256"] = hashlib.sha256(json.dumps(solution["files"], sort_keys=True).encode()).hexdigest()
        manifest["dependency_files"] = solution["files"]
        (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
        export["solution"] = solution
        (output / "world_export.json").write_text(json.dumps(export, indent=2))
        return {"manifest_path": str(output / "render_manifest.json"), "frames": len(frames), "engine": "newton", "export": export}
    finally:
        session.close()
        if frozen is not None:
            freeze_dependencies(code, declared_dependencies(), preimages, frozen)


if __name__ == "__main__":
    print(json.dumps(execute(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv)>3 else None), allow_nan=False))
