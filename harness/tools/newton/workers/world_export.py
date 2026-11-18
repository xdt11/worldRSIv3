"""Native Newton mesh/video export with explicit fixed/moving-camera profiles."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


def encode_video(output, manifest, destination):
    frames = manifest["frames"]
    times = [frame["timestamp_s"] for frame in frames]
    dt = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
    if dt <= 0 or len(times) > 1 and not np.allclose(np.diff(times), dt, atol=1e-6, rtol=0):
        raise ValueError("Native export requires a uniform actual simulation timeline")
    with Image.open(output / frames[0]["path"]) as image:
        width, height = image.size
    command = ["ffmpeg", "-v", "error", "-y", "-xerror", "-f", "image2pipe", "-c:v", "png",
               "-threads", "2", "-framerate", str(1 / dt), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p", "-threads", "2",
               "-frames:v", str(len(frames))]
    # Feed the archived PNGs directly. FFmpeg decodes them in its own threads,
    # without moving an additional uncompressed RGB stream through Python.
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging, tempfile.TemporaryFile() as errors:
        temporary = Path(staging) / destination.name
        with subprocess.Popen([*command, str(temporary)], stdin=subprocess.PIPE, stderr=errors) as process:
            try:
                for frame in frames:
                    png = (output / frame["path"]).read_bytes()
                    with Image.open(io.BytesIO(png)) as image:
                        if image.format != "PNG" or image.size != (width, height):
                            raise ValueError("Native video requires equal-resolution PNG frames")
                        image.verify()
                    process.stdin.write(png)
                process.stdin.close()
                if process.wait(timeout=120):
                    errors.seek(0)
                    raise ValueError("Native video encoding failed: " + errors.read().decode(errors="replace"))
            finally:
                if process.poll() is None:
                    process.kill()
        temporary.replace(destination)


class WorldExport:
    def __init__(self, output):
        self.output = Path(output)
        self.cameras = []
        self.meshes = []
        self.error = None
        self.render_profile = None
        self.unsupported_material_trajectories = []

    def capture(self, session, rendered):
        self.cameras.append(rendered["camera"])
        if self.render_profile:
            return
        if len(self.cameras) > 1 and any(not np.allclose(self.cameras[-1][key], self.cameras[0][key], atol=1e-6, rtol=0)
                                         for key in ("intrinsics", "extrinsic")):
            self.render_profile = "newton-camera-trajectory-v1"
            self.meshes.clear()
            return
        try:
            meshes = session.visual_meshes()
            if not isinstance(meshes, list):
                raise ValueError("visual_meshes must return explicit native mesh records")
            unsupported = [str(mesh["id"]) for mesh in meshes
                           if not mesh.get("material_identity", mesh.get("kind", "rigid") == "rigid")]
            if unsupported:
                self.unsupported_material_trajectories = unsupported
                self.render_profile = "newton-native-render-v1"
                self.meshes.clear()
                return
            captured = []
            previous = self.meshes[-1] if self.meshes else []
            for index, mesh in enumerate(meshes):
                vertices = np.asarray(mesh["vertices"], dtype=np.float32)
                faces = np.asarray(mesh["faces"], dtype=np.int32)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError("Native mesh needs (N,3) vertices and (M,3) faces")
                if not np.isfinite(vertices).all() or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices))):
                    raise ValueError("Native mesh contains invalid coordinates or topology")
                identity = str(mesh["id"])
                prior = previous[index] if index < len(previous) and previous[index]["id"] == identity else None
                # Keep owned snapshots of mutable scene arrays. Static scenery
                # and constant triangle indices can share them across frames.
                vertices = prior["vertices"] if prior and np.array_equal(vertices, prior["vertices"]) else vertices.copy()
                faces = prior["faces"] if prior and np.array_equal(faces, prior["faces"]) else faces.copy()
                captured.append({"id": identity, "semantic_id": mesh.get("semantic_id"), "vertices": vertices, "faces": faces,
                                 "kind": mesh.get("kind", "rigid"), "material_identity": mesh.get("material_identity", mesh.get("kind", "rigid") == "rigid")})
            if len({mesh["id"] for mesh in captured}) != len(captured):
                raise ValueError("Native visual mesh identities must be unique")
            self.meshes.append(captured)
        except Exception as exc:
            self.error = str(exc)
            self.meshes.append([])

    def finish(self, manifest):
        world = self.output / "world"
        world.mkdir()
        (world / "meshes").mkdir()
        (world / "dynamics").mkdir()
        times = [frame["timestamp_s"] for frame in manifest["frames"]]
        receipt = {"status": "unavailable", "protocol": "4dcodebench-newton-v1",
                   "renderer": "Python Warp Render", "official_cycles_render": False,
                   "timestamps_s": times, "frame_count": len(times)}
        try:
            encode_video(self.output, manifest, world / "render.mp4")
            first = self.cameras[0]
            fixed = all(all(np.allclose(camera[key], first[key], atol=1e-6, rtol=0)
                            for key in ("intrinsics", "extrinsic")) for camera in self.cameras)
            camera_profile = "fixed" if fixed else "trajectory"
            receipt["camera_profile"] = camera_profile
            trajectory = [{"timestamp_s": t, **camera} for t, camera in zip(times, self.cameras)]
            (world / "camera_trajectory.json").write_text(json.dumps(trajectory, indent=2))
            if fixed and not self.render_profile:
                (world / "camera.json").write_text(json.dumps({key: first[key] for key in ("intrinsics", "extrinsic")}, indent=2))
            elif not fixed:
                receipt.update(protocol="newton-camera-trajectory-v1",
                               metric_scope="Official render-only semantic/geophys metrics; fixed-camera geometry metrics are not applicable")
            if self.error:
                raise ValueError(self.error)
            if self.render_profile:
                receipt.update(status="ok", protocol=self.render_profile,
                               metric_scope="Render-only; outside the fixed-camera material-correspondence geometry contract",
                               scope="Actual native video and optical cameras; geometry sequences omitted, scene code and world remain rebuildable")
                if self.unsupported_material_trajectories:
                    receipt["unsupported_material_trajectories"] = self.unsupported_material_trajectories
                (self.output / "world_export.json").write_text(json.dumps(receipt, indent=2))
                return receipt
            if not self.meshes or not self.meshes[0]:
                raise ValueError("No native visual geometry was exported")
            initial = [mesh["id"] for mesh in self.meshes[0]]
            if any([mesh["id"] for mesh in meshes] != initial for meshes in self.meshes):
                raise ValueError("Visual identities changed during the formal timeline")
            bindings = {}
            for index, identity in enumerate(initial):
                oid = index + 1
                if oid > 65535:
                    raise ValueError("Official mesh IDs exceed uint16 capacity")
                records = [meshes[index] for meshes in self.meshes]
                frames = {}
                for frame_index, mesh in enumerate(records):
                    frames[f"vertices_{frame_index:04d}"] = mesh["vertices"]
                    frames[f"faces_{frame_index:04d}"] = mesh["faces"]
                np.savez_compressed(world / "meshes" / f"{oid}.npz", **frames)
                bindings[str(oid)] = {"runtime_mesh_id": identity, "semantic_id": records[0]["semantic_id"], "kind": records[0]["kind"]}
                changed = any(not np.array_equal(records[0]["vertices"], mesh["vertices"]) for mesh in records[1:])
                same_topology = all(np.array_equal(records[0]["faces"], mesh["faces"])
                                    and records[0]["vertices"].shape == mesh["vertices"].shape for mesh in records[1:])
                if changed and (not same_topology or not all(mesh["material_identity"] for mesh in records)):
                    receipt.setdefault("unsupported_material_trajectories", []).append(identity)
                elif changed:
                    prefix = "rigid" if records[0]["kind"] == "rigid" else "material"
                    np.savez_compressed(world / "dynamics" / f"{prefix}_{oid}.npz",
                                        pos=np.stack([mesh["vertices"] for mesh in records]),
                                        ids=np.array([oid], np.uint16), faces=records[0]["faces"])
            if receipt.get("unsupported_material_trajectories"):
                receipt.update(protocol="newton-native-render-v1", metric_scope="Render-only; unsupported native material-point trajectories")
                # A mesh animation is not proof of persistent fluid material IDs.
                (world / "camera.json").unlink(missing_ok=True)
            receipt.update(status="ok", object_bindings=bindings,
                           scope="Native visual geometry, actual optical cameras and simulation ticks; collision geometry remains a separate runtime query")
        except Exception as exc:
            receipt["reason"] = str(exc)
            if (world / "render.mp4").is_file():
                receipt.update(status="render_only", metric_scope="Native video only; geometry export failed explicitly")
        (self.output / "world_export.json").write_text(json.dumps(receipt, indent=2))
        return receipt
