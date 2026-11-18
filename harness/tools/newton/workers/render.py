"""Headless Warp rendering of geometry logged by Newton's native viewer API."""
from __future__ import annotations

import numpy as np
import warp as wp
import newton
from newton.viewer import ViewerNull


# Channel bits are uniform across a launch. An omitted plane has no device
# allocation, output stores or host transfer.
_CHANNELS = {"rgb": (1, wp.vec3ub), "depth": (2, wp.float32),
             "normal": (4, wp.vec3), "face_id": (8, wp.int32)}


@wp.func
def _byte(value: float):
    return wp.uint8(wp.clamp(value * 255.0, 0.0, 255.0))


@wp.kernel
def _trace(mesh_id: wp.uint64, has_mesh: int, transform: wp.mat44, inverse_k: wp.mat33,
           colors: wp.array(dtype=wp.vec3), face_ends: wp.array(dtype=int), channels: int,
           rgb: wp.array2d(dtype=wp.vec3ub), depth: wp.array2d(dtype=float),
           normal: wp.array2d(dtype=wp.vec3), face_id: wp.array2d(dtype=int)):
    y, x = wp.tid()
    local = wp.normalize(inverse_k * wp.vec3(float(x) + 0.5, float(y) + 0.5, 1.0))
    origin = wp.vec3(transform[0, 3], transform[1, 3], transform[2, 3])
    direction = wp.normalize(wp.transform_vector(transform, local))
    color = wp.vec3(0.075, 0.095, 0.13)
    distance = float(0.0)
    face = int(-1)
    surface_normal = wp.vec3(0.0)
    if has_mesh != 0:
        hit = wp.mesh_query_ray(mesh_id, origin, direction, 1000000.0)
        if hit.result:
            face = hit.face
            if (channels & 1) != 0:
                light = wp.normalize(wp.vec3(-0.4, -0.6, 1.0))
                material = hit.face
                if face_ends:
                    material = wp.lower_bound(face_ends, hit.face + 1)
                color = colors[material] * (0.25 + 0.75 * wp.abs(wp.dot(hit.normal, light)))
            if (channels & 2) != 0:
                distance = hit.t * local[2]
            if (channels & 4) != 0:
                # Optical +Z forward/+Y down to OpenGL +Y up/+Z backward.
                n = hit.normal
                surface_normal = wp.vec3(
                    transform[0, 0]*n[0] + transform[1, 0]*n[1] + transform[2, 0]*n[2],
                    -(transform[0, 1]*n[0] + transform[1, 1]*n[1] + transform[2, 1]*n[2]),
                    -(transform[0, 2]*n[0] + transform[1, 2]*n[1] + transform[2, 2]*n[2]))
    if (channels & 1) != 0:
        rgb[y, x] = wp.vec3ub(_byte(color[0]), _byte(color[1]), _byte(color[2]))
    if (channels & 2) != 0:
        depth[y, x] = distance
    if (channels & 4) != 0:
        normal[y, x] = surface_normal
    if (channels & 8) != 0:
        face_id[y, x] = face


@wp.struct
class _InstanceArrays:
    poses: wp.array(dtype=wp.transform)
    scales: wp.array(dtype=wp.vec3)
    colors: wp.array(dtype=wp.vec3)


@wp.kernel
def _gather_instances(batches: wp.array(dtype=_InstanceArrays), ends: wp.array(dtype=int),
                      poses: wp.array(dtype=wp.transform), scales: wp.array(dtype=wp.vec3),
                      colors: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    batch = wp.lower_bound(ends, i + 1)
    start = int(0)
    if batch > 0:
        start = ends[batch - 1]
    arrays = batches[batch]
    poses[i] = arrays.poses[i - start]
    scales[i] = arrays.scales[i - start]
    colors[i] = arrays.colors[i - start]


def _array(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _transformed(points, pose, scale):
    matrix = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=np.float32).reshape(3, 3)
    return (_array(points) * scale) @ matrix.T + pose[:3]


class NativeViewer(ViewerNull):
    """Consume Newton log_mesh/log_instances, preserving its geometry support.

    This implements the viewer interface consumed by native Newton
    Example.render methods, including MPM surface extraction and cloth meshes.
    """

    def __init__(self, device):
        self.meshes = {}
        self.instances = {}
        self.points = {}
        self._pending_instances = None
        self._packed_instances = None
        with wp.ScopedDevice(device):
            super().__init__(num_frames=1)
        self.show_particles = True

    def set_model(self, model):
        self.meshes.clear()
        self.instances.clear()
        self.points.clear()
        self._packed_instances = None
        super().set_model(model)
        # Viewer layout offsets are presentation conveniences, not scene poses.
        if self.world_offsets is not None:
            self.world_offsets.zero_()

    def log_mesh(self, name, points, indices, normals=None, uvs=None, texture=None,
                 hidden=False, backface_culling=True, color=None, roughness=None,
                 metallic=None, dynamic=False, opacity=None):
        self.meshes[name] = {"vertices": _array(points).copy(),
                             "faces": _array(indices).reshape(-1, 3).copy(),
                             "color": color, "hidden": hidden, "dynamic": dynamic}

    def log_state(self, state):
        # Newton writes the current poses and material values to many small
        # device arrays. Gather those exact outputs after its viewer traversal.
        if len(self._shape_instances) < 2:
            return super().log_state(state)
        self._pending_instances = {}
        try:
            super().log_state(state)
            if self._pending_instances:
                self._read_instances(self._pending_instances)
        finally:
            self._pending_instances = None

    def _read_instances(self, pending):
        entries = list(pending.values())
        signature = [(poses.ptr, scales.ptr, colors.ptr, len(poses))
                     for mesh, poses, scales, colors, hidden in entries]
        if self._packed_instances is None or self._packed_instances[0] != signature:
            arrays = []
            ends = np.cumsum([row[3] for row in signature], dtype=np.int32)
            for mesh, poses, scales, colors, hidden in entries:
                desc = _InstanceArrays()
                desc.poses, desc.scales, desc.colors = poses, scales, colors
                arrays.append(desc)
            dev = entries[0][1].device
            table = wp.array(arrays, dtype=_InstanceArrays, device=dev)
            offsets = wp.array(ends, dtype=wp.int32, device=dev)
            outputs = [wp.empty(int(ends[-1]), dtype=dtype, device=dev)
                       for dtype in (wp.transform, wp.vec3, wp.vec3)]
            self._packed_instances = (signature, ends, table, offsets, outputs)
        _, ends, table, offsets, outputs = self._packed_instances
        wp.launch(_gather_instances, len(outputs[0]), [table, offsets, *outputs], device=table.device)
        poses, scales, colors = (output.numpy() for output in outputs)
        start = 0
        for (name, (mesh, _, _, _, hidden)), end in zip(pending.items(), ends):
            self.instances[name] = {"mesh": mesh, "xforms": poses[start:end], "scales": scales[start:end],
                                    "colors": colors[start:end], "hidden": hidden}
            start = end

    def log_instances(self, name, mesh, xforms, scales, colors, materials,
                      hidden=False, opacities=None):
        if self._pending_instances is not None:
            # Public log_instances calls outside native log_state retain their
            # immediate snapshot semantics. Other array layouts use that path too.
            if (isinstance(xforms, wp.array) and xforms.dtype == wp.transform and xforms.device.is_cuda and
                    isinstance(scales, wp.array) and isinstance(colors, wp.array) and
                    scales.dtype == colors.dtype == wp.vec3 and xforms.device == scales.device == colors.device == self.device and
                    xforms.is_contiguous and scales.is_contiguous and colors.is_contiguous and
                    xforms.shape == scales.shape == colors.shape and xforms.ndim == 1 and len(xforms)):
                self._pending_instances[name] = (mesh, xforms, scales, colors, hidden)
                return
            if self._pending_instances:
                self._read_instances(self._pending_instances)
                self._pending_instances.clear()
        previous = self.instances.get(name, {})
        self.instances[name] = {
            "mesh": mesh, "xforms": None if xforms is None else _array(xforms).copy(),
            "scales": None if scales is None else _array(scales).copy(),
            "colors": previous.get("colors") if colors is None else _array(colors).copy(),
            "hidden": hidden,
        }

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        previous = self.points.get(name, {})
        self.points[name] = {"points": None if points is None else _array(points).copy(),
                             "radii": np.asarray(.01) if radii is None else _array(radii),
                             "colors": previous.get("colors") if colors is None else _array(colors), "hidden": hidden}

    def visual_meshes(self, shape_semantics):
        result = []
        instanced = {instance["mesh"] for instance in self.instances.values()}
        batches = {batch.name: batch for batch in self._shape_instances.values()}
        for name, instance in self.instances.items():
            mesh = self.meshes.get(instance["mesh"])
            if mesh is None or instance["hidden"] or instance["xforms"] is None:
                continue
            batch = batches.get(name)
            shapes = batch.model_shapes if batch is not None else []
            for index, pose in enumerate(instance["xforms"]):
                scale = instance["scales"][index] if instance["scales"] is not None else np.ones(3)
                color = instance["colors"][index] if instance["colors"] is not None else [.55, .65, .75]
                shape = shapes[index] if index < len(shapes) else None
                result.append({"id": f"shape_{shape}" if shape is not None else f"{name}:{index}",
                               "semantic_id": shape_semantics.get(shape),
                               "vertices": _transformed(mesh["vertices"], pose, scale), "faces": mesh["faces"],
                               "color": color, "kind": "rigid", "material_identity": True})
        for name, mesh in self.meshes.items():
            if name not in instanced and not mesh["hidden"]:
                result.append({"id": name, "vertices": mesh["vertices"], "faces": mesh["faces"],
                               "color": mesh["color"] if mesh["color"] is not None else [.55, .65, .75],
                               "kind": "deformable" if mesh["dynamic"] else "rigid",
                               "material_identity": not mesh["dynamic"]})
        # Native particle displays remain honest point proxies. Fluid scenes can
        # use the native MPM Example.render surface or explicit visual_meshes.
        sphere = None
        for name, cloud in self.points.items():
            if cloud["hidden"] or cloud["points"] is None:
                continue
            if sphere is None:
                sphere = newton.Mesh.create_sphere(1., num_latitudes=4, num_longitudes=6, compute_inertia=False)
            for index, point in enumerate(cloud["points"]):
                radii = cloud["radii"]
                radius = float(radii if radii.ndim == 0 else radii[index])
                colors = cloud["colors"]
                color = [.25, .55, .85] if colors is None else colors if colors.ndim == 1 else colors[index]
                result.append({"id": f"{name}:{index}", "vertices": _array(sphere.vertices) * radius + point,
                               "faces": _array(sphere.indices).reshape(-1, 3), "color": color,
                               "kind": "particle_proxy", "material_identity": False})
        return result


class WarpRenderer:
    def __init__(self, device):
        self.device = wp.get_device(device)
        self.close()

    def _prepare(self, meshes):
        vertices, indices, colors, face_ends = [], [], [], []
        face_colors = []
        vertex_count = face_count = 0
        for mesh in meshes:
            points = np.asarray(mesh["vertices"], dtype=np.float32)
            faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1, 3)
            if not len(points) or not len(faces):
                continue
            vertices.append(points)
            indices.append(faces + vertex_count)
            vertex_count += len(points)
            face_count += len(faces)
            face_ends.append(face_count)
            colors.append(np.asarray(mesh.get("color", [.55, .65, .75]))[:3])
            face_colors.append(mesh.get("face_colors"))

        if not vertices:
            self.mesh = self.vertices = self.topology = None
            self.colors = self.face_ends = None
            return

        points = np.concatenate(vertices)
        faces = np.concatenate(indices).reshape(-1)
        if self.mesh is None or len(points) != len(self.vertices) or not np.array_equal(faces, self.topology):
            self.mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3), indices=wp.array(faces, dtype=wp.int32))
            self.topology = faces
            self.vertices = points
        elif not np.array_equal(points, self.vertices):
            self.mesh.points.assign(points)
            self.mesh.refit()
            self.vertices = points

        palette = np.asarray(colors, dtype=np.float32)
        if any(values is not None for values in face_colors):
            # A scene-owned Warp kernel can shade triangles in their original
            # order. Keep that data on device even when mixed with solid meshes.
            self.face_ends = None
            if self.colors is None or len(self.colors) != face_count:
                self.colors = wp.empty(face_count, dtype=wp.vec3)
            start = 0
            for end, solid, values in zip(face_ends, palette, face_colors):
                count = end - start
                if values is None:
                    self.colors[start:end].fill_(wp.vec3(*solid))
                elif isinstance(values, wp.array):
                    if values.dtype != wp.vec3 or values.shape != (count,) or not values.is_contiguous:
                        raise ValueError("face_colors needs a contiguous Warp vec3 per triangle")
                    wp.copy(self.colors, values, dest_offset=start, count=count)
                else:
                    host = np.asarray(values, dtype=np.float32)
                    if host.shape != (count, 3):
                        raise ValueError("face_colors needs one RGB triplet per triangle")
                    self.colors[start:end].assign(host)
                start = end
            return

        # Solid meshes need just one color each. Each ray resolves its mesh
        # from the ordered face ranges without expanding colors per triangle.
        ends = np.asarray(face_ends, dtype=np.int32)
        if self.face_ends is None or len(self.colors) != len(palette):
            self.colors = wp.array(palette, dtype=wp.vec3)
            self.face_ends = wp.array(ends, dtype=wp.int32)
        else:
            self.colors.assign(palette)
            self.face_ends.assign(ends)

    def render(self, meshes, camera, *, channels=None, snapshot=None):
        channels = tuple(_CHANNELS if channels is None else channels)
        if not channels or len(set(channels)) != len(channels) or set(channels).difference(_CHANNELS):
            raise ValueError("Select distinct render channels from rgb, depth, normal and face_id")
        shape = (camera["height"], camera["width"])
        with wp.ScopedDevice(self.device):
            # Only the session supplies a snapshot token, valid until its next
            # geometry capture. Direct callers can freely modify input arrays.
            if snapshot is None or self.snapshot is not snapshot:
                self.snapshot = None
                self._prepare(meshes)
                self.snapshot = snapshot
            if self.shape != shape:
                self.buffers.clear()
                self.shape = shape
            for name in channels:
                if name not in self.buffers:
                    self.buffers[name] = wp.empty(shape, dtype=_CHANNELS[name][1])
            matrix = np.asarray(camera["camera_to_world"], dtype=np.float32)
            inverse_k = np.linalg.inv(np.asarray(camera["intrinsics"], dtype=np.float32))
            mask = sum(_CHANNELS[name][0] for name in channels)
            wp.launch(_trace, dim=shape, inputs=[
                self.mesh.id if self.mesh is not None else 0, int(self.mesh is not None),
                wp.mat44(*matrix.ravel()), wp.mat33(*inverse_k.ravel()), self.colors, self.face_ends, mask,
                *(self.buffers.get(name) for name in _CHANNELS)])
            result = {name: self.buffers[name].numpy() for name in channels}
            if self.device.is_cpu:
                # numpy() aliases CPU Warp storage. Keep returned frames stable
                # when the next camera overwrites our reusable device buffers.
                result = {name: pixels.copy() for name, pixels in result.items()}
        return {**result, "renderer": "python-warp-raytrace",
                "geometry_source": "native_warp_ray_intersection; no RGB estimation",
                "normal_convention": "camera_opengl_xyz; unit geometric triangle normal; zero background",
                "depth_convention": "optical_axis_m; zero means no intersection"}

    def close(self):
        self.mesh = self.topology = self.vertices = None
        self.colors = self.face_ends = None
        self.buffers = {}
        self.shape = self.snapshot = None
