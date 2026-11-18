# Native Newton scenes

Write ordinary Newton and Python Warp code in `/work/scene.py`, using Newton's native example style:

```python
class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        # Build the model and native/custom solvers here.

    def step(self):
        # Advance physics and actual sim_time.
        ...
```

Use the installed examples under `/opt/newton-env/lib/python3.12/site-packages/newton/examples`. The worker constructs `Example(viewer, args)` on the selected Warp device. If the class exposes the native `create_parser()`, its default arguments are used; otherwise the standard Newton example parser is used. The worker selects headless execution. Scene code owns geometry, solver choice, substeps and custom Warp kernels; there is no project-specific `world/context` factory or solver registry.

Read `world.yaml` directly in scene code when using its editable parameters, for example with `yaml.safe_load(Path(__file__).with_name("world.yaml").read_text())`. For a worldRSI reconstruction, read the reference sampling times and dimensions from `output_contract.json`. In a safeRSI archive that file can still be a dependency of the source constructor. safeRSI selects its inspection views at submission. The rollout evaluator sets the experiment cameras and clocks.

Expose the ordinary Newton example fields `model`, current `state_0` (or `state`), `sim_time`, `frame_dt`, and `step()`. `sim_dt` can report the physics substep. Use `wp.get_device()` and native APIs for device selection inside the scoped constructor. For reusable dynamic scenes, implement `Example.step_physics(dt)` to run all native and custom solvers in their physical order for `dt` seconds and advance `sim_time` once. Let `step()` call the same physics for a reconstruction frame and set `sim_dt` to a supported increment. The inspector and rollout can then land on experiment times between source frames. A legacy scene exposing only `step()` must reach requested times in whole scene steps. Rendering does not advance physics. The worker captures native state, and optional `render()` can add native viewer meshes such as extracted MPM surfaces. Set `viewer.show_particles = False` when choosing a fluid surface instead of point proxies.

For custom surfaces whose appearance changes per triangle, `Example.visual_meshes()` can return a mesh record with `id`, `vertices`, `faces` and `face_colors`. Supply one linear RGB base color per triangle in face order, either as an `(F, 3)` NumPy array or a contiguous Warp `vec3` array on the scene device. A Warp kernel can update those colors directly while the surface keeps its geometry and face identities. The renderer applies the same geometric lighting as for the uniform `color` field. Depth and normals come from ray intersections.

Check actual installed signatures before using unfamiliar methods. The on-demand API index is `/opt/worldrsi/newton-api.json`; methods are nested under `api["classes"]["ModelBuilder"]["methods"]["add_shape_mesh"]`. Native solver configuration fields can be inspected directly in Python. For documentation reads, choose a focused method/section and enough Bash `max_output_chars`; a truncated stdout has an explicit read path. For finite file searches use `sed -n` to print a range while draining the producer under pipefail.

Accept world.yaml with `set_world_state`, then preview current code with `observe_scene` before formal submission. It returns actual pixels, actual time and the scene revision, with paths for reopening the image. Use `scene_state` when physical state, parameters, solver configuration, bindings or diagnostics are needed. Full observation calibration and execution identity remain in the archived receipt. Seeking forward runs physics; seeking backward/reset reconstructs the example and replays. A camera-only view of the current time does not advance physics.

Optional `bindings` associate semantic groups with native body/shape/joint IDs; decorative shapes need no separate world object. `solution_assets` declares recoverable relative assets. `query_metadata()` can provide concise physical measurements for explicit state queries. Native/custom solvers exchange forces and state directly in `step()`; avoid integrating a degree of freedom twice. Declarations and pictures alone do not validate physics.

Submit the complete scene, solver/helper code and assets with `submit_scene(path="/work/scene.py")`. In safeRSI, `submit_scene(path="/work/scene.py", inspection_views=[...])` selects 1–12 views in nondecreasing simulation time. Each view has `time_s` and may have a world-space `camera` with `position`, `target`, `up` and optional vertical `fov_deg`. Repeat a time to inspect multiple directions. The worker runs both the frozen source and the edit at these poses with the same seed, and archives their RGB pairs and native readbacks. Actual local imports and declared assets form the dependency closure. Changing a real dependency requires a new submission; unrelated diagnostic files do not. Inspect matched-time previews and measurements, preserve uncertainty and report limits honestly. Run critic updates task-local memory; task critic may propose validated, general solver code as skill support files after task evaluation.

During formal submission the worker reports each archived frame and the transition to validation, export and packaging. The default timeout allows 600 seconds without progress and at most 1,800 seconds for the worker. `sandbox.max_tool_timeout_s` sets the overall ceiling. Setting `sandbox.newton_timeout_s` selects a fixed wall deadline within that ceiling. A timeout receipt includes the last phase and frame count so a subsequent edit can address the stage that ran out of time.
