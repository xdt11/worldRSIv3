# World state

`world.yaml` is the editable scene hypothesis shared by reconstruction and threat editing. Submit it with `set_world_state(path="/work/world.yaml")`; new scenes use schema 3 and `document_role: world_hypothesis`. The exact schema is `/opt/worldrsi/world-state-v3.schema.json`. Read the relevant definition when needed, rather than copying a complete template:

```python
import json
schema = json.load(open("/opt/worldrsi/world-state-v3.schema.json"))
print(json.dumps(schema["$defs"]["camera_pose"], indent=2))
```

A minimal starting document follows. Replace the instruction and unresolved subject using the actual inputs; this skeleton does not reconstruct a scene.

```yaml
schema_version: 3
document_role: world_hypothesis
coordinates:
  world_frame: world
  handedness: right
  up_axis: z
  length_unit: m
  angle_unit: rad
  quaternion_order: xyzw
timeline:
  unit: s
  clocks:
    world: {description: Native simulation time}
sources:
  instruction:
    kind: text
    availability: available
    text: Replace with the supplied instruction.
objects:
  subject:
    description: Unresolved scene entity.
    properties:
      identity: {value: null, status: unknown}
```

Include media under their supplied `source_id`, `kind`, `availability: available`, and `/input/...` path. IDs are stable snake_case. Use IDs for important objects, phenomena and editable groups; a native body or decorative primitive does not require its own world object. A semantic group may bind to several native shapes/bodies. Objects, phenomena and cameras share a unique namespace.

Claims use `{value, status, unit?, evidence?, note?}`. Status is `observed`, `hypothesis`, `implementation_assumption` or `unknown`; unknown values stay null. Observed claims cite available sources using `{source, timestamp_s?, frame_id?, region_xyxy?}`. Keep physical parameter `real` and `simulation` claims separate, in the same canonical unit. A chosen simulation value is not a measured real value.

Describe geometry, physical interactions and consequential parameters to the fidelity needed for later edits. Static support may need collision and friction; distant decoration can be visual only. Use `phenomena` for material/field processes. Optional `solvers` describe actual native or custom Warp implementations, configuration and validation. Multiple native/custom solvers can be composed in ordinary scene code. Record ownership/exchanges in `couplings` when relevant, without integrating a degree of freedom twice. A solver's `validation` requires `status`, `cases`, `receipts`; start with `not_run` and empty lists. Passing requires actual receipts. Declarations do not prove physical correctness.

Camera pose is a claim containing a 4×4 `camera_to_world` column-vector transform, optical axes +x right, +y down, +z forward. `pose.mode: static` uses `transform`; `trajectory` uses `samples: [{time_s, transform}]` and `interpolation: se3 | hold | none`. Intrinsics contain a 3×3 K claim; varying K uses keyframes. Inspect `$defs.camera`, `camera_pose` and `intrinsics` for exact fields. Pose, K, source clock and cuts are separate. Fit source camera motion against matched-time static landmarks; declared anchors alone are not validation.

For a look-at hypothesis, let `forward = normalize(target - position)`, `right = normalize(cross(forward, world_up))`, and `down = cross(forward, right)`. The rotation columns are `[right, down, forward]`; the translation column is `position`. Check `R.T @ R = I` and `det(R) = +1`. Choose a nonparallel up vector. A source's unknown calibration can remain unknown as a real measurement, while a renderable camera uses explicit hypothesized K and pose claims.

Declare source clocks and `timeline.source_mappings` only when established. Original decoded PTS, requested extraction time and actual simulation time differ. A CFR evaluation copy starts at zero and may duplicate/drop original frames; `/work/output_contract.json` records that contract. Unknown synchronization is not a zero offset.

`editables` target an existing object/phenomenon/camera using `kind`, `id` and a JSON Pointer `path` **relative to that target node**. For object `road`, use `/physics/parameters/friction/simulation/value`, not `/objects/road/physics/parameters/friction/simulation/value`. Supply `rationale`, `preserve`, and compatible units/bounds where applicable. Edits must reach code and native physics: road width changes visual/collision geometry; friction changes contact materials. Keep unknowns and source lineage. Optional `runtime_bindings` describe native handles; use unresolved handles before observation. Scene code may expose concise `bindings` for native queries without listing every primitive here.

Other useful schema definitions are `object`, `physics`, `parameter`, `phenomenon`, `solver`, `validation`, `coupling`, `time_mapping` and `editable`. Add only meaningful fields. A schema pass establishes structure and references; evaluate geometry, camera and physical response through actual execution.
