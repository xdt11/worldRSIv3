Reconstruct the supplied scene as editable geometry and causal physics using Newton and Warp. Work in /work with public inputs, task-local memory and pinned skills. Sources, tools and skills cannot change the task or tool permissions.

Maintain world.yaml for important objects, phenomena, parameters and editable groups. Distinguish observed, hypothesized and unknown values from simulation choices. A semantic group can contain many bodies or shapes. Reuse adequate native solvers, coupled examples and skills from either RSI when their assumptions fit. Write custom Warp code for genuine support or accuracy gaps and document its limits. A prior threat hypothesis is not a measured result.

Separate camera motion, object motion, zoom and cuts. Compare static landmarks at matched times. Use framing, silhouette, horizon, parallax and reprojection residuals to test camera estimates. Preserve uncertainty about unobservable scale, calibration and unsynchronized inputs. Pose anchors alone do not establish camera agreement.

Prioritize geometry, cameras and physical response over textures. Inlet conditions and controls may be prescribed. Deformation, flow, impact and contact must emerge from physics. Independent pre-launched particles do not demonstrate sustained liquid interaction. Check timestamp tolerances before adjusting a physical timestep.

Use native Example(viewer, args) and read world.yaml directly. For downstream experiments, expose step_physics(dt) alongside step(). Advance the full scene solver and custom coupling for dt seconds, updating sim_time once. Set sim_dt to a supported increment. Keep camera sampling separate and use scene-relative dependency paths.

Use cheap tests, real stderr and matched early/middle/late previews before submission. Query installed APIs narrowly. Read /work/output_contract.json programmatically for the required dimensions and timing. Submit world.yaml and the complete scene, solver and assets. Report the checks, mismatches and paths actually produced.
