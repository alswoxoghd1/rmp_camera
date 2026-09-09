# camera_rmp

## 최근 학습된 다이나믹 모델 컨텍스트

현재 확인된 최신 체크포인트는 다음 LPB visual dynamics 모델이다.

```text
모델 이름: train_lpb_visual_dynamics_real_relative
체크포인트: /home/son_rb/folk_lpb_interactive_diffusion_policy_repo/outputs/2026-08-26_00-20-22/checkpoints/latest.ckpt
정규화 파일: /home/son_rb/folk_lpb_interactive_diffusion_policy_repo/outputs/2026-08-26_00-20-22/normalizer.pth
```

이 모델은 로봇의 물리 파라미터를 직접 추정하는 모델이 아니라, `image0`와
proprioception(position, quaternion, gripper), action을 이용해 LPB latent/visual
dynamics를 예측하는 `LPBVisualDynamicsModel`이다. 이미지 encoder는 frozen이고
ViT predictor를 학습했다. 설정은 `frameskip=6`, `num_hist=1`, `num_pred=1`,
100 epochs이며 마지막 epoch의 train loss는 약 `0.0003886`이다.

학습 데이터는 다음 HDF5이다.

```text
/home/son_rb/rb_ws/src/robotory_rb10_ros2/data/0825_cowork_vr_override/cowork_vr_override_dp0010_training_ready_relative_10hz.hdf5
```

데이터는 `lpb_goal_only_vr_override` 실행에서 생성된 cowork 조작 기록으로,
3개 demo와 총 12,464개 샘플(10 Hz 기준 약 20분 46초)을 포함한다. base policy
`/home/son_rb/Downloads/low_epoch_ckpt_DP/epoch=0010-train_loss=0.0075.ckpt`를
실행하면서 사용자가 일부 구간을 Vive/VR teleop으로 override한 데이터다.
`control_mode=3`인 VR override 구간은 3,526 samples(약 28.3%)이며 나머지는
자동/base 구간이다. `avoidance_flag`는 전체 샘플에서 0이므로 RMP avoidance
개입 데이터셋은 아니다. 모든 action은 변환 과정에서 보정되어 저장되었다.

## Offline robot self-filter validation from recorded TF

Validation bags that contain raw depth, CameraInfo, `/tf`, and `/tf_static`
can be checked even when they do not contain JointState. The validator applies
the same collision-sphere surface renderer and tolerance classifier used by
`robot_depth_mask_node`, selecting the closest recorded robot TF for every
sampled depth frame.

Use the dynamic experiment launch as the entry point:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py \
  source:=bag \
  bag_path:=/home/son_rb/bags/nvblox_validation/robot_people \
  offline_self_filter_validation_only:=true
```

The default checks every tenth depth frame, writes a JSON report under
`/tmp/rmp_camera_self_filter_validation`, and saves diagnostic images with:

- measured depth;
- predicted robot depth;
- classification (`red=removed`, `green=foreground kept`, `blue=behind`);
- robot-masked depth.

Use `offline_self_filter_sample_every:=1` for every depth frame. To run the
validator while the existing bag/Nvblox experiment also plays, set
`run_offline_self_filter_validation:=true` instead of validation-only mode.

The CLI can also check multiple bags directly:

```bash
ros2 run rmp_camera validate_robot_self_filter_bag \
  /home/son_rb/bags/nvblox_validation/only_robot \
  /home/son_rb/bags/nvblox_validation/robot_people \
  --sample-every 10 \
  --debug-dir /tmp/rmp_camera_self_filter_validation \
  --json-report /tmp/rmp_camera_self_filter_validation/report.json
```

## Static dense-ESDF medial spheres

The optional `esdf_medial_sphere_node` queries nvblox's
`/nvblox_node/get_esdf_and_gradient` service over a configurable 3D AABB. It
uses only observed voxels with `esdf_distance < -inside_epsilon_m`, separates
them with 18-connectivity, selects spaced local minima (including multiple
representatives on long plateaus), and adds raw ESDF-radius spheres until each
component reaches the default `target_coverage=0.95` or a configured hard
limit. The `safety_margin_m` is added only to published radii.

No surface-point fallback is performed when the dense grid contains no
negative ESDF voxels. Existing obstacle nodes and
`/rmp_camera/camera_obstacle_spheres` are unchanged.

Run it with the static launch:

```bash
ros2 launch rmp_camera d435_nvblox_static.launch.py \
  run_esdf_medial_spheres:=True
```

RViz/debug outputs:

- `/rmp_camera/esdf_medial_sphere_markers` (`visualization_msgs/MarkerArray`)
- `/rmp_camera/esdf_medial_query_bounds` (`visualization_msgs/MarkerArray`),
  the translucent fill and wireframe of the exact configured query AABB
- `/rmp_camera/esdf_medial_sphere_cloud` (`sensor_msgs/PointCloud2`)
- `/rmp_camera/esdf_medial_inside_voxels` (`sensor_msgs/PointCloud2`)
- `/rmp_camera/esdf_medial_uncovered_voxels` (`sensor_msgs/PointCloud2`)

The main tuning parameters are the AABB minimum/size values,
`inside_epsilon_m`, `target_coverage`, `coverage_tolerance_m`,
`plateau_epsilon_m`, `minimum_center_spacing_m`, `min_component_voxels`,
`min_raw_sphere_radius_m`, `max_raw_sphere_radius_m`, `safety_margin_m`, and
the per-component/total sphere limits. The optional bounded
`enable_post_merge_overlap_pruning` pass removes final spheres involved in
excessive output-volume overlap only when volume and surface-shell coverage
remain above their configured guards.

The dynamic launch can shrink individual query-box faces without editing the
YAML. The six `esdf_trim_{x,y,z}_{min,max}_cm` arguments are centimetres
removed from the configured AABB; the cyan query-bounds marker shows the
effective result.

For static-sphere redundancy and dynamic self-filter ghost diagnosis, enable
the opt-in diagnostic recorder in the same launch:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py \
  run_robot_self_filter:=true \
  record_debug_data:=true \
  debug_bag_name:=hand_ghost_test_01
```

It records the exact self-filter depth input/output, predicted robot depth,
removed robot points, joints and robot markers, native Nvblox dynamic points,
static/dynamic retained and rejected voxels, every sphere-cloud stage, TF, and
ROS diagnostics. A manifest beside the bag captures the effective ESDF AABB,
launch trims, topic list, and source config. Raw color is omitted by default to
limit disk use; add `debug_record_color:=true` only when needed. Stop shortly
after reproducing the event because full-rate depth bags are large.

Core algorithm tests do not require ROS:

```bash
python3 -m pytest -q test/test_esdf_medial_sphere_core.py
```

## One-Nvblox dynamic sphere fusion experiment

`d435_nvblox_dynamic_spheres.launch.py` starts one D435 include and one
`nvblox::NvbloxNode` with `mapping_type=dynamic`. Nvblox's `MultiMapper`
maintains the static background and dynamic foreground in that node. The
existing dense static-ESDF medial sphere node is unchanged. A new NumPy-only
adapter voxelizes `/nvblox_node/dynamic_points`, constructs adaptive/fallback
spheres, tracks them through short occlusions, and fuses their PointCloud2
output with the latest static sphere cloud.

All experiment values are in one file:

```text
config/d435_nvblox_dynamic_experiment.yaml
```

Build and locate the installed copy:

```bash
cd ~/rmp_camera_dynamic_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install --packages-select rmp_camera \
  --event-handlers console_direct+
source install/setup.bash
ros2 pkg prefix --share rmp_camera
```

Run with the installed default YAML:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py
```

Enable the depth-aware robot self-filter in live-camera mode:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py \
  run_robot_self_filter:=true
```

This starts the measured RB10 joint source by default, builds the configured
URDF collision spheres at the joint timestamp, renders their nearest camera-Z
surface, and removes a depth pixel only when its measured depth matches that
surface. Measurements clearly in front of the robot are preserved. Disable
`run_rb10_joint_state_source` when an external node already publishes
`/rmp_camera/joint_states_urdf`.

Self-filter debug outputs:

- predicted robot surface: `/rmp_camera/robot_predicted_depth/image_rect_raw`
- removed measured points: `/rmp_camera/robot_depth_mask_removed_points`
- filtered Nvblox input: `/rmp_camera/robot_surface_filtered_depth/image_rect_raw`

Record a repeatable live input bag for self-filter/Nvblox regression tests:

```bash
ros2 launch rmp_camera record_nvblox_validation_bag.launch.py
```

The command starts the live D435/RB10 pipeline by default and records splitter
depth, camera calibration, measured/normalized joints, robot collision markers,
and TF under `~/bags/nvblox_validation`. Stop it with `Ctrl+C`.
Use `start_pipeline:=false` when the live pipeline is already running.

## Markerless robot-depth extrinsic calibration

A robot-only validation bag can also estimate `base_link -> camera0_link`
without a ChArUco/checkerboard target. The offline calibrator selects stationary,
distinct joint poses, aligns the RB10 collision meshes to measured depth, and
checks the candidate on held-out poses:

```bash
ros2 run rmp_camera calibrate_robot_depth_extrinsic \
  ~/bags/nvblox_validation/markerless_calibration_20260902 \
  --output /tmp/rmp_camera_markerless_result.yaml \
  --debug-dir /tmp/rmp_camera_markerless_debug
```

The command never edits package configuration. Inspect the red/green CAD
overlays and the held-out metrics in the result first. If the candidate passes
visual validation, copy its `launch_arguments` to the experiment's
`static_tf_*` values and set every `/robot_sphere_marker_correction_node`
translation/rotation to zero. Applying both corrections would double-correct
the robot-to-camera relationship.

## Manual TCP-click extrinsic refinement

For a more precise refinement, record 15--20 stationary TCP locations and click
the known TCP origin in each color frame. Keep the camera fixed for the entire
recording. A useful 18-pose layout is a 3-D grid in the camera view:

- near / middle / far from the camera;
- at each depth, screen-left / center / screen-right;
- at each horizontal location, one low and one high TCP position.

Hold each pose completely still for 1.5--2 seconds and keep the TCP at least
about 100 pixels from the image boundary. The robot may move freely between
poses; moving frames are discarded automatically.

```bash
ros2 launch rmp_camera record_nvblox_validation_bag.launch.py \
  bag_name:=tcp_click_calibration_20260902
```

After stopping the recorder with `Ctrl+C`, collect clicks and solve from the
current markerless candidate:

```bash
ros2 run rmp_camera calibrate_tcp_click_extrinsic \
  ~/bags/nvblox_validation/tcp_click_calibration_20260902 \
  --initial-calibration ~/rb_ws/calibration_results/markerless_20260902.yaml \
  --clicks ~/rb_ws/calibration_results/tcp_click_20260902_clicks.yaml \
  --output ~/rb_ws/calibration_results/tcp_click_20260902.yaml \
  --debug-dir ~/rb_ws/calibration_results/tcp_click_20260902_debug
```

The red cross is the current calibration's predicted TCP. Click the actual TCP
origin in the full image, refine the same point in the zoom window, then press
Enter. `X` skips an unusable frame, `B` returns to the previous frame, and `Q`
saves progress and exits. Re-running the same command resumes the unfinished
click file. Use `--review-all` to revisit saved clicks or `--solve-only` to
optimize without opening the GUI.

If the clicked point is a different rigid feature, pass its measured TCP-frame
coordinates with `--feature-offset-tcp X Y Z`. As a fallback,
`--estimate-feature-offset` can jointly estimate that offset, but then the bag
must also contain substantially different wrist orientations.

Bag playback runs once by default and exits cleanly. Add `repeat_bag:=true`
only when repeated passes are needed; each repeated pass intentionally restarts
the complete Nvblox graph so maps and sphere tracks begin from a clean state.

Bag self-filtering requires time-aligned joint states or recorded collision
sphere markers. A depth-only bag cannot reconstruct a moving robot pose.

Or pass an edited copy explicitly:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py \
  experiment_config:=/absolute/path/d435_nvblox_dynamic_experiment.yaml
```

Inspect launch arguments and native/derived topics:

```bash
ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py --show-args
ros2 topic info /nvblox_node/dynamic_points --verbose
ros2 topic echo /rmp_camera/esdf_medial_sphere_cloud --once
ros2 topic echo /rmp_camera/dynamic_obstacle_sphere_cloud --once
ros2 topic echo /rmp_camera/combined_obstacle_sphere_cloud --once
```

Important outputs:

- static inside voxels: `/rmp_camera/esdf_medial_inside_voxels`
- static medial spheres: `/rmp_camera/esdf_medial_sphere_markers`
- native dynamic points: `/nvblox_node/dynamic_points`
- dynamic voxels/spheres: `/rmp_camera/dynamic_obstacle_voxels` and
  `/rmp_camera/dynamic_obstacle_sphere_markers`
- source-colored fusion: `/rmp_camera/combined_obstacle_sphere_markers`

Optional PeopleSemSegNet branch
--------------------------------

The normal Nvblox static/dynamic paths continue to represent all obstacles.
The optional semantic branch segments people in RGB, registers that mask to
depth, tracks a separate human sphere stream, and removes only spatially
overlapping duplicate spheres during final fusion. Human spheres are green in
the combined marker stream and use `source_type=2` in the combined cloud.

```bash
cd /home/son_rb/rb_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch rmp_camera d435_nvblox_dynamic_spheres.launch.py \
  source:=camera \
  run_people_segmentation:=true \
  run_human_obstacle_spheres:=true \
  run_robot_self_filter:=false \
  run_rviz:=true
```

The default TensorRT engine is
`~/rb_ws/isaac_ros_assets/models/peoplesemsegnet/deployable_quantized_vanilla_unet_onnx_v2.0/1/model.plan`.
Override it with `people_model_path:=/absolute/path/model.plan`. The semantic
branch publishes:

- mask: `/camera0/segmentation/people_mask`
- registered human points: `/rmp_camera/human_points`
- human spheres: `/rmp_camera/human_sphere_cloud`
- fused static/dynamic/human spheres: `/rmp_camera/combined_obstacle_sphere_cloud`

Set `record_debug_data:=true debug_record_color:=true` to record RGB, semantic
mask, registered points, all three sphere sources, fusion output, TF, and the
algorithm logs in one diagnostic bag.

Human/static ghost filtering is enabled with the human branch by default
(`run_human_static_filter:=true`). It excludes recent registered human voxels
before the static sphere solve, including a bounded band behind the observed
surface. A five-second sparse history locates possible old human geometry;
expired human detection alone never proves empty space. Free-space removal
still requires fresh valid depth behind the entire projected voxel.

With `human_static_occlusion_hold_enabled:=true` (default), the sphere outputs
also preserve **semantic ownership**, separately from free-space evidence:

- For up to `human_static_ownership_hold_s` (2 s), a previously observed human
  location does not become static just because the person/mask moved. At least
  75% of its projected voxel box must have fresh valid depth. Any newly measured
  nonhuman surface inside the voxel's camera-space bounding box plus clearance
  immediately vetoes this additional exclusion. Missing depth alone is not
  evidence; this is a brief semantic-label hold, not proof that hidden space is
  empty. A fully occluded new obstacle cannot be identified from these inputs.
- Between that hold and the 5 s history limit, every observed non-free pixel
  must match current measured human surface points within 5 cm, and the depth
  gap behind that surface is limited to 45 cm. If up to 25% depth holes exist,
  at least 60% of the entire voxel footprint must be measured human surface.
- Human/depth source stamps must differ by at most 100 ms and both be at most
  250 ms old. Unknown background with no human history is not excluded by this
  addition. It neither extends history timestamps through inference nor creates
  new Human spheres/long-lived tracks. Per-call footprint work is bounded to
  131072 pixels; unprocessed candidates are preserved, not dropped.

These tests run before the static solve and on matched cached support in
Fusion. `human_owned` in `Human static voxel evidence` reports this additional
semantic exclusion, not physically free space. Use
`human_static_occlusion_hold_enabled:=false` for the prior policy (recent
surface exclusion and depth/ESDF-confirmed clearing remain enabled).

Static generation also publishes `/rmp_camera/static_sphere_support` with the
exact generation stamp and each sphere's supporting voxel coordinates. Fusion
can suppress a cached static sphere without waiting for the next 1 Hz solve
only when all its support is excluded. Missing support protects the sphere.
With `human_static_refit_enabled:=true` (default), mixed human/ordinary-obstacle
balls are instead tightened around **all retained support**. Four candidate
centres are evaluated; a replacement must remain entirely inside its original
output ball and cover every retained centre with the static solver's existing
coverage tolerance. It preserves the safety margin and minimum radius and must
cover fewer excluded centres. It never increases the number of spheres.
If a safe tighter ball cannot be fitted, the original is retained until the
next ordinary static solve can rebuild/split the component. This is not a
guarantee of zero image-plane overlap: separate objects behind a person may
legitimately project onto the same pixels. Unknown support is never removed
merely to make the view cleaner. Existing Fusion overlap priorities still
apply afterward; this refit is not a new global coverage guarantee. Refits
are cached for an unchanged support
generation/exclusion mask and do not run Minimum-K again. The support topic is
included in debug recordings. In addition, each new static solve checks the
last three support generations against the **unmodified** ESDF and publishes
`/rmp_camera/static_sphere_free_support`. Observed positive distance must exceed
the voxel half-diagonal plus the clearance; unknown and negative ESDF values
never count as free. Fusion can retire an old sphere immediately when all its
support is confirmed free, bypassing the ordinary empty-frame confirmation
delay. A retired generation cannot reappear when the evidence ages out; a new
static generation can still represent a newly placed obstacle at that location.
Free-support evidence is also included in debug recordings.

This changes sphere outputs, not the underlying
Nvblox TSDF/ESDF map, and does not enable the pre-Nvblox robot depth filter.
Shared settings are in `/human_static_filter` in
`config/d435_nvblox_dynamic_experiment.yaml`; use
`run_human_static_filter:=false` for an otherwise identical A/B replay.
To compare just mixed-ball refitting, use `human_static_refit_enabled:=false`;
this keeps the existing human voxel/depth-evidence filters on. The launch
automatically uses `/esdf_medial_sphere_node.min_raw_sphere_radius_m` and
`coverage_tolerance_m` for the fast refit as well. RViz's default view keeps
the raw Static/Dynamic reconstruction groups off and displays Fusion, so
unfiltered intermediate markers are not drawn over the final result.

Static disappearance handling now separates successful results from observation
failures. `/rmp_camera/static_sphere_result_status` (`diagnostic_msgs/DiagnosticArray`)
pairs with the static cloud by **exact generation stamp**, independently of
topic arrival order. `unknown` (ESDF/TF/sync failure, missing observed support,
or stale camera depth with the human filter enabled) does not publish an empty
sphere cloud or advance Fusion's empty-frame counter. Missing/mismatched/stale
status also cannot clear cached geometry. This topic is included in debug bags.

The experiment uses `static_fast_empty_clear:=true`: a successful empty result
whose old support locations remain observed clears Fusion on its first valid
update, including DELETE markers if the bag clock has just stopped. "Empty"
means no obstacle admitted by the configured voxel/component/radius rules; it
does **not** claim every voxel in the room is physically free. Those geometry
thresholds have not been relaxed.

`empty_check_rate_hz: 5.0` in `/esdf_medial_sphere_node` adds empty-only checks
using the full solver's same inside threshold, component size, and robot
component rejection. They do not run sphere fitting, merging, or Minimum-K.
`update_rate_hz: 1.0` still bounds full optimization. A possible ordinary
obstacle component blocks the shortcut. Old support that becomes unknown or
out of bounds blocks empty confirmation as well. These extra ESDF requests
have a nonzero cost; they are not a free 5x increase in sphere generation rate.
When the last static output is already empty, requests fall back to the normal
solve cadence because there is no old geometry to retire.

For a controlled A/B comparison, use `static_fast_empty_clear:=false
static_empty_check_rate_hz:=0.0` for the prior debounce/1Hz behavior. The new
validity/error distinction remains active in both runs. A legacy standalone
static publisher without status messages requires
`static_require_result_status:=false` on Fusion (not recommended for this launch).
The helper `scripts/validate_static_clear_replay.py` runs a headless, hardware-off
replay on a separate ROS domain, saves RGB/status/count logs, and uses the bag's
fixed start timestamp so different runs can be compared.

The fusion node does not exact-sync its different-rate static/dynamic/human
sources. Dynamic output has a short hold and an absolute ghost limit;
`keep_dynamic_until_static_overlap` can optionally extend handover up to that
limit. All overlap decisions use actual 3D center distance and output radii.

Dynamic tracking can optionally run bounded, coverage-preserving overlap
pruning after association. `dynamic_post_tracking_max_spheres`,
`dynamic_post_tracking_max_removals`, and
`dynamic_post_tracking_max_matrix_elements` are hard CPU/memory bounds. A
suppressed track is removed from the tracker as well as the published cloud,
so it cannot reappear on the following timer tick.

### Human input latency and equivalent sphere-generation optimization

`human_low_latency:=true` is the experiment default. Human projection retains
depth history for inference synchronization, but uses a one-message DDS mask
queue and prioritizes the newest mask that has a compatible newer depth frame.
Both input streams advance monotonically; delayed packets cannot move a track
back to an older source frame. An actual ROS clock rewind resets the pair
buffers and watermarks. The original `max_sync_delta_s: 0.05` remains unchanged.

With this mode enabled, pairs older than `max_frame_age_s: 0.25` are discarded
as unavailable, **not** published as newly observed empty scenes. Source stamps
more than `max_sync_delta_s` ahead of the ROS clock are also rejected (including
old-epoch packets after a bag rewind). Rewinding the whole pipeline still
requires the existing clean-restart bag workflow because Nvblox retains maps.

The launch caps OpenBLAS/OpenMP/MKL threads to one **only for Human projection
and Human sphere node processes**. Projection also caps OpenCV threads to one.
This avoids many threads contending over small per-frame CPU operations;
segmentation model settings, RViz, depth filtering and global environment are
unchanged. Use `human_low_latency:=false` to compare the old queue policy and
inherited thread settings. Restart after changing the mode.

The shared Dynamic/Human sphere generator now accumulates seed coverage and
center-spacing exclusions incrementally, and reuses already-computed overlap
metrics unless a global sphere cap actually truncates a component. This avoids
repeated identical work; it does not disable Minimum-K or loosen coverage,
overlap, radius or component-size constraints. The equivalent-computation
optimization remains active even when `human_low_latency:=false`.

That equivalent-computation optimization adds no velocity extrapolation,
warm-start geometry reuse, tracking TTL change or unsupported-track immediate
deletion. The separately selectable refinement path below adds validated
template reuse. Marker publication can remain
30Hz while real position updates are slower: new synchronized depth/mask input
is still required. The replay helper accepts `--human-low-latency true|false`
(default false for controlled comparisons), `--capture-human-points`, and
captures semantic-mask source/publication timing as well as Human output.

### Single-frame depth edge validation

`d435_nvblox_dynamic_spheres.launch.py` now supports `run_depth_edge_filter`.
The experiment YAML enables it by default. This is **not** the robot depth
self-filter: it needs no robot markers, TF, RGB, or previous depth frames.

Raw depth goes to `/rmp_camera/edge_filtered_depth/image_rect_raw`, then to
Nvblox (optionally through the existing robot mask). Human projection and the
static depth-evidence checks also receive validated depth **before** the robot
mask. Setting `run_depth_edge_filter:=false` restores the previous depth routing
and does not start the new node. Existing tracking/TTL/Minimum-K settings are
unchanged.

The filter examines a 7x7 window from the original current frame. It rejects
an intermediate sample only if it has at least two neighbors >=10cm nearer,
at least two >=10cm farther, and fewer than 12 neighbors within +/-4cm of its
own depth. The center does not count as its own support. Invalid depth does not
count as either surface; image borders are retained. A clean foreground hand
edge against a background is not rejected just for being an edge.

Rejected pixels become **0 (unknown)**, never interpolated depths or free-space
measurements. Image timestamp/frame, dimensions, encoding, endian and row
padding are preserved. There is no temporal confirmation delay or rate limiter;
input/output queues keep only one frame. Processing and transport still have a
nonzero cost. Settings in `/depth_edge_filter_node` are startup-only and require
a restart after editing the YAML.

This conservative heuristic cannot remove all stereo errors: a coherent wrong
surface can look well-supported, and a real small surface between two other
depth layers can also look suspicious. Validate retained hand/obstacle geometry
before relying on the output for robot avoidance.

For an A/B test, add `run_depth_edge_filter:=true` or `false` to the same launch
command. The headless helper accepts `--depth-edge-filter true|false` (default
`false` to preserve its historical baseline). Replay raw input topics only;
do not replay previously generated filtered depth or sphere outputs alongside
the nodes that regenerate them.

The node logs input/output/error counts, rejection fraction, and processing
mean/max every two seconds. It publishes the same metrics on
`/rmp_camera/depth_edge_filter/status` (`diagnostic_msgs/msg/DiagnosticArray`).
With `record_debug_data:=true`, bags include raw/validated depth, rejection mask,
and filter status. The rejection mask is otherwise not published.

Run all ROS-independent core tests with:

```bash
python3 -m pytest -q \
  test/test_esdf_medial_sphere_core.py \
  test/test_dynamic_obstacle_sphere_core.py \
  test/test_obstacle_sphere_fusion_core.py
```

### Bounded foreground / background sphere refinement

The experiment enables `async_sphere_refinement:=true` for Dynamic and Human
spheres. Static generation, robot-filter radii, noise thresholds, depth
validation and tracker smoothing/TTL are unchanged. Restart the launch after
changing the setting. Add `async_sphere_refinement:=false` to restore the original
synchronous search budgets for a comparison.

Current voxelization, robot component rejection, adaptive seeds and greedy
set-cover still run on each new input. The **merge/Minimum-K portion only** gets
a shared `dynamic_foreground_refinement_budget_ms: 5.0` per frame; short searches
finish normally. Seed generation keeps the original processing guard: reducing
that guard indiscriminately can create many small fallback spheres.

An unfinished search can submit a snapshot to one spawned worker per node, at
most `dynamic_refinement_rate_hz: 3.0`. There is only one outstanding job and no
backlog of old frames. The worker retains `dynamic_processing_budget_ms` (300ms
Dynamic, 80ms Human in this experiment). Its result is **never published directly**.
The parent reuses a shape only on a new current, robot-filtered component when:

- its source age is at most `dynamic_refinement_max_age_s: 0.5`;
- relative voxel-shape support is >=65% in both directions with <=30cm translation;
- current target coverage and every voxel covered by the fresh cover are retained;
- every reused sphere has current support and passes radius/workspace/empty-space checks;
- the final count is strictly lower, with no worse total overlap. If the fresh
  solver already returned an overlap-infeasible fallback, reuse may improve it
  without reaching the configured overlap limit; diagnostics keep that fact.

Small shape changes may be repaired using current seed spheres. A cache miss,
validation budget exhaustion, split/disappearance, time rewind or worker failure
keeps current geometry. Empty frames and robot-sync failure invalidate historical
templates. The application checks are bounded to 8192 snapshot voxels and a 4ms
soft validation budget. No velocity prediction or unsupported-track deletion is
introduced. These are **soft computation budgets**, not end-to-end deadlines:
voxelization, seeding, tracking, DDS and robot-marker synchronization can still
take longer. Real scene-update Hz remains limited by new depth/mask input.

DiagnosticArray topics (also included in debug bags) are:

- `/rmp_camera/dynamic_obstacle_sphere_cloud/status`
- `/rmp_camera/human_sphere_cloud/status`

They publish only on new observations (or a synchronization rejection), with
input `source_stamp_ns`, `source_age_ms`, processing time, generated coverage,
fast/final counts and worker/reuse statistics. A timer-only 30Hz republication
does not count as a new observation. Existing PointCloud2 fields are unchanged.
Source age is relative to the input point-cloud header, not necessarily the
camera's original exposure time.

For controlled headless playback use `scripts/validate_static_clear_replay.py`
with `--async-refinement true|false`. `scripts/benchmark_sphere_refinement.py`
compares the two search budgets on identical captured XYZ frames. With a moving
or deforming object, historical templates may correctly be rejected; fewer
spheres are not guaranteed on every intermediate frame or globally optimal.

### Latest input, component-aware tracking and coverage-preserving Fusion

Three independently selectable experiment defaults are enabled. Restart after
changing them; a launch override of `false` restores that individual old path:

```bash
latest_sphere_input:=true
component_sphere_tracking:=true
fusion_coverage_guard:=true
```

`latest_sphere_input` uses a best-effort DDS queue of one and a bounded raw-message
buffer of five. It chooses the **newest transformable, robot-synchronized** frame,
not blindly the newest unsynchronized frame. Only the selected cloud is decoded.
Strided NumPy decoding handles endian, field offsets, row padding and malformed
payloads without per-point Python objects. The default input age bound is
`dynamic_input_max_age_s: 0.25`. New arriving unsynchronized frames cannot reset
the existing 0.25s synchronization wait indefinitely. Clock rewind clears queued
frames and tracking; malformed/expired input is reported as an invalid observation,
not a confirmed empty environment. Existing robot geometry and timestamp tolerances
are unchanged. The existing status topics now include selection, decoding and queue
wait times plus cumulative superseded/expired/out-of-order counts.

`component_sphere_tracking` first associates connected components using centroid,
AABB overlap, extent and voxel count. Sphere association is restricted to matched
components and uses measured component displacement for matching, **not velocity
extrapolation**. `dynamic_tracking_max_radius_ratio: 2.0` prevents a much larger
old torso sphere from blending into a new hand sphere. New identities are allowed
when shape changes substantially; this is not semantic person re-identification.
Within the existing coverage-guarded overlap pruning, equally redundant old tracks
are considered for removal before current observations. Spatially separate old
tracks are not immediately removed for lacking support. Smoothing, radius limits,
minimum component sizes, TTL and missed-update limits remain unchanged.

`fusion_coverage_guard` replaces the old rule "any cross-source overlap deletes
the lower-priority sphere". A lower-priority sphere can be removed when **all of
its retained measured voxel support** is covered by the union of already retained
higher-priority spheres, using `fusion_coverage_tolerance_m: 0.02`. Static support
is taken after the existing Human ownership/free-evidence filter and mixed-ball
refit. Dynamic/Human support is paired to the exact sphere-message generation and
must carry an observation timestamp no older than 0.25s. Missing, stale or excessive
support only allows sufficient analytic whole-sphere containment, never mere
intersection or a sparse sample test. Validation timeout preserves the candidate.

New support topics are `/rmp_camera/dynamic_obstacle_sphere_cloud/support` and
`/rmp_camera/human_sphere_cloud/support` (`PointCloud2`, fields x/y/z/sphere_index/
evidence_stamp). They are functional inputs when the guard is enabled, not debug
visualizations; existing sphere cloud layouts and marker topics stay unchanged.
They and `/rmp_camera/combined_obstacle_sphere_cloud/status` are included in debug
bags. Fusion diagnostics report checked support, removed duplicates, retained
partial overlaps, missing support, computation time and output-cap violations.

This can intentionally retain more spheres than the old overlap-only deletion.
Two bounding spheres may visibly intersect while covering different real objects.
The guard preserves *previously represented retained support*, not unseen surfaces
or voxels already removed upstream. The existing robot post-filter still follows
Fusion. The configured `fusion_max_total_spheres` remains a hard cap; if it forces
truncation, diagnostics explicitly set `coverage_valid=False` and warn. No coverage
guarantee is made in that overflow case. The measured bag remains well below it.

Headless A/B replay supports `--latest-input`, `--component-tracking` and
`--coverage-guard` with `true|false`, all defaulting to `false` for controlled old
path comparisons. Keep other flags, raw bag topics and playback rate identical.

### Local-width sphere reconstruction

The experiment now enables `local_width_sphere_cover:=true` for Static, Dynamic
and Human. Set it to `false` to restore the prior geometry path independently
of asynchronous Minimum-K, component tracking and Fusion. Restart the launch.

This is a bounded **voxel-based reconstruction pass**, not anatomical tracking:

1. Partition the current component along its longest principal direction.
2. Estimate visible local width from the second principal extent. Tighten that
   estimate at each candidate center so a torso does not lend its width to a
   distal arm. Long narrow parts produce chains of smaller candidate balls.
3. Try larger supported candidates first, then smaller scales if the occupancy
   guard rejects them. Include surface-supported centers for partial depth views.
4. Select a smaller cover from new candidates and the previous solution. Keep
   **every voxel center covered by the previous solution**, with the same raw
   radius coverage tolerance. Thus coverage cannot regress in a local hand/arm
   region just because a torso dominates the total voxel count.
5. Publish a replacement only if K strictly decreases and overlap checks pass.
   Otherwise keep the previous solution, including on deadline or size limits.

With `integrated_sphere_cover:=true` (experiment default), Dynamic/Human run
local selection after seed/greedy cover and before a single refinement stage,
reusing the existing coverage rows. `false` retains the original post-refinement
pass. Static still uses the post-refinement pass. Support preservation is relative
to the current input cover at that stage, not equality with a differently ordered
full solve. Tracking still follows refinement.
Existing smoothing, TTL, robot masks, small-component floors and Fusion remain
unchanged. This is not immediate deletion of unsupported old tracks. Regional
preservation applies at reconstruction time, not to subsequent smoothing,
intentional robot/Human filtering or a hard-cap overflow.

Dynamic/Human new candidates use `dynamic_local_width_ratio: 1.4` times visible
half-width, limited by the **unchanged absolute** `dynamic_max_raw_radius_m`
and `dynamic_merge_max_radius_m`. This local cap replaces the old whole-component
`dynamic_component_radius_scale * voxel_size * cbrt(N)` heuristic **only for new
local-width candidates** and their validated merges/reuse. Candidate provenance
is internal; the published cloud fields do not change. Generation, merging and
background reuse recheck the same current local-width, absolute-radius and
occupancy constraints. Original candidates retain their old constraints.
The existing empty-space guard still rejects unsupported enlarged balls; no
hidden thickness or free-space occupancy is invented. Pair-overlap feasibility
is preserved if the old solution was feasible, and total pair overlap cannot
increase. A low overlap threshold can therefore limit how much K can fall.

Static uses `local_width_ratio: 1.4`, the absolute raw/coarse/merge radius caps,
the existing coarse density limit and the stricter coarse/merge ESDF free-space
limit. It preserves the old voxel support, including previously covered shell
voxels. Unknown ESDF surface samples are not treated as occupied evidence.

Additional **soft, shared per-update** budgets:

- Dynamic/Human: `dynamic_local_width_budget_ms: 5.0`.
- Static: `local_width_budget_ms: 15.0`.

Work also has limits of 8192 component voxels, 64 old spheres, 64 new candidates
and 500000 coverage-matrix elements. A bounded NumPy operation or final overlap
check can finish after the soft deadline; these are not hard real-time bounds.
The pass adds CPU work and does not promise lower end-to-end latency or a global
minimum K. It cannot guarantee one sphere per hand or three per torso under all
radius, free-space and coverage constraints.

Dynamic/Human status topics add `local_width_saved`, `local_width_applied`,
`local_width_ms`, `local_width_reasons`. Static `/rmp_camera/static_sphere_result_status`
and `/rosout` include reconstruction statistics. Debug manifests record the launch flag.
`scripts/benchmark_local_width_cover.py` compares identical captured frames with
exact old-support preservation assertions; headless replay accepts
`--local-width-cover true|false` (default false for explicit A/B comparisons).

Integrated Dynamic/Human also memoize unsuccessful extra local searches for at
most 0.25 wall seconds: same seed count, at least 90% relative voxel similarity,
and at most 0.30m translation. `recent_no_gain` means **fresh seed generation and
refinement still run**, not that old obstacle positions were published. Large
shape/count changes, disappearance or source clock rewind retry; deadline and
incomplete searches are not memoized. No immediate unsupported-track deletion
was added. This is a bounded CPU/optimization-quality tradeoff, not a global-K proof.

### Experimental current-depth sphere validation

`sphere_observation_guard:=off` is the default and creates no extra depth
subscription. `audit` compares density decisions with coherent registered depth
without replacing the acceptance decision. It does consume the bounded search
budget, so deadline-driven candidate choices can differ. `observed` is an explicit
experimental alternative for local candidates/reuse; it is **not enabled by default**.
This is not a general ghost-removal filter and does not modify input/map voxels.

Depth must be within 50ms of the current cloud, within 250ms of ROS and receipt
time, and have valid matching rectified intrinsics and a timestamped TF available
without waiting. Invalid or missing evidence falls back to density, never stalls
or drops the cloud. Zero depth and occluded rays are not labeled free. An
alternative acceptance additionally needs predominantly matching current-component
surface endpoints, bounded occlusion, and current voxel support. No ray snapshot
is queued in the background worker. Static ESDF validation is unchanged.

Dynamic/Human status adds `ray_evidence_available`, `ray_candidates`,
`ray_supported`, `ray_free_rejected`, `ray_alternative_accept/reject` when enabled.
Use headless replay `--integrated-cover true|false --observation-guard off|audit|observed`
and `scripts/summarize_integrated_cover.py` for measured comparisons. Audit/one bag
cannot establish ghost-free operation; validate independent motions/occlusions
before enabling the observed alternative.
