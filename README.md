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
- `/rmp_camera/esdf_medial_sphere_cloud` (`sensor_msgs/PointCloud2`)
- `/rmp_camera/esdf_medial_inside_voxels` (`sensor_msgs/PointCloud2`)
- `/rmp_camera/esdf_medial_uncovered_voxels` (`sensor_msgs/PointCloud2`)

The main tuning parameters are the AABB minimum/size values,
`inside_epsilon_m`, `target_coverage`, `coverage_tolerance_m`,
`plateau_epsilon_m`, `minimum_center_spacing_m`, `min_component_voxels`,
`min_raw_sphere_radius_m`, `safety_margin_m`, and the per-component/total
sphere limits. See `esdf_medial_sphere_node.py` for units and tuning
consequences.

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

The fusion node does not exact-sync different-rate inputs. Static output is
kept without a normal TTL and requires consecutive empty confirmations to
clear. Dynamic output has both a short occlusion hold and an absolute ghost
limit; after a moving object stops, its stale dynamic sphere remains until an
overlapping static sphere takes over. All overlap decisions use actual 3D
center distance and output radii.

Run all ROS-independent core tests with:

```bash
python3 -m pytest -q \
  test/test_esdf_medial_sphere_core.py \
  test/test_dynamic_obstacle_sphere_core.py \
  test/test_obstacle_sphere_fusion_core.py
```
