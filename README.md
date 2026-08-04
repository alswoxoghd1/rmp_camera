# camera_rmp

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
