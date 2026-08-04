import numpy as np

from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicSphere,
    DynamicSphereParameters,
    DynamicSphereTracker,
    _remove_redundant,
    connected_components,
    generate_dynamic_spheres,
    voxel_centers,
)


def params(**overrides):
    values = dict(
        voxel_size_m=0.1,
        min_component_voxels=1,
        max_component_voxels=10000,
        max_components=20,
        max_input_points=10000,
        connectivity=18,
        dilation_voxels=0,
        closing_iterations=0,
        minimum_center_spacing_m=0.05,
        min_raw_radius_m=0.02,
        max_raw_radius_m=0.5,
        min_useful_adaptive_radius_m=0.05,
        enable_fixed_radius_fallback=True,
        fixed_radius_m=0.14,
        target_coverage=0.9,
        coverage_tolerance_m=0.02,
        safety_margin_m=0.01,
        redundancy_tolerance_m=0.001,
        max_spheres_per_component=100,
        max_iterations_per_component=500,
        max_total_spheres=200,
        processing_budget_ms=1000.0,
        max_local_grid_voxels=100000,
        min_x_m=-1.0, max_x_m=3.0,
        min_y_m=-1.0, max_y_m=3.0,
        min_z_m=-1.0, max_z_m=3.0,
        max_range_from_base_m=0.0,
    )
    values.update(overrides)
    return DynamicSphereParameters(**values)


def points_for(indices, p=None):
    p = p or params()
    return voxel_centers(np.asarray(indices, dtype=np.int64), p)


def box(nx, ny, nz, offset=(0, 0, 0)):
    return [
        (x + offset[0], y + offset[1], z + offset[2])
        for x in range(nx) for y in range(ny) for z in range(nz)
    ]


def signature(result):
    return [
        tuple(round(value, 8) for value in (
            sphere.x, sphere.y, sphere.z, sphere.raw_radius,
            sphere.output_radius, sphere.component_coverage))
        for sphere in result.spheres
    ]


def test_solid_cube_generates_adaptive_spheres():
    result = generate_dynamic_spheres(points_for(box(5, 5, 5)), params())
    assert result.spheres
    assert result.components[0].coverage >= 0.9
    assert max(s.raw_radius for s in result.spheres) >= 0.2
    assert all(s.output_radius > s.raw_radius for s in result.spheres)


def test_long_box_reaches_coverage_with_multiple_spheres():
    result = generate_dynamic_spheres(points_for(box(12, 3, 3)), params())
    assert len(result.spheres) >= 2
    assert result.components[0].coverage >= 0.9


def test_separated_components_remain_separate():
    indices = box(2, 2, 2) + box(2, 2, 2, (8, 0, 0))
    result = generate_dynamic_spheres(points_for(indices), params())
    assert len(result.components) == 2
    assert {s.component_id for s in result.spheres} == {0, 1}


def test_corner_only_voxels_are_separate_with_18_connectivity():
    components = connected_components(np.asarray([(0, 0, 0), (1, 1, 1)]), 18)
    assert [len(component) for component in components] == [1, 1]


def test_face_and_edge_voxels_connect_with_18_connectivity():
    indices = np.asarray([(0, 0, 0), (1, 0, 0), (2, 1, 0)])
    assert len(connected_components(indices, 18)) == 1


def test_small_noise_component_is_removed():
    indices = box(2, 2, 2) + [(12, 12, 12)]
    p = params(min_component_voxels=2, max_x_m=3.0, max_y_m=3.0, max_z_m=3.0)
    result = generate_dynamic_spheres(points_for(indices, p), p)
    assert len(result.components) == 1
    assert len(result.voxel_centers) == 8


def test_thin_shell_uses_fixed_radius_fallback():
    shell = [(x, y, 0) for x in range(5) for y in range(5)]
    p = params(min_useful_adaptive_radius_m=0.25, fixed_radius_m=0.13)
    result = generate_dynamic_spheres(points_for(shell, p), p)
    assert result.spheres
    assert result.components[0].termination_reason == "thin_component_fallback"
    assert all(np.isclose(s.raw_radius, 0.13) for s in result.spheres)


def test_target_coverage_is_reported_and_uncovered_matches():
    result = generate_dynamic_spheres(points_for(box(7, 4, 3)), params(target_coverage=0.95))
    component = result.components[0]
    assert component.coverage >= 0.95
    assert np.isclose(
        component.coverage,
        1.0 - len(component.uncovered_voxels) / len(component.voxel_centers))


def test_redundancy_removal_removes_duplicate_sphere():
    p = params(target_coverage=1.0)
    centers = np.asarray([[0.0, 0.0, 0.0]])
    sphere = DynamicSphere(0.0, 0.0, 0.0, 0.2, 0.19)
    assert len(_remove_redundant(centers, [sphere, sphere], p)) == 1


def test_sphere_limits_are_enforced():
    p = params(
        target_coverage=1.0, fixed_radius_m=0.05,
        min_useful_adaptive_radius_m=1.0,
        max_spheres_per_component=2, max_total_spheres=2)
    result = generate_dynamic_spheres(points_for(box(12, 1, 1), p), p)
    assert len(result.spheres) == 2


def test_generation_is_deterministic():
    points = points_for(box(6, 5, 3))
    assert signature(generate_dynamic_spheres(points, params())) == signature(
        generate_dynamic_spheres(points[::-1], params()))


def test_empty_input_returns_empty_result():
    result = generate_dynamic_spheres(np.empty((0, 3)), params())
    assert not result.spheres
    assert result.termination_reason == "empty_input"


def test_oversized_component_uses_bounded_fallback():
    p = params(max_component_voxels=5, max_spheres_per_component=4)
    result = generate_dynamic_spheres(points_for(box(3, 3, 3), p), p)
    assert result.components[0].termination_reason == "oversized_component_fallback"
    assert 0 < len(result.spheres) <= 4


def test_zero_processing_budget_uses_fallback():
    p = params(processing_budget_ms=0.0)
    result = generate_dynamic_spheres(points_for(box(3, 3, 3), p), p)
    assert result.components[0].termination_reason == "processing_budget_fallback"
    assert result.spheres


def detection(x, radius=0.1):
    return DynamicSphere(x, 0.0, 0.0, radius, radius, confidence=0.5)


def test_tracker_preserves_id_for_small_motion():
    tracker = DynamicSphereTracker(0.3, 1.0, 1.0, 5)
    first = tracker.update([detection(0.0)], 0.0)
    second = tracker.update([detection(0.1)], 0.1)
    assert first[0].track_id == second[0].track_id


def test_tracker_creates_new_id_for_large_motion():
    tracker = DynamicSphereTracker(0.2, 1.0, 1.0, 5)
    first_id = tracker.update([detection(0.0)], 0.0)[0].track_id
    ids = {sphere.track_id for sphere in tracker.update([detection(1.0)], 0.1)}
    assert first_id in ids
    assert len(ids) == 2


def test_tracker_holds_short_occlusion_then_expires_by_ttl():
    tracker = DynamicSphereTracker(0.2, 1.0, 0.5, 10)
    track_id = tracker.update([detection(0.0)], 0.0)[0].track_id
    assert tracker.update([], 0.2)[0].track_id == track_id
    assert tracker.update([], 0.6) == []


def test_tracker_expires_by_missed_update_limit():
    tracker = DynamicSphereTracker(0.2, 1.0, 10.0, 1)
    tracker.update([detection(0.0)], 0.0)
    assert tracker.update([], 0.1)
    assert tracker.update([], 0.2) == []


def test_tracker_exponential_smoothing():
    tracker = DynamicSphereTracker(2.0, 0.25, 1.0, 5)
    tracker.update([detection(0.0, 0.1)], 0.0)
    result = tracker.update([detection(1.0, 0.5)], 0.1)[0]
    assert np.isclose(result.x, 0.25)
    assert np.isclose(result.raw_radius, 0.2)
