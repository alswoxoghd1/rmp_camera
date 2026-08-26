from time import monotonic

import numpy as np

from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicSphere,
    DynamicSphereParameters,
    DynamicSphereTracker,
    agglomerative_merge_dynamic_spheres,
    build_dynamic_sphere_search_state,
    _greedy_set_cover,
    _remove_redundant,
    _sphere_coverage_masks,
    connected_components,
    dynamic_sphere_search_state_rank_key,
    generate_dynamic_spheres,
    minimum_k_dynamic_sphere_search,
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
        enable_greedy_set_cover=True,
        enable_single_sphere_replacement=True,
        single_sphere_max_radius_m=0.18,
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
    p = params(
        min_useful_adaptive_radius_m=0.25, fixed_radius_m=0.13,
        enable_greedy_set_cover=False, enable_single_sphere_replacement=False)
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


def test_greedy_set_cover_removes_duplicate_candidates():
    centers = np.asarray([[float(x), 0.0, 0.0] for x in range(6)])
    spheres = [
        DynamicSphere(1.5, 0.0, 0.0, 1.51, 1.51),
        DynamicSphere(1.5, 0.0, 0.0, 1.51, 1.51),
        DynamicSphere(4.5, 0.0, 0.0, 0.51, 0.51),
    ]
    masks = _sphere_coverage_masks(centers, spheres, 0.0)
    selected = _greedy_set_cover(spheres, masks, 1.0)
    assert selected == [0, 2]
    assert np.all(np.any(masks[selected], axis=0))


def test_compact_thin_component_is_replaced_by_one_sphere():
    indices = box(4, 2, 1)
    common = dict(
        min_useful_adaptive_radius_m=0.25,
        fixed_radius_m=0.09,
        target_coverage=0.9,
        single_sphere_max_radius_m=0.18,
    )
    legacy_params = params(
        **common, enable_greedy_set_cover=False,
        enable_single_sphere_replacement=False)
    optimized_params = params(**common)
    before = generate_dynamic_spheres(points_for(indices, legacy_params), legacy_params)
    after = generate_dynamic_spheres(points_for(indices, optimized_params), optimized_params)

    assert len(before.spheres) > 1
    assert len(after.spheres) == 1
    assert after.components[0].coverage >= optimized_params.target_coverage
    assert after.spheres[0].raw_radius <= optimized_params.single_sphere_max_radius_m


def test_elongated_component_is_not_forced_into_one_sphere():
    indices = box(10, 1, 1)
    p = params(
        min_useful_adaptive_radius_m=0.25,
        fixed_radius_m=0.09,
        target_coverage=0.9,
        single_sphere_max_radius_m=0.18,
    )
    result = generate_dynamic_spheres(points_for(indices, p), p)
    assert len(result.spheres) > 1
    assert result.components[0].coverage >= p.target_coverage


def test_optimization_features_can_be_disabled():
    indices = box(4, 2, 1)
    p = params(
        min_useful_adaptive_radius_m=0.25,
        fixed_radius_m=0.09,
        target_coverage=0.9,
        enable_greedy_set_cover=False,
        enable_single_sphere_replacement=False,
    )
    result = generate_dynamic_spheres(points_for(indices, p), p)
    assert result.spheres
    assert len(result.spheres) > 1
    assert result.components[0].termination_reason == "thin_component_fallback"
    assert result.components[0].coverage >= p.target_coverage


def test_optimized_generation_is_deterministic_for_reversed_input():
    indices = box(4, 2, 1)
    p = params(min_useful_adaptive_radius_m=0.25, fixed_radius_m=0.09)
    points = points_for(indices, p)
    assert signature(generate_dynamic_spheres(points, p)) == signature(
        generate_dynamic_spheres(points[::-1], p))


def test_output_radius_is_exactly_raw_radius_plus_safety_margin():
    p = params(safety_margin_m=0.037)
    result = generate_dynamic_spheres(points_for(box(5, 3, 2), p), p)
    assert result.spheres
    assert all(np.isclose(
        sphere.output_radius, sphere.raw_radius + p.safety_margin_m)
        for sphere in result.spheres)


def test_optimization_preserves_coverage_and_uncovered_count():
    indices = box(4, 2, 1)
    common = dict(
        min_useful_adaptive_radius_m=0.25,
        fixed_radius_m=0.09,
        target_coverage=0.9,
    )
    legacy_params = params(
        **common, enable_greedy_set_cover=False,
        enable_single_sphere_replacement=False)
    optimized_params = params(**common)
    before = generate_dynamic_spheres(points_for(indices, legacy_params), legacy_params)
    after = generate_dynamic_spheres(points_for(indices, optimized_params), optimized_params)

    for result, configured in (
            (before, legacy_params), (after, optimized_params)):
        component = result.components[0]
        assert component.coverage >= configured.target_coverage
        assert np.isclose(
            component.coverage,
            1.0 - len(component.uncovered_voxels) / len(component.voxel_centers))


def test_single_sphere_radius_validation():
    for invalid in (0.0, -0.1):
        try:
            params(single_sphere_max_radius_m=invalid).validate()
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive single-sphere limit must fail")
    try:
        params(
            min_raw_radius_m=0.2,
            single_sphere_max_radius_m=0.1).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("single-sphere limit below minimum raw radius must fail")


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


def merge_dynamic_sphere(x, radius=0.10):
    return DynamicSphere(x, 0.0, 0.0, radius, radius + 0.01, 0)


def run_dynamic_merge(centers, spheres, **overrides):
    values = dict(
        dynamic_merge_max_radius_m=0.30,
        dynamic_merge_max_radius_growth_ratio=1.50,
        dynamic_merge_max_gap_m=0.08)
    values.update(overrides)
    configured = params(**values)
    return agglomerative_merge_dynamic_spheres(
        np.asarray(centers, dtype=np.float64), spheres, configured,
        monotonic() + 1.0)


def test_dynamic_agglomerative_merge_reduces_compact_set_and_keeps_coverage():
    centers = np.asarray(((0.0, 0.0, 0.0), (0.08, 0.0, 0.0),
                          (0.16, 0.0, 0.0)))
    spheres = [merge_dynamic_sphere(x) for x in (0.0, 0.08, 0.16)]
    before = _sphere_coverage_masks(centers, spheres, 0.02).any(axis=0)
    merged = run_dynamic_merge(centers, spheres)
    after = _sphere_coverage_masks(centers, merged, 0.02).any(axis=0)
    assert len(merged) == 1
    assert np.count_nonzero(after) >= np.count_nonzero(before)
    assert np.isclose(merged[0].output_radius, merged[0].raw_radius + 0.01)


def test_dynamic_elongated_set_does_not_collapse_into_giant_sphere():
    coordinates = tuple(0.08 * index for index in range(6))
    centers = np.asarray([(x, 0.0, 0.0) for x in coordinates])
    spheres = [merge_dynamic_sphere(x, 0.08) for x in coordinates]
    merged = run_dynamic_merge(
        centers, spheres, dynamic_merge_max_radius_m=0.20)
    assert 1 < len(merged) < len(spheres)
    assert max(s.raw_radius for s in merged) <= 0.20 + 1e-12


def test_dynamic_radius_growth_and_gap_guards_reject_pairs():
    cases = (
        (((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)),
         (merge_dynamic_sphere(0.0), merge_dynamic_sphere(0.08)),
         {"dynamic_merge_max_radius_m": 0.13}),
        (((0.0, 0.0, 0.0), (0.10, 0.0, 0.0)),
         (merge_dynamic_sphere(0.0), merge_dynamic_sphere(0.10)),
         {"dynamic_merge_max_radius_m": 1.0,
          "dynamic_merge_max_radius_growth_ratio": 1.40}),
        (((0.0, 0.0, 0.0), (0.20, 0.0, 0.0)),
         (merge_dynamic_sphere(0.0, 0.05), merge_dynamic_sphere(0.20, 0.05)),
         {"dynamic_merge_max_radius_m": 1.0,
          "dynamic_merge_max_radius_growth_ratio": 10.0,
          "dynamic_merge_max_gap_m": 0.05}),
    )
    for centers, spheres, overrides in cases:
        assert len(run_dynamic_merge(centers, spheres, **overrides)) == 2


def test_dynamic_merge_respects_expired_deadline_and_returns_valid_input():
    centers = np.asarray(((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)))
    spheres = [merge_dynamic_sphere(0.0), merge_dynamic_sphere(0.08)]
    assert agglomerative_merge_dynamic_spheres(
        centers, spheres, params(), monotonic()) == spheres


def test_dynamic_optional_empty_space_guard_rejects_sparse_merge():
    centers = np.asarray(((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)))
    spheres = [merge_dynamic_sphere(0.0), merge_dynamic_sphere(0.08)]
    assert len(run_dynamic_merge(centers, spheres)) == 1
    assert len(run_dynamic_merge(
        centers, spheres,
        dynamic_merge_enable_empty_space_guard=True,
        dynamic_merge_max_empty_fraction=0.10)) == 2


def test_dynamic_feature_off_keeps_pre_merge_geometry():
    common = dict(
        dynamic_enable_agglomerative_merge=False,
        dynamic_enable_min_k_search=False,
        min_useful_adaptive_radius_m=0.25,
        fixed_radius_m=0.09,
        enable_single_sphere_replacement=False,
        enable_greedy_set_cover=False,
        processing_budget_ms=1000.0)
    disabled = params(**common)
    common.update(
        dynamic_enable_agglomerative_merge=True,
        dynamic_merge_max_radius_m=0.01)
    blocked = params(**common)
    points = points_for(box(5, 2, 1), disabled)
    disabled_result = generate_dynamic_spheres(points, disabled)
    blocked_result = generate_dynamic_spheres(points, blocked)
    assert signature(disabled_result) == signature(blocked_result)
    assert disabled_result.components[0].pre_merge_sphere_count == len(
        disabled_result.spheres)


def test_dynamic_merge_geometry_is_deterministic_for_reversed_order():
    centers = np.asarray([(x, 0.0, 0.0) for x in (0.0, 0.08, 0.16, 0.24)])
    spheres = [merge_dynamic_sphere(x) for x in (0.0, 0.08, 0.16, 0.24)]
    def geometry(values):
        return [tuple(round(v, 12) for v in (
            s.x, s.y, s.z, s.raw_radius, s.output_radius)) for s in values]
    assert geometry(run_dynamic_merge(centers, spheres)) == geometry(
        run_dynamic_merge(centers, spheres[::-1]))


def test_dynamic_merge_parameter_validation():
    for overrides in (
        {"dynamic_merge_max_radius_m": 0.0},
        {"dynamic_merge_max_radius_growth_ratio": 0.9},
        {"dynamic_merge_max_gap_m": -0.1},
        {"dynamic_merge_max_empty_fraction": 1.1},
        {"dynamic_merge_max_validation_voxels": 0},
    ):
        try:
            params(**overrides).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid merge parameters accepted: {overrides}")


def run_min_k_search(centers, spheres, **overrides):
    values = dict(
        target_coverage=1.0,
        coverage_tolerance_m=0.0,
        dynamic_merge_max_radius_m=0.30,
        dynamic_merge_max_radius_growth_ratio=2.0,
        dynamic_merge_max_gap_m=0.08,
        dynamic_min_k_enable_overlap_constraint=False,
    )
    values.update(overrides)
    configured = params(**values)
    return minimum_k_dynamic_sphere_search(
        np.asarray(centers, dtype=np.float64),
        spheres,
        configured,
        monotonic() + 1.0,
    )


def test_overlap_aware_state_ranking_prefers_lower_overlap_at_equal_k():
    configured = params(dynamic_min_k_use_output_overlap=True)
    high_overlap = build_dynamic_sphere_search_state([
        DynamicSphere(0.0, 0.0, 0.0, 0.10, 0.11, 0),
        DynamicSphere(0.08, 0.0, 0.0, 0.10, 0.11, 0),
    ])
    low_overlap = build_dynamic_sphere_search_state([
        DynamicSphere(0.0, 0.0, 0.0, 0.10, 0.11, 0),
        DynamicSphere(0.22, 0.0, 0.0, 0.10, 0.11, 0),
    ])
    assert len(high_overlap.spheres) == len(low_overlap.spheres)
    coverage_points = np.asarray(((0.0, 0.0, 0.0),))
    high_coverage = _sphere_coverage_masks(
        coverage_points, high_overlap.spheres, 0.0).any(axis=0)
    low_coverage = _sphere_coverage_masks(
        coverage_points, low_overlap.spheres, 0.0).any(axis=0)
    assert np.array_equal(high_coverage, low_coverage)
    assert high_coverage.all()

    assert dynamic_sphere_search_state_rank_key(
        low_overlap, configured
    ) < dynamic_sphere_search_state_rank_key(high_overlap, configured)


def test_min_k_search_reduces_compact_set_and_preserves_coverage():
    centers = np.asarray((
        (0.0, 0.0, 0.0),
        (0.08, 0.0, 0.0),
        (0.16, 0.0, 0.0),
    ))
    spheres = [merge_dynamic_sphere(x) for x in (0.0, 0.08, 0.16)]
    before = _sphere_coverage_masks(centers, spheres, 0.0).any(axis=0)
    result = run_min_k_search(centers, spheres)
    after = _sphere_coverage_masks(
        centers, result.spheres, 0.0).any(axis=0)
    assert len(result.spheres) == 1
    assert np.count_nonzero(after) >= np.count_nonzero(before)
    assert result.overlap_constraint_satisfied


def test_bounded_search_escapes_greedy_merge_order_trap():
    coordinates = (
        0.0,
        0.1441209636081885,
        0.2344194989337105,
        0.3537910818681569,
    )
    radii = (
        0.07167951434830674,
        0.09462533464404828,
        0.0821887855824767,
        0.06762221353458232,
    )
    centers = np.asarray([(x, 0.0, 0.0) for x in coordinates])
    spheres = [
        DynamicSphere(x, 0.0, 0.0, radius, radius + 0.01, 0)
        for x, radius in zip(coordinates, radii)
    ]
    configured = params(
        target_coverage=1.0,
        coverage_tolerance_m=0.0,
        dynamic_merge_max_radius_m=0.17680082144659784,
        dynamic_merge_max_radius_growth_ratio=1.6856652026068732,
        dynamic_merge_max_gap_m=0.058939303977189085,
        dynamic_min_k_enable_overlap_constraint=False,
    )
    greedy = agglomerative_merge_dynamic_spheres(
        centers, spheres, configured, monotonic() + 1.0)
    bounded = minimum_k_dynamic_sphere_search(
        centers, spheres, configured, monotonic() + 1.0)
    assert len(greedy) == 3
    assert len(bounded.spheres) == 2
    assert len(bounded.spheres) < len(greedy)


def triangle_spheres(side):
    coordinates = (
        (0.0, 0.0, 0.0),
        (side, 0.0, 0.0),
        (0.5 * side, np.sqrt(3.0) * 0.5 * side, 0.0),
    )
    spheres = [
        DynamicSphere(x, y, z, 0.10, 0.11, 0)
        for x, y, z in coordinates
    ]
    return np.asarray(coordinates), spheres


def test_overlap_constraint_keeps_more_spheres_when_smaller_k_is_unsafe():
    centers, spheres = triangle_spheres(0.22)
    common = dict(
        dynamic_merge_max_radius_m=0.22,
        dynamic_merge_max_radius_growth_ratio=3.0,
        dynamic_merge_max_gap_m=0.05,
        dynamic_min_k_use_output_overlap=False,
    )
    unconstrained = run_min_k_search(
        centers,
        spheres,
        **common,
        dynamic_min_k_enable_overlap_constraint=False,
    )
    constrained = run_min_k_search(
        centers,
        spheres,
        **common,
        dynamic_min_k_enable_overlap_constraint=True,
        dynamic_max_allowed_overlap_fraction=0.05,
    )
    assert len(unconstrained.spheres) == 2
    assert unconstrained.state.max_raw_overlap_fraction > 0.05
    assert len(constrained.spheres) == 3
    assert constrained.state.max_raw_overlap_fraction <= 0.05
    assert constrained.overlap_constraint_satisfied


def test_impossible_overlap_constraint_returns_lowest_violation_fallback():
    centers, spheres = triangle_spheres(0.18)
    before = _sphere_coverage_masks(centers, spheres, 0.0).any(axis=0)
    result = run_min_k_search(
        centers,
        spheres,
        dynamic_merge_max_radius_m=0.20,
        dynamic_merge_max_radius_growth_ratio=3.0,
        dynamic_merge_max_gap_m=0.05,
        dynamic_min_k_use_output_overlap=False,
        dynamic_min_k_enable_overlap_constraint=True,
        dynamic_max_allowed_overlap_fraction=0.0,
    )
    after = _sphere_coverage_masks(
        centers, result.spheres, 0.0).any(axis=0)
    assert result.spheres
    assert np.count_nonzero(after) >= np.count_nonzero(before)
    assert not result.overlap_constraint_satisfied
    assert result.termination_reason == "overlap_constraint_unmet"
    assert len(result.spheres) == 3


def test_min_k_expired_deadline_returns_valid_coverage_fallback():
    centers = np.asarray((
        (0.0, 0.0, 0.0),
        (0.08, 0.0, 0.0),
        (0.16, 0.0, 0.0),
    ))
    spheres = [merge_dynamic_sphere(x) for x in (0.0, 0.08, 0.16)]
    configured = params(
        target_coverage=1.0,
        coverage_tolerance_m=0.0,
        dynamic_min_k_enable_overlap_constraint=False,
    )
    before = _sphere_coverage_masks(centers, spheres, 0.0).any(axis=0)
    result = minimum_k_dynamic_sphere_search(
        centers, spheres, configured, monotonic())
    after = _sphere_coverage_masks(
        centers, result.spheres, 0.0).any(axis=0)
    assert result.termination_reason == "deadline"
    assert result.states_explored == 0
    assert np.count_nonzero(after) >= np.count_nonzero(before)


def test_min_k_max_states_is_a_hard_bound():
    coordinates = (0.0, 0.08, 0.16, 0.24)
    centers = np.asarray([(x, 0.0, 0.0) for x in coordinates])
    spheres = [merge_dynamic_sphere(x) for x in coordinates]
    result = run_min_k_search(
        centers,
        spheres,
        dynamic_min_k_max_states=1,
    )
    assert result.states_explored == 1
    assert result.termination_reason == "max_states"


def test_min_k_search_is_deterministic_for_reversed_sphere_order():
    coordinates = (0.0, 0.08, 0.16, 0.24)
    centers = np.asarray([(x, 0.0, 0.0) for x in coordinates])
    spheres = [merge_dynamic_sphere(x) for x in coordinates]
    forward = run_min_k_search(centers, spheres)
    reversed_result = run_min_k_search(centers, spheres[::-1])
    assert (
        forward.state.canonical_signature
        == reversed_result.state.canonical_signature
    )
    assert np.isclose(
        forward.state.max_output_overlap_fraction,
        reversed_result.state.max_output_overlap_fraction,
    )
    assert np.isclose(
        forward.state.total_output_overlap,
        reversed_result.state.total_output_overlap,
    )


def test_min_k_elongated_set_respects_existing_radius_guard():
    coordinates = tuple(0.08 * index for index in range(6))
    centers = np.asarray([(x, 0.0, 0.0) for x in coordinates])
    spheres = [merge_dynamic_sphere(x, 0.08) for x in coordinates]
    result = run_min_k_search(
        centers,
        spheres,
        dynamic_merge_max_radius_m=0.20,
    )
    assert 1 < len(result.spheres) < len(spheres)
    assert max(
        sphere.raw_radius for sphere in result.spheres
    ) <= 0.20 + 1e-12


def test_min_k_parameter_validation():
    for overrides in (
        {"dynamic_min_k_beam_width": 0},
        {"dynamic_min_k_max_states": 0},
        {"dynamic_max_allowed_overlap_fraction": -0.01},
        {"dynamic_max_allowed_overlap_fraction": 1.01},
    ):
        try:
            params(**overrides).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid min-k parameters accepted: {overrides}")


def test_min_k_generation_is_deterministic_for_shuffled_points():
    configured = params(processing_budget_ms=1000.0)
    points = points_for(box(6, 4, 2), configured)
    shuffled = points.copy()
    np.random.default_rng(2026).shuffle(shuffled)
    results = (
        generate_dynamic_spheres(points, configured),
        generate_dynamic_spheres(points[::-1], configured),
        generate_dynamic_spheres(shuffled, configured),
    )
    assert signature(results[0]) == signature(results[1]) == signature(results[2])
    metrics = [
        (
            result.components[0].max_raw_overlap_fraction,
            result.components[0].max_output_overlap_fraction,
            result.components[0].total_raw_overlap,
            result.components[0].total_output_overlap,
        )
        for result in results
    ]
    assert np.allclose(metrics[0], metrics[1])
    assert np.allclose(metrics[0], metrics[2])


def test_min_k_search_reuses_optional_empty_space_guard():
    centers = np.asarray(((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)))
    spheres = [merge_dynamic_sphere(0.0), merge_dynamic_sphere(0.08)]
    unguarded = run_min_k_search(centers, spheres)
    guarded = run_min_k_search(
        centers,
        spheres,
        dynamic_merge_enable_empty_space_guard=True,
        dynamic_merge_max_empty_fraction=0.10,
    )
    assert len(unguarded.spheres) == 1
    assert len(guarded.spheres) == 2
