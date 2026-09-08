"""ROS-free unit tests for dense signed-ESDF medial spheres."""

from time import monotonic

import numpy as np

from rmp_camera.esdf_medial_sphere_core import (
    Sphere,
    agglomerative_merge_static_spheres,
    add_spheres_until_coverage,
    calculate_component_coverage,
    calculate_mask_coverage,
    connected_components_18,
    create_initial_spheres,
    extract_inside_mask,
    find_local_minimum_candidates,
    general_coverage_pruning,
    generate_medial_spheres,
    merged_sphere_passes_esdf_guard,
    greedy_set_cover_spheres,
    group_or_reduce_plateaus,
    make_surface_shell_mask,
    make_neighbor_offsets_18,
    optimize_component_spheres,
    minimum_k_static_sphere_search,
    prune_overlapping_static_spheres,
    remove_redundant_spheres,
    sphere_coverage_masks,
    static_component_robot_overlap_fraction,
    static_voxel_robot_sphere_overlap_mask,
)


def signed_box_esdf(
    shape,
    box_min_m,
    box_max_m,
    voxel_size_m=1.0,
    origin_m=(0.0, 0.0, 0.0),
):
    """Return an axis-aligned box's Euclidean signed distance at voxel centres."""

    indices = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0)
    points = np.asarray(origin_m) + (indices + 0.5) * voxel_size_m
    box_min = np.asarray(box_min_m, dtype=np.float64)
    box_max = np.asarray(box_max_m, dtype=np.float64)
    center = 0.5 * (box_min + box_max)
    half_extent = 0.5 * (box_max - box_min)
    q = np.abs(points - center) - half_extent
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def make_sphere(index, radius, component_id=0, voxel_size_m=1.0):
    center = (np.asarray(index, dtype=np.float64) + 0.5) * voxel_size_m
    return Sphere(center, radius, radius, component_id, tuple(index))


def sphere_signature(result):
    return [
        (
            sphere.component_id,
            sphere.source_index,
            round(sphere.raw_radius, 12),
            tuple(np.round(sphere.center, 12)),
        )
        for sphere in result.spheres
    ]


def test_neighbor_offsets_are_exactly_the_18_face_and_edge_neighbors():
    offsets = make_neighbor_offsets_18()
    assert len(offsets) == 18
    assert len(set(offsets)) == 18
    assert all(0 < sum(abs(value) for value in offset) <= 2 for offset in offsets)


def test_face_touching_voxels_share_component():
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[0, 0, 0] = mask[1, 0, 0] = True
    assert [len(component) for component in connected_components_18(mask)] == [2]


def test_edge_touching_voxels_share_component():
    mask = np.zeros((2, 2, 1), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 0] = True
    assert [len(component) for component in connected_components_18(mask)] == [2]


def test_corner_only_touching_voxels_are_separate_components():
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    assert [len(component) for component in connected_components_18(mask)] == [1, 1]


def test_two_separated_cubes_are_two_components():
    mask = np.zeros((7, 3, 3), dtype=bool)
    mask[0:2, 0:2, 0:2] = True
    mask[5:7, 0:2, 0:2] = True
    assert [len(component) for component in connected_components_18(mask)] == [8, 8]


def test_static_robot_overlap_uses_voxel_aabb_not_only_center():
    indices = np.asarray([[0, 0, 0]], dtype=np.int64)
    mask = static_voxel_robot_sphere_overlap_mask(
        indices,
        (0.0, 0.0, 0.0),
        0.1,
        [[0.101, 0.05, 0.05]],
        [0.002],
    )
    assert mask.tolist() == [True]


def test_static_component_robot_overlap_fraction_counts_voxels():
    indices = np.asarray([[x, 0, 0] for x in range(10)], dtype=np.int64)
    fraction, mask = static_component_robot_overlap_fraction(
        indices,
        (0.0, 0.0, 0.0),
        1.0,
        [[0.5, 0.5, 0.5]],
        [0.01],
    )
    assert np.isclose(fraction, 0.1)
    assert np.count_nonzero(mask) == 1


def test_static_robot_overlap_rejects_entire_component_at_threshold():
    grid = np.ones((14, 1, 1), dtype=np.float64)
    grid[0:10, 0, 0] = -0.2
    grid[12:14, 0, 0] = -0.2

    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        robot_sphere_centers=[[0.5, 0.5, 0.5]],
        robot_sphere_radii=[0.01],
        robot_component_overlap_threshold=0.10,
    )

    assert result.input_component_count == 2
    assert result.robot_rejected_component_count == 1
    assert result.robot_rejected_component_ids == [0]
    assert result.robot_rejected_voxel_count == 10
    assert np.isclose(result.robot_component_overlap_fractions[0], 0.1)
    assert np.count_nonzero(result.inside_mask) == 2
    assert {component.component_id for component in result.components} == {1}
    assert {sphere.component_id for sphere in result.spheres} == {1}


def test_static_robot_overlap_below_threshold_keeps_component():
    grid = np.full((10, 1, 1), -0.2, dtype=np.float64)
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        robot_sphere_centers=[[0.5, 0.5, 0.5]],
        robot_sphere_radii=[0.01],
        robot_component_overlap_threshold=0.11,
    )

    assert result.robot_rejected_component_count == 0
    assert len(result.components) == 1
    assert result.spheres


def test_unobserved_sentinel_is_not_inside():
    grid = np.array([[[-1000.0, -0.2]]])
    mask = extract_inside_mask(grid, -1000.0, 0.005)
    assert mask.tolist() == [[[False, True]]]


def test_positive_esdf_is_not_inside():
    grid = np.array([[[-0.2, 0.0, 0.2]]])
    mask = extract_inside_mask(grid, -1000.0, 0.005)
    assert mask.tolist() == [[[True, False, False]]]


def test_signed_cube_has_a_local_minimum_near_its_center():
    grid = signed_box_esdf((9, 9, 9), (1.0, 1.0, 1.0), (8.0, 8.0, 8.0))
    component = connected_components_18(extract_inside_mask(grid, -1000.0, 0.0))[0]
    minima = find_local_minimum_candidates(grid, component)
    assert any(np.all(index == np.array([4, 4, 4])) for index in minima)


def test_long_plateau_keeps_multiple_stably_spaced_centers():
    grid = signed_box_esdf((11, 5, 5), (1.0, 1.0, 1.0), (10.0, 4.0, 4.0))
    component = connected_components_18(extract_inside_mask(grid, -1000.0, 0.0))[0]
    minima = find_local_minimum_candidates(grid, component)
    reduced_a = group_or_reduce_plateaus(grid, minima, 1.0, 1e-9, 2.0)
    reduced_b = group_or_reduce_plateaus(grid, minima, 1.0, 1e-9, 2.0)
    assert len(reduced_a) >= 3
    assert np.array_equal(reduced_a, reduced_b)
    pair_distances = np.linalg.norm(
        reduced_a[:, None, :] - reduced_a[None, :, :], axis=2
    )
    pair_distances += np.eye(len(reduced_a)) * 1e9
    assert np.min(pair_distances) >= 2.0


def test_initial_sphere_radius_is_negative_esdf_at_center():
    grid = np.ones((3, 3, 3), dtype=np.float64)
    grid[1, 1, 1] = -0.375
    spheres = create_initial_spheres(
        grid, np.array([[1, 1, 1]]), (0.0, 0.0, 0.0), 0.1, 0.01, 0.02, 7
    )
    assert len(spheres) == 1
    assert spheres[0].raw_radius == 0.375
    assert spheres[0].output_radius == 0.395


def test_static_raw_radius_cap_clips_deep_esdf_candidates():
    grid = np.full((3, 1, 1), -0.50)
    spheres = create_initial_spheres(
        grid,
        np.array([[1, 0, 0]]),
        (0.0, 0.0, 0.0),
        0.1,
        0.01,
        0.005,
        0,
        0.25,
    )
    assert len(spheres) == 1
    assert spheres[0].raw_radius == 0.25
    assert spheres[0].output_radius == 0.255


def test_static_raw_radius_cap_applies_to_complete_generation():
    grid = signed_box_esdf(
        (15, 7, 7),
        (0.5, 0.5, 0.5),
        (7.0, 3.0, 3.0),
        voxel_size_m=0.5,
    )
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        0.5,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        min_raw_sphere_radius_m=0.05,
        max_raw_sphere_radius_m=0.75,
        safety_margin_m=0.01,
        minimum_center_spacing_m=0.5,
        target_coverage=0.90,
        enable_agglomerative_merge=True,
        merge_max_radius_m=0.75,
    )
    assert result.spheres
    assert max(sphere.raw_radius for sphere in result.spheres) <= 0.75
    assert max(sphere.output_radius for sphere in result.spheres) <= 0.76


def test_post_merge_overlap_pruning_preserves_volume_and_shell_coverage():
    grid = np.full((3, 1, 1), -0.05)
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    spheres = [
        Sphere(np.array((1.5, 0.5, 0.5)), 1.2, 1.2, 0, (1, 0, 0), True),
        Sphere(np.array((1.5, 0.5, 0.5)), 1.0, 1.0, 0, (1, 0, 0), True),
    ]
    pruned, removed = prune_overlapping_static_spheres(
        grid,
        component,
        spheres,
        (0.0, 0.0, 0.0),
        1.0,
        0.0,
        1.0,
        0.0,
        True,
        0.10,
        1.0,
        0.0,
        100,
        0.20,
        8,
        4,
    )
    coverage, _ = calculate_component_coverage(
        component, pruned, (0.0, 0.0, 0.0), 1.0, 0.0)
    assert removed == 1
    assert len(pruned) == 1
    assert pruned[0].raw_radius == 1.0
    assert coverage == 1.0


def test_spheres_are_added_until_target_coverage():
    grid = np.full((5, 1, 1), -0.51)
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    initial = create_initial_spheres(
        grid, np.array([[0, 0, 0]]), (0.0, 0.0, 0.0), 1.0, 0.1, 0.0, 0
    )
    spheres, coverage, reason, _ = add_spheres_until_coverage(
        grid,
        component,
        initial,
        (0.0, 0.0, 0.0),
        1.0,
        0.95,
        0.0,
        0.1,
        0.1,
        0.0,
        10,
        10,
        0,
    )
    assert len(spheres) == 5
    assert coverage >= 0.95
    assert reason == "target_coverage"


def test_max_spheres_terminates_without_reaching_target():
    grid = np.full((5, 1, 1), -0.51)
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    spheres, coverage, reason, _ = add_spheres_until_coverage(
        grid,
        component,
        [],
        (0.0, 0.0, 0.0),
        1.0,
        0.95,
        0.0,
        0.1,
        0.1,
        0.0,
        2,
        100,
        0,
    )
    assert len(spheres) == 2
    assert coverage < 0.95
    assert reason == "max_spheres_per_component"


def test_no_inside_voxels_returns_no_spheres():
    result = generate_medial_spheres(
        np.ones((4, 4, 4)), (0.0, 0.0, 0.0), 0.1, min_component_voxels=1
    )
    assert not result.components
    assert not result.spheres


def test_small_component_is_removed():
    grid = np.ones((5, 5, 5))
    grid[2, 2, 2] = -0.1
    result = generate_medial_spheres(
        grid, (0.0, 0.0, 0.0), 0.1, min_component_voxels=2
    )
    assert result.removed_small_components == 1
    assert not result.components


def test_redundant_sphere_is_removed_when_coverage_stays_at_target():
    component = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    large = make_sphere((1, 0, 0), 2.0)
    small = make_sphere((1, 0, 0), 0.5)
    spheres, coverage = remove_redundant_spheres(
        component, [large, small], (0.0, 0.0, 0.0), 1.0, 0.95, 0.0, 0.0
    )
    assert spheres == [large]
    assert coverage == 1.0


def test_redundant_sphere_is_kept_if_removal_loses_target_coverage():
    component = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    left = make_sphere((0, 0, 0), 1.0)
    right = make_sphere((2, 0, 0), 0.1)
    spheres, coverage = remove_redundant_spheres(
        component, [left, right], (0.0, 0.0, 0.0), 1.0, 0.95, 0.0, 2.0
    )
    assert len(spheres) == 2
    assert coverage == 1.0


def test_total_sphere_limit_uses_deterministic_coverage_ranking():
    grid = np.full((9, 1, 1), -0.51)
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        minimum_center_spacing_m=0.1,
        min_raw_sphere_radius_m=0.1,
        coverage_tolerance_m=0.0,
        target_coverage=1.0,
        max_spheres_per_component=20,
        max_iterations_per_component=20,
        max_total_spheres=3,
    )
    assert result.total_limit_applied
    assert len(result.spheres) == 3
    assert result.coverage_lost_component_ids == [0]
    assert result.components[0].termination_reason == "max_total_spheres"
    assert result.components[0].coverage == 3.0 / 9.0


def test_repeated_generation_has_identical_sphere_order_and_values():
    grid = signed_box_esdf((12, 7, 7), (1.0, 1.0, 1.0), (11.0, 6.0, 6.0))
    kwargs = dict(
        origin_m=(0.0, 0.0, 0.0),
        voxel_size_m=1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        minimum_center_spacing_m=2.0,
        min_raw_sphere_radius_m=0.1,
        coverage_tolerance_m=0.01,
    )
    first = generate_medial_spheres(grid, **kwargs)
    second = generate_medial_spheres(grid, **kwargs)
    assert sphere_signature(first) == sphere_signature(second)
    assert [component.coverage for component in first.components] == [
        component.coverage for component in second.components
    ]


def test_static_single_sphere_replacement_uses_esdf_candidate():
    grid = np.full((5, 1, 1), -0.6)
    grid[2, 0, 0] = -2.1
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    candidates = [
        make_sphere((0, 0, 0), 0.6),
        make_sphere((2, 0, 0), 2.1),
        make_sphere((4, 0, 0), 0.6),
    ]
    spheres, volume, shell, _, applied = optimize_component_spheres(
        grid, component, candidates, (0.0, 0.0, 0.0), 1.0, 0.0, 1.0, 0.0,
        surface_shell_thickness_m=0.6,
        target_shell_coverage=1.0,
        shell_coverage_loss_tolerance=0.0,
    )
    assert applied
    assert len(spheres) == 1
    assert spheres[0].source_index == (2, 0, 0)
    assert spheres[0].raw_radius == -grid[spheres[0].source_index]
    assert volume == 1.0
    assert shell == 1.0


def test_static_greedy_set_cover_reduces_overlapping_candidates():
    grid = np.full((6, 1, 1), -0.51)
    grid[1, 0, 0] = -1.51
    grid[2, 0, 0] = -2.51
    grid[4, 0, 0] = -1.51
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    candidates = [
        make_sphere((1, 0, 0), 1.51),
        make_sphere((2, 0, 0), 2.51),
        make_sphere((4, 0, 0), 1.51),
    ]
    masks = sphere_coverage_masks(
        component, candidates, (0.0, 0.0, 0.0), 1.0, 0.0)
    selected = greedy_set_cover_spheres(
        candidates, masks, np.zeros(len(component), dtype=bool),
        1.0, 0.0, False)
    assert selected == [1, 2]
    assert calculate_mask_coverage(np.any(masks[selected], axis=0)) == 1.0


def test_elongated_static_component_is_not_forced_to_one_sphere():
    grid = np.full((10, 1, 1), -1.1)
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    candidates = [
        make_sphere((1, 0, 0), 1.1),
        make_sphere((4, 0, 0), 1.1),
        make_sphere((7, 0, 0), 1.1),
        make_sphere((9, 0, 0), 1.1),
    ]
    spheres, volume, _, _, _ = optimize_component_spheres(
        grid, component, candidates, (0.0, 0.0, 0.0), 1.0, 0.0, 0.9, 0.0,
        enable_surface_shell_guard=False)
    assert len(spheres) > 1
    assert volume >= 0.9


def test_surface_shell_guard_preserves_thin_protrusion_sphere():
    grid = np.full((20, 1, 1), -1.0)
    grid[9, 0, 0] = -9.1
    grid[19, 0, 0] = -0.1
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    candidates = [
        make_sphere((9, 0, 0), 9.1),
        make_sphere((19, 0, 0), 0.1),
    ]
    without_guard = optimize_component_spheres(
        grid, component, candidates, (0.0, 0.0, 0.0), 1.0, 0.005, 0.95, 0.0,
        enable_single_sphere_replacement=False,
        enable_greedy_set_cover=False,
        enable_surface_shell_guard=False,
        surface_shell_thickness_m=0.2,
    )
    with_guard = optimize_component_spheres(
        grid, component, candidates, (0.0, 0.0, 0.0), 1.0, 0.005, 0.95, 0.0,
        enable_single_sphere_replacement=False,
        enable_greedy_set_cover=False,
        enable_surface_shell_guard=True,
        surface_shell_thickness_m=0.2,
        target_shell_coverage=1.0,
        shell_coverage_loss_tolerance=0.0,
    )
    assert len(without_guard[0]) == 1
    assert without_guard[2] == 0.0
    assert len(with_guard[0]) == 2
    assert with_guard[2] == 1.0


def test_empty_surface_shell_disables_guard_without_division_by_zero():
    grid = np.full((5, 1, 1), -1.0)
    grid[2, 0, 0] = -3.0
    component = np.argwhere(np.ones_like(grid, dtype=bool))
    shell_mask = make_surface_shell_mask(grid, component, 0.005, 0.1)
    assert not np.any(shell_mask)
    sphere = make_sphere((2, 0, 0), 3.0)
    result = optimize_component_spheres(
        grid, component, [sphere], (0.0, 0.0, 0.0), 1.0, 0.005, 1.0, 0.0,
        surface_shell_thickness_m=0.1)
    assert result[1] == 1.0
    assert result[2] == 1.0


def test_general_pruning_removes_non_contained_coverage_redundancy():
    spheres = [
        make_sphere((0, 0, 0), 0.1),
        make_sphere((2, 0, 0), 0.1),
        make_sphere((4, 0, 0), 0.1),
    ]
    masks = np.asarray([
        [True, True, True, False, False],
        [False, False, True, True, True],
        [True, False, False, False, True],
    ])
    selected = general_coverage_pruning(
        spheres, masks, np.zeros(5, dtype=bool), [0, 1, 2],
        1.0, 0.0, False)
    assert selected == [0, 1]


def test_all_static_optimization_features_can_be_disabled():
    grid = signed_box_esdf(
        (9, 7, 7), (1.0, 1.0, 1.0), (8.0, 6.0, 6.0))
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        enable_single_sphere_replacement=False,
        enable_greedy_set_cover=False,
        enable_general_coverage_pruning=False,
        enable_surface_shell_guard=False,
    )
    assert result.spheres
    assert result.components[0].coverage >= 0.95


def test_optimized_static_generation_is_deterministic():
    grid = signed_box_esdf(
        (12, 7, 7), (1.0, 1.0, 1.0), (11.0, 6.0, 6.0))
    kwargs = dict(
        origin_m=(0.0, 0.0, 0.0),
        voxel_size_m=1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        minimum_center_spacing_m=1.0,
    )
    signatures = [
        sphere_signature(generate_medial_spheres(grid, **kwargs))
        for _ in range(3)
    ]
    assert signatures[0] == signatures[1] == signatures[2]


def test_final_static_spheres_keep_exact_safety_margin_and_esdf_radius():
    grid = signed_box_esdf(
        (10, 7, 7), (1.0, 1.0, 1.0), (9.0, 6.0, 6.0))
    safety_margin = 0.037
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        safety_margin_m=safety_margin,
    )
    assert result.spheres
    for sphere in result.spheres:
        assert sphere.raw_radius == -grid[sphere.source_index]
        assert np.isclose(
            sphere.output_radius, sphere.raw_radius + safety_margin)


def test_component_coverage_and_uncovered_indices_match_recalculation():
    grid = signed_box_esdf(
        (11, 7, 7), (1.0, 1.0, 1.0), (10.0, 6.0, 6.0))
    result = generate_medial_spheres(
        grid, (0.0, 0.0, 0.0), 1.0,
        inside_epsilon_m=0.0, min_component_voxels=1)
    for component in result.components:
        coverage, covered = calculate_component_coverage(
            component.voxel_indices,
            component.spheres,
            (0.0, 0.0, 0.0),
            1.0,
            0.01,
        )
        assert np.isclose(component.coverage, coverage)
        assert np.array_equal(
            component.uncovered_indices, component.voxel_indices[~covered])


def test_optimization_matrix_limit_falls_back_without_coverage_loss():
    grid = np.full((7, 1, 1), -0.51)
    result = generate_medial_spheres(
        grid,
        (0.0, 0.0, 0.0),
        1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        target_coverage=1.0,
        minimum_center_spacing_m=0.0,
        max_optimization_matrix_elements=1,
    )
    assert result.components[0].coverage == 1.0


def test_static_optimization_parameter_validation():
    invalid_kwargs = [
        {"surface_shell_thickness_m": -0.1},
        {"target_shell_coverage": -0.1},
        {"target_shell_coverage": 1.1},
        {"shell_coverage_loss_tolerance": -0.1},
        {"shell_coverage_loss_tolerance": 1.1},
        {"max_optimization_matrix_elements": 0},
    ]
    grid = np.full((1, 1, 1), -0.1)
    for kwargs in invalid_kwargs:
        try:
            generate_medial_spheres(
                grid, (0.0, 0.0, 0.0), 1.0,
                min_component_voxels=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid optimization parameters accepted: {kwargs}")


def merge_sphere(center, radius, source_index, component_id=0, safety=0.02):
    return Sphere(
        np.asarray(center, dtype=np.float64),
        radius,
        radius + safety,
        component_id,
        source_index,
    )


def static_merge(
    grid, component, spheres, *, guard=False, max_radius=0.35,
    growth=1.5, gap=0.05,
):
    return agglomerative_merge_static_spheres(
        grid,
        component,
        spheres,
        (0.0, 0.0, 0.0),
        0.05,
        0.0,
        0.02,
        merge_max_radius_m=max_radius,
        merge_max_radius_growth_ratio=growth,
        merge_max_gap_m=gap,
        merge_enable_esdf_guard=guard,
        merge_max_free_space_distance_m=0.08,
        merge_surface_sample_count=64,
        merge_min_observed_surface_fraction=0.70,
    )


def test_static_agglomerative_merge_reduces_compact_set_and_preserves_coverage():
    grid = np.full((20, 20, 20), -0.1)
    component = np.asarray([(4, 4, 4), (5, 4, 4), (6, 4, 4)])
    spheres = [
        merge_sphere((0.225, 0.225, 0.225), 0.10, (4, 4, 4)),
        merge_sphere((0.305, 0.225, 0.225), 0.10, (6, 4, 4)),
    ]
    before, _ = calculate_component_coverage(
        component, spheres, (0.0, 0.0, 0.0), 0.05, 0.0)
    merged = static_merge(grid, component, spheres)
    after, _ = calculate_component_coverage(
        component, merged, (0.0, 0.0, 0.0), 0.05, 0.0)
    assert len(merged) == 1
    assert merged[0].is_merged
    assert after + 1e-12 >= before
    assert np.isclose(
        merged[0].output_radius, merged[0].raw_radius + 0.02)


def test_static_elongated_set_does_not_collapse_into_one_giant_sphere():
    grid = np.full((30, 10, 10), -0.1)
    component = np.asarray([(x, 2, 2) for x in range(3, 14)])
    spheres = [
        merge_sphere((0.175 + 0.08 * index, 0.125, 0.125), 0.08,
                     (3 + 2 * index, 2, 2))
        for index in range(6)
    ]
    merged = static_merge(
        grid, component, spheres, max_radius=0.20, growth=1.5, gap=0.02)
    assert 1 < len(merged) < len(spheres)
    assert max(sphere.raw_radius for sphere in merged) <= 0.20 + 1e-12


def test_static_same_component_large_gap_is_not_merged():
    grid = np.full((30, 10, 10), -0.1)
    component = np.asarray([(x, 2, 2) for x in range(3, 14)])
    spheres = [
        merge_sphere((0.2, 0.125, 0.125), 0.05, (3, 2, 2)),
        merge_sphere((0.6, 0.125, 0.125), 0.05, (11, 2, 2)),
    ]
    assert len(static_merge(
        grid, component, spheres, max_radius=1.0, growth=10.0, gap=0.05
    )) == 2


def test_static_esdf_guard_rejects_free_space_merge_but_off_allows_it():
    grid = np.full((20, 20, 20), 0.20)
    component = np.asarray([(8, 8, 8), (9, 8, 8), (10, 8, 8)])
    spheres = [
        merge_sphere((0.425, 0.425, 0.425), 0.10, (8, 8, 8)),
        merge_sphere((0.505, 0.425, 0.425), 0.10, (10, 8, 8)),
    ]
    without_guard = static_merge(grid, component, spheres, guard=False)
    with_guard = static_merge(grid, component, spheres, guard=True)
    assert len(without_guard) == 1
    assert len(with_guard) == 2


def test_static_merge_is_deterministic_for_reversed_sphere_order():
    grid = np.full((20, 20, 20), -0.1)
    component = np.asarray([(x, 4, 4) for x in range(4, 11)])
    spheres = [
        merge_sphere((0.225 + 0.08 * index, 0.225, 0.225), 0.10,
                     (4 + 2 * index, 4, 4))
        for index in range(4)
    ]

    def result_signature(values):
        return [(
            tuple(np.round(sphere.center, 12)),
            round(sphere.raw_radius, 12),
            sphere.component_id,
        ) for sphere in values]

    assert result_signature(static_merge(grid, component, spheres)) == (
        result_signature(static_merge(grid, component, spheres[::-1])))


def test_static_min_k_search_escapes_greedy_merge_order_trap():
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
    spheres = [
        merge_sphere((x, 0.0, 0.0), radius, (index, 0, 0), safety=0.01)
        for index, (x, radius) in enumerate(zip(coordinates, radii))
    ]
    voxel_size = 0.001
    origin = (-0.0005, -0.0005, -0.0005)
    component = np.asarray([
        (round(x / voxel_size), 0, 0) for x in coordinates
    ], dtype=np.int64)
    grid = np.full((400, 3, 3), -0.1)
    merge_args = (
        grid,
        component,
        spheres,
        origin,
        voxel_size,
        0.0,
        0.01,
        -1000.0,
        0.17680082144659784,
        1.6856652026068732,
        0.058939303977189085,
        False,
        0.08,
        64,
        0.70,
    )
    greedy = agglomerative_merge_static_spheres(*merge_args)
    bounded = minimum_k_static_sphere_search(
        *merge_args, 16, 512, monotonic() + 1.0)

    assert len(greedy) == 3
    assert len(bounded.spheres) == 2
    assert bounded.states_explored <= 512
    before, _ = calculate_component_coverage(
        component, spheres, origin, voxel_size, 0.0)
    after, _ = calculate_component_coverage(
        component, bounded.spheres, origin, voxel_size, 0.0)
    assert after + 1e-12 >= before


def test_static_min_k_search_is_deterministic_for_reversed_input():
    grid = np.full((20, 20, 20), -0.1)
    component = np.asarray([(x, 4, 4) for x in range(4, 11)])
    spheres = [
        merge_sphere((0.225 + 0.08 * index, 0.225, 0.225), 0.10,
                     (4 + 2 * index, 4, 4))
        for index in range(4)
    ]

    def run(values):
        return minimum_k_static_sphere_search(
            grid,
            component,
            values,
            (0.0, 0.0, 0.0),
            0.05,
            0.0,
            0.02,
            -1000.0,
            0.35,
            1.5,
            0.05,
            False,
            0.08,
            64,
            0.70,
            8,
            128,
            monotonic() + 1.0,
        )

    assert sphere_signature(run(spheres)) == sphere_signature(run(spheres[::-1]))


def test_static_feature_off_preserves_pre_merge_geometry():
    grid = signed_box_esdf(
        (12, 7, 7), (1.0, 1.0, 1.0), (11.0, 6.0, 6.0))
    common = dict(
        origin_m=(0.0, 0.0, 0.0),
        voxel_size_m=1.0,
        inside_epsilon_m=0.0,
        min_component_voxels=1,
        minimum_center_spacing_m=1.0,
    )
    disabled = generate_medial_spheres(
        grid, **common, enable_agglomerative_merge=False)
    blocked = generate_medial_spheres(
        grid,
        **common,
        enable_agglomerative_merge=True,
        merge_max_radius_m=0.01,
    )
    assert sphere_signature(disabled) == sphere_signature(blocked)
    assert disabled.components[0].pre_merge_sphere_count == len(disabled.spheres)


def test_component_coarse_cover_reduces_large_thin_component():
    grid = np.full((24, 24, 16), 0.10, dtype=np.float64)
    grid[5:19, 5:19, 7:9] = -0.05
    common = dict(
        origin_m=(0.0, 0.0, 0.0),
        voxel_size_m=0.05,
        inside_epsilon_m=0.005,
        target_coverage=0.90,
        coverage_tolerance_m=0.02,
        minimum_center_spacing_m=0.05,
        min_component_voxels=1,
        min_raw_sphere_radius_m=0.04,
        max_raw_sphere_radius_m=0.32,
        safety_margin_m=0.005,
        surface_shell_thickness_m=0.10,
        target_shell_coverage=0.94,
        shell_coverage_loss_tolerance=0.02,
        enable_agglomerative_merge=False,
        merge_max_radius_m=0.32,
        enable_min_k_search=False,
        enable_post_merge_overlap_pruning=False,
    )
    baseline = generate_medial_spheres(
        grid, enable_component_coarse_cover=False, **common)
    coarse = generate_medial_spheres(
        grid,
        enable_component_coarse_cover=True,
        component_coarse_min_voxels=120,
        component_coarse_radius_scale=0.45,
        component_coarse_max_radius_m=0.32,
        component_coarse_max_empty_fraction=0.75,
        component_coarse_max_free_space_distance_m=0.15,
        **common,
    )

    assert baseline.components[0].coverage >= common["target_coverage"]
    assert coarse.components[0].coverage >= common["target_coverage"]
    assert len(coarse.spheres) < len(baseline.spheres)
    assert coarse.components[0].coarse_candidate_count > 0
    assert coarse.components[0].coarse_selected_count > 0
    assert max(sphere.raw_radius for sphere in coarse.spheres) <= 0.32


def test_component_coarse_cover_does_not_change_small_component():
    grid = np.full((16, 16, 12), 0.10, dtype=np.float64)
    grid[4:11, 4:11, 5:7] = -0.05  # 98 voxels, below the 120-voxel gate.
    result = generate_medial_spheres(
        grid,
        origin_m=(0.0, 0.0, 0.0),
        voxel_size_m=0.05,
        inside_epsilon_m=0.005,
        target_coverage=0.90,
        coverage_tolerance_m=0.02,
        minimum_center_spacing_m=0.10,
        min_component_voxels=1,
        min_raw_sphere_radius_m=0.04,
        max_raw_sphere_radius_m=0.32,
        enable_component_coarse_cover=True,
        component_coarse_min_voxels=120,
        component_coarse_radius_scale=0.45,
        component_coarse_max_radius_m=0.32,
        component_coarse_max_empty_fraction=0.75,
        component_coarse_max_free_space_distance_m=0.15,
        merge_max_radius_m=0.32,
        enable_agglomerative_merge=False,
        enable_min_k_search=False,
        enable_post_merge_overlap_pruning=False,
    )

    assert result.components[0].coarse_candidate_count == 0
    assert result.components[0].coarse_selected_count == 0


def test_static_merge_parameter_validation():
    grid = np.full((1, 1, 1), -0.1)
    invalid_kwargs = [
        {"merge_max_radius_m": 0.0},
        {"merge_max_radius_growth_ratio": 0.9},
        {"merge_max_gap_m": -0.1},
        {"merge_max_free_space_distance_m": -0.1},
        {"merge_surface_sample_count": 0},
        {"merge_min_observed_surface_fraction": 1.1},
        {"min_k_beam_width": 0},
        {"min_k_max_states": 0},
        {"min_k_processing_budget_ms": 0.0},
        {"component_coarse_min_voxels": 0},
        {"component_coarse_radius_scale": -0.1},
        {"component_coarse_max_radius_m": 0.0},
        {"component_coarse_max_empty_fraction": 1.1},
        {"component_coarse_max_free_space_distance_m": -0.1},
    ]
    for kwargs in invalid_kwargs:
        try:
            generate_medial_spheres(
                grid, (0.0, 0.0, 0.0), 1.0,
                min_component_voxels=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid merge parameters accepted: {kwargs}")


def test_static_esdf_guard_rejects_unobserved_surface_samples():
    grid = np.full((20, 20, 20), -1000.0)
    assert not merged_sphere_passes_esdf_guard(
        grid,
        (0.5, 0.5, 0.5),
        0.1,
        (0.0, 0.0, 0.0),
        0.05,
        -1000.0,
        0.08,
        64,
        0.70,
    )
