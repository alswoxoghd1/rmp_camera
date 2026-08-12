"""ROS-free unit tests for dense signed-ESDF medial spheres."""

import numpy as np

from rmp_camera.esdf_medial_sphere_core import (
    Sphere,
    add_spheres_until_coverage,
    calculate_component_coverage,
    calculate_mask_coverage,
    connected_components_18,
    create_initial_spheres,
    extract_inside_mask,
    find_local_minimum_candidates,
    general_coverage_pruning,
    generate_medial_spheres,
    greedy_set_cover_spheres,
    group_or_reduce_plateaus,
    make_surface_shell_mask,
    make_neighbor_offsets_18,
    optimize_component_spheres,
    remove_redundant_spheres,
    sphere_coverage_masks,
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
