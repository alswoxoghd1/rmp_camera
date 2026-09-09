from dataclasses import replace
from time import monotonic

import numpy as np
import pytest

import rmp_camera.dynamic_obstacle_sphere_core as core


@pytest.mark.parametrize('seed', range(10))
def test_incremental_seed_selection_matches_original_scalar_loop(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    indices = np.unique(rng.integers(0, 9, size=(200, 3)), axis=0)
    params = core.DynamicSphereParameters(
        voxel_size_m=.05, min_useful_adaptive_radius_m=.025,
        minimum_center_spacing_m=.1, max_iterations_per_component=256,
        enable_fixed_radius_fallback=False, enable_greedy_set_cover=False,
        enable_single_sphere_replacement=False, dynamic_enable_agglomerative_merge=False,
        dynamic_enable_min_k_search=False, target_coverage=.92, processing_budget_ms=10000.)
    centers = core.voxel_centers(indices, params)
    radii = np.clip(core._boundary_depths(indices, params.connectivity) * params.voxel_size_m,
                    params.min_raw_radius_m, params.max_raw_radius_m)
    order = sorted(range(len(centers)), key=lambda i: (-float(radii[i]),) + tuple(centers[i]))
    expected = []
    uncovered = np.ones(len(centers), dtype=bool)
    iterations = 0
    for idx in order:
        if iterations >= params.max_iterations_per_component or len(expected) >= params.max_spheres_per_component:
            break
        iterations += 1
        if not uncovered[idx] or radii[idx] < params.min_useful_adaptive_radius_m:
            continue
        center = centers[idx]
        if expected and min(np.linalg.norm(center - s.center) for s in expected) < params.minimum_center_spacing_m:
            continue
        expected.append(core.DynamicSphere(*center, float(radii[idx]),
                                           float(radii[idx]) + params.safety_margin_m, 0))
        uncovered = ~core._covered_mask(centers, expected, params.coverage_tolerance_m)
        if float(np.mean(~uncovered)) + 1e-12 >= params.target_coverage:
            break
    monkeypatch.setattr(core, '_optimize_component_spheres', lambda centers, spheres, *args: spheres)
    result = core._component_spheres(indices, params, 0, monotonic() + 10)
    actual = [replace(s, component_coverage=0., confidence=0.) for s in result.spheres]
    assert actual == expected
    np.testing.assert_array_equal(result.uncovered_voxels, centers[uncovered])
