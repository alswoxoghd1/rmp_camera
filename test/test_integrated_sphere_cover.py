from dataclasses import replace
from time import monotonic

import numpy as np
import pytest

import rmp_camera.dynamic_obstacle_sphere_core as core
from rmp_camera.local_width_sphere_cover import (
    LocalWidthAttemptCache, coverage_matrix, local_width_cover,
)
from rmp_camera.dynamic_sphere_refinement import reuse_refinement, RefinementOptions


def lattice(shape=(8, 8, 2)):
    return (np.indices(shape).reshape(3, -1).T + .5) * .05


def parameters(**kwargs):
    return core.DynamicSphereParameters(min_x_m=0., min_y_m=0., min_z_m=0.,
        min_component_voxels=2, closing_iterations=0, processing_budget_ms=1000.,
        local_width_budget_ms=1000., local_width_cover_enabled=True,
        local_width_integrated=True, **kwargs)


def test_local_candidate_reuse_uses_width_not_old_component_volume_cap():
    points = lattice((4, 4, 4))
    params = parameters(component_radius_scale=.1)
    center = points.mean(axis=0)
    sphere = core.DynamicSphere(*center, .131, .131, local_width=True)
    validator = core.DynamicCandidateValidator(points, params)
    assert validator(center, .131, monotonic()+1, local_width=True)
    assert not validator(center, .131, monotonic()+1, local_width=False)
    seeds = [core.DynamicSphere(*p, .025, .025) for p in points]
    current = core.DynamicComponentResult(1, points, seeds, 1., np.empty((0, 3)), 'fresh')
    template = replace(current, spheres=[sphere])
    reused = reuse_refinement(current, template, params, RefinementOptions(), monotonic()+1)
    assert reused is not None and reused.spheres[0].local_width
    assert reuse_refinement(current, replace(template, spheres=[replace(sphere, local_width=False)]),
                            params, RefinementOptions(), monotonic()+1) is None


def test_current_width_guard_rejects_torso_sphere_on_narrow_arm():
    points = lattice((20, 2, 2))
    validator = core.DynamicCandidateValidator(points, parameters())
    assert not validator(points.mean(axis=0), .2, monotonic()+1)
    assert not validator(np.array([-.1, 0, 0]), .05, monotonic()+1)
    assert not validator(np.array([1., 1., 1.]), .05, monotonic()+1)
    assert not validator(points[0], .5, monotonic()+1)


def test_local_merge_keeps_provenance_and_uses_local_cap():
    points = lattice((4, 4, 4))
    params = parameters(component_radius_scale=.1)
    # Deliberately make the legacy component cap smaller than either parent.
    # Local parents still satisfy current visible width and absolute caps.
    center = points.mean(axis=0)
    parents = [core.DynamicSphere(*(center+[dx, 0, 0]), .1, .1, local_width=True)
               for dx in (-.025, .025)]
    result = core.agglomerative_merge_dynamic_spheres(points, parents, params, monotonic()+1)
    assert len(result) == 1 and result[0].local_width
    assert core.DynamicCandidateValidator(points, params)(result[0].center, result[0].raw_radius, monotonic()+1)
    required = core._covered_mask(points, parents, params.coverage_tolerance_m)
    assert core._covered_mask(points, result, params.coverage_tolerance_m)[required].all()


def test_integrated_selection_runs_before_one_refinement(monkeypatch):
    events = []
    original_local, original_refine = core._apply_local_width_cover, core.refine_dynamic_component
    def local(*args, **kwargs):
        assert kwargs['coverage_masks'] is not None
        events.append('local')
        return original_local(*args, **kwargs)
    def refine(*args, **kwargs):
        events.append('refine')
        return original_refine(*args, **kwargs)
    monkeypatch.setattr(core, '_apply_local_width_cover', local)
    monkeypatch.setattr(core, 'refine_dynamic_component', refine)
    result = core.generate_dynamic_spheres(lattice(), parameters())
    assert events == ['local', 'refine']
    assert result.components[0].coverage >= parameters().target_coverage


def test_cached_coverage_rows_match_full_matrix_result():
    points = lattice((4, 4, 2))
    radii = np.full(len(points), .025)
    kwargs = dict(voxel_size=.05, min_radius=.025, max_radius=.28,
                  tolerance=.02, target_coverage=.92, budget_ms=1000.)
    a = local_width_cover(points, points, radii, **kwargs)
    b = local_width_cover(points, points, radii,
                         old_coverage_masks=coverage_matrix(points, points, radii, .02), **kwargs)
    np.testing.assert_array_equal(a.selected, b.selected)
    np.testing.assert_allclose(a.radii, b.radii)


def test_negative_cache_shape_count_expiry_and_rewind():
    cache = LocalWidthAttemptCache()
    points = lattice()
    desc = lambda p, count=4: cache.descriptor(p, .05, np.zeros(3), count)
    cache.remember(desc(points), 'no_count_improvement', 1.)
    assert cache.matches(desc(points + [.05, 0, 0]), 1.1)
    assert not cache.matches(desc(points[:30]), 1.1)
    assert not cache.matches(desc(points, 5), 1.1)
    assert not cache.matches(desc(points + [.5, 0, 0]), 1.1)
    assert not cache.matches(desc(points), 1.3)
    cache.remember(desc(points), 'no_valid_candidates', 2.)
    assert not cache.matches(desc(points), 1.9)


@pytest.mark.parametrize('reason', ['deadline', 'size_limit', 'applied', 'coverage_rejected'])
def test_negative_cache_never_memoizes_incomplete_work(reason):
    cache = LocalWidthAttemptCache()
    desc = cache.descriptor(lattice(), .05, np.zeros(3), 4)
    cache.remember(desc, reason, 1.)
    assert not cache.matches(desc, 1.1)


def test_negative_cache_skips_extra_search_not_current_geometry(monkeypatch):
    params = parameters()
    points = lattice()
    cache = LocalWidthAttemptCache()
    seeds = [core.DynamicSphere(*p, .05, .05) for p in points[::8]]
    desc = cache.descriptor(points, .05, np.zeros(3), len(seeds))
    cache.remember(desc, 'no_valid_candidates', monotonic())
    shift = np.array([.05, 0., 0.])
    current = core.DynamicComponentResult(1, points+shift,
        [replace(s, x=s.x+.05) for s in seeds], .9, np.empty((0, 3)), 'fresh')
    fresh = list(current.spheres)
    def unexpected(*args, **kwargs):
        raise AssertionError('memoized no-gain should not repeat extra search')
    monkeypatch.setattr(core, 'local_width_cover', unexpected)
    core._apply_local_width_cover(current, params, attempt_cache=cache)
    assert current.local_width_reason == 'recent_no_gain'
    assert current.spheres == fresh


@pytest.mark.parametrize('kind', ['empty', 'noise', 'robot'])
def test_no_obstacle_clears_negative_cache(kind):
    params = parameters()
    cache = LocalWidthAttemptCache()
    points = lattice()
    cache.remember(cache.descriptor(points, .05, np.zeros(3), 4), 'no_valid_candidates', monotonic())
    kwargs = {}
    if kind == 'empty':
        points = np.empty((0, 3))
    elif kind == 'noise':
        points = points[:1]
    else:
        kwargs = dict(robot_sphere_centers=np.array([[.2, .2, .05]]), robot_sphere_radii=np.array([1.]))
    core.generate_dynamic_spheres(points, params, local_width_cache=cache, **kwargs)
    assert not cache.entries
