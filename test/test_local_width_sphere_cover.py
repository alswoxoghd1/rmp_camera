from dataclasses import replace
from time import monotonic

import numpy as np
import pytest

from rmp_camera.local_width_sphere_cover import local_width_cover, coverage_matrix, VoxelSupportGuard
from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicSphereParameters, generate_dynamic_spheres, dynamic_merge_empty_fraction,
    _covered_mask, _apply_local_width_cover,
)


def lattice(shape, offset=(0, 0, 0), voxel=.05):
    return (np.indices(shape).reshape(3,-1).T + .5) * voxel + offset


def cover(points, **kwargs):
    defaults = dict(voxel_size=.05, min_radius=.025, max_radius=.28,
                    tolerance=.02, target_coverage=.92, budget_ms=1000.)
    defaults.update(kwargs)
    return local_width_cover(points, points, np.full(len(points), .025), **defaults)


def test_broad_object_uses_fewer_larger_spheres_and_preserves_each_voxel():
    points = lattice((8,8,3))
    result = cover(points)
    assert result.applied and len(result.radii) < 10
    assert max(result.radii) > .1
    assert coverage_matrix(points, result.centers, result.radii, .02).any(axis=0).all()


def test_thin_arm_has_a_chain_not_a_single_torso_size_ball():
    points = lattice((20,2,2))
    result = cover(points)
    assert result.applied and len(result.radii) >= 4
    assert max(result.radii) < .12
    assert result.centers[:,0].ptp() > .6
    assert coverage_matrix(points, result.centers, result.radii, .02).any(axis=0).all()


def test_connected_torso_and_arm_keep_distal_support_and_variable_radii():
    torso = lattice((8,8,3))
    arm = lattice((12,2,2), offset=(.4,.15,0))
    points = np.vstack((torso,arm))
    result = cover(points)
    assert result.applied
    distal = result.centers[:,0] > .65
    assert distal.any()
    assert max(result.radii[distal]) < max(result.radii[~distal])
    assert coverage_matrix(arm, result.centers, result.radii,.02).any(axis=0).all()


@pytest.mark.parametrize('kwargs,reason', [({'budget_ms':0},'deadline'),
    ({'max_points':10},'size_limit'), ({'max_matrix_elements':10},'size_limit'),
    ({'validator':lambda *a: False},'no_valid_candidates')])
def test_limits_and_failed_guards_keep_exact_legacy_cover(kwargs,reason):
    points = lattice((4,4,2))
    result = cover(points,**kwargs)
    assert not result.applied and result.reason == reason
    np.testing.assert_array_equal(result.centers,points)
    np.testing.assert_array_equal(result.radii,np.full(len(points),.025))


def test_region_floor_does_not_trade_away_small_hand_support_for_torso():
    torso = lattice((8,8,3))
    hand = lattice((4,2,2),offset=(.8,.15,0))
    points = np.vstack((torso,hand))
    old = np.vstack((torso,hand[::2]))
    result = local_width_cover(points,old,np.full(len(old),.025),
        voxel_size=.05,min_radius=.025,max_radius=.25,tolerance=.02,
        target_coverage=.9,budget_ms=1000.)
    assert result.applied
    assert coverage_matrix(hand[::2],result.centers,result.radii,.02).any(axis=0).all()


@pytest.mark.parametrize('radius',[.026,.05,.083,.12,.21])
def test_vectorized_occupancy_guard_matches_existing_exact_lattice(radius):
    params = DynamicSphereParameters(min_x_m=0,min_y_m=0,min_z_m=0)
    points = lattice((5,5,3))
    center = np.asarray((.13,.13,.08))
    fraction = dynamic_merge_empty_fraction(points,center,radius,params)
    for limit in (0.,.3,.7,1.):
        guard = VoxelSupportGuard(points,(0,0,0),.05,limit)
        assert guard(center,radius,monotonic()+1) == (fraction is not None and fraction <= limit+1e-12)


def test_dynamic_wrapper_preserves_raw_cap_margin_and_covered_voxels():
    points = lattice((9,9,3))
    params = DynamicSphereParameters(closing_iterations=0,min_x_m=0,min_y_m=0,min_z_m=0,
        min_component_voxels=2,processing_budget_ms=1000,dynamic_enable_min_k_search=False,
        dynamic_enable_agglomerative_merge=False,enable_single_sphere_replacement=False,
        max_raw_radius_m=.2,local_width_budget_ms=1000.)
    result = generate_dynamic_spheres(points,params)
    component = result.components[0]
    required = _covered_mask(component.voxel_centers,component.spheres,params.coverage_tolerance_m)
    _apply_local_width_cover(component,params)
    assert np.all(_covered_mask(component.voxel_centers,component.spheres,params.coverage_tolerance_m)[required])
    assert all(s.raw_radius <= .2 and abs(s.output_radius-s.raw_radius-params.safety_margin_m)<1e-12
               for s in component.spheres)


@pytest.mark.parametrize('free_distance',[.02,.5])
def test_static_integration_preserves_coverage_esdf_guard_and_radius_cap(free_distance):
    from rmp_camera.esdf_medial_sphere_core import generate_medial_spheres, calculate_component_coverage
    grid = np.full((18,18,12),free_distance)
    grid[4:12,4:12,4:8] = -.05
    options = dict(min_component_voxels=2, target_coverage=.9,
        max_raw_sphere_radius_m=.22, min_raw_sphere_radius_m=.025,
        enable_agglomerative_merge=False, enable_min_k_search=False,
        enable_component_coarse_cover=False, enable_surface_shell_guard=False,
        merge_max_radius_m=.22, component_coarse_max_radius_m=.22,
        local_width_budget_ms=1000., merge_max_free_space_distance_m=.10,
        component_coarse_max_free_space_distance_m=.10)
    baseline = generate_medial_spheres(grid,(0,0,0),.05,**options)
    candidate = generate_medial_spheres(grid,(0,0,0),.05,enable_local_width_cover=True,**options)
    assert len(candidate.spheres) <= len(baseline.spheres)
    if free_distance < .1:
        assert sum(c.local_width_saved_spheres for c in candidate.components) > 0
    for a,b in zip(baseline.components,candidate.components):
        _,required=calculate_component_coverage(a.voxel_indices,a.spheres,(0,0,0),.05,.01)
        _,covered=calculate_component_coverage(b.voxel_indices,b.spheres,(0,0,0),.05,.01)
        assert covered[required].all()
    assert all(s.raw_radius <= .22 for s in candidate.spheres)


def test_local_width_disabled_is_identical_to_legacy_and_zero_budget_falls_back():
    points=lattice((8,8,3))
    params=DynamicSphereParameters(processing_budget_ms=1000.,closing_iterations=0,
        dynamic_enable_min_k_search=False,dynamic_enable_agglomerative_merge=False)
    a=generate_dynamic_spheres(points,params)
    b=generate_dynamic_spheres(points,replace(params,local_width_cover_enabled=True,local_width_budget_ms=0.))
    assert a.spheres == b.spheres


def test_guard_deadline_is_fail_closed():
    points=lattice((6,6,3))
    guard=VoxelSupportGuard(points,(0,0,0),.05,.7)
    assert not guard(points[0],.05,monotonic()-1)


@pytest.mark.parametrize('kwargs',[{'width_ratio':.9},{'budget_ms':-1},{'max_radius':float('nan')},
                                  {'target_coverage':1.1}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        cover(lattice((3,3,3)),**kwargs)
