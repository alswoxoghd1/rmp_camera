"""Unit tests for conservative two-sphere merge geometry."""

import numpy as np

from rmp_camera.sphere_merge_core import (
    build_pair_merge_candidate,
    minimum_enclosing_sphere_pair,
)


def candidate(c1, r1, c2, r2, max_radius=10.0, growth=10.0, gap=10.0):
    return build_pair_merge_candidate(
        0,
        1,
        np.asarray((c1, c2), dtype=np.float64),
        np.asarray((r1, r2), dtype=np.float64),
        np.asarray((3, 3), dtype=np.int64),
        max_radius,
        growth,
        gap,
    )


def assert_contains(center, radius, child_center, child_radius):
    assert np.linalg.norm(center - child_center) + child_radius <= radius + 1e-12


def test_overlapping_equal_spheres_are_both_contained():
    c1 = np.asarray((0.0, 0.0, 0.0))
    c2 = np.asarray((1.0, 0.0, 0.0))
    center, radius = minimum_enclosing_sphere_pair(c1, 1.0, c2, 1.0)
    assert np.allclose(center, (0.5, 0.0, 0.0))
    assert np.isclose(radius, 1.5)
    assert_contains(center, radius, c1, 1.0)
    assert_contains(center, radius, c2, 1.0)


def test_contained_sphere_returns_larger_sphere_exactly():
    large_center = np.asarray((1.0, 2.0, 3.0))
    center, radius = minimum_enclosing_sphere_pair(
        large_center, 2.0, (1.2, 2.0, 3.0), 0.5)
    assert np.array_equal(center, large_center)
    assert radius == 2.0


def test_same_center_is_finite_and_uses_larger_radius():
    center, radius = minimum_enclosing_sphere_pair(
        (0.0, 0.0, 0.0), 0.2, (0.0, 0.0, 0.0), 0.3)
    assert np.isfinite(center).all()
    assert np.array_equal(center, np.zeros(3))
    assert radius == 0.3


def test_gap_guard_rejects_distant_pair():
    assert candidate(
        (0.0, 0.0, 0.0), 0.1, (1.0, 0.0, 0.0), 0.1, gap=0.2
    ) is None


def test_radius_growth_guard_rejects_pair():
    assert candidate(
        (0.0, 0.0, 0.0), 0.1, (0.3, 0.0, 0.0), 0.1,
        growth=1.4,
    ) is None


def test_absolute_radius_cap_rejects_pair():
    assert candidate(
        (0.0, 0.0, 0.0), 0.2, (0.2, 0.0, 0.0), 0.2,
        max_radius=0.25,
    ) is None


def test_different_components_are_never_merged():
    result = build_pair_merge_candidate(
        0,
        1,
        np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))),
        np.asarray((0.1, 0.1)),
        np.asarray((0, 1)),
        1.0,
        2.0,
        1.0,
    )
    assert result is None
