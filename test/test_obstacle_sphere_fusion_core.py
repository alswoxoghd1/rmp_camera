from rmp_camera.obstacle_sphere_fusion_core import (
    FusionParameters,
    FusionSphere,
    SphereFusionCache,
    fuse_spheres,
    spheres_from_field_rows,
)


def sphere(x, source, radius=0.2, z=0.0):
    return FusionSphere(x, 0.0, z, radius, radius, source)


def params(**overrides):
    values = dict(
        target_frame="base_link",
        overlap_tolerance_m=0.0,
        dynamic_priority=True,
        hysteresis_sec=0.5,
        max_total_spheres=50,
        static_cache_expire_enabled=False,
        static_empty_confirmation_frames=2,
        dynamic_handover_grace_sec=0.5,
        keep_dynamic_until_static_overlap=True,
        dynamic_absolute_max_ttl_sec=3.0,
    )
    values.update(overrides)
    return FusionParameters(**values)


def test_non_overlapping_static_and_dynamic_are_retained():
    result = fuse_spheres([sphere(0.0, 0)], [sphere(1.0, 1)], params())
    assert {item.source_type for item in result} == {0, 1}
    assert len(result) == 2


def test_real_3d_overlap_prefers_dynamic():
    result = fuse_spheres([sphere(0.0, 0)], [sphere(0.2, 1)], params())
    assert len(result) == 1
    assert result[0].source_type == 1


def test_same_camera_ray_but_separated_in_3d_keeps_static():
    result = fuse_spheres(
        [sphere(0.0, 0, radius=0.1, z=0.0)],
        [sphere(0.0, 1, radius=0.1, z=1.0)], params())
    assert len(result) == 2


def test_static_cache_survives_message_delay():
    cache = SphereFusionCache(params())
    cache.update_static([sphere(0.0, 0)], "base_link", 0.0)
    assert any(s.source_type == 0 for s in cache.combined(100.0))


def test_one_static_empty_does_not_clear_cache():
    cache = SphereFusionCache(params(static_empty_confirmation_frames=2))
    cache.update_static([sphere(0.0, 0)], "base_link", 0.0)
    cache.update_static([], "base_link", 1.0)
    assert cache.static_spheres


def test_confirmed_static_empty_clears_cache():
    cache = SphereFusionCache(params(static_empty_confirmation_frames=2))
    cache.update_static([sphere(0.0, 0)], "base_link", 0.0)
    cache.update_static([], "base_link", 1.0)
    cache.update_static([], "base_link", 2.0)
    assert cache.static_spheres == []


def test_dynamic_absolute_ttl_removes_ghost():
    cache = SphereFusionCache(params(dynamic_absolute_max_ttl_sec=1.0))
    cache.update_dynamic([sphere(0.0, 1)], "base_link", 0.0)
    assert cache.combined(0.5)
    assert cache.combined(1.1) == []


def test_handover_has_no_combined_gap():
    cache = SphereFusionCache(params())
    cache.update_dynamic([sphere(0.0, 1)], "base_link", 0.0)
    assert cache.combined(1.0)[0].source_type == 1
    cache.update_static([sphere(0.0, 0)], "base_link", 1.1)
    result = cache.combined(1.1)
    assert len(result) == 1
    assert result[0].source_type == 0


def test_stale_dynamic_without_handover_is_removed():
    cache = SphereFusionCache(params(keep_dynamic_until_static_overlap=False))
    cache.update_dynamic([sphere(0.0, 1)], "base_link", 0.0)
    assert cache.combined(0.6) == []


def test_frame_mismatch_is_rejected_without_corrupting_cache():
    cache = SphereFusionCache(params())
    assert cache.update_static([sphere(0.0, 0)], "map", 0.0) is False
    assert cache.static_spheres == []
    assert "frame mismatch" in cache.last_rejection_reason


def test_pointcloud_field_order_is_name_based():
    names = [
        "output_radius", "z", "track_id", "x", "confidence", "raw_radius", "y"]
    rows = [[0.25, 3.0, 42, 1.0, 0.8, 0.30, 2.0]]
    result = spheres_from_field_rows(names, rows, source_type=1)
    item = result[0]
    assert (item.x, item.y, item.z) == (1.0, 2.0, 3.0)
    assert item.raw_radius == 0.30
    assert item.output_radius == 0.25
    assert item.track_id == 42
    assert item.confidence == 0.8
