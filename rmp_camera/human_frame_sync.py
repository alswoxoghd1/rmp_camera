"""Pure current-epoch, freshness-first depth/semantic-mask pair selection."""

import math


def select_latest_depth_mask_pair(depth_stamps, mask_stamps, max_delta_s,
                                  last_depth_stamp=None, last_mask_stamp=None):
    """Newest usable mask, then nearest newer depth (newer depth wins ties).

    Each stream must advance independently: receiving an older DDS message is
    not a clock rewind. The caller resets these watermarks only on a ROS clock
    jump, and can enforce a maximum source age before calling this function.
    """
    if not math.isfinite(max_delta_s) or max_delta_s < 0:
        raise ValueError('max_delta_s must be finite and non-negative')
    candidates = []
    for mask_index, mask_stamp in enumerate(mask_stamps):
        if not math.isfinite(mask_stamp) or (
                last_mask_stamp is not None and mask_stamp <= last_mask_stamp):
            continue
        for depth_index, depth_stamp in enumerate(depth_stamps):
            if not math.isfinite(depth_stamp) or (
                    last_depth_stamp is not None and depth_stamp <= last_depth_stamp):
                continue
            delta = abs(depth_stamp - mask_stamp)
            if delta <= max_delta_s:
                candidates.append((-mask_stamp, delta, -depth_stamp,
                                   depth_index, mask_index))
    if not candidates:
        return None
    _, delta, _, depth_index, mask_index = min(candidates)
    return depth_index, mask_index, delta
