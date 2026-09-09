"""Summarize captured Fusion geometry; intersections use true 3D, not image overlap."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(folder):
    records = json.loads((folder / 'records.json').read_text())
    rows = [row for row in records if row['label'] == 'fusion' and 5 <= row['t'] <= 26]
    counts, volumes, radii, near_counts = [], [], [], []
    intersections = 0
    for row in rows:
        geometry = np.asarray(row['geometry'], dtype=float).reshape(-1, 6)
        static = geometry[geometry[:, 5] == 0]
        human = geometry[geometry[:, 5] == 2]
        if not len(human):
            continue
        counts.append(len(static))
        volumes.append(float(np.sum(4. * np.pi / 3. * static[:, 4] ** 3)))
        radii.extend(static[:, 4].tolist())
        gaps = (np.linalg.norm(static[:, None, :3] - human[None, :, :3], axis=2)
                - static[:, None, 4] - human[None, :, 4])
        intersections += int(np.count_nonzero(gaps < -1e-6))
        near_counts.append(int(np.count_nonzero(np.any(gaps <= .15, axis=1))))
    final_static = [r for r in records if r['label'] == 'fusion']
    last_index = next((i for i in range(len(final_static) - 1, -1, -1)
                       if final_static[i]['static']), None)
    clear = (final_static[last_index + 1]['t']
             if last_index is not None and last_index + 1 < len(final_static) else None)
    return dict(run=str(folder), frames_with_human=len(counts),
        mean_static_count=float(np.mean(counts)) if counts else None,
        mean_static_ball_volume_sum_m3=float(np.mean(volumes)) if volumes else None,
        static_radius_median_m=float(np.median(radii)) if radii else None,
        mean_static_near_human_15cm=float(np.mean(near_counts)) if near_counts else None,
        static_human_intersection_pairs=intersections, final_static_clear_s=clear)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([summarize(folder) for folder in args.folders], indent=2))
