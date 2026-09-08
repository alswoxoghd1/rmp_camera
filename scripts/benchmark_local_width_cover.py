"""Compare local-width reconstruction on identical precomputed voxel covers."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from rmp_camera.dynamic_obstacle_sphere_node import DynamicObstacleSphereNode
from rmp_camera.dynamic_obstacle_sphere_core import generate_dynamic_spheres, _apply_local_width_cover, _covered_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('--samples', type=int, default=120)
    parser.add_argument('--budget-ms', type=float, default=5.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = {}
    for label, section in (('dynamic', '/dynamic_obstacle_sphere_node'), ('human', '/human_obstacle_sphere_node')):
        params = replace(DynamicObstacleSphereNode._core_parameters(SimpleNamespace(**config[section]['ros__parameters'])),
                         local_width_cover_enabled=False, refinement_budget_ms=5.)
        data = np.load(args.capture / (label + '_points.npz'))
        offsets = data['offsets']
        captured_points = data['points']
        eligible = np.flatnonzero(np.diff(offsets) > 20)
        selected = eligible[np.linspace(0, len(eligible) - 1, min(args.samples, len(eligible)), dtype=int)]
        rows, reasons = [], Counter()
        for index in selected:
            result = generate_dynamic_spheres(captured_points[offsets[index]:offsets[index+1]], params)
            trial = deepcopy(result)
            remaining = args.budget_ms
            for old, component in zip(result.components, trial.components):
                _apply_local_width_cover(component, replace(params, local_width_budget_ms=remaining))
                remaining = max(0., remaining - component.local_width_elapsed_ms)
                reasons[component.local_width_reason] += 1
                required = _covered_mask(old.voxel_centers, old.spheres, params.coverage_tolerance_m)
                covered = _covered_mask(component.voxel_centers, component.spheres, params.coverage_tolerance_m)
                assert np.all(covered[required]), 'lost an old supported voxel'
                if component.local_width_applied:
                    assert component.coverage + 1e-12 >= old.coverage
                    assert len(component.spheres) < len(old.spheres)
            rows.append(dict(index=int(index), voxels=len(result.voxel_centers), baseline_count=len(result.spheres),
                local_count=sum(len(c.spheres) for c in trial.components), baseline_ms=result.elapsed_ms,
                local_ms=sum(c.local_width_elapsed_ms for c in trial.components),
                applied=sum(c.local_width_applied for c in trial.components),
                baseline_coverage=1-len(result.uncovered_voxels)/max(1,len(result.voxel_centers)),
                local_coverage=1-sum(len(c.uncovered_voxels) for c in trial.components)/max(1,len(result.voxel_centers))))
        summary = {key: dict(mean=float(np.mean([r[key] for r in rows])),
                            p95=float(np.percentile([r[key] for r in rows],95)),
                            max=float(np.max([r[key] for r in rows])))
                   for key in ('baseline_count','local_count','baseline_ms','local_ms','baseline_coverage','local_coverage')}
        summary['reasons'] = dict(reasons)
        summary['improved_frames'] = sum(r['local_count'] < r['baseline_count'] for r in rows)
        output[label] = dict(summary=summary, frames=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(output,file,indent=2)
    print(json.dumps({key: value['summary'] for key,value in output.items()},indent=2))


if __name__ == '__main__':
    main()
