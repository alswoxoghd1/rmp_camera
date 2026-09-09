"""Compare post-pass and integrated covers on identical stored XYZ frames."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml

from rmp_camera.dynamic_obstacle_sphere_node import DynamicObstacleSphereNode
from rmp_camera.dynamic_obstacle_sphere_core import generate_dynamic_spheres


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=160)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = {}
    for label in ('dynamic', 'human'):
        params = replace(DynamicObstacleSphereNode._core_parameters(SimpleNamespace(
            **config['/' + label + '_obstacle_sphere_node']['ros__parameters'])),
            local_width_cover_enabled=True, refinement_budget_ms=5.)
        data = np.load(args.capture / (label + '_points.npz'))
        points, offsets = data['points'], data['offsets']
        valid = np.flatnonzero(np.diff(offsets) > 20)
        selected = valid[np.linspace(0, len(valid)-1, min(args.samples, len(valid)), dtype=int)]
        rows = []
        for order, index in enumerate(selected):
            row = dict(index=int(index))
            # Alternate ordering to reduce systematic cache/thermal advantage.
            modes = (False,True) if order%2 else (True,False)
            for integrated in modes:
                result = generate_dynamic_spheres(points[offsets[index]:offsets[index+1]],
                    replace(params, local_width_integrated=integrated))
                row['integrated' if integrated else 'post'] = dict(
                    ms=result.elapsed_ms, count=len(result.spheres),
                    coverage=1.-len(result.uncovered_voxels)/max(1,len(result.voxel_centers)),
                    local_ms=sum(c.local_width_elapsed_ms for c in result.components),
                    local_saved=sum(c.local_width_saved_spheres for c in result.components),
                    reasons=[c.local_width_reason for c in result.components])
                assert all(c.coverage+1e-12 >= params.target_coverage for c in result.components)
            rows.append(row)
        summary = {}
        for mode in ('post','integrated'):
            summary[mode] = {key: dict(mean=float(np.mean([r[mode][key] for r in rows])),
                p95=float(np.percentile([r[mode][key] for r in rows],95)),
                max=float(np.max([r[mode][key] for r in rows])))
                for key in ('ms','count','coverage','local_ms')}
            summary[mode]['reasons']=dict(Counter(x for r in rows for x in r[mode]['reasons']))
        report[label]=dict(summary=summary,frames=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v['summary'] for k,v in report.items()},indent=2))


if __name__=='__main__':
    main()
