#!/usr/bin/env python3
"""Compare full/short search on identical captured XYZ frames (no ROS launch)."""
import argparse
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
    parser.add_argument("capture", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--refinement-ms", type=float, default=5.)
    parser.add_argument("--output", type=Path, help="New JSON report path")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = {}
    for label, section in (("dynamic", "/dynamic_obstacle_sphere_node"),
                           ("human", "/human_obstacle_sphere_node")):
        # Same adapter mapping as production, including all radius/coverage
        # and component limits. Robot filtering isn't rerun on this comparison:
        # both variants receive the exact same captured obstacle XYZ arrays.
        params = DynamicObstacleSphereNode._core_parameters(
            SimpleNamespace(**config[section]["ros__parameters"]))
        data = np.load(args.capture / (label + "_points.npz"))
        eligible = np.flatnonzero(np.diff(data["offsets"]) > 20)
        selected = eligible[np.linspace(0, len(eligible) - 1,
                                        min(args.samples, len(eligible)), dtype=int)]
        frames = []
        for index in selected:
            points = data["points"][data["offsets"][index]:data["offsets"][index + 1]]
            row = dict(index=int(index))
            for name, variant in (("full", params), ("fast", replace(params,
                    refinement_budget_ms=args.refinement_ms))):
                result = generate_dynamic_spheres(points, variant)
                row[name] = dict(ms=result.elapsed_ms, count=len(result.spheres),
                    coverage=1. - len(result.uncovered_voxels) / max(1, len(result.voxel_centers)),
                    stops=[c.min_k_search_termination for c in result.components])
            frames.append(row)
        output[label] = dict(frames=frames)
        for name in ("full", "fast"):
            output[label][name] = {key: dict(mean=float(np.mean([r[name][key] for r in frames])),
                p50_p95_max=np.percentile([r[name][key] for r in frames], [50, 95, 100]).tolist())
                for key in ("ms", "count", "coverage")}
    if args.output:
        with args.output.open("x") as file:
            json.dump(output, file, indent=2)
    print(json.dumps({label: {k: v for k, v in results.items() if k != "frames"}
                      for label, results in output.items()}, indent=2))


if __name__ == "__main__":
    main()
