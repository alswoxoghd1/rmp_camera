"""Summarize raw-input bag runs; observed input-to-result age is not control latency."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def distribution(values):
    return ({'n': len(values), 'mean': float(np.mean(values)),
             'p50': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
             'max': float(np.max(values))} if values else {'n': 0})


def summarize(folder):
    metadata = json.loads((folder/'summary.json').read_text())
    records = json.loads((folder/'records.json').read_text())
    active = [r for r in records if 5 <= r['t'] <= 26]
    result = {'final_clear': metadata['fusion']['final_clear']}
    for label in ('dynamic', 'human', 'static', 'fusion'):
        result[label] = {'output_count': distribution([r['count'] for r in active if r['label'] == label])}
        radii = [p[3] for r in active if r['label'] == label for p in r.get('geometry', [])]
        result[label]['raw_radius_m'] = distribution(radii)
    for label in ('dynamic', 'human'):
        rows = [r for r in active if r['label'] == label+'_status'
                and r['details'].get('observation_valid') == 'True']
        for key in ('processing_ms', 'fast_core_ms', 'local_width_ms', 'generated_sphere_count',
                    'generated_coverage', 'refinement_validation_ms'):
            result[label][key] = distribution([float(r['details'][key]) for r in rows if key in r['details']])
        ages = [(r['t'] - (int(r['details']['source_stamp_ns'])/1e9 - metadata['bag_start']))*1000 for r in rows]
        result[label]['pointcloud_to_result_ms'] = distribution(ages)
        result[label]['local_reasons'] = dict(Counter(
            reason for r in rows for reason in r['details'].get('local_width_reasons', '').split(',') if reason))
        result[label]['async_applied'] = sum(int(r['details'].get('refinement_applied_components', 0)) for r in rows)
        result[label]['ray_evidence_available'] = sum(r['details'].get('ray_evidence_available') == 'True' for r in rows)
        for key in ('ray_candidates', 'ray_supported', 'ray_free_rejected', 'ray_alternative_accept', 'ray_alternative_reject'):
            result[label][key] = sum(int(r['details'].get(key, 0)) for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {str(folder): summarize(folder) for folder in args.directories}
    content = json.dumps(result, indent=2)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content+'\n')
    print(content)


if __name__ == '__main__':
    main()
