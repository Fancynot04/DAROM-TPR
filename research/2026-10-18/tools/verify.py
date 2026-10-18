"""Verify archived evidence offline using only the Python standard library."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
MOTION = ('mean_speed_mps', 'mean_jerk_mps3', 'mean_abs_jerk_mps3', 'rms_jerk_mps3')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(actual is None and expected is None or
            actual is not None and expected is not None and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), label)


def metrics(records):
    def rate(fn):
        return 100 * sum(fn(r) for r in records) / len(records)
    result = {
        'success_pct': rate(lambda r: r['success']),
        'collision_pct': rate(lambda r: r['outcome'] == 'Collision'),
        'timeout_pct': rate(lambda r: r['outcome'] == 'Max Steps'),
        'unmerged_timeout_pct': rate(lambda r: r['outcome'] == 'Max Steps' and not r['merged']),
        'merged_timeout_pct': rate(lambda r: r['outcome'] == 'Max Steps' and r['merged']),
        'raw_arrival_pct': rate(lambda r: r['outcome'] == 'Arrived'),
        'teleported_arrival_pct': rate(lambda r: r['outcome'] == 'Arrived' and r['ego_teleported']),
        'ego_teleport_pct': rate(lambda r: r['ego_teleported']),
        'emergency_stopping_pct': rate(lambda r: r['outcome'] == 'Emergency Stopping'),
        'mean_return': mean(r['return'] for r in records),
        'mean_safety_interventions': mean(r['safety_interventions'] for r in records),
    }
    for key in MOTION:
        values = [r['motion'][key] for r in records if r['motion'][key] is not None]
        result[key] = mean(values) if values else None
    return result


def main():
    checksums = json.loads((ROOT/'checksums.json').read_text())
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
              and p.name != 'checksums.json' and '__pycache__' not in p.parts}
    require(actual == set(checksums), 'File list differs from manifest')
    for name, expected in checksums.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'Hash: '+name)
    base = ROOT/'antwerp/evaluation'
    manifest = json.loads((base/'manifest.json').read_text())
    groups = defaultdict(list)
    seen = set()
    for line in (base/'episodes.jsonl').read_text().splitlines():
        r = json.loads(line)
        j = r['job']
        key = (j['scenario'], j['safety'], j['batch'], j['seed'])
        require(key not in seen, 'Duplicate episode')
        seen.add(key)
        require(r['checkpoint_sha256'] == manifest['checkpoints'][j['scenario']]['checkpoint_sha256'], 'Model hash')
        ep = r['episode']
        require(ep['seed'] == j['seed'], 'Episode seed mismatch')
        require(ep['success'] == (ep['outcome'] == 'Arrived' and not ep['ego_teleported']), 'Success definition')
        for scene in r['scene_files'].values():
            require(scene['sha256'] == manifest['source_sha256'][scene['path']], 'Scene hash')
        groups[(j['scenario'], j['safety'])].append(r)
    expected = {(s, sc, b, seed) for s in ('easy','medium','hard') for sc in (False,True)
                for b,start in enumerate((710001,720001,730001),1) for seed in range(start,start+500)}
    require(seen == expected, 'Evaluation schedule incomplete')
    table = json.loads((base/'table3.json').read_text())
    require(table['complete'] and len(table['rows']) == 6, 'Table incomplete')
    for row in table['rows']:
        reports = groups[(row['scenario'],row['safety'])]
        require(row['episodes'] == len(reports) == 1500, 'Row count')
        records = [r['episode'] for r in reports]
        recomputed = metrics(records)
        batches = []
        for b in (1,2,3):
            part = [r['episode'] for r in reports if r['job']['batch'] == b]
            require(len(part) == row['batches'][str(b)]['episodes'] == 500, 'Batch count')
            batch = metrics(part)
            for key, value in batch.items():
                close(value, row['batches'][str(b)][key], 'Batch metric '+key)
            batches.append(batch)
        for key,value in recomputed.items():
            close(value,row['metrics'][key],'Metric '+key)
            vals = [b[key] for b in batches]
            close(stdev(vals) if all(v is not None for v in vals) else None,
                  row['std_across_three_evaluation_batches'][key], 'Batch std '+key)
        for key in MOTION:
            require(sum(r['motion'][key] is None for r in records) == row['motion_missing'][key], 'Missing motion')
        p = recomputed['success_pct']/100
        n, z = len(records), 1.959963984540054
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        for got,want in zip((100*(center-half),100*(center+half)),row['success_ci95_wilson_pct']):
            close(got,want,'Wilson interval')
    snapshot = json.loads((ROOT/'reproduction/metric_snapshot.json').read_text())
    steps = defaultdict(list)
    previous = -1
    with (ROOT/'reproduction/d_metrics.csv').open() as f:
        for row in csv.DictReader(f):
            step = int(row['env_steps'])
            require(step >= previous, 'Steps decrease across segments')
            previous = step
            steps[row['segment']].append(step)
            require(all(math.isfinite(float(v)) for k,v in row.items() if k != 'segment' and v), 'Nonfinite scalar')
    for segment in snapshot['segments']:
        values = steps[segment['segment']]
        require(len(values) == segment['rows'] and max(values) == segment['last_env_steps'], 'Metric snapshot')
    for path in (ROOT/'reproduction/run_metadata').glob('*/source-hashes.json'):
        for name, expected_hash in json.loads(path.read_text()).items():
            require(hashlib.sha256((ROOT/'reproduction/source'/name).read_bytes()).hexdigest() == expected_hash, 'D source '+name)
    print(f'PASS: {len(checksums)} file hashes; 9000 paired episodes; 6 table rows and batch statistics; D source hashes and {sum(map(len,steps.values()))} metric rows.')


if __name__ == '__main__':
    main()
