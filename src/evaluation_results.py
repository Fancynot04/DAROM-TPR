"""Canonical two-level evaluation output and offline consistency validation."""
import csv
import io
import json
from pathlib import Path

from src.evaluation_metrics import aggregate, joint_record
from src.experiment_tracking import atomic_write, atomic_json

EVALUATION_PROTOCOL_VERSION = 2


def episode_rows(records):
    rows = []
    for record in records:
        row = {key: record[key] for key in ('seed', 'global_steps', 'agent_steps_sum')}
        for ego, prefix in (('ego_o3', 'o3'), ('ego_e74', 'e74')):
            row.update({f'{prefix}_{key}': value for key, value in record['per_agent'][ego].items()})
        row.update({f'joint_{key}': value for key, value in record['joint'].items()})
        rows.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return rows


def write_results(path, summary):
    path = Path(path)
    rows = episode_rows(summary['records'])
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # Publish summary last. A crash between files is detected by validation.
    atomic_write(path.with_name('episodes.csv'), lambda f: f.write(stream.getvalue().encode('utf-8')))
    atomic_json(path, summary)


def validate_results(path):
    path = Path(path)
    summary = json.loads(path.read_text())
    records = summary['records']
    if len(records) != summary['episodes'] or not records:
        raise ValueError('Incomplete episode records')
    with path.with_name('episodes.csv').open(newline='', encoding='utf-8') as stream:
        actual = list(csv.DictReader(stream))
    expected_stream = io.StringIO(newline='')
    rows = episode_rows(records)
    writer = csv.DictWriter(expected_stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    expected_stream.seek(0)
    if actual != list(csv.DictReader(expected_stream)):
        raise ValueError('JSON/CSV episode mismatch')
    if len({r['seed'] for r in records}) != len(records):
        raise ValueError('Duplicate evaluation seeds')
    for record in records:
        if joint_record(record['per_agent']) != record['joint']:
            raise ValueError('Joint episode mismatch')
        if sum(r['agent_steps'] for r in record['per_agent'].values()) != record['agent_steps_sum']:
            raise ValueError('Agent step total mismatch')
        for row in record['per_agent'].values():
            if not row.get('terminal_reason') or row['terminal_reason'] in ('Unknown', 'UnknownRemoval'):
                raise ValueError('Incomplete terminal reason')
    for key in ('global_steps', 'agent_steps_sum'):
        if key in summary and sum(r[key] for r in records) != summary[key]:
            raise ValueError('Global/agent step aggregate mismatch')
    for ego in ('ego_o3', 'ego_e74'):
        if aggregate([r['per_agent'][ego] for r in records]) != summary['per_agent'][ego]:
            raise ValueError('Per-agent aggregate mismatch')
    if aggregate([r['joint'] for r in records]) != summary['joint']:
        raise ValueError('Joint aggregate mismatch')
    return summary
