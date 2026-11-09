"""Integrity and non-intrusion checks for trajectory CSV recording.

  python3 -m scripts.verify_trajectory_csv --trajectory-dir D --report-on on.json \
      [--report-off off.json]
Exit code 1 if any check fails.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

VOLATILE = {'trajectory_dir', 'output', 'wall_seconds'}


def vector_columns(row, prefix):
    """Return only numbered vector fields, excluding similarly named metadata."""
    return [key for key in row if key.startswith(prefix) and key[len(prefix):].isdigit()]


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajectory-dir', type=Path, required=True)
    ap.add_argument('--report-on', type=Path, required=True)
    ap.add_argument('--report-off', type=Path)
    ap.add_argument('--require-route-diagnostics', action='store_true',
                    help='Require each trajectory to contain valid D3 distance/reachability/lane-change samples')
    args = ap.parse_args()
    d = args.trajectory_dir
    steps = list(csv.DictReader((d / 'trajectory_steps.csv').open()))
    trans = list(csv.DictReader((d / 'trajectory_transitions.csv').open()))
    report = json.loads(args.report_on.read_text())
    manifest = json.loads((d / 'trajectory_manifest.json').read_text())
    results, ok = [], True

    def check(name, condition, detail=''):
        nonlocal ok
        ok &= bool(condition)
        results.append(f"{'PASS' if condition else 'FAIL'}  {name} {detail}")

    check('manifest complete', manifest['status'] == 'ok'
          and all(e['complete'] for e in manifest['episodes']))
    check('manifest has no diagnostic read errors', not manifest['diagnostic_read_errors'],
          repr(manifest['diagnostic_read_errors']))
    check('manifest has no unknown reward terms', not manifest['reward_terms_not_in_config'],
          repr(manifest['reward_terms_not_in_config']))
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((d / name).read_bytes()).hexdigest()
        check(f'{name} hash matches manifest', actual == expected)
    check('steps/transitions same length', len(steps) == len(trans), f'({len(steps)}/{len(trans)})')
    episodes = [e for run in report['runs'] for e in run['episode_results']]
    by_id = {}
    for row in steps:
        by_id.setdefault(row['trajectory_id'], []).append(row)
    tids = {row['trajectory_id'] for row in steps}
    check('one trajectory per episode', len(by_id) == len(episodes), f'({len(by_id)}/{len(episodes)})')
    for ep, man in zip(episodes, manifest['episodes']):
        rows = by_id.get(man['trajectory_id'], [])
        tag = f"[{man['trajectory_id']}]"
        if not rows:
            check(f'{tag} trajectory rows present', False)
            continue
        check(f'{tag} rows == env steps', len(rows) == ep['steps'], f"({len(rows)}/{ep['steps']})")
        check(f'{tag} step contiguous from 0', [int(r['step']) for r in rows] == list(range(len(rows))))
        flags = [int(r['is_terminal_step']) for r in rows]
        check(f'{tag} exactly one terminal row, and it is last', sum(flags) == 1 and flags[-1] == 1)
        check(f'{tag} terminal reason == evaluator outcome', rows[-1]['terminal_reason'] == ep['outcome'],
              f"({rows[-1]['terminal_reason']!r})")
        check(f'{tag} return == evaluator return', math.isclose(
            float(rows[-1]['return_cum']), ep['policy_reward_return'], abs_tol=1e-9))
        check(f'{tag} reward ledger consistent', max(float(r['reward_terms_sum_abs_err']) for r in rows) < 1e-9)
        t = [float(r['sim_time_s']) for r in rows if r['sim_time_s'] != '']
        check(f'{tag} sim time strictly increasing by 0.1 s',
              all(abs((b - a) - 0.1) < 1e-6 for a, b in zip(t, t[1:])))
        odo = [float(r['ego_odometer_m']) for r in rows if r['ego_odometer_m'] != '']
        check(f'{tag} odometer non-decreasing', all(b >= a - 1e-6 for a, b in zip(odo, odo[1:])))
        speed = [float(r['ego_speed_mps']) for r in rows if r['ego_speed_mps'] != '']
        check(f'{tag} speed in [0, 40] m/s', all(0 <= s <= 40 for s in speed),
              f'(max {max(speed):.2f})')
        removed = [r for r in rows if int(r['ego_removed_from_sim'])]
        check(f'{tag} removal only on the terminal row', all(int(r['is_terminal_step']) for r in removed))
        check(f'{tag} no after-state fabricated when removed',
              all(r['post_state_source'] == 'unavailable' for r in removed))
        if args.require_route_diagnostics:
            end_distance = [float(r['route_remaining_to_d3_end_m']) for r in rows
                            if r['route_remaining_to_d3_end_m'] != '']
            reachability = [r['exit_reachable'] for r in rows if r['exit_reachable'] != '']
            changes = [float(r['min_lane_changes_required']) for r in rows
                       if r['min_lane_changes_required'] != '']
            check(f'{tag} D3 remaining distance is sampled and non-negative',
                  bool(end_distance) and all(math.isfinite(v) and v >= 0 for v in end_distance))
            check(f'{tag} exit reachability is sampled and binary',
                  bool(reachability) and all(v in ('0', '1') for v in reachability))
            check(f'{tag} minimum lane changes is sampled and non-negative integer',
                  bool(changes) and all(math.isfinite(v) and v >= 0 and v.is_integer() for v in changes))
        results.append(f"INFO  {tag} lane requests={sum(r['lc_request_e76_e77'] not in ('0', '') for r in rows)}, "
                       f"commands sent={sum(r['lc_command_sent'] not in ('0', '') for r in rows)}, "
                       f"physical lane changes={sum(r['lane_change_physical'] == '1' for r in rows)}")
    summary_path = d / 'episode_summary.csv'
    if summary_path.exists():
        with summary_path.open() as handle:
            summary = list(csv.DictReader(handle))
        summary_by_id = {row['trajectory_id']: row for row in summary}
        check('episode summary has one row per trajectory', len(summary) == len(by_id))
        for tid, rows in by_id.items():
            item = summary_by_id.get(tid)
            check(f'[{tid}] episode summary present', item is not None)
            if item is not None:
                check(f'[{tid}] episode summary steps/outcome',
                      int(item['steps']) == len(rows) and item['terminal_reason'] == rows[-1]['terminal_reason'])
                check(f'[{tid}] episode summary return', math.isclose(
                    float(item['policy_return']), float(rows[-1]['return_cum']), abs_tol=1e-9))
    else:
        results.append('INFO  episode_summary.csv absent (legacy trajectory artifact)')
    tr_by = {}
    for row in trans:
        tr_by.setdefault(row['trajectory_id'], []).append(row)
    for tid, rows in tr_by.items():
        obs_keys = vector_columns(rows[0], 'obs_')
        nxt_keys = vector_columns(rows[0], 'next_obs_')
        check(f'[{tid}] obs/next_obs dim 134', len(obs_keys) == 134 == len(nxt_keys))
        check(f'[{tid}] action dim 2 and finite', all(
            math.isfinite(float(r[k])) for r in rows for k in ('action_policy_0', 'action_policy_1')))
        check(f'[{tid}] all obs finite', all(math.isfinite(float(r[k])) for r in rows
                                           for k in obs_keys + nxt_keys))
        check(f'[{tid}] obs_t+1 == next_obs_t', all(
            [a[k] for k in nxt_keys] == [b[k] for k in obs_keys] for a, b in zip(rows, rows[1:])))
        step_last = by_id[tid][-1]
        expected_valid = '1' if step_last['ego_on_road_after'] == '1' else '0'
        check(f'[{tid}] terminal next_obs validity matches Ego availability',
              rows[-1]['next_obs_valid'] == expected_valid,
              f"(actual={rows[-1]['next_obs_valid']}, expected={expected_valid}, "
              f"reason={rows[-1]['terminal_reason']})")
    if args.report_off:
        off = strip(json.loads(args.report_off.read_text()))
        on = strip(report)
        check('evaluation report identical with recording OFF vs ON', off == on)
    print('\n'.join(results))
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
