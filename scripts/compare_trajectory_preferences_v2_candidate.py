"""Validate a P3-only v2 candidate against immutable P1-A/P1-B data.

The v1 implementation and CSV remain untouched.  P1, P2 and P4 are copied by
calling the same helpers and constants as v1.  Stable n=0 rules are enabled
only if the supplied successful trajectories contain no false-negative marker.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_trajectory_preferences import three_cycles
from scripts.compare_trajectory_preferences import (
    SAFETY_STEP_MARGIN,
    TIME_STEP_MARGIN,
    choose as choose_v1,
    comfort_features,
    comparable,
    load_group,
    number,
    parse_group,
    physical_directions,
    safety_features,
)


RULE_VERSION = 'p1b-lexicographic-v2-candidate'
EARLY_COMPLETION_MARGIN_M = 50.0
FIELDS = ('pair_id', 'eval_seed', 'trajectory_a', 'trajectory_b', 'comparable',
          'preference', 'deciding_priority', 'reason', 'rule_version')


def stable_zero_features(rows):
    stable = []
    any_zero = []
    for index, row in enumerate(rows):
        need = number(row, 'min_lane_changes_required')
        distance = number(row, 'route_remaining_to_d3_start_m')
        if need == 0:
            any_zero.append((index, distance, row.get('route_status', ''),
                             row.get('route_state_settled', '')))
            if row.get('route_state_settled') == '1' and distance is not None:
                stable.append((index, distance))
    return {
        'completed': bool(stable),
        'first_distance_m': stable[0][1] if stable else None,
        'any_zero': bool(any_zero),
        'zero_only_unsettled': bool(any_zero) and not stable,
        'any_zero_rows': len(any_zero),
    }


def audit_stable_zero(trajectories):
    by_group = defaultdict(Counter)
    missing_success_ids = []
    for trajectory in trajectories:
        features = stable_zero_features(trajectory.rows)
        counts = by_group[trajectory.group]
        counts['trajectories'] += 1
        if trajectory.success:
            counts['successful'] += 1
            if features['completed']:
                counts['successful_with_stable_zero'] += 1
            else:
                counts['successful_without_stable_zero'] += 1
                missing_success_ids.append(trajectory.id)
                if features['zero_only_unsettled']:
                    counts['missing_but_has_unsettled_zero'] += 1
                elif not features['any_zero']:
                    counts['missing_and_no_zero_observation'] += 1
    totals = Counter()
    for counts in by_group.values():
        totals.update(counts)
    reliable = totals['successful_without_stable_zero'] == 0
    return {
        'reliable_for_completion_preference': reliable,
        'reason': ('all successful trajectories have a stable n=0 observation'
                   if reliable else
                   'successful trajectories without stable n=0 create false negatives'),
        'definition': ('min_lane_changes_required=0 with route_state_settled=1; '
                       'route_state_settled requires non-pending, non-connector, status=ok, '
                       'so at_destination is excluded'),
        'groups': {group: dict(counts) for group, counts in sorted(by_group.items())},
        'totals': dict(totals),
        'missing_success_ids': missing_success_ids,
    }


def route_features_v2(rows):
    stable = stable_zero_features(rows)
    stable['directions'] = physical_directions(rows)
    return stable


def choose_v2(a, b, *, enable_stable_zero):
    """P1/P2/P4 match v1 exactly; only P3 ordering is a candidate change."""
    ok, reason = comparable(a, b)
    if not ok:
        return False, 'unresolved', 'none', reason

    # P1: byte-for-byte equivalent decision condition to v1.
    if a.success != b.success:
        return True, 'A' if a.success else 'B', 'P1', \
            'Arrived is preferred to non-Arrived terminal outcome'

    # P2: same collision and TTC dominance rules and thresholds as v1.
    sa, sb = safety_features(a), safety_features(b)
    if sa['collision'] != sb['collision']:
        return True, 'A' if not sa['collision'] else 'B', 'P2', \
            'non-collision is preferred when P1 ties'
    if ((sa['danger_steps'] + SAFETY_STEP_MARGIN <= sb['danger_steps']
         and sa['hard_braking'] <= sb['hard_braking'])
            or (sb['danger_steps'] + SAFETY_STEP_MARGIN <= sa['danger_steps']
                and sb['hard_braking'] <= sa['hard_braking'])):
        winner = 'A' if sa['danger_steps'] < sb['danger_steps'] else 'B'
        return True, winner, 'P2', \
            'at least 5 fewer TTC<2s steps with no more hard-braking steps'

    # P3 candidate: stable completion first, then >=50 m earlier completion,
    # then reverse lane changes.  The first two branches are gated by the
    # dataset audit because stable n=0 can be a false negative.
    ra, rb = route_features_v2(a.rows), route_features_v2(b.rows)
    if enable_stable_zero:
        if ra['completed'] != rb['completed']:
            return True, 'A' if ra['completed'] else 'B', 'P3', \
                'stable completion of required lane changes is preferred'
        if ra['completed'] and rb['completed']:
            difference = ra['first_distance_m'] - rb['first_distance_m']
            if abs(difference) >= EARLY_COMPLETION_MARGIN_M:
                return True, 'A' if difference > 0 else 'B', 'P3', \
                    'stable completion occurs at least 50 m earlier'
    if ra['directions']['away'] != rb['directions']['away']:
        return True, ('A' if ra['directions']['away'] < rb['directions']['away'] else 'B'), \
            'P3', 'fewer physical lane changes away from the exit side'

    # P4: same 50-step travel-time rule as v1.
    ca, cb = comfort_features(a), comfort_features(b)
    if abs(ca['steps'] - cb['steps']) >= TIME_STEP_MARGIN:
        return True, 'A' if ca['steps'] < cb['steps'] else 'B', 'P4', \
            'at least 5 seconds shorter recorded travel time'
    return True, 'similar', 'none', 'no reliable lexicographic difference under P1-P4'


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def invert(value):
    return {'A': 'B', 'B': 'A', 'similar': 'similar', 'unresolved': 'unresolved'}[value]


def make_row(seed, a, b, result):
    ok, preference, priority, reason = result
    return {
        'pair_id': f'{seed}_{a.group}_vs_{b.group}', 'eval_seed': seed,
        'trajectory_a': a.id, 'trajectory_b': b.id, 'comparable': int(ok),
        'preference': preference, 'deciding_priority': priority,
        'reason': reason, 'rule_version': RULE_VERSION,
    }


def compare(groups, v1_preferences, output_dir):
    trajectories = [trajectory for group, directory in groups
                    for trajectory in load_group(group, directory)]
    by_id = {trajectory.id: trajectory for trajectory in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError('duplicate trajectory_id')
    v1_rows = read_csv(v1_preferences)
    v1_by_pair = {row['pair_id']: row for row in v1_rows}
    if len(v1_by_pair) != 300 or len(v1_rows) != 300:
        raise ValueError(f'expected 300 unique v1 pairs, got {len(v1_rows)}')

    stable_audit = audit_stable_zero(trajectories)
    enable_stable = stable_audit['reliable_for_completion_preference']
    by_seed = defaultdict(list)
    for trajectory in trajectories:
        by_seed[trajectory.seed].append(trajectory)

    v2_rows = []
    changes = []
    replay_mismatches = []
    swap_issues = []
    missing_issues = []
    counts = Counter()
    for seed, items in sorted(by_seed.items()):
        for index, a in enumerate(items):
            for b in items[index + 1:]:
                pair_id = f'{seed}_{a.group}_vs_{b.group}'
                original = v1_by_pair.get(pair_id)
                if original is None:
                    raise ValueError(f'missing v1 pair: {pair_id}')
                replay = choose_v1(a, b)
                recorded = (original['comparable'] == '1', original['preference'],
                            original['deciding_priority'], original['reason'])
                if replay != recorded:
                    replay_mismatches.append(pair_id)
                result = choose_v2(a, b, enable_stable_zero=enable_stable)
                row = make_row(seed, a, b, result)
                v2_rows.append(row)
                counts[row['preference']] += 1
                if row['deciding_priority'] != 'none':
                    counts[row['deciding_priority']] += 1
                reverse = choose_v2(b, a, enable_stable_zero=enable_stable)
                if (reverse[0] != result[0] or reverse[1] != invert(result[1])
                        or reverse[2] != result[2]):
                    swap_issues.append(pair_id)
                if not result[0]:
                    missing_issues.append({'pair_id': pair_id, 'reason': result[3]})
                if (original['preference'], original['deciding_priority'], original['reason']) != (
                        row['preference'], row['deciding_priority'], row['reason']):
                    changes.append({
                        'pair_id': pair_id,
                        'v1_preference': original['preference'],
                        'v1_priority': original['deciding_priority'],
                        'v1_reason': original['reason'],
                        'v2_preference': row['preference'],
                        'v2_priority': row['deciding_priority'],
                        'v2_reason': row['reason'],
                    })

    cycles = three_cycles(v2_rows)
    key_pairs = []
    for pair_id in ('300016_B1_vs_B2', '300003_B0_vs_B2', '300014_B0_vs_B2'):
        v1 = v1_by_pair[pair_id]
        v2 = next(row for row in v2_rows if row['pair_id'] == pair_id)
        a, b = by_id[v2['trajectory_a']], by_id[v2['trajectory_b']]
        ra, rb = route_features_v2(a.rows), route_features_v2(b.rows)
        key_pairs.append({
            'pair_id': pair_id,
            'v1': f"{v1['preference']} / {v1['deciding_priority']} / {v1['reason']}",
            'v2': f"{v2['preference']} / {v2['deciding_priority']} / {v2['reason']}",
            'a_stable_zero_m': ra['first_distance_m'],
            'b_stable_zero_m': rb['first_distance_m'],
            'a_away': ra['directions']['away'],
            'b_away': rb['directions']['away'],
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'trajectory_preferences_v2_candidate.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(v2_rows)
    report = render_report(stable_audit, counts, changes, replay_mismatches,
                           swap_issues, cycles, missing_issues, key_pairs)
    (output_dir / 'p1d_v1_v2_report.md').write_text(report)
    return {
        'rule_version': RULE_VERSION,
        'stable_n0_enabled': enable_stable,
        'changed_pairs': len(changes),
        'similar': counts['similar'],
        'unresolved': counts['unresolved'],
        'swap_issues': len(swap_issues),
        'cycles': len(cycles),
    }


def render_report(stable, counts, changes, replay_mismatches, swap_issues, cycles,
                  missing_issues, key_pairs):
    totals = stable['totals']
    group_lines = []
    for group, values in stable['groups'].items():
        group_lines.append(
            f"- {group}: 成功 {values.get('successful', 0)}，其中无 stable n=0 "
            f"{values.get('successful_without_stable_zero', 0)}；无 marker 但出现未稳定 n=0 "
            f"{values.get('missing_but_has_unsettled_zero', 0)}。")
    change_lines = [
        f"- `{item['pair_id']}`: {item['v1_preference']}/{item['v1_priority']} → "
        f"{item['v2_preference']}/{item['v2_priority']}；{item['v2_reason']}"
        for item in changes
    ] or ['- 无变化。']
    key_lines = []
    for item in key_pairs:
        key_lines.append(
            f"- `{item['pair_id']}`: v1={item['v1']}；v2={item['v2']}；"
            f"stable n=0 A/B={item['a_stable_zero_m']}/{item['b_stable_zero_m']} m；"
            f"反向换道 A/B={item['a_away']}/{item['b_away']}。")
    decision = (
        'stable n=0 在本批数据中存在成功轨迹假阴性，因此 v2 候选未启用“是否完成”和“50 m提前完成”分支。'
        if not stable['reliable_for_completion_preference'] else
        '本批成功轨迹均有 stable n=0；v2 候选启用了完成状态和50 m提前完成分支。'
    )
    return f"""# P1-D 简单轨迹偏好规则 v2 验证

## stable n=0 审计

定义：{stable['definition']}。

成功轨迹 {totals.get('successful', 0)} 条，其中 {totals.get('successful_without_stable_zero', 0)} 条从未出现 stable n=0；这些轨迹中 {totals.get('missing_but_has_unsettled_zero', 0)} 条出现过 n=0，但当时处于未稳定、连接器或待处理状态。由于 `at_destination` 不满足 `status=ok`，进入 D3 后也不会补记 stable n=0。

{chr(10).join(group_lines)}

结论：{decision}

## v1/v2 结果

- v2 规则版本：`{RULE_VERSION}`。
- stable n=0 相关分支启用：{str(stable['reliable_for_completion_preference']).lower()}。
- 变化配对：{len(changes)} / 300。
- v2 A/B/similar/unresolved：{counts['A']}/{counts['B']}/{counts['similar']}/{counts['unresolved']}。
- v2 P1/P2/P3/P4：{counts['P1']}/{counts['P2']}/{counts['P3']}/{counts['P4']}。

{chr(10).join(change_lines)}

## 重点配对

{chr(10).join(key_lines)}

## 一致性

- v1 原CSV复算不一致：{len(replay_mismatches)}。
- A/B反转问题：{len(swap_issues)}。
- v2严格三元循环：{len(cycles)}。
- 缺失值导致不可比：{len(missing_issues)}。

本次只验证候选规则行为。变化数量不作为优劣判断；`300001–300060` 仍是保留数据，未用于任何模型拟合。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', type=parse_group, required=True)
    parser.add_argument('--v1-preferences', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.group, args.v1_preferences, args.output_dir),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
