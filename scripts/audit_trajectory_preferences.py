"""Audit p1b-lexicographic-v1 against immutable trajectory exports.

This script replays the existing comparator for consistency checks and writes
review tables only.  It does not change preference rules or source CSV files.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from scripts.compare_trajectory_preferences import (
    RULE_VERSION,
    Trajectory,
    choose,
    comparable,
    number,
    physical_directions,
    route_features,
    safety_features,
)


CONTROL_STEP_S = 0.1
SAMPLE_QUOTAS = {'P1': 4, 'P2': 4, 'P3': 7, 'P4': 3, 'similar': 2}
REQUIRED_STEP_FIELDS = (
    'trajectory_id', 'step', 'ego_road_id', 'ego_road_id_after', 'ego_lane_index',
    'ego_lane_index_after', 'lane_change_physical', 'ttc_leader_s', 'ttc_leader_status',
    'ttc_follower_s', 'ttc_follower_status', 'min_lane_changes_required',
    'route_state_settled', 'route_remaining_to_d3_start_m',
    'route_remaining_to_d3_end_m',
)
REQUIRED_SUMMARY_FIELDS = (
    'trajectory_id', 'eval_seed', 'scenario', 'terminal_reason', 'hard_braking_steps',
)
REASON_BRANCHES = {
    'P1': {
        'Arrived is preferred to non-Arrived terminal outcome': 'task_success',
    },
    'P2': {
        'non-collision is preferred when P1 ties': 'non_collision',
        'at least 5 fewer TTC<2s steps with no more hard-braking steps': 'ttc_margin_with_hard_braking_guard',
    },
    'P3': {
        'fewer physical lane changes away from the exit side': 'fewer_away_lane_changes',
        'reached settled zero required lane changes while the other trajectory did not': 'settled_required_lane_changes_zero',
    },
    'P4': {
        'at least 5 seconds shorter recorded travel time': 'time_margin_50_steps',
    },
    'similar': {
        'no reliable lexicographic difference under P1-P4': 'no_reliable_difference',
    },
}


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_inputs(groups, preferences_dir):
    trajectories = {}
    group_by_id = {}
    field_info = {}
    for group, directory in groups:
        directory = Path(directory)
        summaries = read_csv(directory / 'episode_summary.csv')
        rows_by_id = defaultdict(list)
        step_rows = read_csv(directory / 'trajectory_steps.csv')
        for row in step_rows:
            rows_by_id[row['trajectory_id']].append(row)
        for summary in summaries:
            trajectory = Trajectory(group, summary, rows_by_id[summary['trajectory_id']])
            if trajectory.id in trajectories:
                raise ValueError(f'duplicate trajectory_id: {trajectory.id}')
            trajectories[trajectory.id] = trajectory
            group_by_id[trajectory.id] = group
        field_info[group] = {
            'episodes': len(summaries),
            'steps': len(step_rows),
            'summary_fields': list(summaries[0]) if summaries else [],
            'step_fields': list(step_rows[0]) if step_rows else [],
            'missing_summary_fields': sorted(set(REQUIRED_SUMMARY_FIELDS) - set(summaries[0] if summaries else ())),
            'missing_step_fields': sorted(set(REQUIRED_STEP_FIELDS) - set(step_rows[0] if step_rows else ())),
        }
    preferences_dir = Path(preferences_dir)
    preferences = read_csv(preferences_dir / 'trajectory_preferences.csv')
    with (preferences_dir / 'trajectory_preferences_summary.json').open() as handle:
        preference_summary = json.load(handle)
    return trajectories, group_by_id, field_info, preferences, preference_summary


def pair_type(a, b):
    if a.success and b.success:
        return 'success_vs_success'
    if not a.success and not b.success:
        return 'failure_vs_failure'
    return 'success_vs_failure'


def result_bucket(row):
    return row['deciding_priority'] if row['preference'] in ('A', 'B') else row['preference']


def invert_preference(value):
    return {'A': 'B', 'B': 'A', 'similar': 'similar', 'unresolved': 'unresolved'}[value]


def reason_branch(priority, preference, reason):
    key = preference if preference in ('similar', 'unresolved') else priority
    return REASON_BRANCHES.get(key, {}).get(reason, 'unknown')


def strict_edges(rows):
    edges = defaultdict(set)
    for row in rows:
        if row['preference'] == 'A':
            edges[row['eval_seed']].add((row['trajectory_a'], row['trajectory_b']))
        elif row['preference'] == 'B':
            edges[row['eval_seed']].add((row['trajectory_b'], row['trajectory_a']))
    return edges


def three_cycles(rows):
    cycles = []
    for seed, edges in strict_edges(rows).items():
        nodes = sorted({node for edge in edges for node in edge})
        for a, b, c in combinations(nodes, 3):
            if ((a, b) in edges and (b, c) in edges and (c, a) in edges):
                cycles.append((seed, a, b, c))
            elif ((a, c) in edges and (c, b) in edges and (b, a) in edges):
                cycles.append((seed, a, c, b))
    return cycles


def invalid_ttc_rows(trajectory):
    issues = []
    for row in trajectory.rows:
        for side in ('leader', 'follower'):
            value_text = row.get(f'ttc_{side}_s', '')
            status = row.get(f'ttc_{side}_status', '')
            value = number(row, f'ttc_{side}_s')
            invalid = False
            if value_text not in ('', None) and value is None:
                invalid = True
            if value is not None and value < 0:
                invalid = True
            if status != 'ok' and value is not None:
                invalid = True
            if status == 'ok' and value is None:
                invalid = True
            if invalid:
                issues.append({
                    'trajectory_id': trajectory.id,
                    'step': row.get('step', ''),
                    'field': f'ttc_{side}_s',
                    'value': value_text,
                    'status': status,
                    'counted_danger': value is not None and value < 2,
                })
    return issues


def lane_direction_issues(trajectory):
    issues = []
    for row in trajectory.rows:
        physical = row.get('lane_change_physical') == '1'
        before = number(row, 'ego_lane_index')
        after = number(row, 'ego_lane_index_after')
        same_road = row.get('ego_road_id') == row.get('ego_road_id_after')
        changed = before is not None and after is not None and before != after
        if physical != (same_road and changed):
            issues.append({
                'trajectory_id': trajectory.id,
                'step': row.get('step', ''),
                'field': 'lane_change_physical',
                'value': row.get('lane_change_physical', ''),
                'reason': f'road_same={same_road}, lane_before={before}, lane_after={after}',
            })
        if physical and abs(after - before) != 1:
            issues.append({
                'trajectory_id': trajectory.id,
                'step': row.get('step', ''),
                'field': 'ego_lane_index_after',
                'value': row.get('ego_lane_index_after', ''),
                'reason': f'physical lane-index delta is {after - before}',
            })
    return issues


def lane_timeline(trajectory, limit=12):
    events = []
    for row in trajectory.rows:
        if row.get('lane_change_physical') != '1':
            continue
        before = number(row, 'ego_lane_index')
        after = number(row, 'ego_lane_index_after')
        remaining = number(row, 'route_remaining_to_d3_start_m')
        if before is None or after is None:
            continue
        direction = 'toward' if after < before else 'away'
        distance = '' if remaining is None else f'@{remaining:.1f}m'
        events.append(f"s{row.get('step', '')}:{before:.0f}>{after:.0f}:{direction}{distance}")
    if len(events) > limit:
        events = events[:limit] + [f'...({len(events) - limit} more)']
    return ';'.join(events)


def trajectory_review_fields(prefix, trajectory):
    safety = safety_features(trajectory)
    route = route_features(trajectory.rows)
    return {
        f'group_{prefix}': trajectory.group,
        f'terminal_{prefix}': trajectory.summary['terminal_reason'],
        f'collision_{prefix}': int(safety['collision']),
        f'danger_ttc_steps_{prefix}': safety['danger_steps'],
        f'hard_braking_steps_{prefix}': safety['hard_braking'],
        f'toward_changes_{prefix}': route['directions']['toward'],
        f'away_changes_{prefix}': route['directions']['away'],
        f'first_settled_zero_distance_m_{prefix}': route['first_settled_zero_distance_m'],
        f'episode_steps_{prefix}': len(trajectory.rows),
        f'episode_time_s_{prefix}': round(len(trajectory.rows) * CONTROL_STEP_S, 1),
        f'lane_change_timeline_{prefix}': lane_timeline(trajectory),
    }


def choose_samples(audited_rows, trajectories):
    selected = []
    used_pair_ids = set()
    for bucket, quota in SAMPLE_QUOTAS.items():
        candidates = [row for row in audited_rows if result_bucket(row) == bucket]
        def score(row):
            a, b = trajectories[row['trajectory_a']], trajectories[row['trajectory_b']]
            pair = tuple(sorted((a.group, b.group)))
            route = route_features(a.rows), route_features(b.rows)
            b2 = int('B2' in pair)
            away_gap = abs(route[0]['directions']['away'] - route[1]['directions']['away'])
            return (-b2 if bucket == 'P3' else 0, -away_gap if bucket == 'P3' else 0,
                    pair, int(row['eval_seed']), row['pair_id'])
        candidates.sort(key=score)
        seen_branches = set()
        seen_pairs = set()
        ordered = []
        for row in candidates:
            branch = row['trigger_branch']
            if branch not in seen_branches:
                ordered.append(row)
                seen_branches.add(branch)
        for row in candidates:
            a, b = trajectories[row['trajectory_a']], trajectories[row['trajectory_b']]
            combo = tuple(sorted((a.group, b.group)))
            if combo not in seen_pairs and row not in ordered:
                ordered.append(row)
                seen_pairs.add(combo)
        ordered.extend(row for row in candidates if row not in ordered)
        for row in ordered:
            if row['pair_id'] in used_pair_ids:
                continue
            selected.append(row)
            used_pair_ids.add(row['pair_id'])
            if sum(result_bucket(item) == bucket for item in selected) == quota:
                break
        if sum(result_bucket(item) == bucket for item in selected) != quota:
            raise ValueError(f'not enough sample pairs for {bucket}')
    return selected


def write_csv(path, rows, fields):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(groups, preferences_dir, output_dir):
    trajectories, group_by_id, field_info, preferences, preference_summary = load_inputs(
        groups, preferences_dir)
    if preference_summary.get('rule_version') != RULE_VERSION:
        raise ValueError('preference summary rule version mismatch')
    if preference_summary.get('total_pairs') != len(preferences):
        raise ValueError('preference summary pair count mismatch')

    audited = []
    pair_counts = Counter()
    trigger_counts = Counter()
    p2_guard_counts = Counter()
    swap_issues = []
    replay_issues = []
    similar_issues = []
    initial_issues = []
    for row in preferences:
        if row['trajectory_a'] not in trajectories or row['trajectory_b'] not in trajectories:
            raise ValueError(f"unknown trajectory in {row['pair_id']}")
        a, b = trajectories[row['trajectory_a']], trajectories[row['trajectory_b']]
        actual = choose(a, b)
        recorded = (row['comparable'] == '1', row['preference'], row['deciding_priority'], row['reason'])
        if actual != recorded:
            replay_issues.append({'pair_id': row['pair_id'], 'recorded': recorded, 'actual': actual})
        decision = dict(row)
        decision.update(comparable=str(int(actual[0])), preference=actual[1],
                        deciding_priority=actual[2], reason=actual[3])
        reverse = choose(b, a)
        if (reverse[0] != actual[0] or reverse[1] != invert_preference(actual[1])
                or reverse[2] != actual[2]):
            swap_issues.append({'pair_id': row['pair_id'], 'forward': actual, 'reverse': reverse})
        ok, initial_reason = comparable(a, b)
        if not ok:
            initial_issues.append({'pair_id': row['pair_id'], 'reason': initial_reason})
        kind = pair_type(a, b)
        bucket = result_bucket(decision)
        branch = reason_branch(decision['deciding_priority'], decision['preference'], decision['reason'])
        pair_counts[(kind, bucket)] += 1
        trigger_counts[(bucket, branch, decision['reason'])] += 1
        if branch == 'ttc_margin_with_hard_braking_guard':
            safety_a, safety_b = safety_features(a), safety_features(b)
            winner, loser = ((safety_a, safety_b) if decision['preference'] == 'A'
                             else (safety_b, safety_a))
            detail = ('winner_fewer_hard_braking_steps'
                      if winner['hard_braking'] < loser['hard_braking']
                      else 'equal_hard_braking_steps')
            p2_guard_counts[detail] += 1
        enriched = decision
        enriched.update(pair_type=kind, trigger_branch=branch,
                        group_a=group_by_id[a.id], group_b=group_by_id[b.id])
        audited.append(enriched)
        if decision['preference'] == 'similar':
            ok, reason = comparable(a, b)
            required = ('ego_road_id', 'ego_lane_index', 'ego_lane_pos_m', 'ego_speed_mps',
                        'route_remaining_to_d3_end_m')
            missing = [key for key in required if not a.rows[0].get(key) or not b.rows[0].get(key)]
            if not ok or missing:
                similar_issues.append({'pair_id': row['pair_id'], 'reason': reason, 'missing': missing})

    cycles = three_cycles(audited)
    ttc_issues = [issue for trajectory in trajectories.values() for issue in invalid_ttc_rows(trajectory)]
    lane_issues = [issue for trajectory in trajectories.values() for issue in lane_direction_issues(trajectory)]

    stats = []
    all_buckets = ('P1', 'P2', 'P3', 'P4', 'similar', 'unresolved')
    all_types = ('success_vs_failure', 'success_vs_success', 'failure_vs_failure')
    for kind in all_types:
        for bucket in all_buckets:
            stats.append({'section': 'pair_type_cross_tab', 'comparison_type': kind,
                          'deciding_priority': bucket, 'trigger': '',
                          'count': pair_counts[(kind, bucket)], 'details': ''})
        stats.append({'section': 'pair_type_total', 'comparison_type': kind,
                      'deciding_priority': '', 'trigger': '',
                      'count': sum(pair_counts[(kind, b)] for b in all_buckets), 'details': ''})
    for (bucket, branch, reason), count in sorted(trigger_counts.items()):
        stats.append({'section': 'rule_trigger', 'comparison_type': '',
                      'deciding_priority': bucket, 'trigger': branch,
                      'count': count, 'details': reason})
    for detail, count in sorted(p2_guard_counts.items()):
        stats.append({'section': 'p2_hard_braking_guard', 'comparison_type': '',
                      'deciding_priority': 'P2', 'trigger': detail,
                      'count': count,
                      'details': 'TTC branch winner must have no more hard-braking steps'})
    checks = {
        'replay_mismatch': len(replay_issues),
        'swap_inversion_issue': len(swap_issues),
        'strict_three_cycle': len(cycles),
        'initial_comparability_issue': len(initial_issues),
        'similar_missing_or_incomparable': len(similar_issues),
        'lane_direction_issue': len(lane_issues),
        'invalid_ttc_issue': len(ttc_issues),
    }
    for name, count in checks.items():
        stats.append({'section': 'consistency_check', 'comparison_type': '',
                      'deciding_priority': '', 'trigger': name, 'count': count,
                      'details': 'pass' if count == 0 else 'review required'})

    selected = choose_samples(audited, trajectories)
    review_rows = []
    for index, row in enumerate(selected, 1):
        a, b = trajectories[row['trajectory_a']], trajectories[row['trajectory_b']]
        review = {
            'sample_index': index,
            'sample_category': result_bucket(row),
            'pair_id': row['pair_id'],
            'eval_seed': row['eval_seed'],
            'trajectory_a': a.id,
            'trajectory_b': b.id,
        }
        review.update(trajectory_review_fields('a', a))
        review.update(trajectory_review_fields('b', b))
        review.update(preference=row['preference'], deciding_priority=row['deciding_priority'],
                      trigger_branch=row['trigger_branch'], reason=row['reason'])
        review_rows.append(review)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    stat_fields = ('section', 'comparison_type', 'deciding_priority', 'trigger', 'count', 'details')
    write_csv(output_dir / 'p1c_statistics.csv', stats, stat_fields)
    review_fields = tuple(review_rows[0])
    write_csv(output_dir / 'p1c_review_samples.csv', review_rows, review_fields)

    totals = {kind: sum(pair_counts[(kind, bucket)] for bucket in all_buckets) for kind in all_types}
    same_success = {bucket: pair_counts[('success_vs_success', bucket)] for bucket in all_buckets}
    p3_triggers = {branch: count for (bucket, branch, _), count in trigger_counts.items() if bucket == 'P3'}
    issue_examples = {
        'replay': replay_issues[:5], 'swap': swap_issues[:5], 'cycles': cycles[:5],
        'initial': initial_issues[:5], 'similar': similar_issues[:5],
        'lane': lane_issues[:5], 'ttc': ttc_issues[:5],
    }
    report = render_report(totals, pair_counts, same_success, p3_triggers, trigger_counts,
                           p2_guard_counts,
                           checks, issue_examples, field_info, review_rows)
    (output_dir / 'p1c_audit.md').write_text(report)
    return {'totals': totals, 'same_success': same_success, 'p3_triggers': p3_triggers,
            'checks': checks, 'samples': len(review_rows)}


def render_report(totals, pair_counts, same_success, p3_triggers, trigger_counts,
                  p2_guard_counts, checks, issue_examples, field_info, review_rows):
    p2_lines = []
    p4_lines = []
    for (bucket, branch, reason), count in sorted(trigger_counts.items()):
        if bucket == 'P2':
            p2_lines.append(f'- `{branch}`：{count} 对（{reason}）')
        if bucket == 'P4':
            p4_lines.append(f'- `{branch}`：{count} 对（{reason}）')
    p3_main = max(p3_triggers.items(), key=lambda item: item[1]) if p3_triggers else ('none', 0)
    all_checks_pass = not any(checks.values())
    conclusion = (
        '代码一致性检查通过，可冻结该提交作为候选 v1 规则，用于下一阶段独立数据的受控试标注；'
        '但不建议直接作为无需人工复核的最终标注器。P3 依赖反向换道计数，且当前规则不判定“足够早”与“末端连续换道”的细粒度差异。'
        if all_checks_pass else
        '发现一致性问题，当前规则不应冻结用于自动标注；应先人工核查下列具体样本，但本审计未修改规则。'
    )
    issue_lines = []
    for category, examples in issue_examples.items():
        if examples:
            issue_lines.append(f'- {category}: `{json.dumps(examples, ensure_ascii=False)}`')
    if not issue_lines:
        issue_lines = ['- 未发现需要定位到具体 trajectory_id 的一致性异常。']
    field_lines = [
        f"- {group}: {info['episodes']} 条轨迹、{info['steps']} 个 step；"
        f"summary {len(info['summary_fields'])} 列、steps {len(info['step_fields'])} 列；"
        f"缺失必需字段：summary={info['missing_summary_fields']}，steps={info['missing_step_fields']}。"
        for group, info in field_info.items()
    ]
    sample_lines = [
        f"- {row['sample_category']} / seed {row['eval_seed']} / {row['group_a']}–{row['group_b']} / "
        f"`{row['pair_id']}`：{row['reason']}"
        for row in review_rows
    ]
    return f"""# P1-C 轨迹偏好规则质量审计

## 数据与字段

- 规则版本：`{RULE_VERSION}`。
- 原始字段直接复用 `episode_summary.csv` 与 `trajectory_steps.csv`；偏好分支通过 `choose()` 逐对复算。
- 车道编号约定为 0 表示最右侧出口车道，因此编号减小为向出口换道，编号增大为反向换道。
{chr(10).join(field_lines)}

## 比较类型

- 成功—失败：{totals['success_vs_failure']} 对。
- 成功—成功：{totals['success_vs_success']} 对。
- 失败—失败：{totals['failure_vs_failure']} 对。
- 同样成功的轨迹由 P2 区分 {same_success['P2']} 对、P3 区分 {same_success['P3']} 对、P4 区分 {same_success['P4']} 对；similar {same_success['similar']} 对。

完整交叉表见 `p1c_statistics.csv`。

## 实际触发分支

### P2

{chr(10).join(p2_lines)}

急刹不是独立获胜分支；它只作为 TTC 分支的“不更多急刹步”守卫条件。在实际 TTC 分支中，获胜者急刹更少 {p2_guard_counts.get('winner_fewer_hard_braking_steps', 0)} 对，急刹相同 {p2_guard_counts.get('equal_hard_braking_steps', 0)} 对。

### P3

- `fewer_away_lane_changes`：{p3_triggers.get('fewer_away_lane_changes', 0)} 对。
- `settled_required_lane_changes_zero`：{p3_triggers.get('settled_required_lane_changes_zero', 0)} 对。

P3 的主要决定分支是 `{p3_main[0]}`，共 {p3_main[1]} 对。

### P4

{chr(10).join(p4_lines)}

因此 P4 的全部配对都由至少 50 控制步的时间差决定。

## 一致性检查

- 原结果逐对复算不一致：{checks['replay_mismatch']}。
- A/B 交换后未正确反转：{checks['swap_inversion_issue']}。
- 严格三元循环：{checks['strict_three_cycle']}。
- 同评估 seed 配对的初始条件不可比：{checks['initial_comparability_issue']}。
- similar 中初始字段缺失或不可比：{checks['similar_missing_or_incomparable']}。
- 实际换道标志或车道编号方向异常：{checks['lane_direction_issue']}。
- TTC 值与状态不一致、负值或可能被错误计为危险：{checks['invalid_ttc_issue']}。

{chr(10).join(issue_lines)}

## 20 对人工复核样本

{chr(10).join(sample_lines)}

详细指标和简要车道变化时间线见 `p1c_review_samples.csv`。P3 样本优先覆盖 B2 及较大的反向换道差异；这些样本最值得人工确认“少反向换道”是否与驾驶语义一致。similar 样本用于确认未决不是由字段缺失造成。

## 冻结建议

{conclusion}

本结论只评价规则实现的一致性和现有 300 对内部行为，不证明该规则对新分布的语义有效性，也不将 300001–300060 用于模型拟合。
"""


def parse_group(value):
    name, sep, directory = value.partition('=')
    if not sep or not name or not directory:
        raise argparse.ArgumentTypeError('NAME=DIRECTORY required')
    return name, Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', type=parse_group, required=True)
    parser.add_argument('--preferences-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.group, args.preferences_dir, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
