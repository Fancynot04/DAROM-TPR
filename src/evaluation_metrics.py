"""Read-only evaluation statistics; reference: E76 commit 913a73f."""
import math

REFERENCE_COMMIT = '913a73fef22be64ca1ac0ade5426874374b566cd'
KNOWN_OUTCOMES = ('Arrived', 'Collision', 'Ego Teleported', 'Missed Exit', 'Timeout')


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite evaluation measurement')
    return value


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def measured_jerk(before_acc, after_acc, before_time, after_time, step, valid):
    if step == 0 or not valid or before_acc is None or after_acc is None:
        return None
    dt = finite(after_time) - finite(before_time)
    if dt <= 0:
        raise ValueError('Invalid SUMO step length for measured jerk')
    return (finite(after_acc) - finite(before_acc)) / dt


def wilson(successes, total, scale=100):
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid proportion numerator/denominator")
    if total == 0:
        return dict(numerator=successes, denominator=total, value=None, ci95=None, method='Wilson')
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return dict(numerator=successes, denominator=total, value=scale*p,
                ci95=[scale*max(0, center-half), scale*min(1, center+half)], method='Wilson')


class AgentMetrics:
    def __init__(self):
        self.speeds = []
        self.jerks = []
        self.braking_samples = 0
        self.hard_braking_steps = 0
        self.emergency_steps = 0
        self.teleported = False
        self.steps = 0
        self.transitions = 0
        self.event_samples = 0
        self.stopped = 0
        self.requests = self.toward = self.away = 0
        self.physical = self.physical_pairs = 0
        self.first_e77_lane = None
        self.depart_time = self.final_time = None
        self.speed_modes = set()
        self.behavior_available = False

    def observe(self, sample, terminal):
        if sample is None:
            return
        self.event_samples += 1
        pre = sample['pre']
        self.depart_time = sample.get('depart_time', self.depart_time)
        self.final_time = sample['post_time']
        if pre is not None and pre.get('road') == 'E77' and self.first_e77_lane is None:
            self.first_e77_lane = pre['lane']
        if 'post_road' in sample:
            self.behavior_available = True
            if sample['post_road'] == 'E77' and self.first_e77_lane is None:
                self.first_e77_lane = sample['post_lane']
        if pre is not None and 'road' in pre:
            self.speed_modes.add(pre['speed_mode'])
            if pre['road'] == 'E77' and self.first_e77_lane is None:
                self.first_e77_lane = pre['lane']
            direction = pre['request_direction'] if sample['agent_step'] else 0
            self.requests += int(direction != 0)
            self.toward += int(direction < 0 and pre['lane'] > 0)
            self.away += int(direction > 0)
            if (sample.get('post_road') == pre['road'] and sample.get('post_lane') is not None
                    and not sample['teleport_started'] and not sample['teleport_ended']):
                self.physical_pairs += 1
                self.physical += int(sample['post_lane'] != pre['lane'])
        self.teleported |= sample['teleport_started'] or sample['teleport_ended']
        self.emergency_steps += int(sample['emergency_stopping'])
        after = sample['post_acceleration']
        if after is not None:
            after = finite(after)
            self.braking_samples += 1
            self.hard_braking_steps += int(after <= -4.5)
        if pre is not None:
            self.speeds.append(finite(pre['speed']))
            self.stopped += int(pre['speed'] < 0.5)
            finite(pre['acceleration'])
            finite(pre['time'])
            jerk = measured_jerk(pre['acceleration'], after, pre['time'],
                                 sample['post_time'], self.transitions,
                                 not terminal and sample['on_road'])
            if jerk is not None:
                self.jerks.append(abs(jerk))
            self.transitions += 1
        self.steps += int(sample['agent_step'])

    def finish(self, outcome, reward):
        if not isinstance(outcome, str) or not outcome.strip() or outcome in ('Unknown', 'UnknownRemoval'):
            raise RuntimeError(f'Missing or invalid terminal reason: {outcome!r}')
        result = dict(terminal_reason=outcome,
                    outcome=outcome if outcome in KNOWN_OUTCOMES else 'Other',
                    success=outcome == 'Arrived' and not self.teleported,
                    raw_arrival=outcome == 'Arrived', ego_teleported=self.teleported,
                    event_samples=self.event_samples,
                    agent_steps=self.steps, mean_policy_reward_return=finite(reward),
                    mean_speed_mps=mean(self.speeds), speed_samples=len(self.speeds),
                    mean_abs_jerk_mps3=mean(self.jerks), jerk_samples=len(self.jerks),
                    hard_braking_samples=self.braking_samples,
                    hard_braking_steps=self.hard_braking_steps,
                    hard_braking_episode=self.hard_braking_steps > 0,
                    sumo_emergency_stopping_steps=self.emergency_steps,
                    sumo_emergency_stopping_episode=self.emergency_steps > 0)
        result.update(stopped_steps=self.stopped, stopped_speed_threshold_mps=0.5,
                      stopped_step_fraction=self.stopped / len(self.speeds) if self.speeds else None,
                      stopped_step_fraction_stats=wilson(self.stopped, len(self.speeds), scale=1),
                      success_exit_time_s=(self.final_time - self.depart_time
                          if result['success'] and self.final_time is not None and self.depart_time is not None else None),
                      lane_change_requests=self.requests if self.behavior_available else None,
                      toward_requests=self.toward if self.behavior_available else None,
                      away_requests=self.away if self.behavior_available else None,
                      physical_lane_changes=self.physical if self.behavior_available else None,
                      physical_lane_change_valid_pairs=self.physical_pairs,
                      first_e77_lane=self.first_e77_lane,
                      speed_modes=sorted(self.speed_modes),
                      behavior_note='physical changes count only observed same-edge lane-index transitions; cross-edge pairs excluded'
                          if self.behavior_available else 'behavior fields unavailable in input diagnostics')
        result['missing_reasons'] = {}
        if result['success_exit_time_s'] is None:
            result['missing_reasons']['success_exit_time_s'] = ('not_strict_success' if not result['success']
                else 'missing_departure_or_terminal_time')
        if result['first_e77_lane'] is None:
            result['missing_reasons']['first_e77_lane'] = 'no_valid_E77_observation'
        if not self.behavior_available:
            for key in ('lane_change_requests', 'toward_requests', 'away_requests', 'physical_lane_changes'):
                result['missing_reasons'][key] = 'diagnostics_unavailable'
        return result



def joint_record(agents):
    rows = list(agents.values())
    result = dict(success=all(r['success'] for r in rows),
                  raw_arrival=all(r['raw_arrival'] for r in rows))
    for key in ('ego_teleported', 'hard_braking_episode', 'sumo_emergency_stopping_episode'):
        result[key] = any(r[key] for r in rows)
    for key in ('event_samples', 'agent_steps', 'speed_samples', 'jerk_samples', 'hard_braking_samples',
                'hard_braking_steps', 'sumo_emergency_stopping_steps'):
        result[key] = sum(r[key] for r in rows)
    for key in ('mean_speed_mps', 'mean_abs_jerk_mps3'):
        result[key] = mean([r[key] for r in rows])
        result[key + '_valid_agents'] = sum(r[key] is not None for r in rows)
    for outcome, key in (('Collision', 'collision'), ('Timeout', 'timeout'), ('Missed Exit', 'missed_exit')):
        result[key] = any(r['terminal_reason'] == outcome for r in rows)
    result['collision_free'] = not result['collision']
    result['mean_policy_reward_return'] = sum(r['mean_policy_reward_return'] for r in rows)
    return result


def aggregate(rows):
    n = len(rows)
    if not n:
        raise ValueError('No completed episodes')
    result = {'episodes': n, 'rate_unit': 'percent', 'hard_braking_threshold_mps2': -4.5}
    for key, name in (('success', 'success_rate'), ('raw_arrival', 'raw_arrival_rate'),
                      ('ego_teleported', 'ego_teleport_rate'),
                      ('hard_braking_episode', 'hard_braking_episode_rate'),
                      ('sumo_emergency_stopping_episode', 'sumo_emergency_stopping_episode_rate')):
        result[name] = 100 * sum(r[key] for r in rows) / n
        result[name + "_stats"] = wilson(sum(r[key] for r in rows), n)
    for key in ('mean_speed_mps', 'mean_abs_jerk_mps3'):
        result[key] = mean([r[key] for r in rows])
        result[key + '_valid_episodes'] = sum(r[key] is not None for r in rows)
    for key in ('event_samples', 'agent_steps', 'speed_samples', 'jerk_samples', 'hard_braking_samples',
                'hard_braking_steps', 'sumo_emergency_stopping_steps'):
        result[key] = sum(r[key] for r in rows)
    for key in ('hard_braking_episode', 'sumo_emergency_stopping_episode'):
        result[key + 's'] = sum(r[key] for r in rows)
    if 'terminal_reason' in rows[0]:
        result['outcome_counts'] = {key: sum(r['outcome'] == key for r in rows)
                                    for key in (*KNOWN_OUTCOMES, 'Other')}
        for outcome, name in (('Collision', 'collision_rate'), ('Timeout', 'timeout_rate'),
                              ('Missed Exit', 'missed_exit_rate')):
            result[name] = 100 * result['outcome_counts'][outcome] / n
            result[name + "_stats"] = wilson(result["outcome_counts"][outcome], n)
        result['mean_policy_reward_return'] = mean([r['mean_policy_reward_return'] for r in rows])
        for key in ('success_exit_time_s', 'lane_change_requests', 'toward_requests', 'away_requests', 'physical_lane_changes'):
            values = [r.get(key) for r in rows]
            result['mean_' + key] = mean(values)
            result[key + '_valid_episodes'] = sum(v is not None for v in values)
        result['physical_lane_change_valid_pairs'] = sum(r.get('physical_lane_change_valid_pairs', 0) for r in rows)
        stopped = sum(r.get('stopped_steps', 0) for r in rows)
        result['stopped_step_fraction_stats'] = wilson(stopped, result['speed_samples'], scale=1)
        result['stopped_step_fraction'] = result['stopped_step_fraction_stats']['value']
        lanes = [r.get('first_e77_lane') for r in rows if r.get('first_e77_lane') is not None]
        result['first_e77_lane_distribution'] = {str(lane): wilson(lanes.count(lane), len(lanes)) for lane in sorted(set(lanes))}
        result['first_e77_lane_valid_episodes'] = len(lanes)
    else:
        for key in ('collision', 'timeout', 'missed_exit'):
            result[key + '_rate'] = 100 * sum(r[key] for r in rows) / n
            result[key + '_rate_stats'] = wilson(sum(r[key] for r in rows), n)
        for key in ('mean_speed_mps', 'mean_abs_jerk_mps3'):
            result[key + '_valid_agent_episodes'] = sum(r[key + '_valid_agents'] for r in rows)
    if 'collision_free' in rows[0]:
        result['collision_free_rate_stats'] = wilson(sum(r['collision_free'] for r in rows), n)
        result['collision_free_rate'] = result['collision_free_rate_stats']['value']
        result['mean_policy_reward_return'] = mean([r['mean_policy_reward_return'] for r in rows])
    result['mean_policy_reward_return_valid_episodes'] = n
    return result
