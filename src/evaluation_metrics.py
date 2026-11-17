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

    def observe(self, sample, terminal):
        if sample is None:
            return
        self.event_samples += 1
        pre = sample['pre']
        self.teleported |= sample['teleport_started'] or sample['teleport_ended']
        self.emergency_steps += int(sample['emergency_stopping'])
        after = sample['post_acceleration']
        if after is not None:
            after = finite(after)
            self.braking_samples += 1
            self.hard_braking_steps += int(after <= -4.5)
        if pre is not None:
            self.speeds.append(finite(pre['speed']))
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
        return dict(terminal_reason=outcome,
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
        result['mean_policy_reward_return'] = mean([r['mean_policy_reward_return'] for r in rows])
    else:
        for key in ('collision', 'timeout', 'missed_exit'):
            result[key + '_rate'] = 100 * sum(r[key] for r in rows) / n
        for key in ('mean_speed_mps', 'mean_abs_jerk_mps3'):
            result[key + '_valid_agent_episodes'] = sum(r[key + '_valid_agents'] for r in rows)
    return result
