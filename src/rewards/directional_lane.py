"""E76-to-D3 reward component."""
import math


def adaptive_lane_reward(direction, distance, mode):
    """Use the distance captured when the lane-change command was accepted."""
    if mode not in ('l1', 'l2'):
        raise ValueError(f'Unknown adaptive lane reward mode: {mode!r}')
    if distance is None or not math.isfinite(distance) or distance < 0:
        return 0.0, 'invalid_distance'
    if direction not in ('toward', 'away'):
        return 0.0, 'neutral'
    d = min(distance, 1067.0)
    if mode == 'l1':
        f = d / 1067.0
        return (3.0 + 2.0 * f if direction == 'toward' else -9.0 + 3.0 * f), 'ok'
    if d <= 176.0:
        return (3.0 + d / 176.0 if direction == 'toward' else -9.0 + d / 176.0), 'ok'
    return (4.0 + (d - 176.0) / 891.0 if direction == 'toward'
            else -8.0 + 2.0 * (d - 176.0) / 891.0), 'ok'

def directional_lane_reward(before, after, reward):
    """Reward one verified same-edge adjacent lane change by route need."""
    if (before is None or after is None or before['road'] != after['road']
            or abs(after['lane'] - before['lane']) != 1):
        return 0.0, None
    delta = after['minimum_lane_changes'] - before['minimum_lane_changes']
    if delta < 0:
        return float(reward['alpha']), 'toward'
    if delta > 0:
        return -float(reward['beta']), 'away'
    return 0.0, 'neutral'
