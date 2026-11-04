"""E76-to-D3 reward component."""

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
