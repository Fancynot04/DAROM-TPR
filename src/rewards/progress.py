"""E76-to-D3 reward component."""
import math

def progress_reward(before_remaining, after_remaining, before_odometer,
                    after_odometer, initial_remaining, reward):
    values = (before_remaining, after_remaining, before_odometer,
              after_odometer, initial_remaining)
    if any(value is None or not math.isfinite(value) for value in values):
        return 0.0
    if before_remaining < 0 or after_remaining < 0 or initial_remaining <= 0:
        return 0.0
    meters = min(max(0.0, before_remaining - after_remaining),
                 max(0.0, after_odometer - before_odometer))
    closeness = 1.0 - min(max(before_remaining / initial_remaining, 0.0), 1.0)
    return reward['progress_per_meter'] * meters * (1.0 + reward['proximity_gain'] * closeness)
