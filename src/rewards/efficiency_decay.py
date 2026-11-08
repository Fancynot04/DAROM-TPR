"""Validated linear decay of SECRM efficiency payment by route progress."""
import math


def remaining_distance_factor(remaining, initial, min_factor=0.0, max_factor=1.0):
    values = (remaining, initial, min_factor, max_factor)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f'Nonfinite efficiency decay input: {values!r}')
    if initial <= 0 or remaining < 0 or not 0 <= min_factor <= max_factor <= 1:
        raise ValueError(f'Invalid efficiency decay input: {values!r}')
    progress = min(1.0, max(0.0, remaining / initial))
    factor = min_factor + (max_factor - min_factor) * progress
    if not math.isfinite(factor):
        raise ValueError(f'Nonfinite efficiency decay factor: {factor!r}')
    return float(factor)
