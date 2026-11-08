"""Traceable reward terms and episode-level reward accounting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RewardTerm:
    name: str
    raw: float
    weight: float
    enabled: bool
    active: bool = True
    clipped: bool = False
    clipped_value: float | None = None
    base_weight: float | None = None

    def __post_init__(self):
        for value in (self.raw, self.weight):
            if not math.isfinite(float(value)):
                raise ValueError(f'Non-finite reward term value: {self.name}')
        if self.clipped_value is not None and not math.isfinite(float(self.clipped_value)):
            raise ValueError(f'Non-finite clipped reward term value: {self.name}')
        if self.base_weight is not None and not math.isfinite(float(self.base_weight)):
            raise ValueError(f'Non-finite base reward term weight: {self.name}')

    @property
    def configured_weight(self) -> float:
        return float(self.weight if self.base_weight is None else self.base_weight)

    @property
    def weighted(self) -> float:
        payable = self.raw if self.clipped_value is None else self.clipped_value
        value = payable * self.weight if self.enabled else 0.0
        if not math.isfinite(value):
            raise ValueError(f'Non-finite weighted reward term: {self.name}')
        return float(value)


class RewardTracker:
    """Episode ledger; active_ratio means raw != 0 on observed steps."""

    def __init__(self, gamma: float):
        if not math.isfinite(float(gamma)) or not 0 <= float(gamma) <= 1:
            raise ValueError('RewardTracker gamma must be finite and in [0, 1]')
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._rows: list[dict[str, RewardTerm]] = []
        self._rewards: list[float] = []

    def add(self, terms: Iterable[RewardTerm], reward: float):
        terms = list(terms)
        row = {term.name: term for term in terms}
        if len(row) != len(terms):
            raise ValueError('Duplicate reward term name')
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError('Non-finite environment reward')
        weighted = sum(term.weighted for term in row.values())
        if not math.isclose(weighted, value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f'Reward ledger mismatch: terms={weighted}, reward={value}')
        self._rows.append(row)
        self._rewards.append(value)

    @staticmethod
    def _percentile(values, percentile):
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if len(values) else 0.0

    def summary(self, terminated: bool = False, truncated: bool = False,
                start: int = 0, end: int | None = None):
        if start < 0 or (end is not None and end < start):
            raise ValueError('Invalid reward summary window')
        rows = self._rows[start:end]
        rewards = self._rewards[start:end]
        # Terminal E76-D3 steps deliberately contain only event terms.  A
        # dense term absent on that step is a non-active zero contribution,
        # rather than a malformed episode ledger.
        names = tuple(dict.fromkeys(name for row in rows for name in row))
        count = len(rows)
        result = {}
        discounts = self.gamma ** np.arange(count, dtype=np.float64)
        for name in names:
            raw = np.asarray([row[name].raw if name in row else 0.0 for row in rows], dtype=np.float64)
            weighted = np.asarray([row[name].weighted if name in row else 0.0 for row in rows], dtype=np.float64)
            abs_values = np.abs(weighted)
            nonzero = abs_values[abs_values > 0]
            result[name] = {
                'raw_sum': float(raw.sum()),
                'weighted_sum': float(weighted.sum()),
                'discounted_sum': float((discounts * weighted).sum()),
                'effective_weight_p95': self._percentile(
                    np.asarray([row[name].weight if name in row else 0.0 for row in rows]), 95),
                'effective_weight_p99': self._percentile(
                    np.asarray([row[name].weight if name in row else 0.0 for row in rows]), 99),
                'configured_weight': float(next((row[name].configured_weight for row in rows
                                                 if name in row), 0.0)),
                'active_ratio': float(np.count_nonzero(raw) / count) if count else 0.0,
                'abs_p95': self._percentile(abs_values, 95),
                'abs_p99': self._percentile(abs_values, 99),
                'nonzero_abs_p95': self._percentile(nonzero, 95),
                'nonzero_abs_p99': self._percentile(nonzero, 99),
                'max_abs': float(abs_values.max()) if count else 0.0,
                'clipped_ratio': float(sum(row[name].clipped for row in rows if name in row) / count) if count else 0.0,
            }
        result['_meta'] = {'steps': count, 'gamma': self.gamma,
                           'terminated': bool(terminated), 'truncated': bool(truncated),
                           'start': start, 'end': end,
                           'weighted_total': float(sum(rewards))}
        return result
