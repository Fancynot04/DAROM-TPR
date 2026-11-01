"""Opt-in C2 reward on the unchanged 134-dimensional B3 environment."""

import hashlib
from pathlib import Path

from . import exit_e76_d3_config as exit_config
from .merging import ANTWERP_DIR, LANE_CHANGE_EDGES, Merging
from .route_urgency import load_route_network, observe_route_urgency


NAVIGATION_PENALTY_PER_CHANGE = 0.01
NETWORK_PATH = ANTWERP_DIR / 'Antwerp.net.xml'


def navigation_contract(path=NETWORK_PATH):
    """Checkpoint metadata in final reward units, stable through JSON saving."""
    return {'reward_profile': 'B3', 'penalty_per_change': NAVIGATION_PENALTY_PER_CHANGE,
            'demand': 'minimum_lane_changes_on_actual_route',
            'pending': 'last_settled_value', 'terminal_reward': 0.0,
            'observation_width': 134,
            'network_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def navigation_reward(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError(f'C2 navigation demand must be a nonnegative integer, got {n!r}')
    return -NAVIGATION_PENALTY_PER_CHANGE * n


class C2Merging(Merging):
    """Add one route-topology penalty after B3 has settled its step reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task != exit_config.TASK_ID or self.reward_profile != 'B3':
            raise ValueError('C2 requires the B3 E76-to-D3 task')
        self.route_network = load_route_network(NETWORK_PATH)
        self.navigation_contract = navigation_contract()
        self._navigation_n = None
        self.navigation_reward_total = 0.0

    def reset_episode_state(self):
        super().reset_episode_state()
        self._navigation_n = None
        self.navigation_reward_total = 0.0

    def _navigation_demand(self):
        estimate = observe_route_urgency(self, self.route_network, LANE_CHANGE_EDGES)
        if estimate.get('valid_for_threshold'):
            n = estimate['settled_minimum_lane_changes']
            navigation_reward(n)  # Reject missing or malformed planner output.
            self._navigation_n = n
            return n, True, 'settled', estimate['status']
        if (estimate.get('status') == 'at_destination'
                and estimate.get('road') == 'D3' and not estimate.get('provisional')):
            self._navigation_n = 0
            return 0, True, 'destination', estimate['status']
        if estimate.get('provisional'):
            if self._navigation_n is None:
                raise ValueError('C2 provisional route state has no settled demand cache: '
                                 f'{estimate.get("status")} {estimate.get("lane_id")}')
            return self._navigation_n, False, 'cached', estimate['status']
        raise ValueError('C2 cannot determine route demand in stable state: '
                         f'{estimate.get("status")} {estimate.get("lane_id")}')

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        try:
            n, valid, source, status = self._navigation_demand()
            if not valid:
                raise ValueError(f'C2 reset requires settled navigation demand: {status}')
        except Exception:
            self.close()
            raise
        info.update(navigation_n=n, navigation_valid=True,
                    navigation_source=source, navigation_status=status,
                    navigation_contract=self.navigation_contract.copy())
        return observation, info

    def collect_step(self, was_on_road, close_on_done=True):
        observation, reward, terminated, truncated, info = super().collect_step(
            was_on_road, close_on_done=close_on_done)
        if terminated or truncated:
            n, valid, source, status = self._navigation_n, False, 'terminal', 'terminal'
            increment = 0.0
        else:
            n, valid, source, status = self._navigation_demand()
            increment = navigation_reward(n)
        self.navigation_reward_total += increment
        self.reward += increment
        self.reward_components['navigation'] = increment
        info['reward_components']['navigation'] = increment
        info.update(navigation_n=n, navigation_valid=valid,
                    navigation_source=source, navigation_status=status,
                    navigation_reward_total=self.navigation_reward_total)
        return observation, float(self.reward), terminated, truncated, info
