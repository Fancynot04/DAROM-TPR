"""Coordinate transforms for captured traffic observations."""

from dataclasses import dataclass
import math

import numpy as np


def driving_axes(angle):
    """SUMO headings are clockwise degrees from north; rows are forward/left."""
    radians = math.radians(angle)
    forward = (math.sin(radians), math.cos(radians))
    return np.asarray((forward, (-forward[1], forward[0])), dtype=float)


def world_velocity(speed, angle):
    return speed * driving_axes(angle)[0]


@dataclass(frozen=True)
class TrafficSnapshot:
    time: float
    ego_position: np.ndarray
    ego_angle: float
    positions: np.ndarray
    velocities: np.ndarray
    valid: np.ndarray


def align_delayed_neighbors(captured, current, ego_state):
    """Predict captured neighbors, then express them in the current ego frame.

    Only the current ego pose/time/state is used. Neighbor positions and full
    velocity vectors come from the received packet, never current traffic.
    Constant world velocity is an approximation between packet timestamps.
    """
    age = max(0.0, current.time - captured.time)
    axes = driving_axes(current.ego_angle)
    observation = np.zeros((len(captured.valid) + 1, 3), dtype=np.float32)
    observation[0] = ego_state
    valid = captured.valid
    positions = captured.positions[valid] + captured.velocities[valid] * age
    neighbors = np.empty((np.count_nonzero(valid), 3))
    neighbors[:, :2] = (positions - current.ego_position) @ axes.T
    neighbors[:, 2] = captured.velocities[valid] @ axes[0] - ego_state[2]
    observation[1:][valid] = neighbors
    return observation
