
"""DAROM 单辆 Ego 的局部交通观测。"""

from dataclasses import dataclass

import numpy as np

from src.geometry import (
    TrafficSnapshot,
    driving_axes,
    world_velocity,
)

@dataclass
class ObservationResult:
    """一次观测的完整结果。"""

    observation: np.ndarray
    snapshot: TrafficSnapshot
    ahead_state: list
    behind_state: list


def build_observation(
    conn,
    ego_id: str,
    radius: float = 50.0,
    num_entities: int = 31,
) -> ObservationResult:
    """生成与原 DAROM 一致的 31×3 基础观测。"""

    if num_entities < 1:
        raise ValueError("num_entities must be positive")

    vehicle = conn.vehicle

    if ego_id not in vehicle.getIDList():
        raise RuntimeError(f"Ego not on road: {ego_id}")

    # 1. Ego 自身状态
    ego_pos = vehicle.getPosition(ego_id)
    ego_speed = vehicle.getSpeed(ego_id)
    ego_angle = vehicle.getAngle(ego_id)

    forward, left = driving_axes(ego_angle)

    ego_state = [
        vehicle.getDistance(ego_id),
        vehicle.getLaneIndex(ego_id),
        ego_speed,
    ]

    # 2. 搜索周围车辆
    intruders = []

    closest_ahead = None
    closest_behind = None

    min_ahead_dist = float("inf")
    min_behind_dist = float("inf")

    for veh_id in vehicle.getIDList():
        if veh_id == ego_id:
            continue

        veh_pos = vehicle.getPosition(veh_id)

        distance = np.sqrt(
            (veh_pos[0] - ego_pos[0]) ** 2
            + (veh_pos[1] - ego_pos[1]) ** 2
        )

        if distance > radius:
            continue

        veh_speed = vehicle.getSpeed(veh_id)
        velocity = world_velocity(
            veh_speed,
            vehicle.getAngle(veh_id),
        )

        dx = veh_pos[0] - ego_pos[0]
        dy = veh_pos[1] - ego_pos[1]

        rel_state = [
            dx * forward[0] + dy * forward[1],
            dx * left[0] + dy * left[1],
            float(velocity @ forward) - ego_speed,
        ]

        intruders.append((rel_state, veh_pos, velocity))

        # 保留原来的最近前车、后车判断
        if rel_state[0] > 0 and abs(rel_state[1]) <= 1.6:
            if rel_state[0] < min_ahead_dist:
                min_ahead_dist = rel_state[0]
                closest_ahead = rel_state

        elif rel_state[0] < 0 and abs(rel_state[1]) <= 1.6:
            if abs(rel_state[0]) < min_behind_dist:
                min_behind_dist = abs(rel_state[0])
                closest_behind = rel_state

    # 3. 先选最近的 30 辆，再进行空间排序
    intruders = sorted(
        intruders,
        key=lambda x: x[0][0] ** 2 + x[0][1] ** 2,
    )[:num_entities - 1]

    intruders = sorted(
        intruders,
        key=lambda x: (x[0][1], x[0][0]),
    )

    # 4. 保存原始交通快照，供延迟观测使用
    positions = np.zeros((num_entities - 1, 2))
    velocities = np.zeros((num_entities - 1, 2))
    valid = np.zeros(num_entities - 1, dtype=bool)

    for index, (_, position, velocity) in enumerate(intruders):
        positions[index] = position
        velocities[index] = velocity
        valid[index] = True

    snapshot = TrafficSnapshot(
        conn.simulation.getTime(),
        np.asarray(ego_pos),
        ego_angle,
        positions,
        velocities,
        valid,
    )

    # 5. 不足 30 辆时补零
    neighbor_states = [item[0] for item in intruders]

    while len(neighbor_states) < num_entities - 1:
        neighbor_states.append([0.0, 0.0, 0.0])

    observation = np.asarray(
        [ego_state] + neighbor_states,
        dtype=np.float32,
    )

    return ObservationResult(
        observation=observation,
        snapshot=snapshot,
        ahead_state=(
            closest_ahead
            if closest_ahead is not None
            else [0.0, 0.0, 0.0]
        ),
        behind_state=(
            closest_behind
            if closest_behind is not None
            else [0.0, 0.0, 0.0]
        ),
    )
