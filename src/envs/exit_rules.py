
"""Route-aware exit rules and the legacy D3 checkpoint rule."""

import math

from .control import MIN_ACCEL


def route_missed_before_step(conn, ego_id, plan, position, *, check_distance=10.0):
    """Check a required route transition that the current lane cannot make."""
    if position not in plan.signed_changes:
        return True
    if position in plan.forward_lanes or position[0] == len(plan.route) - 1:
        return False

    vehicle = conn.vehicle
    if vehicle.getRoadID(ego_id).startswith(":"):
        return False
    remaining = max(0.0, conn.lane.getLength(vehicle.getLaneID(ego_id))
                    - vehicle.getLanePosition(ego_id))
    if remaining > check_distance:
        return False

    lateral_speed = vehicle.getLateralSpeed(ego_id)
    if abs(lateral_speed) > 0.05:
        step = 1 if lateral_speed > 0 else -1
        neighbor = position[0], position[1] + step
        if neighbor in plan.lateral_neighbors[position] and neighbor in plan.forward_lanes:
            return False
    return _stopping_distance(vehicle, ego_id, -MIN_ACCEL) > remaining


def _stopping_distance(vehicle, ego_id, max_action_decel):
    speed = max(0.0, vehicle.getSpeed(ego_id))
    effective_decel = min(vehicle.getEmergencyDecel(ego_id), max_action_decel)
    if not math.isfinite(effective_decel) or effective_decel <= 0:
        raise ValueError("Invalid effective deceleration")
    return speed ** 2 / (2.0 * effective_decel)


def missed_exit_before_step(
    conn,
    ego_id: str,
    *,
    active_change_to_zero: bool = False,
    check_distance: float = 10.0,
    max_action_decel: float = 5.0,
) -> bool:
    """判断 Ego 是否已经无法挽救 D3 出口。

    注意：本函数只读取 SUMO 状态，不推进仿真，
    不执行换道，也不修改车辆状态。
    """

    vehicle = conn.vehicle

    if ego_id not in vehicle.getIDList():
        raise RuntimeError(f"Ego not on road: {ego_id}")

    # 1. 仅在 E77 判断
    road = vehicle.getRoadID(ego_id)

    if road != "E77":
        return False

    # 2. E77_0 可以直接驶离
    lane_index = vehicle.getLaneIndex(ego_id)

    if lane_index == 0:
        return False

    # 3. 计算距离当前车道末端的距离
    lane_id = vehicle.getLaneID(ego_id)

    lane_length = conn.lane.getLength(lane_id)
    lane_position = vehicle.getLanePosition(ego_id)

    remaining = max(0.0, lane_length - lane_position)

    if remaining > check_distance:
        return False

    # 4. 正在向 0 号车道执行有效换道
    if active_change_to_zero:
        return False

    return _stopping_distance(vehicle, ego_id, max_action_decel) > remaining
