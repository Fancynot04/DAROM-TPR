
"""O3 -> D3 驶离任务的终止判定规则。"""

import math


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

    # 5. 判断车辆能否在道路末端前停止
    speed = max(0.0, vehicle.getSpeed(ego_id))

    emergency_decel = vehicle.getEmergencyDecel(ego_id)

    effective_decel = min(
        emergency_decel,
        max_action_decel,
    )

    if (
        not math.isfinite(effective_decel)
        or effective_decel <= 0
    ):
        raise ValueError("Invalid effective deceleration")

    stopping_distance = (
        speed ** 2 / (2.0 * effective_decel)
    )

    return stopping_distance > remaining
