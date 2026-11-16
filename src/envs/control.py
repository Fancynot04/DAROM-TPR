
"""DAROM Ego 动作解析与 SUMO 控制。"""

from dataclasses import dataclass

import numpy as np

from .lane_change import ExitLaneChangeTracker


MAX_ACCEL = 5.0
MIN_ACCEL = -5.0
MAX_SPEED = 120.0 / 3.6

LANE_CHANGE_EDGES = frozenset(
    ("E75", "E76", "E77", "E78", "E82")
)

EXIT_EDGES = frozenset(("E76", "E77"))
EXIT_LANE_CHANGE_DURATION = 4.0


@dataclass
class ControlResult:
    """记录请求动作及实际提交给 SUMO 的指令。"""

    requested_accel: float
    executed_accel: float
    lane_direction: int
    target_lane: int | None
    speed_limited: bool
    


def parse_action(action):
    """将 SAC 的二维动作解析为加速度与换道方向。"""

    values = np.asarray(action, dtype=float)

    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Action must contain two finite values")

    accel = float(np.clip(values[0], MIN_ACCEL, MAX_ACCEL))
    lateral = float(np.clip(values[1], -5.0, 5.0))

    lower = -5.0 + 10.0 / 3.0
    upper = -5.0 + 20.0 / 3.0

    if lateral < lower:
        direction = -1
    elif lateral < upper:
        direction = 0
    else:
        direction = 1

    return accel, direction


def limit_acceleration(command, speed, dt):
    """限制正向加速度，防止突破道路限速。"""

    if command <= 0:
        return command

    return min(
        command,
        max(0.0, (MAX_SPEED - speed) / dt),
    )


class EgoController:
    """控制一辆 Ego，不负责推进 SUMO 时钟。"""

    def __init__(self, conn, ego_id: str, speed_mode=96):
        self.conn = conn
        self.ego_id = ego_id
        self.speed_mode = int(speed_mode)
        self.exit_change = ExitLaneChangeTracker()

    def configure(self):
        """Ego 插入 SUMO 后执行一次。"""

        vehicle = self.conn.vehicle

        vehicle.setLaneChangeMode(self.ego_id, 0)
        vehicle.setSpeedMode(self.ego_id, self.speed_mode)

    def apply_action(self, action) -> ControlResult:
        """向 SUMO 提交动作，但不推进 simulationStep。"""

        vehicle = self.conn.vehicle

        if self.ego_id not in vehicle.getIDList():
            raise RuntimeError(f"Ego not active: {self.ego_id}")

        accel, direction = parse_action(action)

        road = vehicle.getRoadID(self.ego_id)
        lane = vehicle.getLaneIndex(self.ego_id)
        speed = vehicle.getSpeed(self.ego_id)
        dt = float(self.conn.simulation.getDeltaT())

        if not np.isfinite(speed) or dt <= 0:
            raise ValueError("Invalid speed or simulation step")

        if speed > MAX_SPEED + 1e-3:
            raise RuntimeError(
                f"Ego overspeed: {speed:.4f} m/s"
            )

        # 1. 处理横向换道
        target_lane = None

        if direction != 0 and road in LANE_CHANGE_EDGES:
            candidate = lane + direction
            lane_count = self.conn.edge.getLaneNumber(road)

            if 0 <= candidate < lane_count:
                duration = (
                    EXIT_LANE_CHANGE_DURATION
                    if road in EXIT_EDGES
                    else 0.0
                )

                vehicle.changeLane(
                    self.ego_id,
                    candidate,
                    duration,
                )
                target_lane = candidate
                
                if road == "E77":
                    if candidate == 0:
                        self.exit_change.request(self.conn)
                    else:
                        self.exit_change.reset()

        # 2. 处理纵向加速度
        executed_accel = limit_acceleration(
            accel, speed, dt
        )

        vehicle.setAcceleration(
            self.ego_id,
            executed_accel,
            1,
        )

        return ControlResult(
            requested_accel=accel,
            executed_accel=executed_accel,
            lane_direction=direction,
            target_lane=target_lane,
            speed_limited=executed_accel < accel,
        )

    def observe_after_step(self):
        """在 SumoWorld.step() 之后调用。"""
        self.exit_change.observe(
            self.conn, self.ego_id
        )

    @property
    def active_change_to_zero(self):
        """提供给 Missed Exit 判定的状态。"""
        return self.exit_change.active_to_zero(
            self.conn, self.ego_id
        )