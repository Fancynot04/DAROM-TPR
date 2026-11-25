
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
UNIFORM_REQUEST_DURATIONS = (0.0, 0.1, 0.5, 1.0, 4.0)
DEFAULT_UNIFORM_REQUEST_DURATION = 0.1


@dataclass(frozen=True)
class LaneChangeRequest:
    mode: str = "legacy"
    duration_s: float | None = None

    def __post_init__(self):
        if self.mode == "legacy" and self.duration_s is None:
            return
        if self.mode == "uniform" and self.duration_s in UNIFORM_REQUEST_DURATIONS:
            return
        raise ValueError("Expected legacy without a duration, or uniform with a supported duration")

    def for_edge(self, road):
        if self.mode == "uniform":
            return self.duration_s
        return EXIT_LANE_CHANGE_DURATION if road in EXIT_EDGES else 0.0

    def settings(self):
        return {
            "mode": self.mode,
            "duration_s": self.duration_s,
            "duration_by_edge_s": {edge: self.for_edge(edge) for edge in sorted(LANE_CHANGE_EDGES)},
        }

    @classmethod
    def from_checkpoint(cls, config):
        return cls(
            config.get("lane_change_request_mode", "legacy"),
            config.get("lane_change_request_duration"),
        )


def resolve_lane_change_request(args, saved=LaneChangeRequest()):
    mode = getattr(args, "lane_change_request_mode", None)
    duration = getattr(args, "lane_change_request_duration", None)
    if mode is None and duration is None:
        return saved
    mode = mode or "uniform"
    if mode == "uniform" and duration is None:
        duration = DEFAULT_UNIFORM_REQUEST_DURATION
    return LaneChangeRequest(mode, duration)


def add_lane_change_request_arguments(parser):
    parser.add_argument("--lane-change-request-mode", choices=("legacy", "uniform"),
                        help="Request protocol (default: checkpoint protocol, or legacy for a fresh run)")
    parser.add_argument("--lane-change-request-duration", type=float, choices=UNIFORM_REQUEST_DURATIONS,
                        help="Uniform TraCI request validity in seconds (implies uniform; uniform default: 0.1)")


@dataclass
class ControlResult:
    """记录请求动作及实际提交给 SUMO 的指令。"""

    requested_accel: float
    executed_accel: float
    lane_direction: int
    target_lane: int | None
    speed_limited: bool
    lane_duration: float = 0.0
    lane_change_gate: dict | None = None


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

    def __init__(self, conn, ego_id: str, speed_mode=96, lane_change_gate=None, *,
                 lane_change_request=LaneChangeRequest()):
        self.lane_change_request = lane_change_request
        self.conn = conn
        self.ego_id = ego_id
        self.speed_mode = int(speed_mode)
        self.exit_change = ExitLaneChangeTracker()
        self.lane_change_gate = lane_change_gate

    def configure(self):
        """Ego 插入 SUMO 后执行一次。"""

        vehicle = self.conn.vehicle

        vehicle.setLaneChangeMode(self.ego_id, 0)
        vehicle.setSpeedMode(self.ego_id, self.speed_mode)

    def apply_action(self, action, *, active_lane_change=None) -> ControlResult:
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
        duration = 0.0
        gate_decision = None

        # The lane index can flip while the body is still crossing the lane line.
        # A repeated same-direction action must not request another lane at that point.
        extending_change = active_lane_change == (lane, direction)
        if direction != 0 and road in LANE_CHANGE_EDGES and not extending_change:
            candidate = lane + direction
            lane_count = self.conn.edge.getLaneNumber(road)

            valid_target = 0 <= candidate < lane_count
            if valid_target and self.lane_change_gate is not None and active_lane_change != (candidate, direction):
                gate_decision = self.lane_change_gate.check(vehicle, self.ego_id, direction, speed, candidate)

            if valid_target and not (gate_decision and gate_decision["blocked_by"]):
                duration = self.lane_change_request.for_edge(road)

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
            lane_duration=duration,
            lane_change_gate=gate_decision,
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
