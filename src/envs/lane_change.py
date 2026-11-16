
"""跟踪 Ego 在 E77 上向 0 号出口车道换道的状态。"""

from dataclasses import dataclass


@dataclass
class LaneChangeState:
    target_lane: int
    start_time: float
    last_motion_time: float | None = None
    stable_steps: int = 0


class ExitLaneChangeTracker:
    """只跟踪 E77 -> E77_0 的换道，不推进 SUMO。"""

    def __init__(self):
        self.state = None

    def reset(self):
        self.state = None

    def request(self, conn):
        """记录一次指向 0 号车道的请求。"""

        if self.state is None:
            self.state = LaneChangeState(
                target_lane=0,
                start_time=float(
                    conn.simulation.getTime()
                ),
            )

        # 重复相同请求不重置开始时间

    def observe(self, conn, ego_id):
        """SUMO 推进一步后调用，更新实际换道状态。"""

        state = self.state
        if state is None:
            return

        vehicle = conn.vehicle

        if ego_id not in vehicle.getIDList():
            self.reset()
            return

        if vehicle.getRoadID(ego_id) != "E77":
            self.reset()
            return

        now = float(conn.simulation.getTime())
        lane_index = vehicle.getLaneIndex(ego_id)
        lateral_speed = float(
            vehicle.getLateralSpeed(ego_id)
        )

        # 负横向速度：向右侧出口车道移动
        if lateral_speed < -0.05:
            state.last_motion_time = now

        # 判断是否稳定进入 E77_0
        settled = False

        if lane_index == 0:
            lane_id = vehicle.getLaneID(ego_id)

            clearance = (
                conn.lane.getWidth(lane_id)
                - vehicle.getWidth(ego_id)
            ) / 2.0

            settled = (
                clearance >= 0
                and abs(
                    vehicle.getLateralLanePosition(ego_id)
                ) <= clearance + 0.001
                and abs(lateral_speed) <= 0.05
            )

        state.stable_steps = (
            state.stable_steps + 1 if settled else 0
        )

        if state.stable_steps >= 3:
            self.reset()
            return

        # 长时间没有实际横向移动：请求失效
        if state.last_motion_time is None:
            if now - state.start_time >= 6.0:
                self.reset()
        elif now - state.last_motion_time >= 2.0:
            self.reset()

    def active_to_zero(self, conn, ego_id):
        """是否有近期实际向 E77_0 移动的证据。"""

        state = self.state

        if state is None:
            return False

        vehicle = conn.vehicle

        if ego_id not in vehicle.getIDList():
            return False

        if vehicle.getRoadID(ego_id) != "E77":
            return False

        if vehicle.getLaneIndex(ego_id) == 0:
            return False

        if state.last_motion_time is None:
            return False

        now = float(conn.simulation.getTime())

        return (
            now - state.last_motion_time < 2.0
            and (
                now - state.start_time < 6.0
                or vehicle.getLateralSpeed(ego_id) < -0.05
            )
        )
