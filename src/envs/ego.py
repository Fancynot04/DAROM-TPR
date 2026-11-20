
from dataclasses import dataclass
from typing import Optional


@dataclass
class EgoState:
    """单辆 Ego 的运行状态。"""

    ego_id: str
    status: str = "waiting"

    depart_time: Optional[float] = None

    road_id: Optional[str] = None
    lane_id: Optional[str] = None
    lane_index: Optional[int] = None
    speed: Optional[float] = None

    terminal_reason: Optional[str] = None


class Ego:
    """管理单辆 Ego 的状态，不负责推进 SUMO。"""

    def __init__(self, ego_id: str):
        self.state = EgoState(ego_id=ego_id)

    @property
    def ego_id(self):
        return self.state.ego_id

    def update(self, conn):
        """在 SUMO 推进后同步车辆状态。"""

        sim = conn.simulation
        vehicle = conn.vehicle
        ego_id = self.ego_id

        if self.state.status == "finished":
            return self.state

        if ego_id in sim.getDepartedIDList():
            self.state.status = "active"
            self.state.depart_time = sim.getTime()

        if self.state.status == "waiting":
            return self.state

        # 记录本步 SUMO 事件，不能将消失视为成功
        if ego_id in sim.getCollidingVehiclesIDList():
            self.finish("Collision")
        elif ego_id in sim.getArrivedIDList():
            self.finish("Arrived")

        if self.state.status == "finished":
            return self.state

        if ego_id not in vehicle.getIDList():
            # 原因尚未确认，交由环境处理
            self.finish("UnknownRemoval")
            return self.state

        self.state.road_id = vehicle.getRoadID(ego_id)
        self.state.lane_id = vehicle.getLaneID(ego_id)
        self.state.lane_index = vehicle.getLaneIndex(ego_id)
        self.state.speed = vehicle.getSpeed(ego_id)

        return self.state

    def finish(self, reason: str):
        """记录结束原因，后续由环境统一结算奖励。"""
        self.state.status = "finished"
        self.state.terminal_reason = reason
