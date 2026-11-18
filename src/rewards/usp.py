"""USP-v1：无必要停滞惩罚。"""

import math


class USPTracker:
    STOP_SPEED = 0.5
    GRACE_SECONDS = 5.0
    LOOKAHEAD = 80.0
    JUNCTION_MARGIN = 15.0

    ROAD_EDGES = frozenset({
        "O3", "E75", "E76", "E77", "D3"
    })

    def __init__(self, agent_ids, penalty=0.02):
        if penalty <= 0:
            raise ValueError("USP penalty must be positive")

        self.agent_ids = tuple(agent_ids)
        self.penalty = float(penalty)
        self.reset()

    def reset(self):
        self.streak = {
            ego_id: 0 for ego_id in self.agent_ids
        }

    def has_forward_opportunity(self, conn, ego_id):
        vehicle = conn.vehicle

        speed = float(vehicle.getSpeed(ego_id))

        if speed >= self.STOP_SPEED:
            return False

        road = vehicle.getRoadID(ego_id)

        # 不在道路连接器内部施加惩罚
        if road not in self.ROAD_EDGES:
            return False

        lane_id = vehicle.getLaneID(ego_id)

        # 靠近当前车道末端时，不轻易认定为无阻塞
        remaining = (
            conn.lane.getLength(lane_id)
            - vehicle.getLanePosition(ego_id)
        )

        if remaining < self.JUNCTION_MARGIN:
            return False

        leader = vehicle.getLeader(
            ego_id, self.LOOKAHEAD
        )

        # 当前车道前方查询范围内无前车
        if leader is None or not leader[0]:
            return True

        leader_id, gap = leader

        if gap < 0:
            return False

        leader_speed = vehicle.getSpeed(leader_id)
        leader_decel = vehicle.getDecel(leader_id)

        secure_gap = vehicle.getSecureGap(
            ego_id,
            speed,
            leader_speed,
            leader_decel,
            leader_id,
        )

        required_gap = (
            max(
                secure_gap,
                vehicle.getMinGap(ego_id),
            )
            + 2.0
        )

        return gap > required_gap

    def step(self, conn, ego_id):
        if self.has_forward_opportunity(conn, ego_id):
            self.streak[ego_id] += 1
        else:
            self.streak[ego_id] = 0

        dt = float(conn.simulation.getDeltaT())

        grace_steps = math.ceil(
            self.GRACE_SECONDS / dt - 1e-9
        )

        # 前 5 秒不扣，之后每步扣分
        if self.streak[ego_id] > grace_steps:
            return -self.penalty

        return 0.0