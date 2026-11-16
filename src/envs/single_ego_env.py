
"""O3 -> D3 单 Ego Gymnasium 环境。"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sumo_world import SumoWorld
from .ego import Ego
from .control import EgoController
from .observation import build_observation

from src.rewards.events import event_reward
from .exit_rules import missed_exit_before_step

# O3 -> D3 初始 B0 奖励，沿用原驶离任务
B0_EVENTS = {
    "success": 50.0,
    "collision": -100.0,
    "missed_exit": -100.0,
    "teleport": -100.0,
    "timeout": -100.0,
}


def b0_reward(event):
    """计算 B0 事件奖励，不允许未知事件静默结算。"""

    name, value = event_reward(
        event, {"events": B0_EVENTS}
    )

    if event is not None and name is None:
        raise ValueError(f"Unknown B0 event: {event}")

    return value


class SingleEgoEnv(gym.Env):
    """组合 SUMO 世界、Ego 状态、动作控制与观测。"""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, scenario="easy", gui=False,speed_mode=96):
        super().__init__()

        self.scenario = scenario
        self.gui = gui
        self.ego_id = "ego"
        
        self.seen_e77 = False
        self.reached_d3 = False
        
        self.speed_mode = speed_mode

        # 与旧 DAROM 保持一致
        self.observation_space = spaces.Box(
            low=-5000.0,
            high=5000.0,
            shape=(31, 3),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.world = None
        self.ego = None
        self.controller = None

        self.last_observation = None
        self.last_snapshot = None
        self.step_count = 0
        self.done = False

    def reset(self, *, seed=None, options=None):
        """启动 SUMO，等待 Ego 插入并获取初始观测。"""

        super().reset(seed=seed)

        # 关闭之前可能存在的仿真
        self.close()

        sumo_seed = int(
            self.np_random.integers(0, 2**31 - 1)
        )

        self.world = SumoWorld(
            scenario=self.scenario,
            seed=sumo_seed,
            gui=self.gui,
        )

        self.ego = Ego(self.ego_id)
        self.step_count = 0
        self.done = False
        
        self.seen_e77 = False
        self.reached_d3 = False
        
        self.last_observation = None
        self.last_snapshot = None

        try:
            conn = self.world.start()

            self.controller = EgoController(
                conn, 
                self.ego_id,
                speed_mode=self.speed_mode,
            )

            # 与旧环境一样，等待 Ego 实际出发。
            # 1200 秒是插入截止时间，不是回合时限。
            while conn.simulation.getTime() < 1200.0:
                self.world.step()

                state = self.ego.update(conn)

                if state.status == "finished":
                    raise RuntimeError(
                        "Ego terminated during insertion: "
                        f"{state.terminal_reason}"
                    )

                if state.status != "active":
                    continue

                # 接管车辆控制
                self.controller.configure()

                # 生成初始基础观测
                result = build_observation(
                    conn,
                    ego_id=self.ego_id,
                    radius=50.0,
                    num_entities=31,
                )

                self.last_observation = (
                    result.observation.copy()
                )
                self.last_snapshot = result.snapshot

                info = {
                    "ego_id": self.ego_id,
                    "origin": "O3",
                    "destination": "D3",
                    "scenario": self.scenario,
                    "sumo_seed": sumo_seed,
                    "sumo_version": conn.getVersion(),
                    "depart_time": state.depart_time,
                    "route": list(
                        conn.vehicle.getRoute(self.ego_id)
                    ),
                    "lane": state.lane_id,
                    "speed": state.speed,
                    "simulation_time":
                        conn.simulation.getTime(),
                    "speed_mode_requested": self.speed_mode,
                    "speed_mode_actual": conn.vehicle.getSpeedMode(self.ego_id),
                }

                return self.last_observation.copy(), info

            raise RuntimeError(
                "Ego was not inserted before 1200s"
            )

        except Exception:
            self.close()
            raise
    
    
    def _finish_missed_before_step(self):
        """在 SUMO 推进前终止已无法挽救的驶离任务。"""

        conn = self.world.conn
        state = self.ego.state

        self.step_count += 1
        self.done = True
        self.ego.finish("Missed Exit")
        self.controller.exit_change.reset()

        info = {
            "ego_id": self.ego_id,
            "simulation_time": conn.simulation.getTime(),
            "step_count": self.step_count,
            "terminal_reason": "Missed Exit",
            "is_success": False,
            "road_id": state.road_id,
            "lane_id": state.lane_id,
            "speed": state.speed,
            "requested_accel": None,
            "executed_accel": None,
            "lane_direction": None,
            "target_lane": None,
            "speed_limited": False,
            "sumo_advanced": False,
            "pre_step_terminal": True,
        }

        return (
            self.last_observation.copy(),
            b0_reward("Missed Exit"),
            True,
            False,
            info,
        )

    
    def step(self, action):
        """执行一次控制，检查 O3 -> D3 的终止事件。"""

        if self.world is None or self.controller is None:
            raise RuntimeError("Call reset() first")

        if self.done:
            raise RuntimeError("Episode finished; call reset()")

        conn = self.world.conn
        ego_id = self.ego_id

        # 1. SUMO 推进前检查是否已经错过出口
        if missed_exit_before_step(
            conn,
            ego_id,
            active_change_to_zero=(
                self.controller.active_change_to_zero
            ),
        ):
            return self._finish_missed_before_step()

        # 2. 提交当前动作
        control = self.controller.apply_action(action)

        # 3. 统一推进 SUMO 一个仿真步
        try:
            current_time = self.world.step()
            self.controller.observe_after_step()
        except Exception:
            self.close()
            raise

        self.step_count += 1

        sim = conn.simulation

        # 4. 读取原始 SUMO 事件
        collision = (
            ego_id in sim.getCollidingVehiclesIDList()
        )

        teleport = (
            ego_id in sim.getStartingTeleportIDList()
            or ego_id in sim.getEndingTeleportIDList()
        )

        arrived = (
            ego_id in sim.getArrivedIDList()
        )

        # 5. 更新 Ego 状态
        state = self.ego.update(conn)
        current_road = state.road_id

        # 6. 记录实际经过的关键道路
        if current_road == "E77":
            self.seen_e77 = True

        if current_road == "D3":
            self.reached_d3 = True

        # 穿过 Junction 时不立即判定错过出口
        internal_road = (
            isinstance(current_road, str)
            and current_road.startswith(":")
        )

        # 经过 E77 后进入其他普通道路，且从未到达 D3
        wrong_exit = (
            self.seen_e77
            and not self.reached_d3
            and current_road is not None
            and not internal_road
            and current_road not in ("E77", "D3")
        )

        # 7. 终止事件优先级
        if collision:
            event = "Collision"
        elif teleport:
            event = "Ego Teleported"
        elif wrong_exit:
            event = "Missed Exit"
        elif arrived:
            event = "Arrived"
        elif self.step_count >= 4000:
            event = "Timeout"
        else:
            event = None

        # 未知移除是仿真异常，不是正常训练失败
        if state.status == "finished" and event is None:
            raise RuntimeError(
                "Unexpected Ego termination: "
                f"{state.terminal_reason}"
            )

        # SUMO Arrived 必须确实经过 D3
        if event == "Arrived" and not self.reached_d3:
            raise RuntimeError(
                "Ego arrived without reaching D3"
            )

        # 8. 计算 B0 事件奖励
        reward = b0_reward(event)

        terminated = event is not None
        truncated = False

        if terminated:
            self.done = True
            self.ego.finish(event)

        # 9. 更新观测
        can_observe = (
            event not in (
                "Collision",
                "Ego Teleported",
                "Arrived",
            )
            and ego_id in conn.vehicle.getIDList()
        )

        if can_observe:
            result = build_observation(
                conn,
                ego_id=ego_id,
                radius=50.0,
                num_entities=31,
            )

            self.last_observation = (
                result.observation.copy()
            )
            self.last_snapshot = result.snapshot
        else:
            self.last_snapshot = None

        obs = self.last_observation.copy()

        # 10. 组织回合信息
        info = {
            "ego_id": ego_id,
            "simulation_time": current_time,
            "step_count": self.step_count,
            "terminal_reason": event,
            "is_success": event == "Arrived",
            "road_id": current_road,
            "lane_id": state.lane_id,
            "speed": state.speed,
            "requested_accel": control.requested_accel,
            "executed_accel": control.executed_accel,
            "lane_direction": control.lane_direction,
            "target_lane": control.target_lane,
            "speed_limited": control.speed_limited,
            "exit_change_active": (
                self.controller.active_change_to_zero
            ),
            "seen_e77": self.seen_e77,
            "reached_d3": self.reached_d3,
            "sumo_advanced": True,
            "pre_step_terminal": False,
        }

        return obs, reward, terminated, truncated, info

    

    def close(self):
        """关闭当前环境。"""
        if self.world is not None:
            self.world.close()
            self.world = None

        self.controller = None
