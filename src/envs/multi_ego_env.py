
"""多 Ego 环境：统一 SUMO 时钟，独立 Ego 状态。"""

import numpy as np

from .sumo_world import SumoWorld, SUMO_DIR
from .ego import Ego
from .control import EgoController
from .observation import build_observation

from .exit_rules import missed_exit_before_step
from .single_ego_env import b0_reward

from .action_history import MultiEgoActionHistory
from .delay_buffer import MultiEgoDelayBuffer

class MultiEgoEnv:
    """双 Ego 环境的第一版交互接口。"""

    AGENTS = ("ego_o3", "ego_e74")
    OBS_SHAPE = (31, 3)

    def __init__(self, scenario="easy", speed_mode=96, collect_metrics=False):
        # 目前只生成了 Easy 双 Ego 路线
        if scenario != "easy":
            raise ValueError(
                "Only Easy dual-Ego route is available"
            )

        self.collect_metrics = collect_metrics
        self.scenario = scenario
        self.speed_mode = speed_mode

        self.world = None
        self.egos = {}
        self.controllers = {}
        self.observations = {}

        self.active_steps = {}
        self.step_count = 0
        
        self.terminations = {}
        self.terminal_reasons = {}
        self.seen_e77 = {}
        self.reached_d3 = {}
        
        self.action_history = MultiEgoActionHistory(
            self.AGENTS,
            hist_len=20,
        )
        self.delay_buffer = MultiEgoDelayBuffer(
            self.AGENTS,
            max_delay=20,
            seed=42,
        )

        self.policy_observations = {}

    def _sync_agents(self):
        """SUMO 推进后，更新所有 Ego 的状态。"""

        conn = self.world.conn

        for ego_id, ego in self.egos.items():
            previous_status = ego.state.status
            state = ego.update(conn)

            # 新插入的 Ego 立即接管控制
            if (
                previous_status == "waiting"
                and state.status == "active"
            ):
                controller = EgoController(
                    conn,
                    ego_id,
                    speed_mode=self.speed_mode,
                )

                controller.configure()
                self.controllers[ego_id] = controller


    def _finish_agent(self, ego_id, event):
        """结算一辆 Ego，不影响其他 Ego 的状态。"""

        if self.terminations[ego_id]:
            raise RuntimeError(
                f"Duplicate terminal event: {ego_id}"
            )

        self.terminations[ego_id] = True
        self.terminal_reasons[ego_id] = event

        self.egos[ego_id].finish(event)

        controller = self.controllers.get(ego_id)
        if controller is not None:
            controller.exit_change.reset()

        # Ego 到达终点时通常已由 SUMO 移除。
        # Missed / Timeout 时，主动移除已结束的 Ego。
        conn = self.world.conn

        if ego_id in conn.vehicle.getIDList():
            conn.vehicle.remove(ego_id)

        return b0_reward(event)

    
    def _build_output(self):
        """分别生成每辆 Ego 的观测和信息。"""

        conn = self.world.conn
        observations = {}
        infos = {}

        for ego_id, ego in self.egos.items():
            state = ego.state

            if state.status == "active":
                result = build_observation(
                    conn,
                    ego_id=ego_id,
                    radius=50.0,
                    num_entities=31,
                )

                observation = result.observation.copy()
            elif state.status == "finished":
                observation = self.observations.get(
                    ego_id,
                    np.zeros(
                        self.OBS_SHAPE,
                        dtype=np.float32,
                    ),
                ).copy()
            else:
                # 等待中的车辆没有物理观测
                observation = np.zeros(
                    self.OBS_SHAPE,
                    dtype=np.float32,
                )

            observations[ego_id] = observation

            infos[ego_id] = {
                "status": state.status,
                "depart_time": state.depart_time,
                "road_id": state.road_id,
                "lane_id": state.lane_id,
                "active_steps": self.active_steps[ego_id],

                # 独立终止信息
                "terminal_reason": self.terminal_reasons[ego_id],
                "is_success": (
                    self.terminal_reasons[ego_id] == "Arrived"
                ),

                "simulation_time": conn.simulation.getTime(),
            }

        self.observations = observations

        return observations, infos
    
    
    def _update_policy_observations(
        self,
        observations,
        infos,
    ):
        """更新每辆 Ego 的 134 维延迟策略观测。

        每个 SUMO 控制步调用一次。
        Waiting / Finished Ego 不更新延迟队列。
        """

        for ego_id in self.AGENTS:
            status = self.egos[ego_id].state.status

            if status == "active":

                # 刚出发的 Ego：独立初始化延迟队列
                if ego_id not in self.delay_buffer.active:
                    delayed = self.delay_buffer.activate(
                        ego_id,
                        observations[ego_id],
                    )

                else:
                    # 已经在运行：模拟本步数据包传输
                    delayed = self.delay_buffer.update(
                        ego_id,
                        observations[ego_id],
                    )

                

                # 93 维延迟观测
                # + 40 维掩码动作历史
                # + 1 维观测年龄
                policy_obs = self.action_history.encode(
                    ego_id,
                    delayed.observation,
                    delay=delayed.age,
                )

                self.policy_observations[ego_id] = (
                    policy_obs.copy()
                )

                infos[ego_id].update({
                    "observation_ready": True,
                    "observation_age": delayed.age,
                    "sampled_delay": delayed.sampled_delay,
                    "received_packet":
                        delayed.received_packet,
                })

            else:
                # Waiting：零向量
                # Finished：保留此前最后的策略观测
                if ego_id not in self.policy_observations:
                    self.policy_observations[ego_id] = (
                        np.zeros(134, dtype=np.float32)
                    )

                infos[ego_id].update({
                    "observation_ready": False,
                    "observation_age": None,
                    "sampled_delay": None,
                    "received_packet": False,
                })


    def reset(self, seed=42):
        """只等待第一辆 Ego 出发，不要求同时插入。"""

        self.close()

        route_file = (
            SUMO_DIR / "Antwerp_easy_o3_d3_dual.rou.xml"
        )

        self.world = SumoWorld(
            scenario=self.scenario,
            seed=seed,
            route_file=route_file,
        )

        self.egos = {
            ego_id: Ego(ego_id)
            for ego_id in self.AGENTS
        }

        self.controllers = {}
        self.active_steps = {
            ego_id: 0
            for ego_id in self.AGENTS
        }
        self.action_history.reset()
        self.delay_buffer.reset(seed=seed)
        self.policy_observations = {}
        
        self.step_count = 0
        
        self.terminations = {
            ego_id: False for ego_id in self.AGENTS
        }

        self.terminal_reasons = {
            ego_id: None for ego_id in self.AGENTS
        }

        self.seen_e77 = {
            ego_id: False for ego_id in self.AGENTS
        }

        self.reached_d3 = {
            ego_id: False for ego_id in self.AGENTS
        }

        self.observations = {}
        
        try:
            conn = self.world.start()

            # 只推进到第一辆 Ego 插入的时间
            while conn.simulation.getTime() < 1200.0:
                self.world.step()
                self._sync_agents()

                if any(
                    ego.state.status == "active"
                    for ego in self.egos.values()
                ):
                    observations, infos = self._build_output()

                    self._update_policy_observations(
                        observations,
                        infos,
                    )

                    return observations, infos

            raise RuntimeError(
                "No Ego departed before 1200s"
            )

        except Exception:
            self.close()
            raise

    
    def step(self, actions):
        """多辆 Ego 独立决策，SUMO 统一推进一次。"""

        if self.world is None:
            raise RuntimeError("Call reset() first")

        if all(self.terminations.values()):
            raise RuntimeError(
                "All Ego episodes finished; call reset()"
            )

        conn = self.world.conn
        sim = conn.simulation

        active_ids = {
            ego_id
            for ego_id, ego in self.egos.items()
            if ego.state.status == "active"
            and not self.terminations[ego_id]
        }

        if set(actions) != active_ids:
            raise ValueError(
                f"Expected {sorted(active_ids)}, "
                f"got {sorted(actions)}"
            )

        rewards = {
            ego_id: 0.0 for ego_id in self.AGENTS
        }

        pre_terminated = set()

        metric_pre = {}
        if self.collect_metrics:
            for ego_id in self.AGENTS:
                if ego_id in active_ids:
                    metric_pre[ego_id] = {
                        "speed": float(conn.vehicle.getSpeed(ego_id)),
                        "acceleration": float(conn.vehicle.getAcceleration(ego_id)),
                        "time": float(sim.getTime()),
                    }

        # 1. 每辆车独立进行 Missed Exit 预判
        for ego_id in active_ids:
            controller = self.controllers[ego_id]

            missed = missed_exit_before_step(
                conn,
                ego_id,
                active_change_to_zero=(
                    controller.active_change_to_zero
                ),
            )

            if missed:
                rewards[ego_id] = self._finish_agent(
                    ego_id,
                    "Missed Exit",
                )
                pre_terminated.add(ego_id)

        # 2. 仅尚未终止的 Ego 提交动作
        acting_ids = active_ids - pre_terminated

        for ego_id in self.AGENTS:
            if ego_id in acting_ids:
                self.controllers[ego_id].apply_action(
                    actions[ego_id]
                )

                self.action_history.record(
                    ego_id,
                    actions[ego_id],
                )

        # 3. 统一推进 SUMO，一次且仅一次
        try:
            self.world.step()
        except Exception:
            self.close()
            raise

        self.step_count += 1

        for ego_id in acting_ids:
            self.active_steps[ego_id] += 1

        # 4. 获取本步 SUMO 原始事件
        collisions = set(
            sim.getCollidingVehiclesIDList()
        )

        teleports = (
            set(sim.getStartingTeleportIDList())
            | set(sim.getEndingTeleportIDList())
        )

        arrivals = set(
            sim.getArrivedIDList()
        )

        metric_steps = {}
        if self.collect_metrics:
            present = set(conn.vehicle.getIDList())
            started = set(sim.getStartingTeleportIDList())
            ended = set(sim.getEndingTeleportIDList())
            emergency = set(sim.getEmergencyStoppingVehiclesIDList())
            for ego_id in self.AGENTS:
                if self.terminations[ego_id] and ego_id not in pre_terminated:
                    continue
                on_road = (ego_id in present
                           and (ego_id not in started or ego_id in ended))
                metric_steps[ego_id] = {
                    "pre": metric_pre.get(ego_id),
                    "post_acceleration": (float(conn.vehicle.getAcceleration(ego_id))
                                          if on_road else None),
                    "post_time": float(sim.getTime()),
                    "on_road": on_road,
                    "teleport_started": ego_id in started,
                    "teleport_ended": ego_id in ended,
                    "emergency_stopping": ego_id in emergency,
                    "agent_step": ego_id in acting_ids,
                }

        # 5. 更新每辆车的状态
        self._sync_agents()

        for ego_id in acting_ids:
            self.controllers[ego_id].observe_after_step()

        # 6. 分别结算终止事件
        for ego_id in self.AGENTS:
            if self.terminations[ego_id]:
                continue

            state = self.egos[ego_id].state

            if state.status == "waiting":
                continue

            road = state.road_id

            if road == "E77":
                self.seen_e77[ego_id] = True

            if road == "D3":
                self.reached_d3[ego_id] = True

            internal_road = (
                isinstance(road, str)
                and road.startswith(":")
            )

            wrong_exit = (
                self.seen_e77[ego_id]
                and not self.reached_d3[ego_id]
                and road is not None
                and not internal_road
                and road not in ("E77", "D3")
            )

            # 沿用 B0 终止优先级
            if ego_id in collisions:
                event = "Collision"

            elif ego_id in teleports:
                event = "Ego Teleported"

            elif wrong_exit:
                event = "Missed Exit"

            elif ego_id in arrivals:
                event = "Arrived"

            elif self.active_steps[ego_id] >= 4000:
                event = "Timeout"

            else:
                event = None

            # SUMO 车辆无故消失，不能默认为失败
            if state.status == "finished" and event is None:
                raise RuntimeError(
                    f"Unexpected removal: {ego_id}, "
                    f"reason={state.terminal_reason}"
                )

            if event == "Arrived":
                if not self.reached_d3[ego_id]:
                    raise RuntimeError(
                        f"{ego_id} arrived without D3"
                    )

            if event is not None:
                rewards[ego_id] = self._finish_agent(
                    ego_id,
                    event,
                )

        # 7. 生成两辆 Ego 的观测
        observations, infos = self._build_output()

        for ego_id in self.AGENTS:
            if self.collect_metrics:
                infos[ego_id]["evaluation_metrics"] = metric_steps.get(ego_id)
            infos[ego_id]["pre_step_terminal"] = (
                ego_id in pre_terminated
            )
        
        self._update_policy_observations(
            observations,
            infos,
        )
        
        truncations = {
            ego_id: False for ego_id in self.AGENTS
        }

        return (
            observations,
            rewards,
            self.terminations.copy(),
            truncations,
            infos,
        )


    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None

        self.controllers = {}
