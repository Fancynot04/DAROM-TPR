
"""多 Ego 环境：统一 SUMO 时钟，独立 Ego 状态。"""

from xml.etree import ElementTree

import numpy as np

from .sumo_world import SumoWorld, SUMO_DIR
from .ego import Ego
from .control import EgoController
from .observation import build_observation

from .exit_rules import missed_exit_before_step
from src.rewards.events import b0_reward
from src.rewards.masac import MASACRewards, TARGET_EXIT_DIM

from .action_history import MultiEgoActionHistory
from .delay_buffer import MultiEgoDelayBuffer

from src.rewards.usp import USPTracker

class MultiEgoEnv:
    """双 Ego 环境的第一版交互接口。"""

    AGENTS = ("ego_1", "ego_2")
    OBS_SHAPE = (31, 3)

    def __init__(
        self,
        scenario="easy",
        speed_mode=96,
        collect_metrics=False,
        usp_enabled=False,
        usp_penalty=0.02,
        gui=False,
        reward_config=None,
    ):
        # 目前只生成了 Easy 双 Ego 路线
        if scenario != "easy":
            raise ValueError(
                "Only Easy dual-Ego route is available"
            )

        self.collect_metrics = collect_metrics
        self.scenario = scenario
        self.speed_mode = speed_mode
        self.gui = bool(gui)
        self.reward_tracker = MASACRewards(reward_config) if reward_config is not None else None
        self.target_exit_dim = TARGET_EXIT_DIM if self.reward_tracker is not None else 0

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
        
        self.usp_enabled = bool(usp_enabled)

        self.usp = USPTracker(
            self.AGENTS,
            penalty=usp_penalty,
        )

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

        return (b0_reward(event) if self.reward_tracker is None
                else self.reward_tracker.terminal(event))

    
    def target_exit_features(self, ego_id):
        # D1 uses [0, 0], D2 uses [1, 0], D3 uses [0, 1].
        target = self.target_exits[ego_id]
        return np.array([target == "D2", target == "D3"], dtype=np.float32)

    def reward_summary(self):
        values = {"teleport_events": self.teleport_events.copy()}
        if self.reward_tracker is not None:
            values["team_reward"] = self.reward_tracker.team_return
            values["reward_components"] = self.reward_tracker.totals
        return values

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
                "target_exit": self.target_exits[ego_id],
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
        """更新每辆 Ego 的延迟策略观测。

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

                if self.target_exit_dim:
                    policy_obs = np.concatenate([
                        policy_obs[:93], self.target_exit_features(ego_id), policy_obs[93:],
                    ])
                self.policy_observations[ego_id] = policy_obs

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
                    policy_obs = np.zeros(134 + self.target_exit_dim, dtype=np.float32)
                    if self.target_exit_dim:
                        policy_obs[93:93 + self.target_exit_dim] = self.target_exit_features(ego_id)
                    self.policy_observations[ego_id] = policy_obs

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
            SUMO_DIR / "Antwerp_easy_o3_d3_homogeneous_dual.rou.xml"
        )

        trips = {trip.attrib["id"]: trip.attrib for trip in ElementTree.parse(route_file).iter("trip")}
        self.target_exits = {ego_id: trips[ego_id]["to"] for ego_id in self.AGENTS}
        self.needs_merge = {ego_id: trips[ego_id]["from"] == "O3" for ego_id in self.AGENTS}
        self.teleport_events = []
        self.team_reward = 0.0
        if self.reward_tracker is not None:
            self.reward_tracker.reset(self.AGENTS)

        self.world = SumoWorld(
            scenario=self.scenario,
            seed=seed,
            route_file=route_file,
            gui=self.gui,
            disable_teleport=self.reward_tracker is not None,
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
        self.usp.reset()
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

        tracker = self.reward_tracker
        components = {ego_id: tracker.empty() for ego_id in self.AGENTS} if tracker is not None else {}
        pre_terminated = set()

        metric_pre = {}
        if self.collect_metrics:
            for ego_id in self.AGENTS:
                if ego_id in active_ids:
                    metric_pre[ego_id] = {
                        "speed": float(conn.vehicle.getSpeed(ego_id)),
                        "acceleration": float(conn.vehicle.getAcceleration(ego_id)),
                        "time": float(sim.getTime()),
                        "road": conn.vehicle.getRoadID(ego_id),
                        "lane": int(conn.vehicle.getLaneIndex(ego_id)),
                        "speed_mode": int(conn.vehicle.getSpeedMode(ego_id)),
                        "request_direction": 0,
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
                if tracker is not None:
                    components[ego_id]["task"] = rewards[ego_id]
                pre_terminated.add(ego_id)

        # 2. 仅尚未终止的 Ego 提交动作
        acting_ids = active_ids - pre_terminated
        
        usp_candidates = {
            ego_id: 0.0 for ego_id in self.AGENTS
        }

        if self.usp_enabled:
            for ego_id in acting_ids:
                usp_candidates[ego_id] = self.usp.step(
                    conn, ego_id
                )

        for ego_id in self.AGENTS:
            if ego_id in acting_ids:
                control = self.controllers[ego_id].apply_action(
                    actions[ego_id]
                )

                if tracker is not None:
                    tracker.before_step(conn, ego_id, control)

                # 仅统计真正向 SUMO 提交的换道指令
                if self.collect_metrics and ego_id in metric_pre:
                    metric_pre[ego_id]["request_direction"] = (
                        control.lane_direction
                        if control.target_lane is not None
                        else 0
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

        started = set(sim.getStartingTeleportIDList())
        ended = set(sim.getEndingTeleportIDList())
        teleports = started | ended
        teleport_details = {}
        for ego_id in sorted(teleports.intersection(self.AGENTS)):
            detail = {
                "ego_id": ego_id,
                "simulation_time": sim.getTime(),
                "started": ego_id in started,
                "ended": ego_id in ended,
                "cause": "collision_detected" if ego_id in collisions else "unknown",
            }
            teleport_details[ego_id] = detail
            self.teleport_events.append(detail)

        present = set(conn.vehicle.getIDList()) if self.collect_metrics or tracker is not None else set()
        if tracker is not None:
            for ego_id in acting_ids:
                components[ego_id] = tracker.after_step(
                    conn, ego_id, ego_id in present and ego_id not in teleports,
                    self.needs_merge[ego_id],
                )

        arrivals = set(
            sim.getArrivedIDList()
        )

        metric_steps = {}
        if self.collect_metrics:
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
                    "post_road": conn.vehicle.getRoadID(ego_id) if on_road else None,
                    "post_lane": int(conn.vehicle.getLaneIndex(ego_id)) if on_road else None,
                    "depart_time": self.egos[ego_id].state.depart_time,
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
                if tracker is not None:
                    components[ego_id]["task"] = rewards[ego_id]

        if tracker is not None:
            for ego_id, values in components.items():
                rewards[ego_id] = sum(values.values())
                for key, value in values.items():
                    tracker.totals[ego_id][key] += value
            self.team_reward = tracker.settle_team(self.terminal_reasons)

        applied_usp = {
            ego_id: 0.0 for ego_id in self.AGENTS
        }

        for ego_id in acting_ids:
            # 终止步只支付原 B0 事件奖励
            if self.terminations[ego_id]:
                continue

            value = usp_candidates[ego_id]

            rewards[ego_id] += value
            applied_usp[ego_id] = value

        # 7. 生成两辆 Ego 的观测
        observations, infos = self._build_output()

        for ego_id in self.AGENTS:
            if self.collect_metrics:
                sample = metric_steps.get(ego_id)
                if sample is not None:
                    sample["depart_time"] = self.egos[ego_id].state.depart_time
                infos[ego_id]["evaluation_metrics"] = sample
            infos[ego_id]["pre_step_terminal"] = (
                ego_id in pre_terminated
            )
        
            infos[ego_id]["teleport_event"] = teleport_details.get(ego_id)
            if tracker is not None:
                infos[ego_id]["reward_components"] = components[ego_id]
            infos[ego_id]["usp_reward"] = applied_usp[ego_id]

            infos[ego_id]["usp_triggered"] = (
                applied_usp[ego_id] < 0.0
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
