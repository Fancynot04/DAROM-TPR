
"""多 Ego 环境：统一 SUMO 时钟，独立 Ego 状态。"""

from xml.etree import ElementTree

import numpy as np

from .sumo_world import SumoWorld, SUMO_DIR
from .ego import Ego
from .control import EgoController
from .observation import RouteGoal, build_observation, leader_observation, remaining_goal_distance
from .stalled import StalledDetector

from .exit_rules import missed_exit_before_step, route_missed_before_step
from .route_lane_plan import DEFAULT_ROUTE_GUIDANCE_RULES, RouteLaneNetwork, UNREACHABLE_LANE_CHANGES
from .route_preparation import DEFAULT_ROUTE_PREPARATION_STAGES, DEFAULT_ROUTE_PREPARATION_WEIGHTS, RoutePreparation
from src.rewards.events import STALLED_EVENT, b0_reward
from src.rewards.masac import MASACRewardProtocol, MASACRewards, TARGET_EXIT_DIM

from .action_history import MultiEgoActionHistory
from .delay_buffer import MultiEgoDelayBuffer

from src.rewards.usp import USPTracker

class MultiEgoEnv:
    """双 Ego 环境的第一版交互接口。"""

    AGENTS = ("ego_1", "ego_2")
    OBS_SHAPE = (31, 3)
    DEFAULT_MAX_ACTIVE_STEPS = 2500

    def __init__(
        self,
        scenario="easy",
        speed_mode=96,
        collect_metrics=False,
        usp_enabled=False,
        usp_penalty=0.02,
        gui=False,
        reward_config=None,
        max_active_steps=DEFAULT_MAX_ACTIVE_STEPS,
        collect_control_trace=False,
        stalled_shadow=False,
        stalled_terminate=False,
        route_file=None,
        legacy_route_semantics=False,
        actor_leader_features=False,
        reward_protocol=MASACRewardProtocol(),
        route_guidance_rules=DEFAULT_ROUTE_GUIDANCE_RULES,
        stalled_block_grace_s=0.0,
        stalled_match_route_links=False,
        route_preparation_settings=None,
    ):
        # 目前只生成了 Easy 双 Ego 路线
        if scenario != "easy":
            raise ValueError(
                "Only Easy dual-Ego route is available"
            )

        if max_active_steps <= 0:
            raise ValueError("max_active_steps must be positive")

        if stalled_terminate and usp_enabled:
            raise ValueError("Stalled termination and USP cannot both be enabled")

        self.max_active_steps = max_active_steps
        self.collect_metrics = collect_metrics or collect_control_trace
        self.collect_control_trace = collect_control_trace
        self.stalled_shadow = stalled_shadow
        self.stalled_terminate = stalled_terminate
        self.stalled_block_grace_s = stalled_block_grace_s
        self.stalled_match_route_links = stalled_match_route_links
        self.sumo_warning_log_dir = None
        self.stall_detectors = {}
        self.route_file = (route_file if route_file is not None else
                           SUMO_DIR / "Antwerp_easy_o3_d3_homogeneous_dual.rou.xml")
        self.scenario = scenario
        self.speed_mode = speed_mode
        self.gui = bool(gui)
        self.reward_tracker = MASACRewards(reward_config, reward_protocol) if reward_config is not None else None
        self.route_preparation = None
        if reward_config is not None and reward_config.route_preparation:
            self.route_preparation = RoutePreparation(
                route_preparation_settings if route_preparation_settings is not None else {
                    "weights": DEFAULT_ROUTE_PREPARATION_WEIGHTS, "stages": DEFAULT_ROUTE_PREPARATION_STAGES,
                },
            )
        self.route_guidance_rules = {}
        if reward_config is not None and reward_config.route_guidance:
            for rule in route_guidance_rules:
                self.route_guidance_rules[tuple(rule["route"])] = (
                    tuple(rule["critical_connection"]),
                    frozenset(map(tuple, rule["accepted_connections"])),
                )
        self.target_exit_dim = TARGET_EXIT_DIM if self.reward_tracker is not None else 0
        self.legacy_route_semantics = legacy_route_semantics
        self.actor_leader_features = actor_leader_features
        self.lane_network = None

        self.world = None
        self.egos = {}
        self.controllers = {}
        self.observations = {}

        self.active_steps = {}
        self.step_count = 0
        
        self.terminations = {}
        self.terminal_reasons = {}
        self.seen_e77 = {}
        self.reached_target = {}
        
        self.action_history = MultiEgoActionHistory(
            self.AGENTS,
            hist_len=20,
        )
        self.delay_buffer = MultiEgoDelayBuffer(
            self.AGENTS,
            max_delay=20,
            seed=42,
            observation_shape=(self.OBS_SHAPE[0] + int(actor_leader_features), self.OBS_SHAPE[1]),
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
                if self.legacy_route_semantics and not self.route_guidance_rules and self.route_preparation is None:
                    continue
                vehicle = conn.vehicle
                route = tuple(vehicle.getRoute(ego_id))
                if not self.legacy_route_semantics and (not route or route[-1] != self.target_exits[ego_id]):
                    raise RuntimeError(f"{ego_id}: route {route} does not end at {self.target_exits[ego_id]}")
                guidance = self.route_guidance_rules.get(route)
                if not self.legacy_route_semantics or guidance is not None or self.route_preparation is not None:
                    plan = self.lane_network.plan(route, vehicle.getVehicleClass(ego_id))
                    if not self.legacy_route_semantics or self.route_preparation is not None:
                        self.route_lane_plans[ego_id] = plan
                    if self.route_preparation is not None:
                        self.route_preparation.configure(ego_id, self.lane_network, plan)
                    if guidance is not None:
                        self.ego_guidance[ego_id] = plan, *guidance

    def _route_lane_position(self, ego_id):
        vehicle = self.world.conn.vehicle
        plan = self.route_lane_plans[ego_id]
        if tuple(vehicle.getRoute(ego_id)) != plan.route:
            return None
        return plan.position(vehicle, ego_id)

    def _finish_agent(self, ego_id, event):
        """结算一辆 Ego，不影响其他 Ego 的状态。"""

        if self.terminations[ego_id]:
            raise RuntimeError(
                f"Duplicate terminal event: {ego_id}"
            )

        self.terminations[ego_id] = True
        self.terminal_reasons[ego_id] = event

        self.egos[ego_id].finish(event)
        if self.reward_tracker is not None:
            self.reward_tracker.end_change(ego_id, self.world.conn.simulation.getTime(), event)

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
        values = {}
        if self.reward_tracker is not None:
            values["team_reward"] = self.reward_tracker.team_return
            values["reward_components"] = self.reward_tracker.totals
            values["lane_change_events"] = self.reward_tracker.lane_change_events.copy()
            values["route_preparation_diagnostics"] = (
                self.route_preparation.diagnostics if self.route_preparation is not None else {
                    ego_id: {"enabled": False, "reason": "disabled", "phases": {}} for ego_id in self.AGENTS
                }
            )
        if self.stalled_block_grace_s and self.stall_detectors:
            values["stalled_diagnostics"] = {
                ego_id: detector.summary() for ego_id, detector in self.stall_detectors.items()
            }
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

        entity_count, entity_dim = self.OBS_SHAPE
        base_dim = entity_count * entity_dim
        actor_obs_dim = base_dim + entity_dim * int(self.actor_leader_features)
        for ego_id in self.AGENTS:
            status = self.egos[ego_id].state.status

            if status == "active":

                actor_observation = observations[ego_id]
                if self.reward_tracker is not None:
                    distance = remaining_goal_distance(self.world.conn, ego_id, self.route_goals[ego_id])
                    actor_observation = actor_observation.copy()
                    actor_observation[0, 0] = distance
                    infos[ego_id]["remaining_goal_distance_m"] = distance
                    if not self.legacy_route_semantics:
                        plan = self.route_lane_plans[ego_id]
                        changes = plan.signed_changes.get(self.route_lane_positions[ego_id])
                        actor_observation[0, 1] = (
                            UNREACHABLE_LANE_CHANGES if changes is None else changes
                        )
                        infos[ego_id]["signed_min_lane_changes"] = changes

                if self.actor_leader_features:
                    front = leader_observation(self.world.conn.vehicle, ego_id, actor_observation[0, 2])
                    # Only row 0 is refreshed live by the delay buffer; the front
                    # vehicle row travels with the rest of the traffic packet.
                    actor_observation = np.vstack((actor_observation, front))

                # 刚出发的 Ego：独立初始化延迟队列
                if ego_id not in self.delay_buffer.active:
                    delayed = self.delay_buffer.activate(
                        ego_id,
                        actor_observation,
                    )

                else:
                    # 已经在运行：模拟本步数据包传输
                    delayed = self.delay_buffer.update(
                        ego_id,
                        actor_observation,
                    )

                

                # 93 维延迟观测
                # + 40 维掩码动作历史
                # + 1 维观测年龄
                policy_obs = self.action_history.encode(
                    ego_id,
                    delayed.observation[:entity_count],
                    delay=delayed.age,
                )

                if self.actor_leader_features or self.target_exit_dim:
                    nav = self.target_exit_features(ego_id) if self.target_exit_dim else ()
                    policy_obs = np.concatenate([
                        policy_obs[:base_dim], delayed.observation[entity_count:].reshape(-1),
                        nav, policy_obs[base_dim:],
                    ], dtype=np.float32)
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
                    policy_dim = actor_obs_dim + self.target_exit_dim + self.action_history.hist_len * 2 + 1
                    policy_obs = np.zeros(policy_dim, dtype=np.float32)
                    if self.target_exit_dim:
                        policy_obs[actor_obs_dim:actor_obs_dim + self.target_exit_dim] = self.target_exit_features(ego_id)
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

        trips = {trip.attrib["id"]: trip.attrib for trip in ElementTree.parse(self.route_file).iter("trip")}
        self.target_exits = {ego_id: trips[ego_id]["to"] for ego_id in self.AGENTS}
        self.needs_merge = {ego_id: trips[ego_id]["from"] == "O3" for ego_id in self.AGENTS}
        self.stall_detectors = {
            ego_id: StalledDetector(self.stalled_block_grace_s) for ego_id in self.AGENTS
        } if self.stalled_shadow or self.stalled_terminate else {}
        self.team_reward = 0.0
        if self.reward_tracker is not None:
            self.reward_tracker.reset(self.AGENTS)
        if self.route_preparation is not None:
            self.route_preparation.reset(self.AGENTS)

        self.world = SumoWorld(
            scenario=self.scenario,
            seed=seed,
            route_file=self.route_file,
            gui=self.gui,
            warning_log=(self.sumo_warning_log_dir / f"seed-{seed}.log"
                         if self.sumo_warning_log_dir is not None else None),
        )
        if (not self.legacy_route_semantics or self.route_guidance_rules or self.route_preparation is not None
                ) and self.lane_network is None:
            self.lane_network = RouteLaneNetwork(self.world.net_file)
        self.route_lane_plans = {}
        self.route_lane_positions = {}
        self.ego_guidance = {}

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

        self.reached_target = {
            ego_id: False for ego_id in self.AGENTS
        }

        self.observations = {}
        
        try:
            conn = self.world.start()
            if self.reward_tracker is not None:
                self.route_goals = {
                    ego_id: RouteGoal.from_trip(conn, trips[ego_id]) for ego_id in self.AGENTS
                }

            # 只推进到第一辆 Ego 插入的时间
            while conn.simulation.getTime() < 1200.0:
                self.world.step()
                self._sync_agents()

                if any(
                    ego.state.status == "active"
                    for ego in self.egos.values()
                ):
                    if not self.legacy_route_semantics:
                        self.route_lane_positions = {
                            ego_id: self._route_lane_position(ego_id)
                            for ego_id in self.route_lane_plans
                        }
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
        change_event_start = len(tracker.lane_change_events) if tracker is not None else 0

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

        route_positions_before = {}
        # 1. 每辆车独立进行 Missed Exit 预判
        for ego_id in active_ids:
            if self.legacy_route_semantics:
                missed = self.target_exits[ego_id] == "D3" and missed_exit_before_step(
                    conn, ego_id, active_change_to_zero=self.controllers[ego_id].active_change_to_zero,
                )
            else:
                position = self._route_lane_position(ego_id)
                route_positions_before[ego_id] = position
                # Replay attributes a pre-step failure to the previous action.
                missed = self.active_steps[ego_id] > 0 and route_missed_before_step(
                    conn, ego_id, self.route_lane_plans[ego_id], position,
                )

            if missed:
                rewards[ego_id] = self._finish_agent(
                    ego_id,
                    "Missed Exit",
                )
                if tracker is not None:
                    components[ego_id]["task"] = rewards[ego_id]
                    components[ego_id]["merge"] += tracker.settle_merge(ego_id, "Missed Exit")
                pre_terminated.add(ego_id)

        # 2. 仅尚未终止的 Ego 提交动作
        acting_ids = active_ids - pre_terminated
        preparation_samples = {}
        if self.route_preparation is not None:
            self.route_preparation.step_info.clear()
            for ego_id in acting_ids:
                position = (self._route_lane_position(ego_id) if self.legacy_route_semantics
                            else route_positions_before[ego_id])
                sample = self.route_preparation.before_step(
                    conn, ego_id, position, tracker.pending_changes.get(ego_id),
                )
                if sample is not None:
                    preparation_samples[ego_id] = sample
        guidance_before = {
            ego_id: (conn.vehicle.getRouteIndex(ego_id), conn.vehicle.getLaneID(ego_id))
            for ego_id in acting_ids & self.ego_guidance.keys()
            if ego_id not in tracker.guidance_paid
        }
        stall_results = {}
        if self.stall_detectors:
            now = sim.getTime()
            teleports = set()
            if self.stalled_block_grace_s:
                teleports.update(sim.getStartingTeleportIDList())
                teleports.update(sim.getEndingTeleportIDList())
            for ego_id, detector in self.stall_detectors.items():
                if ego_id in teleports:
                    stall_results[ego_id] = detector.observe(now, unavailable_reason="teleport")
                elif ego_id in acting_ids:
                    plan = self.route_lane_plans[ego_id] if self.stalled_match_route_links else None
                    stall_results[ego_id] = detector.sample(conn, ego_id, now, plan)
                else:
                    stall_results[ego_id] = detector.observe(now, unavailable_reason="inactive")

        
        usp_candidates = {
            ego_id: 0.0 for ego_id in self.AGENTS
        }

        if self.usp_enabled:
            for ego_id in acting_ids:
                usp_candidates[ego_id] = self.usp.step(
                    conn, ego_id
                )

        control_traces = {}
        for ego_id in self.AGENTS:
            if ego_id in acting_ids:
                if self.collect_control_trace:
                    vehicle = conn.vehicle
                    leader = vehicle.getLeader(ego_id, 100.0)
                    gap = leader[1] + vehicle.getMinGap(ego_id) if leader and leader[0] else None
                    lane_id = vehicle.getLaneID(ego_id)
                    lane_position = vehicle.getLanePosition(ego_id)
                    control_traces[ego_id] = {
                        "leader_id": leader[0] if gap is not None else None,
                        "leader_gap_m": gap,
                        "lane_position_m": lane_position,
                        "lane_remaining_m": conn.lane.getLength(lane_id) - lane_position,
                        "road": metric_pre[ego_id]["road"],
                        "speed_mps": metric_pre[ego_id]["speed"],
                        "requested_acceleration_mps2": float(actions[ego_id][0]),
                    }
                active_change = None
                if tracker is not None:
                    change = tracker.pending_changes.get(ego_id)
                    if change is not None and change.last_motion is not None:
                        active_change = (change.target_lane, change.direction)
                control = self.controllers[ego_id].apply_action(
                    actions[ego_id], active_lane_change=active_change,
                )

                if self.collect_control_trace:
                    control_traces[ego_id]["commanded_acceleration_mps2"] = control.executed_accel

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
        arrivals = set(
            sim.getArrivedIDList()
        )

        present = set(conn.vehicle.getIDList()) if self.collect_metrics or tracker is not None else set()
        if tracker is not None:
            for ego_id in acting_ids:
                change_events_before = len(tracker.lane_change_events)
                terminal_event = None
                if ego_id in collisions:
                    terminal_event = "Collision"
                elif ego_id in arrivals:
                    terminal_event = "Arrived"
                components[ego_id] = tracker.after_step(
                    conn, ego_id, ego_id in present,
                    self.needs_merge[ego_id], terminal_event=terminal_event,
                )
                if ego_id in preparation_samples:
                    completed_change = None
                    if len(tracker.lane_change_events) > change_events_before:
                        latest = tracker.lane_change_events[-1]
                        if latest["event"] == "completed":
                            completed_change = latest
                    components[ego_id]["route_preparation"] = self.route_preparation.after_step(
                        conn, ego_id, preparation_samples[ego_id], ego_id in present, completed_change,
                    )

        metric_steps = {}
        if self.collect_metrics:
            emergency = set(sim.getEmergencyStoppingVehiclesIDList())
            for ego_id in self.AGENTS:
                if self.terminations[ego_id] and ego_id not in pre_terminated:
                    continue
                on_road = ego_id in present
                metric_steps[ego_id] = {
                    "pre": metric_pre.get(ego_id),
                    "post_acceleration": (float(conn.vehicle.getAcceleration(ego_id))
                                          if on_road else None),
                    "post_time": float(sim.getTime()),
                    "on_road": on_road,
                    "post_road": conn.vehicle.getRoadID(ego_id) if on_road else None,
                    "post_lane": int(conn.vehicle.getLaneIndex(ego_id)) if on_road else None,
                    "depart_time": self.egos[ego_id].state.depart_time,
                    "emergency_stopping": ego_id in emergency,
                    "agent_step": ego_id in acting_ids,
                }
                if ego_id in control_traces:
                    control_traces[ego_id].update(
                        actual_acceleration_mps2=metric_steps[ego_id]["post_acceleration"],
                        post_speed_mps=float(conn.vehicle.getSpeed(ego_id)) if on_road else None,
                        sumo_arrived=ego_id in arrivals,
                    )

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

            target = self.target_exits[ego_id]
            if road == target:
                self.reached_target[ego_id] = True

            if self.legacy_route_semantics:
                if road == "E77":
                    self.seen_e77[ego_id] = True
                wrong_exit = (
                    target == "D3"
                    and self.seen_e77[ego_id]
                    and not self.reached_target[ego_id]
                    and road is not None
                    and not road.startswith(":")
                    and road not in ("E77", "D3")
                )
            else:
                wrong_exit = False
                if state.status == "active":
                    position = self._route_lane_position(ego_id)
                    self.route_lane_positions[ego_id] = position
                    if ego_id in acting_ids:
                        plan = self.route_lane_plans[ego_id]
                        wrong_exit = position not in plan.signed_changes
                        if not wrong_exit and self.active_steps[ego_id] == 1:
                            wrong_exit = route_missed_before_step(conn, ego_id, plan, position)
                elif ego_id in arrivals and ego_id in acting_ids:
                    wrong_exit = (route_positions_before[ego_id]
                                  not in self.route_lane_plans[ego_id].signed_changes)

            # Settle physical SUMO and route events before the shadow trigger.
            if ego_id in collisions:
                event = "Collision"

            elif state.terminal_reason == "UnknownRemoval":
                raise RuntimeError(f"Unexpected removal: {ego_id}, reason={state.terminal_reason}")

            elif wrong_exit:
                event = "Missed Exit"

            elif ego_id in arrivals:
                event = "Arrived"

            elif (
                self.stalled_terminate
                and state.status == "active"
                and stall_results[ego_id]["would_stall"]
            ):
                event = STALLED_EVENT

            elif self.active_steps[ego_id] >= self.max_active_steps:
                event = "Timeout"

            else:
                event = None

            if event == "Arrived" and not self.reached_target[ego_id]:
                # A short arrivalPos can be reached in the same step as edge entry.
                # The acting Ego's route was verified before the step.
                if ego_id not in acting_ids or (self.legacy_route_semantics and tracker is None):
                    raise RuntimeError(f"{ego_id} arrived without reaching {target}")

            if ego_id in guidance_before and (ego_id in present or event == "Arrived"):
                plan, critical, accepted = self.ego_guidance[ego_id]
                connection = plan.completed_connection(
                    guidance_before[ego_id], conn.vehicle if ego_id in present else None,
                    ego_id, critical,
                )
                components[ego_id]["route_guidance"] = tracker.score_route_guidance(
                    ego_id, connection, accepted,
                )

            if event is not None:
                rewards[ego_id] = self._finish_agent(
                    ego_id,
                    event,
                )
                if tracker is not None:
                    components[ego_id]["task"] = rewards[ego_id]
                    components[ego_id]["merge"] += tracker.settle_merge(ego_id, event)

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
            # USP does not pay on an agent's terminal step.
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
        
            if self.collect_control_trace:
                infos[ego_id]["control_trace"] = control_traces.get(ego_id)
            if self.stalled_shadow:
                infos[ego_id]["stalled_shadow"] = stall_results[ego_id]
            if tracker is not None:
                infos[ego_id]["reward_components"] = components[ego_id]
                preparation_info = (self.route_preparation.step_info.get(ego_id)
                                    if self.route_preparation is not None else None)
                infos[ego_id]["route_preparation"] = preparation_info
                infos[ego_id]["following_risks"] = (
                    tracker.following_risks[ego_id] if ego_id in acting_ids else None
                )
                infos[ego_id]["lane_change_events"] = [
                    event for event in tracker.lane_change_events[change_event_start:]
                    if event["ego_id"] == ego_id
                ]
                if ego_id in control_traces:
                    control_traces[ego_id].update(tracker.following_risks[ego_id])
                    if preparation_info is not None:
                        control_traces[ego_id]["route_preparation"] = preparation_info
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
