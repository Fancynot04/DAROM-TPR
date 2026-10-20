"""Two task agents sharing one SUMO clock; no learner or automatic reset.

The dictionary API is intentionally not an SB3 single-agent Gym environment.
Only IDs in active_agents accept actions or produce transitions. Newly departed
agents produce an initial observation, with transition_valid=False.
"""

import hashlib
import uuid

import numpy as np

from .merging import (ANTWERP_DIR, EGO_DEPART_TIME, EGO_DEPARTURE_DEADLINE,
                      SUMO_STEP_LENGTH, Merging, traci)
from .safety_controller import safetyCheck, target_lane_line
from .merging import parseAction
from .wrapper import DelayWrapper


AGENT_TASKS = {"ego_entry": "entry", "ego_exit": "exit"}


class DualMerging:
    def __init__(self, seed=42, scenario="easy", safety=False,
                 delay_mode="uniform", fixed_delay=None):
        if fixed_delay is not None and not 0 <= fixed_delay <= 20:
            raise ValueError("fixed_delay must be between 0 and 20 steps")
        self.seed = seed
        self.scenario = scenario
        self.safety = safety
        self.delay_mode = delay_mode
        self.fixed_delay = fixed_delay
        self._rng = np.random.default_rng(seed)
        self.connection = None
        self.cars = {key: Merging(seed, scenario=scenario, task=task)
                     for key, task in AGENT_TASKS.items()}
        self.delays = {
            key: DelayWrapper(car, max_delay=20, mode="all", delay_mode=delay_mode)
            for key, car in self.cars.items()
        }
        if fixed_delay is not None:
            for wrapper in self.delays.values():
                wrapper._sample_delay = lambda: self.fixed_delay
        self.status = {key: "pending" for key in AGENT_TASKS}
        self.released = set()
        self.finished = True

    @property
    def active_agents(self):
        return tuple(key for key in AGENT_TASKS if self.status[key] == "active")

    def reset(self, seed=None):
        self.close()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_seed = int(self._rng.integers(0, 2**31 - 1))
        self.status = {key: "pending" for key in AGENT_TASKS}
        self.released.clear()
        self.finished = False
        for index, (key, car) in enumerate(self.cars.items()):
            car.reset_episode_state()
            delay_seed = int(np.random.SeedSequence(
                [self.episode_seed, index]).generate_state(1)[0])
            self.delays[key].reset_history(car.observation, {}, seed=delay_seed)
        paths = {"network": ANTWERP_DIR / "Antwerp.net.xml",
                 "routes": ANTWERP_DIR / f"Antwerp_{self.scenario}_dual.rou.xml"}
        command = [
            next(iter(self.cars.values())).sumoBinary,
            "--net-file", str(paths["network"]), "--route-files", str(paths["routes"]),
            "--begin", "0", "--end", "3600", "--step-length", str(SUMO_STEP_LENGTH),
            "--lateral-resolution", "0.8", "--seed", str(self.episode_seed),
            "--no-step-log", "--collision.check-junctions",
            "--collision.action", "remove", "-Q",
        ]
        label = "dual-" + uuid.uuid4().hex
        traci.start(command, label=label, doSwitch=False)
        self.connection = traci.getConnection(label)
        for car in self.cars.values():
            car._connection = self.connection
        try:
            self.connection.simulationStep(EGO_DEPART_TIME - SUMO_STEP_LENGTH)
            observations, infos = {}, {}
            # Stop as soon as ANY ego enters: do not let it drive uncontrolled
            # while waiting for a congested second origin.
            while not observations:
                self.connection.simulationStep()
                observations, infos = self._departures()
                if self.connection.simulation.getTime() >= EGO_DEPARTURE_DEADLINE:
                    if not observations:
                        raise RuntimeError("Neither ego entered before departure deadline")
            self.start_time = self.connection.simulation.getTime()
            return observations, {
                "agents": infos, "scene_done": False, "sumo_seed": self.episode_seed,
                "sumo_time": self.start_time, "sumo_version": list(self.connection.getVersion()),
                "scene_files": {key: {"path": str(path.relative_to(ANTWERP_DIR.parent)),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for key, path in paths.items()},
                "sumo_command": command, "status": self.status.copy(),
                "fixed_delay": self.fixed_delay, "delay_mode": self.delay_mode,
                "safety": self.safety,
            }
        except Exception:
            self.close()
            raise

    def _departures(self):
        observations, infos = {}, {}
        departed = set(self.connection.simulation.getDepartedIDList())
        for index, (key, car) in enumerate(self.cars.items()):
            if self.status[key] != "pending" or key not in departed:
                continue
            if self.connection.vehicle.getRoadID(key) != car.origin:
                raise RuntimeError(f"{key} departed on the wrong origin")
            car.ego_id = key
            car.ego_inserted = car.ego_on_road = True
            car.ego_depart_time = self.connection.simulation.getTime()
            self.connection.vehicle.setLaneChangeMode(key, 0)
            car.observation = car._get_obs()
            car.state = car.observation.copy()
            self.status[key] = "active"
            delay_seed = int(np.random.SeedSequence(
                [self.episode_seed, index]).generate_state(1)[0])
            infos[key] = {"transition_valid": False, "task": car.task,
                          "ego_depart_time": car.ego_depart_time,
                          "delay_seed": delay_seed, "status": "active"}
            observations[key], _ = self.delays[key].reset_history(
                car.observation, infos[key], seed=delay_seed)
        return observations, infos

    def step(self, actions):
        if self.finished or self.connection is None:
            raise RuntimeError("Call reset() before stepping a finished scene")
        active = self.active_agents
        if set(actions) != set(active):
            raise ValueError(f"Expected actions for {active}, got {tuple(actions)}")
        # Validate the entire joint action before applying any command.
        requested = {key: np.asarray(actions[key], dtype=np.float32) for key in active}
        for value in requested.values():
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("Each action must be a finite two-value vector")
            if (np.abs(value) > 5).any():
                raise ValueError("Action values must lie in [-5, 5]")
        applied, before = {}, {}
        # All safety decisions see the same pre-step traffic state.
        for key in active:
            car = self.cars[key]
            applied[key] = requested[key].copy()
            if self.safety and car.ego_on_road:
                line = target_lane_line(key, parseAction(applied[key])[1],
                                        connection=self.connection)
                applied[key], _ = safetyCheck(
                    applied[key], self.delays[key].safety_observation(), False, "all", line)
        for key in active:
            before[key] = self.cars[key].prepare_step(applied[key])
        old_time = self.connection.simulation.getTime()
        self.connection.simulationStep()
        now = self.connection.simulation.getTime()
        if not np.isclose(now - old_time, SUMO_STEP_LENGTH):
            raise RuntimeError("Shared clock did not advance by exactly one control step")

        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        # Read ALL outcomes before releasing any remaining failed vehicle.
        for key in active:
            car = self.cars[key]
            obs, reward, term, trunc, info = car.collect_step(before[key], close_on_done=False)
            observations[key] = self.delays[key].observation(obs)
            rewards[key], terminated[key], truncated[key] = reward, term, trunc
            wrapper = self.delays[key]
            snapshot = wrapper.last_snapshot
            info.update(transition_valid=True, requested_action=requested[key].tolist(),
                        applied_action=np.asarray(applied[key]).tolist(), sumo_time=now,
                        observation_age_steps=int(wrapper.delay_of_received_observation),
                        packet_time=None if snapshot is None else snapshot.time,
                        control_steps=car.counter, task=car.task,
                        safety_intervened=not np.array_equal(requested[key], applied[key]))
            infos[key] = info
            if term or trunc:
                self.status[key] = "success" if info["is_success"] else "failed"
            info["status"] = self.status[key]

        on_road = set(self.connection.vehicle.getIDList())
        for key, car in self.cars.items():
            if car.done and key in on_road and key not in self.released:
                # Non-arrival termination leaves a physical vehicle in traffic.
                # Release RL commands and restore SUMO's standard lane controller.
                self.connection.vehicle.setAcceleration(key, 0, 0)
                self.connection.vehicle.setSpeed(key, -1)
                self.connection.vehicle.setLaneChangeMode(key, 1621)
                self.released.add(key)

        new_obs, new_infos = self._departures()
        observations.update(new_obs)
        infos.update(new_infos)
        for key in new_obs:
            rewards[key], terminated[key], truncated[key] = 0.0, False, False
        for key in AGENT_TASKS:
            if self.status[key] == "pending" and now >= EGO_DEPARTURE_DEADLINE:
                self.status[key] = "not_departed"
                infos[key] = {"transition_valid": False, "status": "not_departed",
                              "message": "Departure Deadline", "sumo_time": now}
        self.finished = all(status not in ("pending", "active")
                            for status in self.status.values())
        infos["__scene__"] = {"scene_done": self.finished, "sumo_time": now,
                              "status": self.status.copy()}
        return observations, rewards, terminated, truncated, infos

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.finished = True
        for car in self.cars.values():
            car._connection = None
