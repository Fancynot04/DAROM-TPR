"""Dual-scene contracts; real SUMO tests opt in with -m tests.test_dual_merging."""

from pathlib import Path
from unittest.mock import Mock, patch
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from src.dual_merging import AGENT_TASKS, DualMerging
from src.merging import Merging


class DualContractTests(unittest.TestCase):
    def test_routes_keep_background_and_define_two_tasks(self):
        directory = Path(__file__).resolve().parents[1] / "sumo_new"
        for scenario in ("easy", "medium", "hard"):
            original = ET.parse(directory / f"Antwerp_{scenario}.rou.xml").getroot()
            dual = ET.parse(directory / f"Antwerp_{scenario}_dual.rou.xml").getroot()
            self.assertEqual([(n.tag, n.attrib) for n in original if n.tag != "trip"],
                             [(n.tag, n.attrib) for n in dual if n.tag != "trip"])
            trips = {node.get("id"): node.attrib for node in dual.findall("trip")}
            self.assertEqual(set(trips), set(AGENT_TASKS))
            self.assertEqual((trips["ego_entry"]["from"], trips["ego_entry"]["to"]),
                             ("O3", "E76"))
            self.assertEqual((trips["ego_exit"]["from"], trips["ego_exit"]["to"]),
                             ("O1", "D3"))

    def test_joint_action_validated_before_any_command(self):
        env = DualMerging()
        env.connection = Mock()
        env.finished = False
        env.status = dict.fromkeys(AGENT_TASKS, "active")
        for actions in ({"ego_entry": [0, 0]},
                        {"ego_entry": [0, 0], "ego_exit": [np.nan, 0]},
                        {"ego_entry": [0, 0], "ego_exit": [6, 0]}):
            with self.assertRaises(ValueError):
                env.step(actions)
        self.assertEqual(env.connection.mock_calls, [])

    def test_failed_and_pending_agents_do_not_accept_actions(self):
        env = DualMerging()
        env.connection = Mock()
        env.finished = False
        env.status = {"ego_entry": "failed", "ego_exit": "pending"}
        self.assertEqual(env.active_agents, ())
        with self.assertRaises(ValueError):
            env.step({"ego_entry": [0, 0]})

    def test_pending_departure_deadline_is_a_scene_failure_not_experience(self):
        env = DualMerging()
        env.connection = Mock()
        env.finished = False
        env.status = {"ego_entry": "success", "ego_exit": "pending"}
        env.connection.simulation.getTime.side_effect = [1199.9, 1200.0]
        env.connection.simulation.getDepartedIDList.return_value = []
        env.connection.vehicle.getIDList.return_value = []
        obs, reward, term, trunc, info = env.step({})
        self.assertEqual((obs, reward, term, trunc), ({}, {}, {}, {}))
        self.assertEqual(info["ego_exit"]["status"], "not_departed")
        self.assertFalse(info["ego_exit"]["transition_valid"])
        self.assertTrue(info["__scene__"]["scene_done"])
        env.connection.simulationStep.assert_called_once_with()

    def test_reward_priority_and_reset(self):
        car = Merging(1, task="exit")
        car._connection = Mock()
        car.reset_episode_state()
        car.ego_id = "ego_exit"
        car.ego_on_road = True
        car.parsed_action = [0, 0]
        car.stage_reached = True
        car.counter = 6000
        car.traci.simulation.getCollidingVehiclesIDList.return_value = []
        car.traci.simulation.getArrivedIDList.return_value = ["ego_exit"]
        car.traci.simulation.getEmergencyStoppingVehiclesIDList.return_value = []
        with patch("src.merging.safe_gap", return_value=(True, True)):
            reward = car.getReward()
        self.assertAlmostEqual(reward, 49.97)
        self.assertEqual(car.reward_components["stage"], 0)
        self.assertEqual(car.reward_components["timeout"], 0)
        self.assertAlmostEqual(sum(car.reward_components.values()), reward)
        car.reset_episode_state()
        self.assertFalse(car.stage_reached)
        self.assertFalse(car.given_merge_reward)
        self.assertEqual(car.counter, 0)

    def test_late_departure_does_not_reset_first_agents_history(self):
        env = DualMerging()
        env.connection = Mock()
        env.episode_seed = 17
        for car in env.cars.values():
            car.reset_episode_state()
            car._get_obs = Mock(return_value=np.zeros((31, 3), dtype=np.float32))
        env.connection.simulation.getDepartedIDList.return_value = ["ego_entry"]
        env.connection.vehicle.getRoadID.return_value = "O3"
        env.connection.simulation.getTime.return_value = 120.1
        obs, _ = env._departures()
        self.assertEqual(set(obs), {"ego_entry"})
        env.delays["ego_entry"].action_history[0][:] = [1, 2]
        env.connection.simulation.getDepartedIDList.return_value = ["ego_exit"]
        env.connection.vehicle.getRoadID.return_value = "O1"
        env.connection.simulation.getTime.return_value = 121.0
        obs, info = env._departures()
        self.assertEqual(set(obs), {"ego_exit"})
        self.assertFalse(info["ego_exit"]["transition_valid"])
        self.assertTrue((obs["ego_exit"][93:] == 0).all())
        np.testing.assert_array_equal(env.delays["ego_entry"].action_history[0], [1, 2])

    def test_collision_and_emergency_are_attributed_by_vehicle_id(self):
        for event in ("getCollidingVehiclesIDList", "getEmergencyStoppingVehiclesIDList"):
            with self.subTest(event=event):
                env = DualMerging(fixed_delay=0)
                env.connection = Mock()
                env.finished = False
                env.status = dict.fromkeys(AGENT_TASKS, "active")
                env.connection.simulation.getTime.side_effect = [120.1, 120.2]
                env.connection.vehicle.getIDList.return_value = []
                for name in ("getStartingTeleportIDList", "getEndingTeleportIDList",
                             "getCollidingVehiclesIDList", "getArrivedIDList",
                             "getEmergencyStoppingVehiclesIDList", "getDepartedIDList"):
                    getattr(env.connection.simulation, name).return_value = []
                getattr(env.connection.simulation, event).return_value = ["ego_entry"]
                for key, car in env.cars.items():
                    car.reset_episode_state()
                    car.ego_id = key
                    car._connection = env.connection
                    env.delays[key].reset_history(car.observation, {}, seed=1)
                with patch("src.merging.safe_gap", return_value=(True, True)):
                    _, rewards, terms, _, info = env.step(dict.fromkeys(AGENT_TASKS, [0, 0]))
                self.assertTrue(terms["ego_entry"])
                self.assertFalse(terms["ego_exit"])
                self.assertLess(rewards["ego_entry"], -99)
                self.assertFalse(info["__scene__"]["scene_done"])
                env.connection.simulationStep.assert_called_once_with()
                env.connection.close.assert_not_called()


@unittest.skipUnless(__name__ == "__main__", "opt-in real SUMO tests")
class DualSumoTests(unittest.TestCase):
    def setUp(self):
        self.env = None

    def tearDown(self):
        if self.env is not None:
            self.env.close()

    def make_env(self, **kwargs):
        self.env = DualMerging(**kwargs)
        obs, info = self.env.reset(seed=579071)
        return obs, info

    def actions(self):
        return {key: [0, 0] for key in self.env.active_agents}

    def test_clock_delays_and_safety(self):
        for fixed_delay in (0, 10, 20, None):
            for safety in (False, True):
                with self.subTest(fixed_delay=fixed_delay, safety=safety):
                    obs, _ = self.make_env(fixed_delay=fixed_delay, safety=safety)
                    self.assertEqual(set(obs), set(AGENT_TASKS))
                    first = self.env.connection.simulation.getTime()
                    for step in range(1, 51):
                        obs, rewards, terms, truncs, infos = self.env.step(self.actions())
                        self.assertAlmostEqual(self.env.connection.simulation.getTime(),
                                               first + step * 0.1)
                        for key in AGENT_TASKS:
                            self.assertFalse(terms[key] or truncs[key])
                            self.assertEqual(obs[key].shape, (134,))
                            self.assertTrue(np.isfinite(obs[key]).all())
                            self.assertAlmostEqual(rewards[key],
                                                   sum(infos[key]["reward_components"].values()))
                            age = infos[key]["observation_age_steps"]
                            self.assertAlmostEqual(infos[key]["sumo_time"] - infos[key]["packet_time"],
                                                   age * 0.1)
                            actions = obs[key][93:133].reshape(20, 2)
                            self.assertTrue((actions[min(age, 20):] == 0).all())
                            if fixed_delay is not None:
                                self.assertEqual(age, min(step, fixed_delay))
                    self.env.close()

    def test_one_arrival_keeps_other_agent_and_connection_alive(self):
        self.make_env(fixed_delay=0)
        conn = self.env.connection
        conn.vehicle.moveTo("ego_entry", "E76_0", conn.lane.getLength("E76_0") - 0.1)
        conn.vehicle.setSpeed("ego_entry", 10)
        _, rewards, terminated, _, infos = self.env.step(self.actions())
        self.assertTrue(terminated["ego_entry"])
        self.assertEqual(infos["ego_entry"]["message"], "Arrived")
        self.assertGreater(rewards["ego_entry"], 49)
        self.assertEqual(self.env.active_agents, ("ego_exit",))
        old_time = conn.simulation.getTime()
        obs, rewards, _, _, _ = self.env.step(self.actions())
        self.assertEqual(set(obs), {"ego_exit"})
        self.assertNotIn("ego_entry", rewards)
        self.assertAlmostEqual(conn.simulation.getTime() - old_time, 0.1)

    def test_timeout_keeps_other_alive_and_releases_vehicle(self):
        self.make_env(fixed_delay=0)
        self.env.cars["ego_entry"].counter = 5999
        _, rewards, terms, truncs, _ = self.env.step(self.actions())
        self.assertTrue(truncs["ego_entry"])
        self.assertFalse(terms["ego_entry"])
        self.assertLess(rewards["ego_entry"], -99)
        self.assertEqual(self.env.connection.vehicle.getLaneChangeMode("ego_entry"), 1621)
        self.assertEqual(self.env.active_agents, ("ego_exit",))
        self.env.step(self.actions())

    def test_distinct_action_histories_and_one_time_exit_reward(self):
        self.make_env(fixed_delay=10)
        for _ in range(4):
            obs, _, _, _, _ = self.env.step({"ego_entry": [0.1, 0], "ego_exit": [0.7, 0]})
        np.testing.assert_allclose(obs["ego_entry"][93:95], [0.1, 0])
        np.testing.assert_allclose(obs["ego_exit"][93:95], [0.7, 0])
        car = self.env.cars["ego_exit"]
        self.env.connection.vehicle.moveTo("ego_exit", "D3_0", 10)
        car.observation = car._get_obs()
        car.state = car.observation.copy()
        _, _, _, _, first = self.env.step(self.actions())
        _, _, _, _, second = self.env.step(self.actions())
        self.assertEqual(first["ego_exit"]["stage_reward"], 50)
        self.assertEqual(second["ego_exit"]["stage_reward"], 0)
        self.assertFalse(self.env.cars["ego_entry"].given_merge_reward)

    def test_teleport_is_local_and_never_earns_success(self):
        from src.merging import traci
        original_start = traci.start
        with patch("src.dual_merging.traci.start",
                   side_effect=lambda command, **kwargs: original_start(
                       command + ["--time-to-teleport", "0.2"], **kwargs)):
            self.make_env(fixed_delay=0)
        conn = self.env.connection
        car = self.env.cars["ego_exit"]
        conn.vehicle.moveTo("ego_exit", "E77_2", conn.lane.getLength("E77_2") - 0.1)
        conn.vehicle.setPreviousSpeed("ego_exit", 0)
        conn.vehicle.setSpeed("ego_exit", 0)
        car.observation = car._get_obs()
        car.state = car.observation.copy()
        detected = False
        for _ in range(40):
            _, rewards, _, _, info = self.env.step(self.actions())
            if info["ego_exit"]["teleport_started"]:
                detected = True
                self.assertLessEqual(rewards["ego_exit"], 0)
            if detected and car.ego_on_road:
                break
        self.assertTrue(detected)
        self.assertTrue(car.ego_on_road)
        self.assertFalse(self.env.cars["ego_entry"].ego_teleported)
        conn.vehicle.moveTo("ego_exit", "D3_0", conn.lane.getLength("D3_0") - 0.1)
        conn.vehicle.setSpeed("ego_exit", 10)
        _, rewards, terms, _, info = self.env.step(self.actions())
        self.assertTrue(terms["ego_exit"])
        self.assertFalse(info["ego_exit"]["is_success"])
        self.assertLessEqual(rewards["ego_exit"], 0)
        self.assertEqual(info["ego_exit"]["stage_reward"], 0)
        self.assertEqual(self.env.active_agents, ("ego_entry",))

    def test_reset_reproduces_observations_and_clears_histories(self):
        self.make_env()
        def trajectory():
            result = []
            for _ in range(30):
                obs, rewards, _, _, _ = self.env.step(self.actions())
                result.append((np.concatenate([obs[k] for k in AGENT_TASKS]),
                               tuple(rewards.values())))
            return result
        first = trajectory()
        self.env.cars["ego_entry"].given_merge_reward = True
        self.env.cars["ego_exit"].ego_teleported = True
        obs, _ = self.env.reset(seed=579071)
        for key, car in self.env.cars.items():
            self.assertEqual(car.counter, 0)
            self.assertFalse(car.given_merge_reward)
            self.assertFalse(car.ego_teleported)
            self.assertTrue((obs[key][93:] == 0).all())
        second = trajectory()
        for (left, lr), (right, rr) in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(lr, rr)

    def test_completed_scene_requires_explicit_reset_and_starts_clean(self):
        initial, first_info = self.make_env(fixed_delay=0)
        initial = {key: value.copy() for key, value in initial.items()}
        conn = self.env.connection
        for key, destination in (("ego_entry", "E76"), ("ego_exit", "D3")):
            lane = f"{destination}_0"
            conn.vehicle.moveTo(key, lane, conn.lane.getLength(lane) - 0.1)
            conn.vehicle.setSpeed(key, 10)
            car = self.env.cars[key]
            car.observation = car._get_obs()
            car.state = car.observation.copy()
        _, _, terms, _, info = self.env.step(self.actions())
        self.assertTrue(all(terms.values()))
        self.assertTrue(info["__scene__"]["scene_done"])
        self.assertEqual(self.env.active_agents, ())
        with self.assertRaises(RuntimeError):
            self.env.step({})
        reset_obs, reset_info = self.env.reset(seed=579071)
        self.assertEqual(reset_info["sumo_seed"], first_info["sumo_seed"])
        self.assertEqual(set(self.env.active_agents), set(AGENT_TASKS))
        for key in AGENT_TASKS:
            np.testing.assert_array_equal(reset_obs[key], initial[key])
            car = self.env.cars[key]
            self.assertFalse(car.done)
            self.assertEqual(car.counter, 0)
            self.assertEqual(car.reward, 0)
            self.assertFalse(car.stage_reached)
            self.assertFalse(car.given_merge_reward)

    def test_connections_are_isolated(self):
        self.make_env(fixed_delay=0)
        second = DualMerging(fixed_delay=0)
        try:
            second.reset(seed=579072)
            before = self.env.connection.simulation.getTime()
            second.step({key: [0, 0] for key in second.active_agents})
            self.assertEqual(self.env.connection.simulation.getTime(), before)
            second.close()
            self.env.step(self.actions())
        finally:
            second.close()


if __name__ == "__main__":
    unittest.main()
