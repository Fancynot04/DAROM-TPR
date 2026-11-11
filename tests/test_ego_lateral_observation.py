import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from src.wrapper import DelayWrapper
from src.delay_encoder import DelayAwareGRUEncoder


class Vehicle:
    def __init__(self): self.position, self.speed = 0., 0.
    def getLateralLanePosition(self, ego): return self.position
    def getLateralSpeed(self, ego): return self.speed
    def getPosition(self, ego): return (100. + self.position, 200. - self.position)
    def getSpeed(self, ego): return 12. + self.position
    def getAcceleration(self, ego): return -0.25
    def getRoadID(self, ego): return 'E76'
    def getDrivingDistance(self, ego, edge, pos): return 500. - self.position


class FakeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-5000, 5000, (31, 3), dtype=np.float32)
        self.action_space = gym.spaces.Box(-5, 5, (2,), dtype=np.float32)
        self.traci = SimpleNamespace(vehicle=Vehicle())
        self.ego_id, self.ego_on_road, self.ego_teleported = 'ego', True, False
        self.action = np.zeros(2, dtype=np.float32)
        self.observation_snapshot = object()
        self.observation = np.arange(93, dtype=np.float32).reshape(31, 3)
    def reset(self, *, seed=None, options=None):
        return self.observation.copy(), {}
    def step(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.observation = self.observation + 1
        self.traci.vehicle.position += .2
        self.traci.vehicle.speed = 2.
        return self.observation.copy(), 0., False, False, {}


class LateralObservationTests(unittest.TestCase):
    def test_delay_and_layout(self):
        old = DelayWrapper(FakeEnv(), max_delay=20, mode='all', seed=5)
        new = DelayWrapper(FakeEnv(), max_delay=20, mode='all', seed=5, observation_mode='ego_lateral_v1')
        a, _ = old.reset(seed=5); b, _ = new.reset(seed=5)
        for step in range(40):
            self.assertEqual(a.shape, (134,)); self.assertEqual(b.shape, (136,))
            np.testing.assert_array_equal(a[:3], b[:3])
            np.testing.assert_array_equal(a[3:], b[5:])
            np.testing.assert_allclose(b[3:5], [new.env.traci.vehicle.position, new.env.traci.vehicle.speed])
            a, *_ = old.step([1., -2.]); b, *_ = new.step([1., -2.])
            self.assertEqual(old.current_observation_delay, new.current_observation_delay)
            self.assertEqual(old.delay_of_received_observation, new.delay_of_received_observation)

    def test_spatial_live_and_delay(self):
        old = DelayWrapper(FakeEnv(), max_delay=20, mode='all', seed=5)
        new = DelayWrapper(FakeEnv(), max_delay=20, mode='all', seed=5, observation_mode='ego_spatial_v2')
        a, _ = old.reset(seed=5); b, _ = new.reset(seed=5)
        seen_delay = False
        for _ in range(40):
            self.assertEqual(b.shape, (137,))
            v = new.env.traci.vehicle
            np.testing.assert_allclose(b[:6], [*v.getPosition('ego'), v.getSpeed('ego'),
                v.position, v.getAcceleration('ego'), v.getDrivingDistance('ego', 'D3', 0)], rtol=1e-6)
            np.testing.assert_array_equal(a[3:], b[6:])
            self.assertEqual(old.delay_of_received_observation, new.delay_of_received_observation)
            seen_delay |= new.delay_of_received_observation > 0
            a, *_ = old.step([1., -2.]); b, *_ = new.step([1., -2.])
        self.assertTrue(seen_delay)
        new.env.done = True
        np.testing.assert_array_equal(new._entities_for_policy(new.env.observation)[:6], np.zeros(6))
        new.env.done = False
        new.env.traci.vehicle.getRoadID = lambda _: 'D3'
        self.assertEqual(new._entities_for_policy(new.env.observation)[5], 0.)
        new.env.traci.vehicle.getRoadID = lambda _: 'E77'
        new.env.traci.vehicle.getDrivingDistance = lambda *_: -1073741824.
        with self.assertRaises(ValueError): new._entities_for_policy(new.env.observation)

    def test_spatial_sac_buffer(self):
        env = DelayWrapper(FakeEnv(), max_delay=20, mode='all', observation_mode='ego_spatial_v2')
        obs, _ = env.reset(seed=9)
        kw = dict(features_extractor_class=DelayAwareGRUEncoder,
                  features_extractor_kwargs=dict(act_dim=2, hist_len=20, delay_dim=1,
                      input_mode='paper_raw', observation_mode='ego_spatial_v2'))
        model = SAC('MlpPolicy', env, policy_kwargs=kw, buffer_size=10, device='cpu')
        tensor = torch.as_tensor(obs[None]); action = model.actor(tensor)
        self.assertEqual(action.shape, (1, 2)); self.assertEqual(len(model.critic(tensor, action)), 2)
        model.replay_buffer.add(obs[None], obs[None], action.detach().numpy(), np.array([0.]), np.array([0.]), [{}])
        self.assertEqual(model.replay_buffer.sample(1).observations.shape, (1, 137))
        with self.assertRaises(ValueError):
            DelayAwareGRUEncoder(env.observation_space, 2, 20, 1, input_mode='paper_raw')
        with self.assertRaises(ValueError):
            DelayAwareGRUEncoder(env.observation_space, 2, 20, 1, observation_mode='ego_spatial_v2')

    def test_sac_and_buffer(self):
        env = DelayWrapper(FakeEnv(), max_delay=20, mode='all', observation_mode='ego_lateral_v1')
        obs, _ = env.reset(seed=9)
        kwargs = dict(features_extractor_class=DelayAwareGRUEncoder,
                      features_extractor_kwargs=dict(act_dim=2, hist_len=20, delay_dim=1,
                          input_mode='paper_raw', observation_mode='ego_lateral_v1'))
        model = SAC('MlpPolicy', env, policy_kwargs=kwargs, buffer_size=10, learning_starts=100, device='cpu')
        tensor = torch.as_tensor(obs[None])
        action = model.actor(tensor)
        self.assertEqual(action.shape, (1, 2))
        self.assertEqual(len(model.critic(tensor, action)), 2)
        model.replay_buffer.add(obs[None], obs[None], action.detach().numpy(), np.array([0.]), np.array([0.]), [{}])
        self.assertEqual(model.replay_buffer.sample(1).observations.shape, (1, 136))
        with self.assertRaises(ValueError):
            DelayAwareGRUEncoder(env.observation_space, 2, 20, 1, input_mode='paper_raw')


if __name__ == '__main__': unittest.main()
