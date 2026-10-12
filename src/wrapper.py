import numpy as np

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper, spaces
from gymnasium.spaces import Box, Discrete
from collections import deque
from .utils import flatten, PredictorDataset, fine_tune_predictor, getDistance
from .safety_controller import safetyCheck, target_lane_line
from .merging import parseAction
from .geometry import align_delayed_neighbors
# from state_predictor import predictor_infer, load_predictor
import torch
ABLATION_MODES = ('full', 'delayed_state_only', 'delayed_state_delay',
                  'delayed_state_action', 'linear_predictor')


class DelayWrapper(ObservationWrapper):
    def __init__(self, env, max_delay=10, mode=['only_delayed_state' 'delayed_state_and_action', 'delayed_state_and_delay', 'all'], drl_ormoc=False, delay_mode='uniform', seed=None, ablation='full'):
        super().__init__(env)
        if ablation not in ABLATION_MODES or (ablation != 'full' and mode != 'all'):
            raise ValueError(f'Unsupported ablation/mode: {ablation}/{mode}')
        self.ablation = ablation
        self.orig_shape = self.env.observation_space.shape
        orig_shape = self.env.observation_space.shape
        self.mode = mode
        self.delay_mode = delay_mode
        self.congested = False  # for bursty mode
        self._delay_seed = seed
        self._delay_episode = -1
        self._delay_rng = np.random.default_rng()

        self.max_delay = max_delay
 
        # self.observation_space = spaces.Dict({
        #     "entities": self.env.observation_space,
        #     "global": spaces.Box(low=-5.0, high=5.0, shape=(self.global_dim,))
        # })
        if drl_ormoc:
            mode = 'only_delayed_state'
        else:
            mode = mode
        if self.mode == 'only_delayed_state':
            self.observation_space = self.env.observation_space
        elif self.mode == 'delayed_state_and_action':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + self.max_delay * self.env.action_space.shape[0],),
                low=-5000,
                high=5000,
                dtype=np.float32
            )
        elif self.mode == 'delayed_state_and_delay':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + 1,),
                low=-5000,
                high=5000,
                dtype=np.float32
            )
        elif self.mode == 'all':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + self.max_delay * self.env.action_space.shape[0] + 1,),
                low=-5000,
                high=5000,
                dtype=np.float32
            )
        self.max_delay = max_delay
        # Keep full-shaped entity observations in history: (num_entities, feature_dim)
        self.observation_history = deque([
            {
                'observation': np.zeros((self.orig_shape[0], self.orig_shape[1]), dtype=np.float32),
                'delay': 0
            }
            for _ in range(self.max_delay + 1)
        ], maxlen=self.max_delay + 1)
        # Keep action history as float32 vectors; avoid shared inner lists
        self.action_history = deque([
            np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            for _ in range(self.max_delay)
        ], maxlen=self.max_delay)
        self.last_observation = None
        self.delay_of_received_observation = 0
        self.delay_of_last_observation = 0
        self.obs = None

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        if kwargs.get('seed') is not None:
            self._delay_seed = kwargs['seed']
            self._delay_episode = 0
        else:
            self._delay_episode += 1
        if self._delay_seed is None:
            self._delay_rng = np.random.default_rng()
        else:
            sequence = np.random.SeedSequence([self._delay_seed, self._delay_episode])
            self._delay_rng = np.random.default_rng(sequence)
        self.current_observation_delay = 0
        self.last_observation = self.obs
        self.last_snapshot = self.env.observation_snapshot
        self.delay_of_received_observation = 0
        self.delay_of_last_observation = 0
        self.congested = False  # reset bursty state
        self.observation_history = deque([
            {
                'observation': np.zeros((self.orig_shape[0], self.orig_shape[1]), dtype=np.float32),
                'delay': 0
            }
            for _ in range(self.max_delay + 1)
        ], maxlen=self.max_delay + 1)
        self.action_history = deque([
            np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            for _ in range(self.max_delay)
        ], maxlen=self.max_delay)
        # self.newX = {
        #     "entities": self.obs,
        #     "global": flatten([[a.tolist() for a in self.action_history], [0]])
        # }
        self.newX = np.asarray(flatten([self.obs, list(self.action_history), [0]]), dtype=np.float32)
        if self.mode == 'only_delayed_state':
            return self.obs, info
        elif self.mode == 'delayed_state_and_action':
            return np.asarray(flatten([self.obs, list(self.action_history)]), dtype=np.float32), info
        elif self.mode == 'delayed_state_and_delay':
            return np.asarray(flatten([self.obs, [0]]), dtype=np.float32), info
        elif self.mode == 'all':
            return self._policy_observation(), info
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        # return flatten([self.obs, list(self.action_history), [0]]), info

    def _sample_delay(self):
        """Sample observation delay based on the configured delay_mode."""
        if self.delay_mode == "uniform":
            return self._delay_rng.integers(0, self.max_delay + 1)

        elif self.delay_mode == "exponential":
            # Favors low delays, long tail clipped to max_delay
            raw = self._delay_rng.exponential(scale=self.max_delay / 3.0)
            return int(np.clip(round(raw), 0, self.max_delay))

        elif self.delay_mode == "triangular":
            # Symmetric triangular distribution peaking at max_delay / 2
            mid = self.max_delay / 2.0
            return int(round(self._delay_rng.triangular(0, mid, self.max_delay)))

        elif self.delay_mode == "bursty":
            # Markov regime-switching: normal vs congested
            if self.congested:
                self.congested = self._delay_rng.random() < 0.9  # 90% stay congested
                return self._delay_rng.integers(self.max_delay // 2, self.max_delay + 1)
            else:
                self.congested = self._delay_rng.random() < 0.05  # 5% chance of burst
                return self._delay_rng.integers(0, self.max_delay // 4 + 1)

        elif self.delay_mode == "bimodal":
            # Mixture of good connection (low delay) and bad connection (high delay)
            if self._delay_rng.random() < 0.6:
                return self._delay_rng.integers(0, max(self.max_delay // 5, 1) + 1)
            else:
                return self._delay_rng.integers(self.max_delay * 3 // 5, self.max_delay + 1)

        else:
            raise ValueError(f"Unknown delay_mode: {self.delay_mode}")

    def _policy_observation(self):
        if self.ablation == 'full':
            return self.newX
        policy_obs = self.newX.copy()
        obs_size = self.orig_shape[0] * self.orig_shape[1]
        if self.ablation == 'linear_predictor':
            entities = policy_obs[:obs_size].reshape(self.orig_shape)
            entities[1:, 0] += entities[1:, 2] * self.delay_of_received_observation * 0.1
        if self.ablation in ('delayed_state_only', 'delayed_state_delay', 'linear_predictor'):
            policy_obs[obs_size:-1] = 0
        if self.ablation in ('delayed_state_only', 'delayed_state_action', 'linear_predictor'):
            policy_obs[-1] = 0
        return policy_obs

    def observation(self, obs):
        self.obs = obs
        self.current_observation_delay = self._sample_delay()
        # self.current_observation_delay = 2
        # self.d = self.max_delay
        self.action_history.pop()
        self.action_history.appendleft(self.env.action)
        snapshot = self.env.observation_snapshot
        self.observation_history[-self.current_observation_delay-1] = {
            'observation': self.obs.copy() if snapshot is not None else np.zeros(self.orig_shape),
            'delay': self.current_observation_delay, 'snapshot': snapshot,
        }
        received_obsWithDelay = self.observation_history.pop()
        received_obs = received_obsWithDelay['observation']
        self.delay_of_received_observation = received_obsWithDelay['delay']
        self.observation_history.appendleft({'observation': np.zeros(self.orig_shape), 'delay': 0})
        # print('Received observation:', received_obs)
        # print('Received observation:', received_obs)
        first_if_statement = not received_obs.any()
        second_if_statement = self.delay_of_last_observation + 1 < self.delay_of_received_observation
        # print('First if statement:', first_if_statement)
        # print('Second if statement:', second_if_statement)
        if_statement = first_if_statement or second_if_statement
        if if_statement:
            received_obs = self.last_observation
            self.delay_of_received_observation = self.delay_of_last_observation + 1
        else:
            self.last_snapshot = received_obsWithDelay['snapshot']
        # Updating the current ego must not mutate captured packets or env.state.
        received_obs = received_obs.copy()
        self.last_observation = received_obs
        self.delay_of_last_observation = self.delay_of_received_observation
        received_obs[0][:] = self.env.observation[0][:] #ego state is not affected by delay
        # Round values in newState to three decimal points 
        # self.newX = flatten([received_obs, list(self.action_history), [self.delay_of_received_observation]])
        # self.newX = {
        #     "entities": received_obs.astype(np.float32),
        #     "global": flatten([[a for a in self.action_history], [self.delay_of_received_observation]])
        # }
        masked_action_history = np.array(list(self.action_history)).copy()
        for i in range(self.delay_of_received_observation, len(masked_action_history)):
            masked_action_history[i] = np.zeros_like(masked_action_history[i])
        self.newX = np.asarray(flatten([received_obs, masked_action_history, [self.delay_of_received_observation]]), dtype=np.float32)
        if self.mode == 'only_delayed_state':
            return received_obs
        elif self.mode == 'delayed_state_and_action':
            return np.asarray(flatten([received_obs, masked_action_history]), dtype=np.float32)
        elif self.mode == 'delayed_state_and_delay':
            return np.asarray(flatten([received_obs, [self.delay_of_received_observation]]), dtype=np.float32)
        elif self.mode == 'all':
            return self._policy_observation()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def safety_observation(self):
        return align_delayed_neighbors(
            self.last_snapshot, self.env.observation_snapshot, self.env.observation[0],
        )

class SafetyWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if not self.env.env.ego_on_road:
            return action
        target_line = target_lane_line(self.env.env.ego_id, parseAction(action)[1])
        # All captured neighbors are predicted and aligned before lane filtering.
        # They already refer to the current instant: do not extrapolate twice.
        action, safety_info = safetyCheck(action, self.env.safety_observation(),
                                        False, 'all', target_line)
        return action
