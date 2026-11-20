
"""多 Ego 独立通信延迟缓存。

沿用 DAROM DelayWrapper 的数据包排队与接收规则。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DelayedObservation:
    observation: np.ndarray
    age: int
    sampled_delay: int
    received_packet: bool


class MultiEgoDelayBuffer:
    """每辆 Ego 独立维护观测队列和通信延迟。"""

    def __init__(self, ego_ids, max_delay=20, seed=42, observation_shape=(31, 3)):
        self.ego_ids = tuple(ego_ids)
        self.max_delay = int(max_delay)
        self.seed = int(seed)
        self.observation_shape = observation_shape

        if not self.ego_ids:
            raise ValueError("No Ego IDs")

        if len(set(self.ego_ids)) != len(self.ego_ids):
            raise ValueError("Duplicate Ego IDs")

        if self.max_delay < 0:
            raise ValueError("Invalid max_delay")

        self.reset()

    def _check_observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)

        if obs.shape != self.observation_shape:
            raise ValueError(f"Expected observation {self.observation_shape}")

        if not np.isfinite(obs).all():
            raise ValueError("Non-finite observation")

        return obs.copy()

    def reset(self, seed=None):
        """新回合：清空所有队列并重建随机数状态。"""

        if seed is not None:
            self.seed = int(seed)

        self.rng = {
            ego_id: np.random.default_rng(
                np.random.SeedSequence([self.seed, i])
            )
            for i, ego_id in enumerate(self.ego_ids)
        }

        self.queues = {
            ego_id: [None] * (self.max_delay + 1)
            for ego_id in self.ego_ids
        }

        self.last_observations = {}
        self.ages = {}
        self.active = set()

    def activate(self, ego_id, observation):
        """Ego 实际出发时，初始化自己的观测历史。"""

        if ego_id not in self.queues:
            raise KeyError(ego_id)

        if ego_id in self.active:
            raise RuntimeError(f"Already active: {ego_id}")

        obs = self._check_observation(observation)

        self.queues[ego_id] = [None] * (self.max_delay + 1)
        self.last_observations[ego_id] = obs.copy()
        self.ages[ego_id] = 0
        self.active.add(ego_id)

        return DelayedObservation(
            observation=obs,
            age=0,
            sampled_delay=0,
            received_packet=True,
        )

    def update(self, ego_id, observation, delay=None):
        """一个 Ego 完成一次控制步后的延迟观测更新。"""

        if ego_id not in self.active:
            raise RuntimeError(
                f"Ego not activated: {ego_id}"
            )

        current = self._check_observation(observation)

        # 每辆车从自己的随机数生成器采样
        if delay is None:
            delay = int(
                self.rng[ego_id].integers(
                    0, self.max_delay + 1
                )
            )

        if (
            not isinstance(delay, (int, np.integer))
            or not 0 <= delay <= self.max_delay
        ):
            raise ValueError("Invalid delay")

        delay = int(delay)
        queue = self.queues[ego_id]

        # 1. 当前采集的数据包按延迟放入队列
        # 最新数据包覆盖同一到达槽位中的旧数据包
        queue[-delay - 1] = (
            current.copy(),
            delay,
        )

        # 2. 取出本步到达的数据包
        packet = queue.pop()
        queue.insert(0, None)

        previous_age = self.ages[ego_id]

        # 3. 无数据包，或者新到达的数据比已用观测过旧
        if (
            packet is None
            or packet[1] > previous_age + 1
        ):
            received = self.last_observations[
                ego_id
            ].copy()

            age = previous_age + 1
            accepted = False

        else:
            received = packet[0].copy()
            age = packet[1]
            accepted = True

        # 4. Ego 自身状态始终使用当前真实状态
        received[0] = current[0]

        # 5. 保存本车最新可用观测
        self.last_observations[ego_id] = received.copy()
        self.ages[ego_id] = age

        return DelayedObservation(
            observation=received,
            age=age,
            sampled_delay=delay,
            received_packet=accepted,
        )
