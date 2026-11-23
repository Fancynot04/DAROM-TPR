
"""CTDE-MASAC 联合经验回放。"""

import json

import numpy as np
import torch


class JointReplayBuffer:
    """每个 SUMO 全局步保存一条双 Ego 联合经验。"""

    AGENT_IDS = ("ego_1", "ego_2")

    SHAPES = {
        "policy_obs": (2, 134),
        "critic_obs": (2, 93),
        "actions": (2, 2),
        "rewards": (2,),
        "dones": (2,),
        "active_mask": (2,),
        "acted_mask": (2,),
        "next_policy_obs": (2, 134),
        "next_critic_obs": (2, 93),
        "next_active_mask": (2,),
    }

    TEAM_MASK_FIELDS = ("team_active", "team_done", "success_mask", "next_success_mask")

    MASK_FIELDS = (
        "dones",
        "active_mask",
        "acted_mask",
        "next_active_mask",
    )

    def __init__(
        self,
        capacity=100_000,
        seed=42,
        agent_ids=AGENT_IDS,
        nav_dim=0,
        team=False,
        policy_obs_dim=None,
        decision_transitions=False,
    ):
        if capacity <= 0:
            raise ValueError("Invalid capacity")

        if tuple(agent_ids) != self.AGENT_IDS:
            raise ValueError("Agent order mismatch")

        if policy_obs_dim is None:
            policy_obs_dim = 134 + nav_dim
        self.shapes = {
            **self.SHAPES,
            "policy_obs": (2, policy_obs_dim),
            "next_policy_obs": (2, policy_obs_dim),
            "critic_obs": (2, 93 + nav_dim),
            "next_critic_obs": (2, 93 + nav_dim),
        }
        self.mask_fields = self.MASK_FIELDS
        if decision_transitions:
            self.shapes["bootstrap_discount"] = ()
        if team:
            self.shapes.update(team_reward=(), team_active=(), team_done=(),
                               success_mask=(2,), next_success_mask=(2,))
            self.mask_fields += self.TEAM_MASK_FIELDS
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0

        self.rng = np.random.default_rng(seed)

        self.data = {
            name: np.empty(
                (self.capacity, *shape),
                dtype=np.float32,
            )
            for name, shape in self.shapes.items()
        }

    def __len__(self):
        return self.size

    def save(self, stream):
        """Serialize only initialized slots, preserving their physical order."""
        metadata = json.dumps(dict(version=1, capacity=self.capacity, size=self.size,
                                   position=self.position, rng=self.rng.bit_generator.state))
        np.savez(stream, metadata=metadata,
                 **{name: values[:self.size] for name, values in self.data.items()})

    def load(self, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            size, position = meta["size"], meta["position"]
            if (meta["version"] != 1 or meta["capacity"] != self.capacity
                    or not 0 <= size <= self.capacity or not 0 <= position < self.capacity
                    or (size < self.capacity and position != size)
                    or set(archive.files) != {"metadata", *self.data}):
                raise ValueError("Incompatible replay buffer metadata")
            for name, target in self.data.items():
                values = archive[name]
                if values.shape != (size, *self.shapes[name]) or values.dtype != target.dtype:
                    raise ValueError(f"Incompatible replay buffer field: {name}")
                target[:size] = values
            self.rng.bit_generator.state = meta["rng"]
            self.size, self.position = size, position

    def add(self, **transition):
        """保存一条联合经验，不保存调用者数组的引用。"""

        if set(transition) != set(self.shapes):
            missing = set(self.shapes) - set(transition)
            extra = set(transition) - set(self.shapes)
            raise ValueError(
                f"Transition fields: missing={missing}, "
                f"extra={extra}"
            )

        values = {}

        for name, shape in self.shapes.items():
            value = np.asarray(
                transition[name],
                dtype=np.float32,
            )

            if value.shape != shape:
                raise ValueError(
                    f"{name}: expected {shape}, "
                    f"got {value.shape}"
                )

            if not np.isfinite(value).all():
                raise ValueError(
                    f"{name}: non-finite values"
                )

            values[name] = value.copy()

        for name in self.mask_fields:
            mask = values[name]

            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError(
                    f"{name}: expected binary mask"
                )

        active = values["active_mask"]
        acted = values["acted_mask"]

        if np.any(acted > active):
            raise ValueError(
                "Inactive Ego cannot execute action"
            )

        # 防止没有执行动作的终止事件被错误归因
        # 给当前请求动作。
        invalid_terminal = (
            (active == 1)
            & (acted == 0)
            & (values["rewards"] != 0)
        )

        if np.any(invalid_terminal):
            raise ValueError(
                "Pre-step terminal reward requires "
                "correct transition attribution"
            )

        index = self.position

        for name, value in values.items():
            self.data[name][index] = value

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(self, batch_size, device="cpu"):
        """随机抽取联合 Transition。"""

        if batch_size <= 0:
            raise ValueError("Invalid batch size")

        if self.size < batch_size:
            raise ValueError(
                "Not enough transitions"
            )

        indices = self.rng.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        return {
            name: torch.as_tensor(
                values[indices],
                device=device,
            )
            for name, values in self.data.items()
        }
