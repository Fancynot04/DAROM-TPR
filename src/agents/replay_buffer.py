
"""CTDE-MASAC 联合经验回放。"""

import numpy as np
import torch


class JointReplayBuffer:
    """每个 SUMO 全局步保存一条双 Ego 联合经验。"""

    AGENT_IDS = ("ego_o3", "ego_e74")

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
    ):
        if capacity <= 0:
            raise ValueError("Invalid capacity")

        if tuple(agent_ids) != self.AGENT_IDS:
            raise ValueError("Agent order mismatch")

        self.capacity = int(capacity)
        self.position = 0
        self.size = 0

        self.rng = np.random.default_rng(seed)

        self.data = {
            name: np.empty(
                (self.capacity, *shape),
                dtype=np.float32,
            )
            for name, shape in self.SHAPES.items()
        }

    def __len__(self):
        return self.size

    def add(self, **transition):
        """保存一条联合经验，不保存调用者数组的引用。"""

        if set(transition) != set(self.SHAPES):
            missing = set(self.SHAPES) - set(transition)
            extra = set(transition) - set(self.SHAPES)
            raise ValueError(
                f"Transition fields: missing={missing}, "
                f"extra={extra}"
            )

        values = {}

        for name, shape in self.SHAPES.items():
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

        for name in self.MASK_FIELDS:
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
                values[indices].copy(),
                device=device,
            )
            for name, values in self.data.items()
        }
