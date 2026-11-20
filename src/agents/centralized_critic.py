
"""CTDE-MASAC 集中式 Twin-Q Critic。

直接使用当前真实交通观测，不使用 GRU。
"""

import torch
from torch import nn


class CentralizedTwinQ(nn.Module):
    def __init__(
        self,
        num_agents=2,
        obs_dim=93,
        action_dim=2,
        hidden_dim=256,
        action_scale=5.0,
        nav_dim=0,
    ):
        super().__init__()

        if num_agents < 1:
            raise ValueError("Invalid num_agents")

        if obs_dim < 3 or obs_dim % 3 != 0:
            raise ValueError("Invalid obs_dim")

        if action_dim < 1:
            raise ValueError("Invalid action_dim")

        if action_scale <= 0:
            raise ValueError("Invalid action_scale")

        self.num_agents = num_agents
        self.obs_dim = obs_dim + nav_dim
        self.action_dim = action_dim
        self.action_scale = float(action_scale)

        # 与 DAROM GRU 的真实交通状态缩放保持一致
        scale = [1000.0, 5.0, 30.0]
        scale += [50.0, 20.0, 30.0] * (
            obs_dim // 3 - 1
        )

        self.register_buffer(
            "observation_scale",
            torch.tensor(scale + [1.0] * nav_dim, dtype=torch.float32),
        )

        # 每辆 Ego：
        # 交通观测及可选目标出口 + 2 维动作 + 1 维 Mask
        input_dim = num_agents * (
            self.obs_dim + action_dim + 1
        )

        def make_network():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_agents),
            )

        self.q1 = make_network()
        self.q2 = make_network()

    def forward(
        self,
        observations,
        actions,
        active_mask,
    ):
        """
        observations: (B, 2, obs_dim)
        actions:      (B, 2, 2)
        active_mask:  (B, 2)

        返回:
            q1: (B, 2)
            q2: (B, 2)
        """

        batch = observations.shape[0]
        n = self.num_agents

        if observations.shape != (
            batch, n, self.obs_dim
        ):
            raise ValueError("Invalid observation shape")

        if actions.shape != (
            batch, n, self.action_dim
        ):
            raise ValueError("Invalid action shape")

        if active_mask.shape != (batch, n):
            raise ValueError("Invalid mask shape")

        if (
            not torch.isfinite(observations).all()
            or not torch.isfinite(actions).all()
            or not torch.isfinite(active_mask).all()
        ):
            raise ValueError("Non-finite Critic input")

        if not torch.all(
            (active_mask == 0) | (active_mask == 1)
        ):
            raise ValueError("Active mask must be binary")

        mask = active_mask.to(
            device=observations.device,
            dtype=observations.dtype,
        )

        # 1. 真实观测归一化
        normalized_obs = (
            observations / self.observation_scale
        )

        # 2. 无效 Ego 的状态和动作清零
        normalized_obs = (
            normalized_obs * mask.unsqueeze(-1)
        )

        normalized_actions = (
            actions / self.action_scale
        ) * mask.unsqueeze(-1)

        # 3. 拼接两辆 Ego 的联合输入
        joint_input = torch.cat(
            [
                normalized_obs.reshape(batch, -1),
                normalized_actions.reshape(batch, -1),
                mask,
            ],
            dim=-1,
        )

        # 4. Twin-Q
        q1 = self.q1(joint_input)
        q2 = self.q2(joint_input)

        # 5. 不输出无效 Ego 的 Q
        q1 = q1 * mask
        q2 = q2 * mask

        return q1, q2
