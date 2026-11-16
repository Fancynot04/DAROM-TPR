
"""CTDE-MASAC 的分散式 Gaussian Actor。"""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from gymnasium import spaces

from src.delay_encoder import DelayAwareGRUEncoder


class GaussianActor(nn.Module):
    """单辆 Ego 的 DAROM-GRU + 随机策略。"""

    def __init__(
        self,
        obs_dim=134,
        action_dim=2,
        action_scale=5.0,
        hidden_dim=256,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_scale = float(action_scale)

        if self.action_scale <= 0:
            raise ValueError("Invalid action scale")

        observation_space = spaces.Box(
            low=-5000.0,
            high=5000.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 复用 DAROM 的延迟感知 GRU
        self.encoder = DelayAwareGRUEncoder(
            observation_space=observation_space,
            act_dim=2,
            hist_len=20,
            delay_dim=1,
            features_dim=256,
            gru_hidden=64,
            input_mode="antwerp_scaled_reversed",
            nav_dim=0,
        )

        self.policy = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim, action_dim
        )

        self.log_std_head = nn.Linear(
            hidden_dim, action_dim
        )

    def forward(self, observation, deterministic=False):
        """返回动作和经过 tanh 修正的 log probability。"""

        if observation.ndim != 2 or observation.shape[1] != 134:
            raise ValueError("Expected observation (B, 134)")

        features = self.encoder(observation)
        hidden = self.policy(features)

        mean = self.mean_head(hidden)

        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, -20.0, 2.0)

        std = log_std.exp()
        distribution = Normal(mean, std)

        if deterministic:
            pre_tanh = mean
        else:
            # 重参数化采样，允许 Actor 反向传播
            pre_tanh = distribution.rsample()

        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale

        # tanh 的 Jacobian 修正，使用数值稳定公式
        log_det_tanh = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

        log_prob = (
            distribution.log_prob(pre_tanh)
            - log_det_tanh
            - math.log(self.action_scale)
        ).sum(dim=-1)

        return action, log_prob


class MultiAgentGaussianActor(nn.Module):
    """两个独立 Actor，分别使用自己的局部观测。"""

    def __init__(
        self,
        ego_ids=("ego_o3", "ego_e74"),
    ):
        super().__init__()

        self.ego_ids = tuple(ego_ids)

        if len(set(self.ego_ids)) != len(self.ego_ids):
            raise ValueError("Duplicate Ego IDs")

        if not self.ego_ids:
            raise ValueError("No Ego IDs")

        self.actors = nn.ModuleDict({
            ego_id: GaussianActor()
            for ego_id in self.ego_ids
        })

    def forward(
        self,
        observations,
        active_mask,
        deterministic=False,
    ):
        """
        observations: (B, N, 134)
        active_mask: (B, N)

        返回:
            actions:   (B, N, 2)
            log_probs: (B, N)
        """

        batch = observations.shape[0]
        n = len(self.ego_ids)

        if observations.shape != (batch, n, 134):
            raise ValueError("Invalid observation shape")

        if active_mask.shape != (batch, n):
            raise ValueError("Invalid active mask shape")

        if not torch.isfinite(observations).all():
            raise ValueError("Non-finite observations")

        mask = active_mask.to(
            device=observations.device,
            dtype=observations.dtype,
        )

        actions = []
        log_probs = []

        for index, ego_id in enumerate(self.ego_ids):
            actor = self.actors[ego_id]

            # 只使用当前 Ego 自己的 134 维观测
            action, log_prob = actor(
                observations[:, index, :],
                deterministic=deterministic,
            )

            valid = mask[:, index]

            # Waiting / Finished Ego 的输出置零
            actions.append(
                action * valid.unsqueeze(-1)
            )

            log_probs.append(
                log_prob * valid
            )

        return (
            torch.stack(actions, dim=1),
            torch.stack(log_probs, dim=1),
        )
