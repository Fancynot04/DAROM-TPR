
"""验证双 Ego 延迟观测与 DAROM GRU 编码器兼容。"""

import numpy as np
import torch

from gymnasium import spaces

from src.delay_encoder import DelayAwareGRUEncoder
from src.envs.multi_ego_env import MultiEgoEnv


def test_multi_ego_gru_forward():
    """真实 SUMO 观测 -> DAROM GRU -> 双 Ego 特征。"""

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    try:
        env.reset(seed=42)

        # 推进 30 步，确保两个 Ego 都已出发
        for _ in range(30):
            actions = {
                ego_id: [0.0, 0.0]
                for ego_id in env.AGENTS
                if (
                    env.egos[ego_id].state.status == "active"
                    and not env.terminations[ego_id]
                )
            }

            env.step(actions)

        assert all(
            env.egos[ego_id].state.status == "active"
            for ego_id in env.AGENTS
        )

        # 获取两辆车各自的 134 维策略输入
        policy_obs = np.stack([
            env.policy_observations[ego_id]
            for ego_id in env.AGENTS
        ])

        assert policy_obs.shape == (2, 134)
        assert np.isfinite(policy_obs).all()

        # 使用 DAROM 原有的编码器
        observation_space = spaces.Box(
            low=-5000.0,
            high=5000.0,
            shape=(134,),
            dtype=np.float32,
        )

        encoder = DelayAwareGRUEncoder(
            observation_space=observation_space,
            act_dim=2,
            hist_len=20,
            delay_dim=1,
            features_dim=256,
            gru_hidden=64,
            input_mode="antwerp_scaled_reversed",
            nav_dim=0,
        )

        encoder.eval()

        # 两辆车作为 batch 输入
        tensor = torch.as_tensor(
            policy_obs,
            dtype=torch.float32,
        )

        with torch.no_grad():
            features = encoder(tensor)

        assert features.shape == (2, 256)
        assert torch.isfinite(features).all()

        # 两辆车的输入与编码特征应当不同
        assert not np.array_equal(
            policy_obs[0],
            policy_obs[1],
        )

        assert not torch.allclose(
            features[0],
            features[1],
        )

        print("\n===== DAROM GRU integration =====")
        print("Policy observation:", policy_obs.shape)
        print("Encoded features:", tuple(features.shape))
        print("GRU forward PASS")

    finally:
        env.close()
