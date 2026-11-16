
"""每辆 Ego 独立的动作历史及策略输入拼接。"""

import numpy as np


class MultiEgoActionHistory:
    def __init__(self, ego_ids, hist_len=20):
        self.ego_ids = tuple(ego_ids)
        self.hist_len = hist_len
        self.histories = {}
        self.reset()

    def reset(self):
        """每辆车都有自己的动作缓冲区。"""
        self.histories = {
            ego_id: np.zeros(
                (self.hist_len, 2),
                dtype=np.float32,
            )
            for ego_id in self.ego_ids
        }

    def record(self, ego_id, action):
        """记录本次提交的二维动作，最新动作在前。"""
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError("Expected action shape (2,)")

        if not np.isfinite(action).all():
            raise ValueError("Non-finite action")

        history = self.histories[ego_id]

        history[1:] = history[:-1].copy()
        history[0] = action

    def encode(self, ego_id, observation, delay=0):
        """拼接基础观测、掩码动作历史和延迟。"""
        obs = np.asarray(observation, dtype=np.float32)

        if obs.shape != (31, 3):
            raise ValueError("Expected observation (31, 3)")

        if not isinstance(delay, (int, np.integer)):
            raise ValueError("Delay must be an integer")

        if not 0 <= delay <= self.hist_len:
            raise ValueError("Invalid delay")

        if not np.isfinite(obs).all():
            raise ValueError("Non-finite observation")

        history = self.histories[ego_id].copy()

        # 与原 DAROM DelayWrapper 的掩码规则一致：
        # 仅保留最近 delay 个动作
        history[delay:] = 0.0

        result = np.concatenate([
            obs.reshape(-1),
            history.reshape(-1),
            np.array([delay], dtype=np.float32),
        ])

        assert result.shape == (134,)
        return result
