
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
        """生成 DAROM 134 维延迟策略输入。"""

        if ego_id not in self.histories:
            raise KeyError(ego_id)

        obs = np.asarray(
            observation, dtype=np.float32
        )

        if obs.shape != (31, 3):
            raise ValueError("Expected observation (31, 3)")

        if not np.isfinite(obs).all():
            raise ValueError("Non-finite observation")

        # 实际观测年龄允许超过动作历史长度
        if (
            isinstance(delay, (bool, np.bool_))
            or not isinstance(delay, (int, np.integer))
            or delay < 0
        ):
            raise ValueError("Invalid observation age")

        age = int(delay)

        # 新动作在前，保留最近 min(age, 20) 个动作
        history = self.histories[ego_id].copy()

        retained = min(age, self.hist_len)
        history[retained:] = 0.0

        encoded = np.concatenate([
            obs.reshape(-1),
            history.reshape(-1),
            np.array([age], dtype=np.float32),
        ])

        assert encoded.shape == (134,)

        return encoded
