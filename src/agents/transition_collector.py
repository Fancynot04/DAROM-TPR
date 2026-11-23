
"""双 Ego 联合 Transition 采集器。

延迟一个全局步提交，以处理 pre-step Missed Exit。
"""

import numpy as np


class JointTransitionCollector:
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer
        self.pending = None

    def _copy_transition(self, transition):
        """检查并复制经验，避免引用被外部修改。"""

        shapes = self.buffer.shapes

        if set(transition) != set(shapes):
            raise ValueError("Transition fields mismatch")

        result = {}

        for name, shape in shapes.items():
            value = np.asarray(
                transition[name],
                dtype=np.float32,
            )

            if value.shape != shape:
                raise ValueError(
                    f"{name}: expected {shape}, got {value.shape}"
                )

            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {name}")

            result[name] = value.copy()

        for name in self.buffer.mask_fields:
            value = result[name]
            if not np.all((value == 0) | (value == 1)):
                raise ValueError(f"Invalid {name}")

        if np.any(
            result["acted_mask"] > result["active_mask"]
        ):
            raise ValueError("Acted mask exceeds active mask")

        return result

    def push(self, transition, pre_step_terminal):
        """处理本步经验，提交上一步，暂存本步。"""

        current = self._copy_transition(transition)

        pre = np.asarray(pre_step_terminal)

        if (
            pre.shape != (len(self.buffer.AGENT_IDS),)
            or not np.all((pre == 0) | (pre == 1))
        ):
            raise ValueError("Invalid pre-step mask")

        pre = pre.astype(bool)

        # 本步没有执行动作且没有发生 pre-step 事件，
        # 却出现非零奖励，属于错误的训练数据。
        invalid_reward = (
            (current["acted_mask"] == 0)
            & (current["rewards"] != 0)
            & ~pre
        )

        if np.any(invalid_reward):
            raise ValueError(
                "Reward without an executed action"
            )

        previous = (
            None if self.pending is None
            else {
                k: v.copy()
                for k, v in self.pending.items()
            }
        )

        for i in np.flatnonzero(pre):
            if previous is None:
                raise RuntimeError(
                    "Pre-step terminal has no previous transition"
                )

            # 当前步：Ego 原本活跃，但未执行动作就终止
            if not (
                current["active_mask"][i] == 1
                and current["acted_mask"][i] == 0
                and current["dones"][i] == 1
                and current["next_active_mask"][i] == 0
            ):
                raise ValueError(
                    "Invalid pre-step terminal transition"
                )

            # 上一步必须确实执行过动作，
            # 并且当时尚未结束。
            if not (
                previous["acted_mask"][i] == 1
                and previous["dones"][i] == 0
                and previous["next_active_mask"][i] == 1
            ):
                raise RuntimeError(
                    "Cannot attribute terminal to previous action"
                )

            # 修正上一步经验：
            # 上一个动作导致的下一状态已经不可挽救。
            previous["rewards"][i] += (
                previous.get("bootstrap_discount", 1.0) * current["rewards"][i]
            )

            previous["dones"][i] = 1
            previous["next_active_mask"][i] = 0

            # 当前步未执行动作，不应重复存储终止奖励。
            current["rewards"][i] = 0
            current["dones"][i] = 0
            current["active_mask"][i] = 0
            current["acted_mask"][i] = 0
            current["next_active_mask"][i] = 0
            current["actions"][i] = 0

        if np.any(pre) and "team_active" in current:
            # 步前的 Missed Exit 已使全员成功不可能，归入上一步的团队结束。
            previous["team_done"][...] = 1
            current["team_active"][...] = 0

        # 现在可以安全提交上一步经验。
        if previous is not None:
            self.buffer.add(**previous)

        # 团队任务仍可成功时保留等待步，团队折扣按 SUMO 共享时钟推进。
        self.pending = (
            current
            if np.any(current["acted_mask"] == 1) or current.get("team_active", 0)
            else None
        )

    def flush(self):
        """完整回合结束时，提交最后一条经验。"""

        if self.pending is not None:
            self.buffer.add(**self.pending)
            self.pending = None
