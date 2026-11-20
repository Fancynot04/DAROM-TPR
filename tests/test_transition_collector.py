
import numpy as np
import pytest

from src.agents.replay_buffer import JointReplayBuffer
from src.agents.transition_collector import (
    JointTransitionCollector,
)


def make_transition(
    active=(1, 1),
    acted=(1, 1),
    rewards=(0, 0),
    dones=(0, 0),
    next_active=(1, 1),
):
    data = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in JointReplayBuffer.SHAPES.items()
    }

    data["active_mask"][:] = active
    data["acted_mask"][:] = acted
    data["rewards"][:] = rewards
    data["dones"][:] = dones
    data["next_active_mask"][:] = next_active

    return data


def test_one_step_delayed_commit():
    buffer = JointReplayBuffer(capacity=10)
    collector = JointTransitionCollector(buffer)

    collector.push(
        make_transition(),
        pre_step_terminal=[0, 0],
    )

    # 第一条经验尚未确认，不立即提交
    assert len(buffer) == 0

    collector.push(
        make_transition(),
        pre_step_terminal=[0, 0],
    )

    assert len(buffer) == 1

    collector.flush()
    assert len(buffer) == 2


@pytest.mark.parametrize("terminal_reward", [-400, 0])
def test_pre_step_missed_exit_attribution(terminal_reward):
    buffer = JointReplayBuffer(capacity=10)
    collector = JointTransitionCollector(buffer)

    # t-1：两辆 Ego 都执行了动作，暂时没有奖励
    collector.push(
        make_transition(),
        pre_step_terminal=[0, 0],
    )

    assert len(buffer) == 0

    # t：Ego X 在执行动作之前 Missed Exit
    # Ego M 仍然正常执行了当前动作
    collector.push(
        make_transition(
            active=(1, 1),
            acted=(1, 0),
            rewards=(0, terminal_reward),
            dones=(0, 1),
            next_active=(1, 0),
        ),
        pre_step_terminal=[0, 1],
    )

    # 上一步已被修正并正式存储
    assert len(buffer) == 1

    assert buffer.data["rewards"][0, 1] == terminal_reward
    assert buffer.data["dones"][0, 1] == 1
    assert buffer.data["next_active_mask"][0, 1] == 0

    # Ego M 在上一步不应受到影响
    assert buffer.data["rewards"][0, 0] == 0
    assert buffer.data["dones"][0, 0] == 0
    assert buffer.data["next_active_mask"][0, 0] == 1

    # 当前步 Ego M 的经验仍然要保留
    collector.flush()

    assert len(buffer) == 2

    # Ego X 的失败只计入一次
    assert buffer.data["rewards"][:2, 1].sum() == terminal_reward

    # 当前步 X 没有执行动作，不参与训练
    assert buffer.data["active_mask"][1, 1] == 0
    assert buffer.data["acted_mask"][1, 1] == 0
    assert buffer.data["rewards"][1, 1] == 0

    # Ego M 正常保留当前步
    assert buffer.data["acted_mask"][1, 0] == 1


def test_pre_step_without_previous_action_rejected():
    buffer = JointReplayBuffer(capacity=10)
    collector = JointTransitionCollector(buffer)

    with pytest.raises(RuntimeError):
        collector.push(
            make_transition(
                acted=(1, 0),
                rewards=(0, -100),
                dones=(0, 1),
                next_active=(1, 0),
            ),
            pre_step_terminal=[0, 1],
        )

    assert len(buffer) == 0


def test_team_replay_keeps_idle_step_between_arrivals():
    buffer = JointReplayBuffer(capacity=4, team=True)
    collector = JointTransitionCollector(buffer)
    for active, next_active, success, next_success, reward, done in [
        ((1, 0), (0, 0), (0, 0), (1, 0), 0, 0),
        ((0, 0), (0, 1), (1, 0), (1, 0), 0, 0),
        ((0, 1), (0, 0), (1, 0), (1, 1), 100, 1),
    ]:
        row = make_transition(active=active, acted=active, next_active=next_active)
        row.update(team_active=1, team_done=done, team_reward=reward,
                   success_mask=success, next_success_mask=next_success)
        collector.push(row, pre_step_terminal=[0, 0])
    collector.flush()
    assert len(buffer) == 3
    np.testing.assert_array_equal(buffer.data["team_reward"][:3], [0, 0, 100])
    np.testing.assert_array_equal(buffer.data["team_done"][:3], [0, 0, 1])
    np.testing.assert_array_equal(buffer.data["acted_mask"][1], [0, 0])


def test_team_replay_closes_on_pre_step_failure():
    buffer = JointReplayBuffer(capacity=4, team=True)
    collector = JointTransitionCollector(buffer)
    team_fields = dict(team_active=1, team_done=0, team_reward=0,
                       success_mask=(0, 0), next_success_mask=(0, 0))
    collector.push({**make_transition(), **team_fields}, pre_step_terminal=[0, 0])
    row = make_transition(acted=(1, 0), rewards=(0, -400), dones=(0, 1), next_active=(1, 0))
    collector.push({**row, **team_fields, "team_done": 1}, pre_step_terminal=[0, 1])
    collector.flush()
    np.testing.assert_array_equal(buffer.data["team_active"][:2], [1, 0])
    np.testing.assert_array_equal(buffer.data["team_done"][:2], [1, 1])
