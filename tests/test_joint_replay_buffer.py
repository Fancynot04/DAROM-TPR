
import numpy as np
import pytest

from src.agents.replay_buffer import JointReplayBuffer


def make_transition():
    return {
        "policy_obs": np.ones((2, 134), dtype=np.float32),
        "critic_obs": np.ones((2, 93), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "rewards": np.array([50.0, 0.0]),
        "dones": np.array([1.0, 0.0]),
        "active_mask": np.array([1.0, 1.0]),
        "acted_mask": np.array([1.0, 1.0]),
        "next_policy_obs": np.ones(
            (2, 134), dtype=np.float32
        ),
        "next_critic_obs": np.ones(
            (2, 93), dtype=np.float32
        ),
        "next_active_mask": np.array([0.0, 1.0]),
    }


def test_joint_replay_shapes():
    buffer = JointReplayBuffer(capacity=10)

    buffer.add(**make_transition())

    assert len(buffer) == 1

    batch = buffer.sample(1)

    assert batch["policy_obs"].shape == (1, 2, 134)
    assert batch["critic_obs"].shape == (1, 2, 93)
    assert batch["actions"].shape == (1, 2, 2)
    assert batch["rewards"].shape == (1, 2)

    assert batch["rewards"][0, 0].item() == 50
    assert batch["rewards"][0, 1].item() == 0

    assert batch["next_active_mask"][0, 0].item() == 0
    assert batch["next_active_mask"][0, 1].item() == 1


def test_buffer_copies_input():
    buffer = JointReplayBuffer(capacity=10)

    transition = make_transition()
    buffer.add(**transition)

    transition["policy_obs"][:] = 999

    batch = buffer.sample(1)

    assert np.all(
        batch["policy_obs"].numpy() == 1
    )


def test_buffer_capacity():
    buffer = JointReplayBuffer(capacity=3)

    for _ in range(5):
        buffer.add(**make_transition())

    assert len(buffer) == 3
    assert buffer.position == 2


def test_invalid_pre_step_terminal():
    buffer = JointReplayBuffer(capacity=10)

    transition = make_transition()

    transition["acted_mask"][0] = 0

    with pytest.raises(ValueError):
        buffer.add(**transition)

    assert len(buffer) == 0


def test_replay_to_critic():
    from src.agents.centralized_critic import (
        CentralizedTwinQ
    )

    buffer = JointReplayBuffer(capacity=10)
    buffer.add(**make_transition())

    batch = buffer.sample(1)

    critic = CentralizedTwinQ()

    q1, q2 = critic(
        batch["critic_obs"],
        batch["actions"],
        batch["active_mask"],
    )

    assert q1.shape == (1, 2)
    assert q2.shape == (1, 2)

    assert q1.isfinite().all()
    assert q2.isfinite().all()