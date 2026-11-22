
import torch

from src.agents.centralized_critic import CentralizedTwinQ


def test_critic_output_shape():
    torch.manual_seed(42)

    critic = CentralizedTwinQ()

    obs = torch.randn(4, 2, 93)
    actions = torch.randn(4, 2, 2)
    mask = torch.ones(4, 2)

    q1, q2 = critic(obs, actions, mask)

    assert q1.shape == (4, 2)
    assert q2.shape == (4, 2)
    assert torch.isfinite(q1).all()
    assert torch.isfinite(q2).all()

    assert not torch.allclose(q1, q2)


def test_self_first_critic_exchanges_values_with_agent_slots():
    critic = CentralizedTwinQ(nav_dim=2, self_first=True)
    obs = torch.randn(3, 2, 95)
    actions = torch.randn(3, 2, 2)
    mask = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])
    q = critic(obs, actions, mask)
    swapped = critic(obs.flip((1,)), actions.flip((1,)), mask.flip((1,)))
    for values, swapped_values in zip(q, swapped):
        torch.testing.assert_close(values, swapped_values.flip((1,)))
        assert (values[mask == 0] == 0).all()


def test_inactive_agent_mask():
    torch.manual_seed(42)

    critic = CentralizedTwinQ()

    obs = torch.randn(1, 2, 93)
    actions = torch.randn(1, 2, 2)

    mask = torch.tensor([[1.0, 0.0]])

    q1_before, q2_before = critic(
        obs, actions, mask
    )

    changed_obs = obs.clone()
    changed_actions = actions.clone()

    changed_obs[:, 1, :] = 999.0
    changed_actions[:, 1, :] = -999.0

    q1_after, q2_after = critic(
        changed_obs, changed_actions, mask
    )

    # Ego X 已结束，其输入不能影响 Ego M
    torch.testing.assert_close(q1_before, q1_after)
    torch.testing.assert_close(q2_before, q2_after)

    assert q1_before[0, 1].item() == 0.0
    assert q2_before[0, 1].item() == 0.0


def test_critic_backward():
    torch.manual_seed(42)

    critic = CentralizedTwinQ()

    obs = torch.randn(
        4, 2, 93,
        requires_grad=True,
    )

    actions = torch.randn(
        4, 2, 2,
        requires_grad=True,
    )

    mask = torch.ones(4, 2)

    q1, q2 = critic(obs, actions, mask)

    loss = q1.square().mean() + q2.square().mean()
    loss.backward()

    assert any(
        p.grad is not None
        and p.grad.abs().sum() > 0
        for p in critic.parameters()
    )

    assert obs.grad is not None
    assert actions.grad is not None
    assert torch.isfinite(obs.grad).all()
    assert torch.isfinite(actions.grad).all()


def test_critic_input_contract():
    critic = CentralizedTwinQ()

    # 93 * 2 + 2 * 2 + 2 = 192
    assert critic.q1[0].in_features == 192
    assert critic.q2[0].in_features == 192

    # 缩放来自 DAROM 原有观测约定
    expected = torch.tensor(
        [1000.0, 5.0, 30.0]
        + [50.0, 20.0, 30.0] * 30
    )

    torch.testing.assert_close(
        critic.observation_scale,
        expected,
    )
