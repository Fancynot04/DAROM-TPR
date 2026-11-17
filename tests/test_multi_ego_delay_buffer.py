
import numpy as np

from src.envs.delay_buffer import MultiEgoDelayBuffer


def make_obs(position, neighbor):
    """构造可辨认的 Ego 与周围车状态。"""

    obs = np.zeros((31, 3), dtype=np.float32)

    obs[0] = [position, 0.0, 10.0]
    obs[1] = [neighbor, 0.0, 0.0]

    return obs


def test_delayed_packet_arrival():
    buffer = MultiEgoDelayBuffer(
        ("ego_o3", "ego_e74"),
        max_delay=20,
        seed=42,
    )

    initial = make_obs(100.0, 10.0)

    buffer.activate("ego_o3", initial)

    # 第 1 步：生成延迟 2 步的数据包
    r1 = buffer.update(
        "ego_o3",
        make_obs(101.0, 11.0),
        delay=2,
    )

    # 尚未收到新包，沿用上次邻车观测
    assert r1.age == 1
    assert not r1.received_packet
    assert r1.observation[0, 0] == 101.0
    assert r1.observation[1, 0] == 10.0

    # 第 2 步：仍未收到第一个数据包
    r2 = buffer.update(
        "ego_o3",
        make_obs(102.0, 22.0),
        delay=2,
    )

    assert r2.age == 2
    assert r2.observation[1, 0] == 10.0

    # 第 3 步：第 1 步的数据包到达
    r3 = buffer.update(
        "ego_o3",
        make_obs(103.0, 33.0),
        delay=2,
    )

    assert r3.received_packet
    assert r3.age == 2

    # 自己的位置是当前的
    assert r3.observation[0, 0] == 103.0

    # 邻车状态来自第 1 步
    assert r3.observation[1, 0] == 11.0


def test_old_packet_is_rejected():
    buffer = MultiEgoDelayBuffer(
        ("ego_o3",),
        max_delay=20,
        seed=42,
    )

    buffer.activate(
        "ego_o3",
        make_obs(100.0, 10.0),
    )

    buffer.update(
        "ego_o3",
        make_obs(101.0, 11.0),
        delay=0,
    )

    buffer.update(
        "ego_o3",
        make_obs(102.0, 22.0),
        delay=2,
    )

    # 更新的数据包立即到达
    r3 = buffer.update(
        "ego_o3",
        make_obs(103.0, 33.0),
        delay=0,
    )

    assert r3.age == 0
    assert r3.observation[1, 0] == 33.0

    # 之前的旧包到达，但不能覆盖更新的数据
    r4 = buffer.update(
        "ego_o3",
        make_obs(104.0, 44.0),
        delay=2,
    )

    assert not r4.received_packet
    assert r4.age == 1
    assert r4.observation[0, 0] == 104.0
    assert r4.observation[1, 0] == 33.0


def test_two_ego_independent():
    buffer = MultiEgoDelayBuffer(
        ("ego_o3", "ego_e74"),
        max_delay=20,
        seed=42,
    )

    buffer.activate(
        "ego_o3",
        make_obs(100.0, 10.0),
    )

    # 第二辆车可以晚出发
    buffer.update(
        "ego_o3",
        make_obs(101.0, 11.0),
        delay=2,
    )

    buffer.activate(
        "ego_e74",
        make_obs(200.0, 50.0),
    )

    a = buffer.update(
        "ego_o3",
        make_obs(102.0, 12.0),
        delay=2,
    )

    b = buffer.update(
        "ego_e74",
        make_obs(201.0, 51.0),
        delay=0,
    )

    assert a.age == 2
    assert a.observation[1, 0] == 10.0

    assert b.age == 0
    assert b.observation[1, 0] == 51.0

    # 重置后两辆车均不再处于 Active 状态
    buffer.reset(seed=42)
    assert not buffer.active
    assert all(
        all(packet is None for packet in queue)
        for queue in buffer.queues.values()
    )


def test_observation_age_can_exceed_max_delay():
    buffer = MultiEgoDelayBuffer(
        ("ego_o3",),
        max_delay=20,
        seed=42,
    )

    obs = make_obs(100.0, 10.0)
    buffer.activate("ego_o3", obs)

    # 模拟已经连续 20 步没有接受更新的数据包
    buffer.ages["ego_o3"] = 20

    result = buffer.update(
        "ego_o3",
        make_obs(101.0, 11.0),
        delay=20,
    )

    assert result.age == 21
    assert result.sampled_delay == 20
    assert not result.received_packet

    # Ego 自身状态仍然实时更新
    assert result.observation[0, 0] == 101.0