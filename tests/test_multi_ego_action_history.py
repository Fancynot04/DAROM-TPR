
import numpy as np

from src.envs.action_history import MultiEgoActionHistory


def test_independent_action_histories():
    history = MultiEgoActionHistory(
        ("ego_o3", "ego_e74"),
        hist_len=20,
    )

    obs = np.ones((31, 3), dtype=np.float32)

    history.record("ego_o3", [1.0, -2.0])
    history.record("ego_o3", [3.0, -4.0])
    history.record("ego_e74", [-1.0, 2.0])

    # 两辆车的历史必须完全独立
    np.testing.assert_array_equal(
        history.histories["ego_o3"][0],
        [3.0, -4.0],
    )

    np.testing.assert_array_equal(
        history.histories["ego_o3"][1],
        [1.0, -2.0],
    )

    np.testing.assert_array_equal(
        history.histories["ego_e74"][0],
        [-1.0, 2.0],
    )

    # 134 维输入
    encoded = history.encode(
        "ego_o3",
        obs,
        delay=2,
    )

    assert encoded.shape == (134,)
    np.testing.assert_array_equal(
        encoded[:93],
        obs.reshape(-1),
    )

    action_part = encoded[93:133].reshape(20, 2)

    np.testing.assert_array_equal(
        action_part[0],
        [3.0, -4.0],
    )

    np.testing.assert_array_equal(
        action_part[1],
        [1.0, -2.0],
    )

    assert np.all(action_part[2:] == 0.0)
    assert encoded[-1] == 2.0

    # delay=0 时动作历史全部掩码
    no_delay = history.encode(
        "ego_o3",
        obs,
        delay=0,
    )
    assert np.all(no_delay[93:133] == 0.0)

    # reset 必须同时清空两个独立缓冲区
    history.reset()
    assert all(
        np.all(buffer == 0.0)
        for buffer in history.histories.values()
    )
    

def test_age_exceeds_action_history():
    history = MultiEgoActionHistory(
        ("ego_o3", "ego_e74"),
        hist_len=20,
    )

    for i in range(20):
        history.record(
            "ego_o3",
            [0.1 * (i + 1), 0.0],
        )

    obs = np.zeros((31, 3), dtype=np.float32)

    encoded = history.encode(
        "ego_o3",
        obs,
        delay=25,
    )

    assert encoded.shape == (134,)
    assert encoded[-1] == 25

    # 超过 20 步时，仍保留全部可用动作历史
    np.testing.assert_array_equal(
        encoded[93:133].reshape(20, 2),
        history.histories["ego_o3"],
    )