
"""真实 SUMO 中的 O3 -> D3 单 Ego 完整回合测试。"""

import numpy as np

from src.envs.single_ego_env import SingleEgoEnv


def test_complete_episode():
    env = SingleEgoEnv(scenario="easy", gui=False)

    try:
        obs, reset_info = env.reset(seed=42)

        print("\nInitial route:", reset_info["route"])

        previous_time = reset_info["simulation_time"]
        last_road = None
        final_info = None
        final_reward = None

        for _ in range(4000):
            # 固定动作：不加速、不主动换道
            action = np.array([0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            assert obs.shape == (31, 3)
            assert np.isfinite(obs).all()

            road = info["road_id"]

            if road != last_road:
                print(
                    f"{info['simulation_time']:.1f}s "
                    f"road={road} "
                    f"lane={info['lane_id']} "
                    f"speed={info['speed']}"
                )
                last_road = road

            # 普通仿真步必须推进一次 0.1s
            if info["sumo_advanced"]:
                assert np.isclose(
                    info["simulation_time"] - previous_time,
                    0.1,
                    atol=1e-6,
                )
            else:
                assert info["pre_step_terminal"]
                assert np.isclose(
                    info["simulation_time"],
                    previous_time,
                    atol=1e-6,
                )

            previous_time = info["simulation_time"]

            if terminated or truncated:
                final_info = info
                final_reward = reward
                break

        assert final_info is not None, "Episode did not terminate"

        allowed_events = {
            "Arrived",
            "Collision",
            "Ego Teleported",
            "Missed Exit",
            "Timeout",
        }

        assert final_info["terminal_reason"] in allowed_events

        if final_info["terminal_reason"] == "Arrived":
            assert final_reward == 50.0
            assert final_info["reached_d3"]
        else:
            assert final_reward == -100.0

        print("\n===== Episode summary =====")
        print("SUMO version:", reset_info["sumo_version"])
        print("Steps:", final_info["step_count"])
        print("Final time:", final_info["simulation_time"])
        print("Outcome:", final_info["terminal_reason"])
        print("Reward:", final_reward)

    finally:
        env.close()
