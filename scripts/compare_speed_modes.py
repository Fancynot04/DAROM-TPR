
from src.envs.single_ego_env import SingleEgoEnv


for mode in (96, 31):
    print(f"\n===== Speed Mode {mode} =====")

    for seed in range(42, 47):
        env = SingleEgoEnv(
            scenario="easy",
            speed_mode=mode,
        )

        try:
            _, reset_info = env.reset(seed=seed)

            speeds = []

            for _ in range(4000):
                _, _, terminated, truncated, info = (
                    env.step([0.0, 0.0])
                )

                if info["speed"] is not None:
                    speeds.append(info["speed"])

                if terminated or truncated:
                    print(
                        f"seed={seed} "
                        f"requested={mode} "
                        f"actual={reset_info['speed_mode_actual']} "
                        f"outcome={info['terminal_reason']} "
                        f"steps={info['step_count']} "
                        f"min_speed={min(speeds):.2f} "
                        f"max_speed={max(speeds):.2f}"
                    )
                    break
            else:
                raise RuntimeError("Episode did not terminate")

        finally:
            env.close()
