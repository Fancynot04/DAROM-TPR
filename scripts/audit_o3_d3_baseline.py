
"""O3 -> D3 固定动作基线及合流区交通检查。"""

from src.envs.single_ego_env import SingleEgoEnv


def run_episode(seed):
    env = SingleEgoEnv(scenario="easy")

    near_merge_samples = 0

    try:
        _, reset_info = env.reset(seed=seed)
        conn = env.world.conn

        for step in range(4000):
            # 每隔 1 秒检查一次 J104 前的合流交通
            if step % 10 == 0:
                ego = "ego"
                vehicle = conn.vehicle

                if (
                    ego in vehicle.getIDList()
                    and vehicle.getRoadID(ego) == "E75"
                    and vehicle.getLaneIndex(ego) == 0
                ):
                    lane_id = vehicle.getLaneID(ego)
                    pos = vehicle.getLanePosition(ego)

                    remaining = (
                        conn.lane.getLength(lane_id) - pos
                    )

                    # 只观察 J104 前最后 100 米
                    if remaining <= 100.0:
                        # E75_1 是一条汇入 E76_0 的主线
                        for other in vehicle.getIDList():
                            if other == ego:
                                continue

                            if (
                                vehicle.getRoadID(other) == "E75"
                                and vehicle.getLaneIndex(other) == 1
                                and abs(
                                    vehicle.getLanePosition(other) - pos
                                ) <= 30.0
                            ):
                                near_merge_samples += 1
                                break

            # 固定动作：零加速度、零换道
            _, reward, terminated, truncated, info = (
                env.step([0.0, 0.0])
            )

            if terminated or truncated:
                return {
                    "seed": seed,
                    "sumo_seed": reset_info["sumo_seed"],
                    "outcome": info["terminal_reason"],
                    "steps": info["step_count"],
                    "reward": reward,
                    "near_merge_samples": near_merge_samples,
                }

        raise RuntimeError("Episode exceeded step budget")

    finally:
        env.close()


def main():
    results = []

    for seed in range(42, 52):
        result = run_episode(seed)
        results.append(result)

        print(
            f"seed={seed} "
            f"outcome={result['outcome']} "
            f"steps={result['steps']} "
            f"near_merge={result['near_merge_samples']}"
        )

    arrived = sum(
        r["outcome"] == "Arrived" for r in results
    )

    print("\n===== Fixed-action baseline =====")
    print(f"Success: {arrived}/{len(results)}")
    print(f"Success rate: {arrived / len(results):.1%}")
    print("Action: [0.0, 0.0]")


if __name__ == "__main__":
    main()
