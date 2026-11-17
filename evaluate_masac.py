
"""独立评估双 Ego CTDE-MASAC 策略。"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
from src.agents.masac import MultiAgentGaussianActor
from train_masac import snapshot, AGENT_IDS


@torch.inference_mode()
def evaluate(args):
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if tuple(checkpoint["agent_ids"]) != AGENT_IDS:
        raise ValueError("Checkpoint agent IDs mismatch")

    actors = MultiAgentGaussianActor(
        ego_ids=AGENT_IDS
    )
    actors.load_state_dict(
        checkpoint["actors"],
        strict=True,
    )
    actors.eval()

    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
    )

    records = []

    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            observations, _ = env.reset(seed=seed)

            outcomes = {}
            rewards_total = {
                ego_id: 0.0 for ego_id in AGENT_IDS
            }

            for step in range(args.max_steps):
                policy_obs, critic_obs, active_mask = snapshot(
                    env, observations
                )

                actions_tensor, _ = actors(
                    torch.from_numpy(
                        policy_obs[None]
                    ),
                    torch.from_numpy(
                        active_mask[None]
                    ),
                    deterministic=not args.stochastic,
                )

                joint_actions = (
                    actions_tensor[0].cpu().numpy()
                )

                actions = {
                    ego_id: joint_actions[i].tolist()
                    for i, ego_id in enumerate(AGENT_IDS)
                    if active_mask[i] == 1
                }
                
                
                # 仅在第一个评估回合采样打印，避免日志过多
                if episode == 0 and step % 100 == 0:
                    for i, ego_id in enumerate(AGENT_IDS):
                        if active_mask[i] == 0:
                            continue

                        # critic_obs 的前三维为：
                        # 累计行驶距离、车道索引、速度
                        distance = float(critic_obs[i, 0])
                        lane = int(critic_obs[i, 1])
                        speed = float(critic_obs[i, 2])

                        acc = float(joint_actions[i, 0])
                        lateral = float(joint_actions[i, 1])

                        print(
                            f"TRACE step={step} "
                            f"ego={ego_id} "
                            f"distance={distance:.1f} "
                            f"lane={lane} "
                            f"speed={speed:.2f} "
                            f"acc={acc:+.2f} "
                            f"lateral={lateral:+.2f}",
                            flush=True,
                        )


                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = env.step(actions)

                for ego_id in AGENT_IDS:
                    rewards_total[ego_id] += float(
                        rewards[ego_id]
                    )

                    if (
                        ego_id not in outcomes
                        and (
                            terminations[ego_id]
                            or truncations[ego_id]
                        )
                    ):
                        outcomes[ego_id] = infos[
                            ego_id
                        ].get("terminal_reason", "Unknown")

                if len(outcomes) == len(AGENT_IDS):
                    break

            else:
                raise RuntimeError(
                    f"Evaluation seed {seed}: "
                    "incomplete episode"
                )

            both_success = all(
                outcomes[ego_id] == "Arrived"
                for ego_id in AGENT_IDS
            )

            record = {
                "seed": seed,
                "steps": step + 1,
                "outcomes": outcomes,
                "rewards": rewards_total,
                "both_success": both_success,
            }
            records.append(record)

            print(
                f"seed={seed} "
                f"outcomes={outcomes} "
                f"both_success={both_success}",
                flush=True,
            )

    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "training_step": checkpoint["global_step"],
        "episodes": len(records),
        "success_rates": {},
        "outcome_counts": {},
        "joint_success_rate": float(
            np.mean([
                r["both_success"] for r in records
            ])
        ),
        "records": records,
    }

    for ego_id in AGENT_IDS:
        reasons = Counter(
            r["outcomes"][ego_id]
            for r in records
        )

        summary["outcome_counts"][ego_id] = dict(reasons)
        summary["success_rates"][ego_id] = (
            reasons["Arrived"] / len(records)
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== MASAC Evaluation =====")
    print("Training step:", checkpoint["global_step"])
    print("Episodes:", len(records))
    print("Success rates:", summary["success_rates"])
    print("Joint success:", summary["joint_success_rate"])
    print("Outcome counts:", summary["outcome_counts"])
    print("Report:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", required=True
    )
    parser.add_argument(
        "--episodes", type=int, default=5
    )
    parser.add_argument(
        "--seed-start", type=int, default=200001
    )
    parser.add_argument(
        "--max-steps", type=int, default=4500
    )
    parser.add_argument(
        "--output",
        default="runs/masac_eval/report.json",
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic Actor actions for diagnosis",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    evaluate(args)
