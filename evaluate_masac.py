
"""独立评估双 Ego CTDE-MASAC 策略。"""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
from src.agents.masac import MultiAgentGaussianActor
from train_masac import snapshot, AGENT_IDS
from src.evaluation_metrics import AgentMetrics, aggregate, joint_record, REFERENCE_COMMIT, wilson
from src.experiment_tracking import sha256, git_metadata, evaluation_fingerprint, runtime_versions
from src.evaluation_results import write_results, EVALUATION_PROTOCOL_VERSION
from sumolib import checkBinary


@torch.inference_mode()
def evaluate(args):
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
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
        collect_metrics=True,
    )

    records = []
    sumo_version = None

    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            observations, _ = env.reset(seed=seed)
            sumo_version = list(env.world.conn.getVersion())

            outcomes = {}
            metrics = {ego_id: AgentMetrics() for ego_id in AGENT_IDS}
            agent_records = {}
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
                if getattr(args, "trace", False) and episode == 0 and step % 100 == 0:
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
                    if ego_id not in outcomes:
                        metrics[ego_id].observe(
                            infos[ego_id]["evaluation_metrics"],
                            terminations[ego_id] or truncations[ego_id],
                        )
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
                        reason = infos[ego_id].get("terminal_reason")
                        agent_records[ego_id] = metrics[ego_id].finish(reason, rewards_total[ego_id])
                        outcomes[ego_id] = reason

                if len(outcomes) == len(AGENT_IDS):
                    break

            else:
                details = {}

                for ego_id in AGENT_IDS:
                    state = env.egos[ego_id].state

                    details[ego_id] = {
                        "status": state.status,
                        "depart_time": state.depart_time,
                        "active_steps": env.active_steps[ego_id],
                        "road_id": state.road_id,
                        "terminated": env.terminations[ego_id],
                        "terminal_reason": env.terminal_reasons[ego_id],
                    }

                raise RuntimeError(
                    f"Evaluation seed {seed}: incomplete episode\n"
                    f"SUMO time: {env.world.conn.simulation.getTime():.1f}\n"
                    f"Recorded outcomes: {outcomes}\n"
                    f"Agent states: {details}"
                )

            both_success = all(
                agent_records[ego_id]["success"]
                for ego_id in AGENT_IDS
            )

            record = {
                "seed": seed,
                "global_steps": step + 1,
                "agent_steps_sum": sum(r["agent_steps"] for r in agent_records.values()),
                "per_agent": agent_records,
                "joint": joint_record(agent_records),
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

    per_agent = {ego_id: aggregate([r["per_agent"][ego_id] for r in records])
                 for ego_id in AGENT_IDS}
    summary = {
        "metric_version": "homogeneous-dual-ego-v1",
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_sha256": sha256(args.checkpoint),
        "evaluation_code_sha256": evaluation_fingerprint(),
        "runtime_versions": runtime_versions(),
        "git": git_metadata(),
        "training_seed": checkpoint.get("config", {}).get("seed"),
        "training_config": checkpoint.get("config", {}),
        "checkpoint_git": checkpoint.get("git"),
        "checkpoint_source_sha256": checkpoint.get("source_sha256"),
        "trainer_config": checkpoint.get("trainer_config"),
        "sumo_version": sumo_version,
        "environment": {
            "sumo_binary_sha256": sha256(checkBinary("sumo")),
            "scenario": "easy", "agent_ids": list(AGENT_IDS), "requested_speed_mode": 96,
            "actual_speed_modes": {key: sorted({mode for record in records for mode in record["per_agent"][key]["speed_modes"]}) for key in AGENT_IDS},
            "step_length_s": 0.1, "max_steps": args.max_steps,
            "net_sha256": sha256(Path(__file__).parent / "sumo_new/Antwerp.net.xml"),
            "route_sha256": sha256(Path(__file__).parent / "sumo_new/Antwerp_easy_o3_d3_homogeneous_dual.rou.xml"),
            "stopped_speed_threshold_mps": 0.5,
        },
        "reference_commit": REFERENCE_COMMIT,
        "rate_unit": "fraction for legacy success_rates/joint_success_rate; percent in per_agent/joint",
        "stochastic": args.stochastic,
        "protocol": {
            "speed": "valid pre-step TraCI speed; episode means equally weighted",
            "jerk": "absolute measured acceleration difference / simulation dt; exclude first and terminal transitions",
            "hard_braking": "valid post-step acceleration <= -4.5 m/s^2, including terminal",
            "success": "Arrived and no starting/ending teleport event",
            "joint_continuous": "equal agents within episode, then equal valid episodes",
            "global_steps": "env.step calls, excludes reset bootstrap simulation steps",
            "agent_steps_sum": "sum of environment active_steps (actions applied); excludes pre-step Missed Exit",
            "missing": "JSON null / CSV empty; separate speed, jerk, braking and event sample counts",
        },
        "seed_start": args.seed_start,
        "actor_sampling_order": list(AGENT_IDS),
        "global_steps": sum(r["global_steps"] for r in records),
        "agent_steps_sum": sum(r["agent_steps_sum"] for r in records),
        "per_agent": per_agent,
        "joint": aggregate([r["joint"] for r in records]),
        "checkpoint": str(args.checkpoint),
        "training_step": checkpoint["global_step"],
        "episodes": len(records),
        "success_rates": {},
        "success_rates_stats": {key: wilson(sum(r["per_agent"][key]["success"] for r in records), len(records), scale=1) for key in AGENT_IDS},
        "joint_success_rate_stats": wilson(sum(r["both_success"] for r in records), len(records), scale=1),
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
            per_agent[ego_id]["success_rate"] / 100
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    write_results(output, summary)

    print("\n===== MASAC Evaluation =====")
    print("Training step:", checkpoint["global_step"])
    print("Episodes:", len(records))
    print("Success rates:", summary["success_rates"])
    print("Joint success:", summary["joint_success_rate"])
    print("Outcome counts:", summary["outcome_counts"])
    print("Report:", output)
    return summary


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
        default="runs/masac_eval/summary.json",
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic Actor actions for diagnosis",
    )

    parser.add_argument("--trace", action="store_true", help="Print first episode trace")
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    evaluate(args)
