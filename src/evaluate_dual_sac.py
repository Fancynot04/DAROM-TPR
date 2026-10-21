"""Deterministic paired-seed evaluation of a dual SAC checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dual_merging import AGENT_TASKS, DualMerging
from .multi_agent_sac import AGENTS, TwoAgentSAC


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scenario", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=700_001)
    parser.add_argument("--safety", action="store_true",
                        help="Enable the environment safety controller during evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed_start < 0 or args.torch_threads < 1:
        parser.error("episodes and threads must be positive; seed-start nonnegative")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    return args


def evaluate(learner, scenario, episodes, seed_start, safety=False):
    env = DualMerging(scenario=scenario, safety=safety, delay_mode="uniform")
    records = []
    try:
        for episode in range(episodes):
            seed = seed_start + episode
            observations, reset_info = env.reset(seed=seed)
            returns = dict.fromkeys(AGENTS, 0.0)
            outcomes = {key: None for key in AGENTS}
            scene_steps = 0
            while not env.finished:
                scene_steps += 1
                if scene_steps > 18_000:
                    raise RuntimeError("Evaluation scene-step guard exceeded")
                active = env.active_agents
                actions = {}
                for i, key in enumerate(AGENTS):
                    if key not in active:
                        continue
                    with torch.no_grad():
                        state = torch.as_tensor(observations[key][None], device=learner.device)
                        value = learner.actors[i].sample(state, deterministic=True)[0][0]
                    actions[key] = (value.cpu().numpy() * 5).tolist()
                observations, rewards, terminated, truncated, infos = env.step(actions)
                for key in active:
                    returns[key] += rewards[key]
                    if terminated[key] or truncated[key]:
                        outcomes[key] = {field: infos[key][field] for field in (
                            "message", "is_success", "ego_teleported", "stage_reached",
                            "stage_reward_paid", "control_steps")}
            for key in AGENTS:
                if outcomes[key] is None:
                    outcomes[key] = {"message": "Departure Deadline", "is_success": False,
                                     "ego_teleported": False, "stage_reached": False,
                                     "stage_reward_paid": False, "control_steps": 0}
            records.append({"episode": episode, "seed": seed, "sumo_seed": reset_info["sumo_seed"],
                            "scene_steps": scene_steps, "returns": returns, "outcomes": outcomes,
                            "both_success": all(outcomes[k]["is_success"] for k in AGENTS),
                            "any_collision": any(outcomes[k]["message"] == "Collision"
                                                 for k in AGENTS)})
            print(f"{scenario} {episode + 1}/{episodes}: "
                  f"entry={outcomes[AGENTS[0]]['message']} "
                  f"exit={outcomes[AGENTS[1]]['message']}", flush=True)
    finally:
        env.close()
    return records


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - radius), float(center + radius)]


def summarize(records):
    count = len(records)
    result = {"episodes": count, "agents": {}}
    for key in AGENTS:
        successes = sum(record["outcomes"][key]["is_success"] for record in records)
        collisions = sum(record["outcomes"][key]["message"] == "Collision"
                         for record in records)
        teleports = sum(record["outcomes"][key]["ego_teleported"] for record in records)
        result["agents"][key] = {
            "successes": successes, "success_rate": successes / count,
            "success_wilson95": wilson_interval(successes, count),
            "collisions": collisions, "collision_rate": collisions / count,
            "teleports": teleports, "teleport_rate": teleports / count,
            "mean_return": float(np.mean([record["returns"][key] for record in records])),
        }
    joint = sum(record["both_success"] for record in records)
    any_collision = sum(record["any_collision"] for record in records)
    result["both_successes"] = joint
    result["both_success_rate"] = joint / count
    result["both_success_wilson95"] = wilson_interval(joint, count)
    result["scenes_with_collision"] = any_collision
    result["scene_collision_rate"] = any_collision / count
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    learner, metadata = TwoAgentSAC.from_checkpoint(args.checkpoint)
    if metadata["scenario"] != "easy":
        raise ValueError("Dual comparison checkpoints must be trained on Easy")
    if metadata["delay_mode"] != "uniform" or metadata["training_safety"]:
        raise ValueError("Unexpected training delay or safety protocol")
    if metadata["actor_gru_input_mode"] != "paper_raw":
        raise ValueError("Unexpected GRU input mode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = evaluate(learner, args.scenario, args.episodes, args.seed_start,
                       safety=args.safety)
    report = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "method": learner.config.mode, "training": metadata,
              "scenario": args.scenario, "delay_mode": "uniform", "safety": args.safety,
              "policy_action": "deterministic_tanh_mean", "seed_start": args.seed_start,
              "summary": summarize(records), "episodes": records}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
