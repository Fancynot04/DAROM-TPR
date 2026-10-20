"""Train two decentralized SAC actors in one shared SUMO scene.

Examples:
python -B -m src.train_dual_sac --mode independent --experience-target 50000 --run-name independent-easy-seed42-50k
python -B -m src.train_dual_sac --mode ctde --experience-target 50000 --run-name ctde-easy-seed42-50k
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dual_merging import AGENT_TASKS, DualMerging
from .multi_agent_sac import AGENTS, OBS_DIM, SACConfig, TwoAgentSAC


ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("independent", "ctde"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experience-target", type=int, default=50_000,
                        help="Valid training transitions per ego")
    parser.add_argument("--scenario", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "models_experiments" / "dual-marl")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if (args.experience_target < 1 or args.checkpoint_every < 1 or
            args.torch_threads < 1 or args.seed < 0):
        parser.error("Target, checkpoint interval and threads must be positive; seed nonnegative")
    if not args.run_name or "/" in args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one path segment")
    return args


def provenance(scenario):
    paths = [ROOT / name for name in (
        "src/dual_merging.py", "src/merging.py", "src/wrapper.py",
        "src/delay_encoder.py", "src/multi_agent_sac.py", "src/train_dual_sac.py",
        f"sumo_new/Antwerp_{scenario}_dual.rou.xml", "sumo_new/Antwerp.net.xml")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run(args):
    torch.set_num_threads(args.torch_threads)
    config = SACConfig(mode=args.mode, seed=args.seed, batch_size=args.batch_size,
                       replay_capacity=args.replay_capacity, device=args.device)
    learner = TwoAgentSAC(config)
    run_dir = args.save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {"mode": args.mode, "scenario": args.scenario,
                "delay_mode": "uniform", "delay_steps": [0, 20],
                "training_safety": False, "experience_target_per_ego": args.experience_target,
                "seed": args.seed, "torch_threads": args.torch_threads,
                "sac_config": vars(config), "source_sha256": provenance(args.scenario),
                "observation_shape_per_ego": [OBS_DIM], "action_shape_per_ego": [2],
                "action_scale": 5.0, "actor_gru_input_mode": "paper_raw",
                "algorithm": "two-independent-twin-Q" if args.mode == "independent"
                             else "centralized-twin-Q-two-reward-heads"}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = DualMerging(seed=args.seed, scenario=args.scenario, safety=False,
                      delay_mode="uniform")
    started = time.monotonic()
    scene_index = 0
    scene_steps = 0
    next_checkpoint = args.checkpoint_every
    episode_rewards = dict.fromkeys(AGENTS, 0.0)
    episode_steps = dict.fromkeys(AGENTS, 0)
    try:
        with (run_dir / "training.jsonl").open("x") as log:
            observations, reset_info = env.reset(seed=args.seed + scene_index)
            log.write(json.dumps({"event": "scene_reset", "scene": scene_index,
                                  "sumo_seed": reset_info["sumo_seed"],
                                  "scene_files": reset_info["scene_files"]}) + "\n")
            while min(learner.experiences) < args.experience_target:
                scene_steps += 1
                if scene_steps > 3 * args.experience_target + 20_000:
                    raise RuntimeError("Scene-step guard exceeded before both egos reached budget")
                before = {key: observations[key] for key in env.active_agents}
                chosen = learner.actions(before)
                env_actions = {key: (action * 5.0).tolist() for key, action in chosen.items()}
                following, rewards, terminated, truncated, infos = env.step(env_actions)
                learner.observe(before, chosen, following, rewards, terminated,
                                truncated, env.active_agents, args.experience_target)
                for key in before:
                    episode_rewards[key] += rewards[key]
                    episode_steps[key] += 1
                    if terminated[key] or truncated[key]:
                        record = {"event": "episode", "scene": scene_index, "agent": key,
                                  "experience": learner.experiences[AGENTS.index(key)],
                                  "return": episode_rewards[key],
                                  "length": episode_steps[key], "outcome": infos[key]["message"],
                                  "success": infos[key]["is_success"],
                                  "ego_teleported": infos[key]["ego_teleported"]}
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        episode_rewards[key] = 0.0
                        episode_steps[key] = 0
                observations = following
                if scene_steps % 1000 == 0:
                    progress = {"event": "progress", "scene_steps": scene_steps,
                                "scene": scene_index, "experiences": learner.experiences,
                                "actor_updates": learner.actor_updates,
                                "critic_updates": learner.critic_updates,
                                "alpha": [float(x.exp().detach().cpu())
                                          for x in learner.log_alpha],
                                "latest_losses": learner.last_losses,
                                "elapsed_seconds": time.monotonic() - started}
                    log.write(json.dumps(progress) + "\n")
                    log.flush()
                    print(json.dumps(progress), flush=True)
                if min(learner.experiences) >= next_checkpoint:
                    path = run_dir / f"checkpoint-{next_checkpoint}.pt"
                    torch.save(learner.checkpoint({**settings, "scene_steps": scene_steps,
                                                   "checkpoint_per_ego": next_checkpoint}), path)
                    next_checkpoint += args.checkpoint_every
                if env.finished and min(learner.experiences) < args.experience_target:
                    scene_index += 1
                    observations, reset_info = env.reset(seed=args.seed + scene_index)
                    log.write(json.dumps({"event": "scene_reset", "scene": scene_index,
                                          "sumo_seed": reset_info["sumo_seed"],
                                          "scene_files": reset_info["scene_files"]}) + "\n")
            final = {**settings, "scene_steps": scene_steps,
                     "scenes_started": scene_index + 1,
                     "experiences": learner.experiences,
                     "actor_updates": learner.actor_updates,
                     "critic_updates": learner.critic_updates,
                     "elapsed_seconds": time.monotonic() - started}
            torch.save(learner.checkpoint(final), run_dir / "final.pt")
            (run_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
            log.write(json.dumps({"event": "complete", **final}) + "\n")
            print(json.dumps({"event": "complete", "path": str(run_dir),
                              "experiences": learner.experiences,
                              "scene_steps": scene_steps}), flush=True)
            return final
    finally:
        env.close()


if __name__ == "__main__":
    run(arguments())
