"""Paired preflight proving that full trajectory export is simulation-neutral."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC

from evaluate_sac_delay import evaluate_runs
from src import exit_e76_d3_config as exit_config
from src.trajectory_export import (SCHEMA_VERSION, TrajectoryDigestProbe,
                                   _sha256_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=200051)
    parser.add_argument("--episodes", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--trajectory-export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--neighbor-range-m", type=float, default=150.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.trajectory_export_dir.exists() or args.output.exists():
        parser.error("Validation outputs must not already exist")
    if (len(args.source_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in args.source_commit.lower())):
        parser.error("--source-commit must be a full 40-hex commit")
    return args


def comparable_episode(record):
    return {key: record.get(key) for key in (
        "seed", "steps", "outcome", "ego_teleported", "stage_reached",
        "success", "toward_requests", "away_requests",
        "lane_change_commands_sent", "toward_commands_sent",
        "away_commands_sent", "same_direction_repeat_requests",
        "short_horizon_request_reversals_1s", "return",
        "lane_change_counts", "lane_change_direction_reversals",
        "reward_totals")}


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = SAC.load(args.checkpoint, device=args.device)
    common = dict(
        scenario="easy", delay_mode="uniform", safety=False, ablation="full",
        episodes=args.episodes, seeds=[args.seed], task=exit_config.TASK_ID,
    )
    baseline_probe = TrajectoryDigestProbe()
    baseline = evaluate_runs(model, trajectory_probe=baseline_probe, **common)
    recorded_probe = TrajectoryDigestProbe()
    recorded = evaluate_runs(
        model, trajectory_probe=recorded_probe,
        trajectory_export_dir=args.trajectory_export_dir,
        trajectory_neighbor_range_m=args.neighbor_range_m,
        trajectory_metadata={
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "device": args.device,
            "torch_threads": args.torch_threads,
            "source_commit": args.source_commit,
            "schema_version": SCHEMA_VERSION,
            "validation_preflight": True,
        }, **common)
    baseline_episodes = [comparable_episode(row)
                         for row in baseline["runs"][0]["episode_results"]]
    recorded_episodes = [comparable_episode(row)
                         for row in recorded["runs"][0]["episode_results"]]
    baseline_digests = baseline_probe.summary()
    recorded_digests = recorded_probe.summary()
    checks = {
        "episode_results_equal": baseline_episodes == recorded_episodes,
        "key_step_trajectory_digests_equal": baseline_digests == recorded_digests,
        "expected_episode_count": len(recorded_episodes) == args.episodes,
        "all_episode_step_counts_match": all(
            item["steps"] == episode["steps"]
            for item, episode in zip(recorded_digests, recorded_episodes)),
    }
    report = {
        "protocol": "darom-pbrl-p0-export-neutrality-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "source_commit": args.source_commit,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "full_export_disabled_run": {
            "episode_results_sha256": _sha256_json(baseline_episodes),
            "trajectory_digests": baseline_digests,
        },
        "full_export_enabled_run": {
            "episode_results_sha256": _sha256_json(recorded_episodes),
            "trajectory_digests": recorded_digests,
            "dataset": str(args.trajectory_export_dir),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "limitation": ("The disabled run uses a validation-only in-memory Ego digest "
                       "probe; it does not execute the full route/neighbor/file exporter."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Trajectory export changed paired evaluation behavior")


if __name__ == "__main__":
    main()
