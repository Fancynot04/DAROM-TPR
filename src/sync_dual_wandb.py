"""Mirror only full-budget dual-SAC training and evaluation metrics to W&B.

This process reads the durable local experiment outputs. It never imports the
training environment, changes a learner, or uploads model checkpoints.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models_experiments" / "dual-marl"
STATE = ROOT / "tmp" / "ai" / "dual-wandb-sync"
RUN_NAMES = {
    "independent": "independent-easy-seed42-1m",
    "ctde": "ctde-easy-seed42-1m",
}
SCENARIOS = ("easy", "medium", "hard")
ENTITY = "arlikache-anhui-university"
PROJECT = "DAROM-TRP"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(RUN_NAMES), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--state-root", type=Path, default=STATE)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--api-key-stdin", action="store_true",
                        help="Read the API key from stdin, keeping it out of files and process arguments")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    return args


def read_key_from_stdin():
    key = sys.stdin.readline().strip()
    if not key:
        raise ValueError("No W&B API key received on stdin")
    os.environ["WANDB_API_KEY"] = key


def verify_formal_config(path, method):
    config = json.loads(path.read_text())
    if (config["mode"] != method or config["scenario"] != "easy"
            or config["experience_target_per_ego"] != 1_000_000
            or config["seed"] != 42 or config["delay_mode"] != "uniform"
            or config["delay_steps"] != [0, 20] or config["training_safety"]):
        raise ValueError("W&B sync accepts only the two approved full-budget runs")
    return config


def initial_state(method):
    run_id = hashlib.sha256(f"dual-sac-1m-seed42:{method}".encode()).hexdigest()[:8]
    return {"run_id": run_id, "training_lines": 0, "evaluations": [],
            "recent": {ego: {"success": [], "return": []}
                       for ego in ("ego_entry", "ego_exit")}}


def store_state(path, state):
    path.write_text(json.dumps(state, sort_keys=True) + "\n")


def training_metric(event, state):
    if event["event"] == "progress":
        data = {"train/scene_steps": event["scene_steps"]}
        for i, name in enumerate(("entry", "exit")):
            losses = event["latest_losses"][i]
            data.update({f"train/{name}_experiences": event["experiences"][i],
                         f"train/{name}_actor_updates": event["actor_updates"][i],
                         f"train/{name}_critic_updates": event["critic_updates"][i],
                         f"train/{name}_alpha": event["alpha"][i]})
            for key in ("critic", "actor", "max_abs_target"):
                if key in losses:
                    data[f"train/{name}_{key}"] = losses[key]
        return data
    if event["event"] == "episode":
        ego = event["agent"]
        name = "entry" if ego == "ego_entry" else "exit"
        recent = state["recent"][ego]
        for key, value in (("success", int(event["success"])),
                           ("return", event["return"])):
            recent[key] = list(deque((*recent[key], value), maxlen=10))
        return {f"episode/{name}_experience": event["experience"],
                f"episode/{name}_return": event["return"],
                f"episode/{name}_success": int(event["success"]),
                f"episode/{name}_return_mean10": sum(recent["return"]) / len(recent["return"]),
                f"episode/{name}_success_rate10": sum(recent["success"]) / len(recent["success"])}
    return None


def sync_training(run, path, state):
    if not path.exists():
        return 0
    synced = 0
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index < state["training_lines"]:
                continue
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            metric = training_metric(event, state)
            if metric is not None:
                run.log(metric)
                synced += 1
            state["training_lines"] = index + 1
    return synced


def evaluation_metrics(report, method, scenario):
    if (report["method"] != method or report["scenario"] != scenario
            or report["summary"]["episodes"] != 200
            or report["seed_start"] != 700_001
            or report["training"]["experience_target_per_ego"] != 1_000_000
            or report["training"]["seed"] != 42
            or report["training"]["delay_mode"] != "uniform"
            or report["training"]["training_safety"]
            or report["training"]["experiences"] != [1_000_000, 1_000_000]):
        raise ValueError("Evaluation does not match the approved paired protocol")
    prefix = f"eval_{scenario}"
    rows = []
    for i, episode in enumerate(report["episodes"]):
        if episode["seed"] != 700_001 + i:
            raise ValueError("Evaluation seeds are not consecutive")
        rows.append({f"{prefix}/episode": i + 1,
                     f"{prefix}/entry_return": episode["returns"]["ego_entry"],
                     f"{prefix}/exit_return": episode["returns"]["ego_exit"],
                     f"{prefix}/entry_success": int(episode["outcomes"]["ego_entry"]["is_success"]),
                     f"{prefix}/exit_success": int(episode["outcomes"]["ego_exit"]["is_success"]),
                     f"{prefix}/both_success": int(episode["both_success"]),
                     f"{prefix}/any_collision": int(episode["any_collision"])})
    summary = report["summary"]
    totals = {f"{prefix}/entry_success_rate": summary["agents"]["ego_entry"]["success_rate"],
              f"{prefix}/exit_success_rate": summary["agents"]["ego_exit"]["success_rate"],
              f"{prefix}/both_success_rate": summary["both_success_rate"],
              f"{prefix}/collision_rate": summary["scene_collision_rate"],
              f"{prefix}/entry_mean_return": summary["agents"]["ego_entry"]["mean_return"],
              f"{prefix}/exit_mean_return": summary["agents"]["ego_exit"]["mean_return"]}
    return rows, totals


def sync_evaluation(run, report_path, method, scenario):
    report = json.loads(report_path.read_text())
    rows, totals = evaluation_metrics(report, method, scenario)
    for row in rows:
        run.log(row)
    run.summary.update(totals)


def main(argv=None):
    args = arguments(argv)
    if args.api_key_stdin:
        read_key_from_stdin()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for online sync")
    run_dir = args.output_root / RUN_NAMES[args.method]
    pipeline = ROOT / "tmp" / "ai" / "dual-sac-transfer"
    while not (run_dir / "config.json").exists():
        if (pipeline / "pipeline.failed").exists():
            raise RuntimeError("Experiment pipeline failed before this formal run started")
        time.sleep(args.poll_seconds)
    config = verify_formal_config(run_dir / "config.json", args.method)
    args.state_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_root / f"{args.method}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else initial_state(args.method)
    import wandb
    with wandb.init(entity=ENTITY, project=PROJECT, id=state["run_id"], resume="allow",
                    name=RUN_NAMES[args.method], group="dual-sac-easy-seed42-1m",
                    job_type="dual-sac-comparison", mode="online",
                    dir=str(args.state_root),
                    save_code=False,
                    config={"method": args.method, "scenario": "easy", "seed": 42,
                            "experience_target_per_ego": 1_000_000,
                            "delay_steps": [0, 20], "safety": False,
                            "source_sha256": config["source_sha256"]},
                    settings=wandb.Settings(disable_git=True, disable_code=True,
                                            x_disable_stats=True,
                                            x_disable_machine_info=True)) as run:
        for name in ("entry", "exit"):
            run.define_metric(f"episode/{name}_experience")
            for metric in ("return", "success", "return_mean10", "success_rate10"):
                run.define_metric(f"episode/{name}_{metric}",
                                  step_metric=f"episode/{name}_experience")
        run.define_metric("train/scene_steps")
        for scenario in SCENARIOS:
            run.define_metric(f"eval_{scenario}/episode")
            for metric in ("entry_return", "exit_return", "entry_success",
                           "exit_success", "both_success", "any_collision"):
                run.define_metric(f"eval_{scenario}/{metric}",
                                  step_metric=f"eval_{scenario}/episode")
        print(json.dumps({"wandb_url": run.url, "method": args.method}), flush=True)
        while True:
            changed = bool(sync_training(run, run_dir / "training.jsonl", state))
            for scenario in SCENARIOS:
                if scenario in state["evaluations"]:
                    continue
                report = args.output_root / f"eval-{args.method}-{scenario}.json"
                if report.is_file():
                    try:
                        sync_evaluation(run, report, args.method, scenario)
                    except json.JSONDecodeError:
                        # The evaluator writes its JSON report once, at the end.
                        # Retry if the file was observed during that short write.
                        continue
                    state["evaluations"].append(scenario)
                    changed = True
                    print(json.dumps({"method": args.method,
                                      "evaluation_synced": scenario}), flush=True)
            if changed:
                store_state(state_path, state)
            if len(state["evaluations"]) == len(SCENARIOS):
                break
            if (pipeline / "pipeline.failed").exists():
                raise RuntimeError("Experiment pipeline failed; W&B sync is incomplete")
            if (pipeline / "pipeline.status").exists():
                raise RuntimeError("Pipeline ended without all six evaluation reports")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
