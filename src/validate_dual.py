"""Phase-0 fixed-action diagnostics, not policy evaluation or learning.

Run: python -B -m src.validate_dual --output tmp/ai/phase0-dual-ego/run-name
Existing outputs are never overwritten. No W&B or external service is used.
"""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .dual_merging import DualMerging, AGENT_TASKS


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run_case(directory, scenario, seed, safety, fixed_delay):
    env = DualMerging(scenario=scenario, safety=safety, fixed_delay=fixed_delay)
    summary = {"scenario": scenario, "seed": seed, "policy": "fixed [0, 0]",
               "safety": safety, "fixed_delay": fixed_delay,
               "returns": dict.fromkeys(AGENT_TASKS, 0.0),
               "transitions": dict.fromkeys(AGENT_TASKS, 0),
               "stage_payments": dict.fromkeys(AGENT_TASKS, 0),
               "outcomes": {}, "joint_active_steps": 0, "within_50m_steps": 0,
               "minimum_distance_m": None}
    try:
        _, reset_info = env.reset(seed=seed)
        summary["reset"] = reset_info
        with (directory / "steps.jsonl").open("x") as trace:
            for scene_step in range(1, 18001):
                active = env.active_agents
                distance = None
                if len(active) == 2:
                    summary["joint_active_steps"] += 1
                    if all(env.cars[key].ego_on_road for key in active):
                        positions = [np.asarray(env.connection.vehicle.getPosition(key))
                                     for key in active]
                        distance = float(np.linalg.norm(positions[0] - positions[1]))
                        previous = summary["minimum_distance_m"]
                        summary["minimum_distance_m"] = distance if previous is None else min(previous, distance)
                        summary["within_50m_steps"] += int(distance <= 50)
                obs, rewards, terms, truncs, infos = env.step(
                    {key: [0, 0] for key in active})
                for key in active:
                    if obs[key].shape != (134,) or not np.isfinite(obs[key]).all():
                        raise AssertionError(f"Invalid observation for {key}")
                    if not np.isfinite(rewards[key]):
                        raise AssertionError(f"Invalid reward for {key}")
                    if not np.isclose(sum(infos[key]["reward_components"].values()), rewards[key]):
                        raise AssertionError(f"Reward decomposition mismatch for {key}")
                    summary["transitions"][key] += 1
                    summary["returns"][key] += rewards[key]
                    summary["stage_payments"][key] += int(infos[key]["stage_reward"] != 0)
                    if summary["stage_payments"][key] > 1:
                        raise AssertionError(f"Repeated stage reward for {key}")
                    car = env.cars[key]
                    infos[key]["position"] = (list(env.connection.vehicle.getPosition(key))
                                              if car.ego_on_road else None)
                    infos[key]["actual_speed"] = (env.connection.vehicle.getSpeed(key)
                                                  if car.ego_on_road else None)
                    infos[key]["actual_acceleration"] = (
                        env.connection.vehicle.getAcceleration(key) if car.ego_on_road else None)
                    if terms[key] or truncs[key]:
                        summary["outcomes"][key] = {
                            field: infos[key][field] for field in
                            ("message", "status", "is_success", "ego_teleported", "control_steps")}
                # The existing single-agent info action is a numpy array.
                for key in AGENT_TASKS:
                    if key in infos and "action" in infos[key]:
                        infos[key]["action"] = np.asarray(infos[key]["action"]).tolist()
                trace.write(json.dumps({"scene_step": scene_step, "distance_m": distance,
                                        "rewards": rewards, "terminated": terms,
                                        "truncated": truncs, "infos": infos}, allow_nan=False) + "\n")
                if env.finished:
                    summary["scene_steps"] = scene_step
                    summary["status"] = env.status.copy()
                    summary["contract_passed"] = True
                    break
            else:
                raise AssertionError("Scene exceeded the departure plus per-agent time bound")
        for key, status in env.status.items():
            if status == "not_departed":
                summary["outcomes"][key] = {"message": "Departure Deadline", "status": status}
        write_json(directory / "summary.json", summary)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=("easy", "medium", "hard"),
                        default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[579071, 579072, 579073])
    parser.add_argument("--fixed-delay", type=int, choices=range(21), default=None,
                        help="Absent: uniform 0..20; fixed delays are diagnostic fixtures")
    parser.add_argument("--safety", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Seeds and scenarios must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in
             ("src/merging.py", "src/dual_merging.py", "src/wrapper.py",
              "src/safety_controller.py", "src/geometry.py", "src/utils.py",
              "src/validate_dual.py")]
    write_json(args.output / "manifest.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "purpose": "environment correctness, not learned-policy performance",
        "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files},
    })
    results = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            directory = args.output / f"{scenario}-{seed}"
            directory.mkdir()
            print(f"Starting {scenario} seed={seed}", flush=True)
            with (directory / "simulation.log").open("x") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run_case(directory, scenario, seed, args.safety, args.fixed_delay)
            results.append(result)
            print(f"Finished {scenario} seed={seed}: {result['status']}", flush=True)
    write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
