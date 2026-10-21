"""Summarize paired dual-Ego evaluations with explicit metric definitions."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .compare_dual import check_pair
from .evaluate_dual_sac import wilson_interval
from .multi_agent_sac import AGENTS


SCENARIOS = ("easy", "medium", "hard")
METHODS = ("independent", "ctde")
SAFETY = (False, True)
EXPECTED_SEEDS = list(range(700001, 700201))
BOOTSTRAP_SAMPLES = 5000


def report_path(directory, method, scenario, safety):
    suffix = f"-safety-on-{scenario}" if safety else f"-{scenario}"
    return directory / f"eval-{method}{suffix}.json"


def load_reports(directory):
    reports = {}
    for scenario in SCENARIOS:
        for safety in SAFETY:
            for method in METHODS:
                path = report_path(directory, method, scenario, safety)
                report = json.loads(path.read_text())
                if (report["method"], report["scenario"], report["safety"]) != (method, scenario, safety):
                    raise ValueError(f"Report identity mismatch: {path}")
                if (report["delay_mode"] != "uniform" or
                        report["policy_action"] != "deterministic_tanh_mean"):
                    raise ValueError(f"Evaluation protocol mismatch: {path}")
                if report["training"]["training_safety"]:
                    raise ValueError(f"Unexpected safety-enabled training: {path}")
                records = report["episodes"]
                if ([record["seed"] for record in records] != EXPECTED_SEEDS or
                        report["summary"]["episodes"] != len(EXPECTED_SEEDS)):
                    raise ValueError(f"Expected 200 evaluation seeds 700001–700200: {path}")
                if (any("mean_speed_mps" not in report["summary"]["agents"][ego]
                        for ego in AGENTS) or
                        any("motion" not in record or
                            any(ego not in record["motion"] or
                                any(metric not in record["motion"][ego] for metric in (
                                    "mean_speed_mps", "mean_jerk_mps3", "mean_abs_jerk_mps3"))
                                for ego in AGENTS)
                            for record in records)):
                    raise ValueError(f"Motion samples missing; reevaluate frozen checkpoint: {path}")
                reports[scenario, safety, method] = report

            left = reports[scenario, safety, METHODS[0]]
            right = reports[scenario, safety, METHODS[1]]
            check_pair(left, right)
            if [r["sumo_seed"] for r in left["episodes"]] != [r["sumo_seed"] for r in right["episodes"]]:
                raise ValueError(f"SUMO seeds differ: {scenario}, safety={safety}")

    for method in METHODS:
        reference = reports[SCENARIOS[0], SAFETY[0], method]
        for scenario in SCENARIOS:
            for safety in SAFETY:
                report = reports[scenario, safety, method]
                if report["checkpoint_sha256"] != reference["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed across evaluations: {method}")
                if ([r["seed"] for r in report["episodes"]] !=
                        [r["seed"] for r in reference["episodes"]]):
                    raise ValueError(f"Evaluation seeds changed: {method}")
    return reports


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": None, "n": 0}
    mean = float(values.mean())
    if len(values) == 1:
        return {"mean": mean, "ci95": None, "n": 1}
    indices = np.random.default_rng(42).integers(0, len(values),
                                                 size=(BOOTSTRAP_SAMPLES, len(values)))
    estimates = values[indices].mean(axis=1)
    return {"mean": mean,
            "ci95": [float(x) for x in np.percentile(estimates, [2.5, 97.5])],
            "n": len(values)}


def paired_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired metric lengths differ")
    valid = np.isfinite(left) & np.isfinite(right)
    return bootstrap_mean(left[valid] - right[valid])


def rate(flags):
    flags = np.asarray(flags, dtype=bool)
    successes = int(flags.sum())
    return {"count": successes, "n": len(flags), "rate": successes / len(flags),
            "ci95": wilson_interval(successes, len(flags))}


def values(records, ego):
    outcomes = [record["outcomes"][ego] for record in records]
    timeout = np.array([outcome["message"] == "Max Steps" for outcome in outcomes])
    motion = [record["motion"][ego] for record in records]

    def motion_values(name):
        return np.array([item[name] if item.get(name) is not None else np.nan
                         for item in motion], dtype=float)

    return {
        "success": np.array([outcome["is_success"] for outcome in outcomes], dtype=float),
        "collision": np.array([outcome["message"] == "Collision" for outcome in outcomes], dtype=float),
        "timeout": timeout.astype(float),
        "teleport": np.array([outcome["ego_teleported"] for outcome in outcomes], dtype=float),
        "unmerged_timeout": np.array([timed_out and not outcome["merged"]
                                       for timed_out, outcome in zip(timeout, outcomes)], dtype=float),
        "return": np.array([record["returns"][ego] for record in records], dtype=float),
        "speed_mps": motion_values("mean_speed_mps"),
        "signed_jerk_mps3": motion_values("mean_jerk_mps3"),
        "abs_jerk_mps3": motion_values("mean_abs_jerk_mps3"),
    }


def summarize_values(metrics):
    output = {name: rate(data) for name, data in metrics.items()
              if name in ("success", "collision", "timeout", "teleport", "unmerged_timeout")}
    output.update({name: bootstrap_mean(data) for name, data in metrics.items()
                   if name in ("return", "speed_mps", "signed_jerk_mps3", "abs_jerk_mps3")})
    return output


def aggregate(reports):
    rows = []
    differences = []
    joint_rows = []
    joint_differences = []
    for scenario in SCENARIOS:
        for safety in SAFETY:
            for ego in AGENTS:
                samples = {}
                for method in METHODS:
                    report = reports[scenario, safety, method]
                    samples[method] = values(report["episodes"], ego)
                    rows.append({"scenario": scenario, "safety": safety,
                                 "method": method, "ego": ego,
                                 "metrics": summarize_values(samples[method])})
                differences.append({"scenario": scenario, "safety": safety, "ego": ego,
                                    "direction": "independent_minus_ctde",
                                    "metrics": {key: paired_difference(samples["independent"][key],
                                                                        samples["ctde"][key])
                                                for key in ("success", "collision", "timeout",
                                                            "return", "speed_mps", "signed_jerk_mps3",
                                                            "abs_jerk_mps3")}})
            joint_samples = {}
            for method in METHODS:
                records = reports[scenario, safety, method]["episodes"]
                joint_samples[method] = {
                    "both_success": np.array([r["both_success"] for r in records], dtype=float),
                    "any_collision": np.array([r["any_collision"] for r in records], dtype=float),
                }
                joint_rows.append({"scenario": scenario, "safety": safety,
                                   "method": method,
                                   "metrics": {key: rate(data) for key, data in joint_samples[method].items()}})
            joint_differences.append({"scenario": scenario, "safety": safety,
                                      "direction": "independent_minus_ctde",
                                      "metrics": {key: paired_difference(joint_samples["independent"][key],
                                                                          joint_samples["ctde"][key])
                                                  for key in joint_samples["independent"]}})
    provenance = {}
    for method in METHODS:
        reference = reports[SCENARIOS[0], SAFETY[0], method]
        training = reference.get("training", {})
        provenance[method] = {
            "checkpoint_sha256": reference.get("checkpoint_sha256"),
            "training_seed": training.get("seed"),
            "experience_target_per_ego": training.get("experience_target_per_ego"),
            "experiences": training.get("experiences"),
            "actor_updates": training.get("actor_updates"),
            "critic_updates": training.get("critic_updates"),
        }
    return {"protocol": {"episodes_per_cell": len(EXPECTED_SEEDS),
                         "seed_start": EXPECTED_SEEDS[0],
                         "seed_end": EXPECTED_SEEDS[-1],
                         "bootstrap_resamples": BOOTSTRAP_SAMPLES,
                         "confidence_level": 0.95,
                         "training_seeds_per_method": 1,
                         "difference_direction": "independent_minus_ctde",
                         "no_merge": "entry Max Steps timeout proxy; unmerged timeout separately retained",
                         "no_exit": "exit Max Steps timeout proxy",
                         "motion": "equal-weight mean across episodes with samples; paired differences use only episodes with samples for both methods",
                         "jerk": "signed SUMO acceleration difference divided by 0.1 s; mean absolute jerk separately reported"},
            "provenance": provenance, "rows": rows, "paired_differences": differences,
            "joint_rows": joint_rows, "joint_paired_differences": joint_differences}


def cell(stat, percent=False):
    point = stat.get("rate", stat.get("mean"))
    if point is None:
        return "—"
    scale = 100 if percent else 1
    digits = 1 if percent else 2
    fmt = f"{{:.{digits}f}}"
    result = fmt.format(point * scale)
    if stat["ci95"] is not None:
        result += " [" + ", ".join(fmt.format(x * scale) for x in stat["ci95"]) + "]"
    return result


def markdown(summary):
    lines = ["# Dual-Ego transfer evaluation (200 paired scenes per condition)", "",
             "Each cell is a point estimate [95% interval]. Rates use Wilson intervals; means and paired differences use 5,000 episode bootstrap resamples.",
             "One training seed per method: intervals describe evaluation-scene variability for these frozen checkpoints, not training-seed variability.",
             "No Merge and No Exit count Max Steps timeouts for entry and exit respectively. No Merge is the historical timeout proxy; true unmerged timeouts are retained in JSON.",
             "Signed jerk can cancel; mean absolute jerk is included as a separate comfort measure. Motion means exclude episodes without motion samples; see each metric's n in JSON.", "",
             "| Scenario | SC | Method | Ego | Success % | Collision % | No Merge % | No Exit % | Avg return | Avg speed m/s | Avg signed jerk m/s³ | Avg absolute jerk m/s³ | Teleport % |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        metric = row["metrics"]
        timeout = cell(metric["timeout"], True)
        lines.append("| " + " | ".join((row["scenario"].title(), "On" if row["safety"] else "Off",
                                     row["method"], "Entry" if row["ego"] == AGENTS[0] else "Exit",
                                     cell(metric["success"], True), cell(metric["collision"], True),
                                     timeout if row["ego"] == AGENTS[0] else "—",
                                     timeout if row["ego"] == AGENTS[1] else "—",
                                     cell(metric["return"]), cell(metric["speed_mps"]),
                                     cell(metric["signed_jerk_mps3"]), cell(metric["abs_jerk_mps3"]),
                                     cell(metric["teleport"], True))) + " |")
    lines += ["", "## Paired method differences", "",
              "Independent SAC minus CTDE on the same 200 seeds. Intervals crossing zero do not establish a clear difference for these fixed checkpoints.", "",
              "| Scenario | SC | Ego | Success Δ pp | Collision Δ pp | Timeout Δ pp | Return Δ | Speed Δ m/s | Abs jerk Δ m/s³ | Paired motion n |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["paired_differences"]:
        metric = row["metrics"]
        lines.append("| " + " | ".join((row["scenario"].title(), "On" if row["safety"] else "Off",
                                     "Entry" if row["ego"] == AGENTS[0] else "Exit",
                                     cell(metric["success"], True), cell(metric["collision"], True),
                                     cell(metric["timeout"], True), cell(metric["return"]),
                                     cell(metric["speed_mps"]), cell(metric["abs_jerk_mps3"]),
                                     str(metric["speed_mps"]["n"]))) + " |")
    lines += ["", "## Joint scene outcomes", "",
              "| Scenario | SC | Method | Both succeed % | Any Ego collision % |",
              "|---|---|---|---:|---:|"]
    for row in summary["joint_rows"]:
        lines.append("| " + " | ".join((row["scenario"].title(), "On" if row["safety"] else "Off",
                                     row["method"], cell(row["metrics"]["both_success"], True),
                                     cell(row["metrics"]["any_collision"], True))) + " |")
    lines += ["", "The paired joint-method differences are in the JSON report. Comparisons across safety settings are descriptive; this table does not attribute their difference to the controller alone."]
    return "\n".join(lines) + "\n"


def write_outputs(summary, output):
    paths = (output, output.with_suffix(".md"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite an existing report")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(markdown(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New .json path; a same-stem .md table is also written")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Wait up to this many seconds for queued reports; 0 fails immediately")
    args = parser.parse_args(argv)
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            reports = load_reports(args.reports_dir)
            break
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Queued evaluation reports did not become complete") from exc
            time.sleep(min(60, remaining))
    summary = aggregate(reports)
    write_outputs(summary, args.output)
    print(json.dumps({"json": str(args.output), "table": str(args.output.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
