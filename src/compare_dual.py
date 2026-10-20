"""Compare paired Independent SAC and CTDE-MASAC evaluation reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from .multi_agent_sac import AGENTS


def check_pair(left, right):
    if (left["method"], right["method"]) != ("independent", "ctde"):
        raise ValueError("Pass independent report first, CTDE report second")
    for field in ("scenario", "delay_mode", "safety", "seed_start"):
        if left[field] != right[field]:
            raise ValueError(f"Mismatched evaluation {field}")
    if left["summary"]["episodes"] != right["summary"]["episodes"]:
        raise ValueError("Mismatched evaluation episode count")
    if [x["seed"] for x in left["episodes"]] != [x["seed"] for x in right["episodes"]]:
        raise ValueError("Evaluation seeds differ")
    for field in ("scenario", "delay_mode", "training_safety", "experience_target_per_ego",
                  "seed", "actor_gru_input_mode"):
        if left["training"][field] != right["training"][field]:
            raise ValueError(f"Mismatched training {field}")
    for field in ("experiences", "actor_updates", "critic_updates"):
        if left["training"][field] != right["training"][field]:
            raise ValueError(f"Mismatched learning budget: {field}")


def plot_curves(reports, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, agent in zip(axes, AGENTS):
        for label, report, color in (("Independent SAC", reports[0], "tab:blue"),
                                     ("CTDE-MASAC", reports[1], "tab:orange")):
            logfile = Path(report["checkpoint"]).parent / "training.jsonl"
            events = [json.loads(line) for line in logfile.open()]
            episodes = [event for event in events if event.get("event") == "episode"
                        and event["agent"] == agent
                        and event["experience"] <= report["training"]["experience_target_per_ego"]]
            x = np.array([event["experience"] for event in episodes], dtype=float)
            y = np.array([event["return"] for event in episodes], dtype=float)
            axis.plot(x, y, ".", color=color, alpha=.3)
            if len(y):
                smoothed = np.convolve(y, np.ones(min(5, len(y))) / min(5, len(y)), "valid")
                axis.plot(x[min(5, len(y)) - 1:], smoothed, color=color, label=label)
        axis.set_title(agent)
        axis.set_ylabel("Episode return")
        axis.grid(alpha=.3)
        axis.legend()
    axes[-1].set_xlabel("Valid training transitions for this ego")
    figure.suptitle("Easy training episodes; trailing 5-episode mean")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("independent", type=Path)
    parser.add_argument("ctde", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="New comparison .json path; adds .md and .png")
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in (args.independent, args.ctde)]
    check_pair(*reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = args.output.with_suffix(".md")
    chart_path = args.output.with_suffix(".png")
    for path in (args.output, markdown_path, chart_path):
        if path.exists():
            raise FileExistsError(path)
    rows = []
    for report in reports:
        summary = report["summary"]
        rows.append({"method": report["method"],
                     "entry_success": summary["agents"][AGENTS[0]]["success_rate"],
                     "exit_success": summary["agents"][AGENTS[1]]["success_rate"],
                     "both_success": summary["both_success_rate"],
                     "scene_collision": summary["scene_collision_rate"],
                     "entry_mean_return": summary["agents"][AGENTS[0]]["mean_return"],
                     "exit_mean_return": summary["agents"][AGENTS[1]]["mean_return"]})
    output = {"scenario": reports[0]["scenario"],
              "episodes": reports[0]["summary"]["episodes"],
              "seed_start": reports[0]["seed_start"],
              "training_experiences": reports[0]["training"]["experiences"],
              "actor_updates": reports[0]["training"]["actor_updates"],
              "critic_updates": reports[0]["training"]["critic_updates"],
              "rows": rows, "individual_reports": [str(args.independent), str(args.ctde)]}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
    header = ("| Method | Entry success | Exit success | Both success | "
              "Any ego collision | Entry return | Exit return |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n")
    body = "".join(f"| {r['method']} | {r['entry_success']:.1%} | "
                   f"{r['exit_success']:.1%} | {r['both_success']:.1%} | "
                   f"{r['scene_collision']:.1%} | {r['entry_mean_return']:.2f} | "
                   f"{r['exit_mean_return']:.2f} |\n" for r in rows)
    markdown_path.write_text(f"# Dual SAC comparison: {output['scenario']}\n\n"
                             f"{output['episodes']} paired evaluation seeds from {output['seed_start']}. "
                             "One training seed per method; descriptive comparison only.\n\n"
                             + header + body)
    plot_curves(reports, chart_path)
    print(json.dumps({"comparison": str(args.output), "table": str(markdown_path),
                      "training_curves": str(chart_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
