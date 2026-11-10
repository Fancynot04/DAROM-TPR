"""Train the first process-only MLP from isolated trajectory preferences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.preference_reward import (FEATURE_DIM, FEATURE_SCHEMA_VERSION,
                                   NEXT_OBS_START, fit_preference_reward,
                                   load_and_audit, normalize_trajectory,
                                   select_success_pairs)


HIDDEN_DIMS = (64, 64)
AGGREGATION = "mean"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_history(path, history):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "train_loss", "validation_loss",
            "train_accuracy", "validation_accuracy"))
        writer.writeheader()
        writer.writerows(history)


def _safe_correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _baseline_report(pairs, trajectories):
    total = len(pairs)
    labels_a = sum(pair.target_a == 1.0 for pair in pairs)
    labels_b = total - labels_a
    shorter_correct = 0
    shorter_ties = 0
    for pair in pairs:
        len_a = len(trajectories[pair.trajectory_a].features)
        len_b = len(trajectories[pair.trajectory_b].features)
        if len_a == len_b:
            shorter_ties += 1
            continue
        predicted_a = len_a < len_b
        shorter_correct += int(predicted_a == (pair.target_a == 1.0))
    return {
        "always_A": {"correct": labels_a, "total": total,
                     "accuracy": labels_a / total},
        "always_B": {"correct": labels_b, "total": total,
                     "accuracy": labels_b / total},
        "majority_direction": {"correct": max(labels_a, labels_b), "total": total,
                               "accuracy": max(labels_a, labels_b) / total},
        "random_expected": {"correct_expected": total / 2, "total": total,
                            "accuracy_expected": 0.5},
        "shorter_trajectory": {
            "correct": shorter_correct, "total_non_tie": total - shorter_ties,
            "ties": shorter_ties,
            "accuracy_non_tie": (shorter_correct / (total - shorter_ties)
                                 if total != shorter_ties else None),
        },
    }


def _diagnostics(output_dir, model, normalizer, trajectories, validation_pairs,
                 result, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_ids = []
    for pair in validation_pairs:
        for tid in (pair.trajectory_a, pair.trajectory_b):
            if tid not in validation_ids:
                validation_ids.append(tid)
    selected_ids = validation_ids[:4]
    per_trajectory = []
    all_rewards, reverse_rewards, non_reverse_rewards = [], [], []
    lengths, scores = [], []
    terminal_abs_fraction = []
    terminal_lane_change_rewards = []
    model.eval()
    with torch.no_grad():
        for tid in validation_ids:
            trajectory = trajectories[tid]
            features = torch.from_numpy(normalize_trajectory(
                trajectory, normalizer["mean"], normalizer["scale"]))
            rewards = model(features).numpy()
            score = float(np.mean(rewards))
            lengths.append(len(rewards))
            scores.append(score)
            all_rewards.extend(rewards.tolist())
            reverse = trajectory.diagnostics["reverse_lane_change"]
            physical_lane_change = trajectory.diagnostics["physical_lane_change"]
            reverse_rewards.extend(rewards[reverse].tolist())
            non_reverse_rewards.extend(rewards[~reverse].tolist())
            tail_start = max(0, len(rewards) - max(1, math.ceil(len(rewards) * 0.1)))
            denominator = float(np.abs(rewards).sum())
            terminal_abs_fraction.append(
                float(np.abs(rewards[tail_start:]).sum() / denominator)
                if denominator else 0.0)
            terminal_lane_change_mask = physical_lane_change.copy()
            terminal_lane_change_mask[:tail_start] = False
            terminal_lane_change_rewards.extend(rewards[terminal_lane_change_mask].tolist())
            per_trajectory.append({
                "trajectory_id": tid, "eval_seed": trajectory.eval_seed,
                "steps": len(rewards), "mean_score": score,
                "sum_reward": float(np.sum(rewards)),
                "min_step_reward": float(np.min(rewards)),
                "max_step_reward": float(np.max(rewards)),
                "positive_step_fraction": float(np.mean(rewards > 0)),
                "longest_positive_reward_run_steps": _longest_true_run(rewards > 0),
                "reverse_lane_change_count": int(np.sum(reverse)),
                "final_10pct_lane_change_count": int(np.sum(terminal_lane_change_mask)),
                "mean_reward_at_final_10pct_lane_change": (
                    float(np.mean(rewards[terminal_lane_change_mask]))
                    if np.any(terminal_lane_change_mask) else None),
                "last_10pct_abs_reward_fraction": terminal_abs_fraction[-1],
            })
            if tid not in selected_ids:
                continue
            steps = trajectory.diagnostics["step"]
            figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
            axes[0].plot(steps, rewards, linewidth=0.8)
            axes[0].axhline(0, color="black", linewidth=0.5)
            axes[0].set_ylabel("learned r")
            axes[1].plot(steps, trajectory.diagnostics["lane_index"], linewidth=0.8)
            axes[1].set_ylabel("lane index")
            axes[2].plot(steps, trajectory.diagnostics["speed_mps"], linewidth=0.8)
            axes[2].set_ylabel("speed m/s")
            axes[3].plot(steps, np.cumsum(rewards), linewidth=0.8)
            axes[3].set_ylabel("cumulative r")
            axes[3].set_xlabel("step")
            for axis in axes:
                for event_step in steps[reverse]:
                    axis.axvline(event_step, color="tab:red", alpha=0.35, linewidth=0.7)
            figure.suptitle(f"{tid} (mean trajectory score={score:.4f})")
            figure.tight_layout()
            figure.savefig(output_dir / f"{tid}.png", dpi=130)
            plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set_ylabel("preference loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation")
    axes[1].set_ylabel("pairwise accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=130)
    plt.close(figure)

    margins = result["validation_metrics"]["margins"]
    length_differences = result["validation_metrics"]["length_differences"]
    summary = {
        "plotted_trajectory_ids": selected_ids,
        "validation_trajectory_count": len(validation_ids),
        "step_reward": {
            "count": len(all_rewards),
            "min": float(np.min(all_rewards)), "max": float(np.max(all_rewards)),
            "mean": float(np.mean(all_rewards)), "std": float(np.std(all_rewards)),
            "max_abs": float(np.max(np.abs(all_rewards))),
        },
        "length_checks": {
            "trajectory_length_vs_mean_score_pearson": _safe_correlation(lengths, scores),
            "pair_length_difference_vs_model_margin_pearson": _safe_correlation(
                length_differences, margins),
        },
        "reverse_lane_change": {
            "event_step_count": len(reverse_rewards),
            "mean_reward_at_event": (float(np.mean(reverse_rewards))
                                     if reverse_rewards else None),
            "mean_reward_other_steps": (float(np.mean(non_reverse_rewards))
                                        if non_reverse_rewards else None),
        },
        "terminal_proximity": {
            "mean_last_10pct_abs_reward_fraction": float(np.mean(terminal_abs_fraction)),
            "note": "A value near 0.1 indicates no concentration under equal absolute rewards.",
        },
        "terminal_lane_changes": {
            "final_10pct_event_count": len(terminal_lane_change_rewards),
            "mean_reward_at_event": (float(np.mean(terminal_lane_change_rewards))
                                     if terminal_lane_change_rewards else None),
        },
        "trajectories": per_trajectory,
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def _format_fraction(metric):
    return f"{metric['correct']}/{metric['total']} ({metric['accuracy']:.3f})"


def _write_report(path, config, result, baselines, diagnostics):
    train = result["train_metrics"]
    validation = result["validation_metrics"]
    lines = [
        "# Preference reward validation report", "", "## Data and selection", "",
        f"- Train pairs: {result['selected_pairs']['train']} explicit Arrived–Arrived A/B pairs.",
        f"- Validation pairs: {result['selected_pairs']['validation']} explicit Arrived–Arrived A/B pairs.",
        "- P1 Test: skipped at source selection and not opened, fitted, or evaluated.",
        "- Similar/unresolved: excluded from fitting and validation metrics.", "",
        "## Model", "",
        f"- Architecture: 270 → 64 → 64 → 1 ReLU MLP ({config['model']['parameter_count']} parameters).",
        "- Features: obs_t(134), executed action_t(2), next_obs_t(134).",
        "- Invalid successor: normalized next-observation slice reset to zero.",
        "- Original environment reward and terminal event fields are excluded.",
        "- Trajectory score: mean of step rewards (not sum), limiting a direct length shortcut.", "",
        "## Selected checkpoint", "",
        f"- Best epoch by validation loss: {result['best_epoch']}.",
        f"- Train loss: {result['train_loss']:.6f}.",
        f"- Validation loss: {result['validation_loss']:.6f}.",
        f"- Train accuracy: {_format_fraction(train)}.",
        f"- Validation accuracy: {_format_fraction(validation)}.", "",
        "### Validation by deciding priority", "",
    ]
    for priority in ("P2", "P3", "P4"):
        metric = validation["by_priority"].get(priority)
        lines.append(f"- {priority}: {_format_fraction(metric) if metric else '0/0 (N/A)'}.")
    lines.extend(["", "## Baselines", ""])
    for name in ("always_A", "always_B", "majority_direction"):
        lines.append(f"- {name}: {_format_fraction(baselines[name])}.")
    lines.append("- random_expected: 0.500 expected accuracy.")
    shorter = baselines["shorter_trajectory"]
    shorter_accuracy = ("N/A" if shorter["accuracy_non_tie"] is None
                        else f"{shorter['accuracy_non_tie']:.3f}")
    lines.append(f"- shorter_trajectory: {shorter['correct']}/{shorter['total_non_tie']} "
                 f"non-ties ({shorter_accuracy}); ties={shorter['ties']}.")
    lines.extend(["", "## Reward diagnostics", ""])
    reward = diagnostics["step_reward"]
    length = diagnostics["length_checks"]
    reverse = diagnostics["reverse_lane_change"]
    terminal = diagnostics["terminal_proximity"]
    terminal_lane_changes = diagnostics["terminal_lane_changes"]
    longest_positive = max(
        item["longest_positive_reward_run_steps"]
        for item in diagnostics["trajectories"])
    lines.extend([
        f"- Step reward range: [{reward['min']:.6f}, {reward['max']:.6f}], max |r|={reward['max_abs']:.6f}.",
        f"- Length vs mean score Pearson r: {length['trajectory_length_vs_mean_score_pearson']}.",
        f"- Pair length difference vs margin Pearson r: {length['pair_length_difference_vs_model_margin_pearson']}.",
        f"- Reverse lane-change event reward: mean={reverse['mean_reward_at_event']}, n={reverse['event_step_count']}; other-step mean={reverse['mean_reward_other_steps']}.",
        f"- Longest uninterrupted positive-reward run: {longest_positive} steps.",
        f"- Mean share of absolute reward in final 10% of steps: {terminal['mean_last_10pct_abs_reward_fraction']:.4f}.",
        f"- Lane changes in final 10%: n={terminal_lane_changes['final_10pct_event_count']}, mean learned reward={terminal_lane_changes['mean_reward_at_event']}.",
        "- These are descriptive checks from a small validation set; causal interpretation is not warranted.", "",
        "## Leakage and semantics", "",
        "- Normalization was fitted only on trajectories referenced by selected Train pairs.",
        "- Validation affected checkpoint selection only; it did not contribute gradients or normalization.",
        "- Train/Validation remain grouped by their pre-existing eval_seed roles.",
        "- The model is a learned process signal only; no terminal bonus is embedded or joined to SAC.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-spec", type=Path, required=True)
    parser.add_argument("--expected-data-spec-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory exists: {args.output_dir}")
    if args.epochs < 1:
        parser.error("epochs must be positive")

    observed_spec_sha256 = hashlib.sha256(args.data_spec.read_bytes()).hexdigest()
    if (args.expected_data_spec_sha256
            and observed_spec_sha256 != args.expected_data_spec_sha256):
        raise SystemExit("dataset manifest SHA256 mismatch")
    args.output_dir.mkdir(parents=True)

    # Role filtering happens before paths or CSVs belonging to P1 Test are read.
    trajectories, pairs, audit = load_and_audit(
        args.data_spec, roles={"train", "validation"},
        required_roles=("train", "validation"))
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    if not audit["ok"] or not audit["fit_ready"]:
        raise SystemExit("Train/Validation data audit failed; see data_audit.json")

    selected = select_success_pairs(trajectories, pairs)
    selected_by_role = Counter(pair.role for pair in selected)
    if not selected_by_role["train"] or not selected_by_role["validation"]:
        raise SystemExit("no explicit Arrived–Arrived A/B pairs in Train or Validation")
    requested_device = args.device
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, normalizer, history, result = fit_preference_reward(
        trajectories, pairs, epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, hidden_dims=HIDDEN_DIMS, seed=args.seed,
        aggregation=AGGREGATION, device=device)
    validation_pairs = [pair for pair in selected if pair.role == "validation"]
    baselines = _baseline_report(validation_pairs, trajectories)
    diagnostics = _diagnostics(args.output_dir / "reward_diagnostics", model, normalizer,
                               trajectories, validation_pairs, result, history)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "data_spec": str(args.data_spec.resolve()),
        "data_spec_sha256": audit["data_spec_sha256"],
        "expected_data_spec_sha256": args.expected_data_spec_sha256,
        "roles_loaded": ["train", "validation"],
        "roles_skipped": audit["skipped_sources"],
        "selection": "both Arrived and preference in {A,B}",
        "model": {"type": "TransitionRewardMLP", "input_dim": FEATURE_DIM,
                  "hidden_dims": list(HIDDEN_DIMS), "activation": "ReLU",
                  "output_dim": 1, "parameter_count": parameter_count,
                  "feature_schema": FEATURE_SCHEMA_VERSION},
        "reward_definition": {
            "kind": "learned process reward", "trajectory_aggregation": AGGREGATION,
            "original_reward_used": False, "terminal_event_features_used": False,
            "invalid_next_obs": f"normalized features[{NEXT_OBS_START}:] reset to zero"},
        "optimizer": {"type": "Adam", "learning_rate": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "epochs": args.epochs, "seed": args.seed,
        "requested_device": requested_device, "actual_device": device,
        "torch_version": torch.__version__, "selected_pairs": dict(selected_by_role),
    }
    _write_history(args.output_dir / "training_metrics.csv", history)
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(
        _jsonable({"result": result, "baselines": baselines,
                   "diagnostics": diagnostics}), indent=2, ensure_ascii=False) + "\n")
    _write_report(args.output_dir / "validation_report.md", config, result,
                  baselines, diagnostics)
    torch.save({
        "model_state_dict": model.state_dict(), "model_config": config["model"],
        "normalizer_mean": normalizer["mean"],
        "normalizer_scale": normalizer["scale"],
        "reward_definition": config["reward_definition"],
        "training_config": config, "best_epoch": result["best_epoch"],
    }, args.output_dir / "reward_model.pt")
    print(json.dumps({
        "output_dir": str(args.output_dir), "selected_pairs": dict(selected_by_role),
        "best_epoch": result["best_epoch"],
        "train_accuracy": result["train_metrics"]["accuracy"],
        "validation_accuracy": result["validation_metrics"]["accuracy"],
        "p1_test_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
