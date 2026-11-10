"""Preference-reward data validation and a small transition MLP.

The loader intentionally separates data roles at the source level.  Sources
marked ``holdout`` are audited but can never contribute normalization
statistics, gradients, early stopping, or model selection.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBS_DIM = 134
ACTION_DIM = 2
DATA_SPEC_VERSION = "preference-reward-data-v1"
FEATURE_SCHEMA_VERSION = "transition-process-mlp-v2"
ALLOWED_ROLES = frozenset({"train", "validation", "test", "holdout"})
FEATURE_DIM = OBS_DIM + ACTION_DIM + OBS_DIM
NEXT_OBS_START = OBS_DIM + ACTION_DIM
# Project evaluation partitions already used for model comparison.  They may
# be evaluated as test/holdout, but never influence fitting or validation.
RESERVED_EVAL_SEEDS = frozenset(range(200001, 200151)) | frozenset(range(300001, 300061))
REQUIRED_PROVENANCE = frozenset({"experiment_id", "git_commit", "evaluation_seeds"})


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    source_id: str
    role: str
    group: str
    eval_seed: int
    features: np.ndarray
    next_obs_valid: np.ndarray
    terminal_reason: str
    diagnostics: dict


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    source_id: str
    role: str
    trajectory_a: str
    trajectory_b: str
    target_a: float
    preference: str
    deciding_priority: str


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"missing or non-numeric {key!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key!r}")
    return value


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_features(row):
    """Return process-only obs/action/next-obs features.

    Invalid successors are represented as zeros here and are reset to the
    normalized mean after train-only normalization.  Terminal flags, terminal
    reasons, and recorded environment rewards are deliberately excluded.
    """
    obs = [_finite(row, f"obs_{i:03d}") for i in range(OBS_DIM)]
    action = [_finite(row, f"action_exec_{i}") for i in range(ACTION_DIM)]
    valid = row.get("next_obs_valid") == "1"
    next_obs = ([_finite(row, f"next_obs_{i:03d}") for i in range(OBS_DIM)]
                if valid else [0.0] * OBS_DIM)
    return np.asarray(obs + action + next_obs, dtype=np.float32)


def _optional_number(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _audit_manifest(directory, errors):
    path = directory / "trajectory_manifest.json"
    if not path.exists():
        errors.append(f"{directory}: missing trajectory_manifest.json")
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: unreadable manifest ({exc})")
        return None
    if manifest.get("status") != "ok":
        errors.append(f"{path}: manifest status is not ok")
    if manifest.get("schema_version") not in (None, "trajectory-csv-v1"):
        errors.append(f"{path}: unsupported trajectory schema {manifest.get('schema_version')!r}")
    episodes = manifest.get("episodes", ())
    if not episodes or not all(item.get("complete") for item in episodes):
        errors.append(f"{path}: incomplete manifest episode")
    if manifest.get("diagnostic_read_errors"):
        errors.append(f"{path}: diagnostic read errors are present")
    if manifest.get("reward_terms_not_in_config"):
        errors.append(f"{path}: unknown reward terms are present")
    for name, expected in manifest.get("files", {}).items():
        target = directory / name
        if not target.exists() or _sha256(target) != expected:
            errors.append(f"{target}: missing or hash mismatch")
    return manifest


def _load_group(source, group, directory, errors):
    directory = Path(directory)
    _audit_manifest(directory, errors)
    summary_path = directory / "episode_summary.csv"
    transition_path = directory / "trajectory_transitions.csv"
    steps_path = directory / "trajectory_steps.csv"
    if not summary_path.exists() or not transition_path.exists() or not steps_path.exists():
        errors.append(f"{directory}: missing summary, step, or transition CSV")
        return [], {}
    summaries = _read_csv(summary_path)
    summary_ids = [row.get("trajectory_id", "") for row in summaries]
    if len(summary_ids) != len(set(summary_ids)):
        errors.append(f"{directory}: duplicate trajectory summary")
    rows_by_id = defaultdict(list)
    for row in _read_csv(transition_path):
        rows_by_id[row.get("trajectory_id", "")].append(row)
    steps_by_id = defaultdict(list)
    for row in _read_csv(steps_path):
        steps_by_id[row.get("trajectory_id", "")].append(row)
    result, seeds = [], {}
    for summary in summaries:
        tid = summary.get("trajectory_id", "")
        rows = rows_by_id.get(tid, [])
        step_rows = steps_by_id.get(tid, [])
        try:
            seed = int(summary["eval_seed"])
            expected_steps = int(summary["steps"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{directory}/{tid}: invalid summary seed or step count")
            continue
        if not tid or len(rows) != expected_steps or len(step_rows) != expected_steps:
            errors.append(f"{directory}/{tid}: step/transition count differs from summary")
            continue
        try:
            indices = [int(row["step"]) for row in rows]
            if indices != list(range(len(rows))):
                raise ValueError("non-contiguous steps")
            step_indices = [int(row["step"]) for row in step_rows]
            if step_indices != indices:
                raise ValueError("step CSV is not aligned with transition CSV")
            terminal = [row.get("is_terminal_step") == "1" for row in rows]
            if sum(terminal) != 1 or not terminal[-1]:
                raise ValueError("terminal flag must occur exactly once at the end")
            if rows[-1].get("terminal_reason") != summary.get("terminal_reason"):
                raise ValueError("terminal reason differs from summary")
            for left, right in zip(rows, rows[1:]):
                if left.get("next_obs_valid") != "1":
                    raise ValueError("non-final invalid next_obs")
                for i in range(OBS_DIM):
                    if left[f"next_obs_{i:03d}"] != right[f"obs_{i:03d}"]:
                        raise ValueError("broken obs/next_obs chain")
            features = np.stack([transition_features(row) for row in rows])
            next_obs_valid = np.asarray(
                [row.get("next_obs_valid") == "1" for row in rows], dtype=bool)
            diagnostics = {
                "step": np.asarray(indices, dtype=np.int64),
                "lane_index": np.asarray(
                    [_optional_number(row, "ego_lane_index") for row in step_rows]),
                "speed_mps": np.asarray(
                    [_optional_number(row, "ego_speed_mps") for row in step_rows]),
                "reverse_lane_change": np.asarray([
                    row.get("lane_change_physical") == "1"
                    and _optional_number(row, "ego_lane_index_after")
                    > _optional_number(row, "ego_lane_index")
                    for row in step_rows], dtype=bool),
                "physical_lane_change": np.asarray(
                    [row.get("lane_change_physical") == "1" for row in step_rows],
                    dtype=bool),
            }
        except (KeyError, ValueError) as exc:
            errors.append(f"{directory}/{tid}: {exc}")
            continue
        result.append(Trajectory(tid, source["source_id"], source["role"], group,
                                 seed, features, next_obs_valid,
                                 summary.get("terminal_reason", ""), diagnostics))
        seeds[tid] = seed
    extras = sorted(set(rows_by_id) - {item.trajectory_id for item in result})
    if extras:
        errors.append(f"{directory}: transitions without valid summary: {extras[:3]}")
    return result, seeds


def load_and_audit(spec_path, *, roles=None, required_roles=None):
    """Load a data specification and return data plus a serializable audit."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != DATA_SPEC_VERSION:
        raise ValueError(f"schema_version must be {DATA_SPEC_VERSION}")
    selected_roles = set(ALLOWED_ROLES if roles is None else roles)
    unknown_roles = selected_roles - ALLOWED_ROLES
    if unknown_roles:
        raise ValueError(f"unknown roles: {sorted(unknown_roles)}")
    errors, warnings, trajectories, pairs = [], [], {}, []
    source_reports = []
    skipped_sources = []
    seed_roles = defaultdict(set)
    sources = spec.get("sources", [])
    if not sources:
        errors.append("data spec has no sources")
    for source in sources:
        source_id, role = source.get("source_id"), source.get("role")
        if not source_id or role not in ALLOWED_ROLES:
            errors.append(f"invalid source_id/role: {source_id!r}/{role!r}")
            continue
        if role not in selected_roles:
            skipped_sources.append({"source_id": source_id, "role": role})
            continue
        provenance = source.get("provenance", {})
        missing_provenance = sorted(REQUIRED_PROVENANCE - set(provenance))
        if missing_provenance:
            errors.append(f"{source_id}: missing provenance fields {missing_provenance}")
        source_trajectories, source_seeds = [], {}
        for group, raw_path in source.get("groups", {}).items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = (spec_path.parent / path).resolve()
            loaded, seeds = _load_group(source, group, path, errors)
            source_trajectories.extend(loaded)
            source_seeds.update(seeds)
        for item in source_trajectories:
            if item.trajectory_id in trajectories:
                errors.append(f"duplicate trajectory_id: {item.trajectory_id}")
            trajectories[item.trajectory_id] = item
            seed_roles[item.eval_seed].add(role)
            if role in {"train", "validation"} and item.eval_seed in RESERVED_EVAL_SEEDS:
                errors.append(f"{source_id}: reserved eval_seed {item.eval_seed} assigned to {role}")
        preference_path = Path(source.get("preferences", ""))
        if not preference_path.is_absolute():
            preference_path = (spec_path.parent / preference_path).resolve()
        preference_counts = Counter()
        seen_pair_ids = set()
        seen_pair_members = set()
        rule_versions = set()
        if not source.get("preferences") or not preference_path.is_file():
            errors.append(f"{source_id}: missing preference CSV {preference_path}")
        else:
            for row in _read_csv(preference_path):
                pair_id = row.get("pair_id", "")
                a, b, preference = (row.get("trajectory_a", ""),
                                    row.get("trajectory_b", ""), row.get("preference", ""))
                if pair_id in seen_pair_ids:
                    errors.append(f"{source_id}: duplicate pair_id {pair_id}")
                    continue
                seen_pair_ids.add(pair_id)
                if a not in source_seeds or b not in source_seeds:
                    errors.append(f"{source_id}/{pair_id}: unknown trajectory reference")
                    continue
                if a == b:
                    errors.append(f"{source_id}/{pair_id}: self-comparison")
                    continue
                member_key = tuple(sorted((a, b)))
                if member_key in seen_pair_members:
                    errors.append(f"{source_id}/{pair_id}: duplicate unordered trajectory pair")
                    continue
                seen_pair_members.add(member_key)
                if source_seeds[a] != source_seeds[b]:
                    errors.append(f"{source_id}/{pair_id}: pair crosses eval seeds")
                    continue
                try:
                    recorded_seed = int(row.get("eval_seed", source_seeds[a]))
                except (TypeError, ValueError):
                    errors.append(f"{source_id}/{pair_id}: invalid recorded eval_seed")
                    continue
                if recorded_seed != source_seeds[a]:
                    errors.append(f"{source_id}/{pair_id}: recorded eval_seed mismatch")
                    continue
                if preference not in {"A", "B", "similar", "unresolved"}:
                    errors.append(f"{source_id}/{pair_id}: invalid preference {preference!r}")
                    continue
                comparable = row.get("comparable")
                if comparable == "0" and preference != "unresolved":
                    errors.append(f"{source_id}/{pair_id}: non-comparable pair has a fitted label")
                    continue
                if row.get("rule_version"):
                    rule_versions.add(row["rule_version"])
                preference_counts[preference] += 1
                if preference == "unresolved":
                    continue
                target = {"A": 1.0, "B": 0.0, "similar": 0.5}[preference]
                pairs.append(PreferencePair(pair_id, source_id, role, a, b, target, preference,
                                            row.get("deciding_priority", "none")))
        source_reports.append({
            "source_id": source_id, "role": role,
            "trajectory_count": len(source_trajectories),
            "eval_seeds": sorted(set(source_seeds.values())),
            "preference_counts": dict(sorted(preference_counts.items())),
            "group_paths": dict(source.get("groups", {})),
            "preferences_path": source.get("preferences"),
            "preference_rule_versions": sorted(rule_versions),
            "provenance": provenance,
        })
    overlaps = {str(seed): sorted(roles) for seed, roles in seed_roles.items() if len(roles) > 1}
    if overlaps:
        errors.append(f"eval_seed leakage across roles: {overlaps}")
    usable_by_role = Counter(pair.role for pair in pairs)
    if not usable_by_role["train"]:
        warnings.append("no independent training preference pairs; fitting is disabled")
    required = tuple(required_roles or ("train", "validation", "test"))
    for role in required:
        if role != "train" and not usable_by_role[role]:
            warnings.append(f"no {role} preference pairs")
    audit = {
        "schema_version": DATA_SPEC_VERSION,
        "data_spec_sha256": _sha256(spec_path),
        "ok": not errors,
        "fit_ready": not errors and all(usable_by_role[role] for role in required),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dim": next(iter(trajectories.values())).features.shape[1] if trajectories else None,
        "sources": source_reports,
        "skipped_sources": skipped_sources,
        "required_roles": list(required),
        "usable_pairs_by_role": dict(sorted(usable_by_role.items())),
        "errors": errors, "warnings": warnings,
    }
    return trajectories, pairs, audit


class TransitionRewardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def select_success_pairs(trajectories, pairs):
    """Keep explicit A/B preferences where both trajectories arrived."""
    return [pair for pair in pairs
            if pair.preference in {"A", "B"}
            and trajectories[pair.trajectory_a].terminal_reason == "Arrived"
            and trajectories[pair.trajectory_b].terminal_reason == "Arrived"]


def _normalizer(trajectories, trajectory_ids):
    items = [trajectories[tid] for tid in sorted(trajectory_ids)]
    steps = np.concatenate([item.features for item in items])
    mean = steps.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = steps.std(axis=0, dtype=np.float64).astype(np.float32)
    valid_next = np.concatenate([
        item.features[item.next_obs_valid, NEXT_OBS_START:] for item in items
    ])
    mean[NEXT_OBS_START:] = valid_next.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale[NEXT_OBS_START:] = valid_next.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize_trajectory(trajectory, mean, scale):
    values = (trajectory.features - mean) / scale
    values = values.astype(np.float32, copy=False)
    values[~trajectory.next_obs_valid, NEXT_OBS_START:] = 0.0
    return values


def trajectory_score(model, features, aggregation="mean"):
    rewards = model(features)
    if aggregation == "mean":
        return rewards.mean()
    if aggregation == "sum":
        return rewards.sum()
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _pair_metrics(model, tensors, trajectories, items, aggregation):
    losses, correct = [], 0
    by_priority = defaultdict(lambda: [0, 0])
    margins, length_differences = [], []
    trajectory_ids = {tid for pair in items
                      for tid in (pair.trajectory_a, pair.trajectory_b)}
    scores = {tid: trajectory_score(model, tensors[tid], aggregation)
              for tid in trajectory_ids}
    for pair in items:
        score_a = scores[pair.trajectory_a]
        score_b = scores[pair.trajectory_b]
        target = torch.tensor(pair.target_a, dtype=score_a.dtype, device=score_a.device)
        margin = score_a - score_b
        losses.append(nn.functional.binary_cross_entropy_with_logits(margin, target))
        predicted_a = float(margin.detach()) > 0
        expected_a = pair.target_a == 1.0
        hit = int(predicted_a == expected_a)
        correct += hit
        by_priority[pair.deciding_priority][0] += hit
        by_priority[pair.deciding_priority][1] += 1
        margins.append(float(margin.detach()))
        length_differences.append(
            len(trajectories[pair.trajectory_a].features)
            - len(trajectories[pair.trajectory_b].features))
    loss = torch.stack(losses).mean()
    return loss, {
        "correct": correct, "total": len(items), "accuracy": correct / len(items),
        "by_priority": {key: {"correct": value[0], "total": value[1],
                              "accuracy": value[0] / value[1]}
                        for key, value in sorted(by_priority.items())},
        "margins": margins, "length_differences": length_differences,
    }


def fit_preference_reward(trajectories, pairs, *, epochs=200, learning_rate=1e-3,
                          weight_decay=1e-4, hidden_dims=(64, 64), seed=42,
                          aggregation="mean", device="cpu"):
    """Fit process reward on train; use validation only for model selection."""
    selected = select_success_pairs(trajectories, pairs)
    train_pairs = [pair for pair in selected if pair.role == "train"]
    validation_pairs = [pair for pair in selected if pair.role == "validation"]
    if not train_pairs or not validation_pairs:
        raise ValueError("explicit success-success train and validation pairs are required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    compute_device = torch.device(device)
    train_ids = {value for pair in train_pairs for value in (pair.trajectory_a, pair.trajectory_b)}
    mean, scale = _normalizer(trajectories, train_ids)
    tensors = {tid: torch.from_numpy(normalize_trajectory(item, mean, scale)).to(compute_device)
               for tid, item in trajectories.items()}
    model = TransitionRewardMLP(FEATURE_DIM, hidden_dims).to(compute_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    history, best = [], None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss, train_metrics = _pair_metrics(
            model, tensors, trajectories, train_pairs, aggregation)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss, validation_metrics = _pair_metrics(
                model, tensors, trajectories, validation_pairs, aggregation)
        history.append({"epoch": epoch + 1, "train_loss": float(train_loss.detach()),
                        "validation_loss": float(validation_loss),
                        "train_accuracy": train_metrics["accuracy"],
                        "validation_accuracy": validation_metrics["accuracy"]})
        if best is None or float(validation_loss) < best["validation_loss"]:
            best = {"epoch": epoch + 1, "validation_loss": float(validation_loss),
                    "state_dict": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        train_loss, train_metrics = _pair_metrics(
            model, tensors, trajectories, train_pairs, aggregation)
        validation_loss, validation_metrics = _pair_metrics(
            model, tensors, trajectories, validation_pairs, aggregation)
    result = {
        "best_epoch": best["epoch"],
        "train_loss": float(train_loss), "validation_loss": float(validation_loss),
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "selected_pairs": {"train": len(train_pairs), "validation": len(validation_pairs)},
        "aggregation": aggregation,
    }
    return model.cpu(), {"mean": mean, "scale": scale}, history, result
