"""Build blinded, paired 10-second PbRL annotation candidates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from .trajectory_segments import load_episode, summarize_segment


PAIR_VERSION = "darom-pbrl-p0-pairs-v1"


def _episode_by_seed(dataset):
    with (Path(dataset) / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        seed = int(row["eval_seed"])
        if seed in result:
            raise ValueError(f"Duplicate eval seed {seed} in {dataset}")
        result[seed] = row
    return result


def select_seeds(results_a, results_b, count=20, conversion_each=8,
                 both_success=4):
    """Select paired seeds from prior evaluation summaries, never trajectories."""
    by_a = {int(row["seed"]): bool(row["success"]) for row in results_a}
    by_b = {int(row["seed"]): bool(row["success"]) for row in results_b}
    common = sorted(by_a.keys() & by_b.keys())
    a_to_b = [seed for seed in common if not by_a[seed] and by_b[seed]]
    b_to_a = [seed for seed in common if by_a[seed] and not by_b[seed]]
    both = [seed for seed in common if by_a[seed] and by_b[seed]]
    selected = a_to_b[:conversion_each] + b_to_a[:conversion_each] + both[:both_success]
    if len(selected) < count:
        remaining = [seed for seed in common if seed not in selected]
        selected.extend(remaining[:count - len(selected)])
    return sorted(selected[:count])


def _minimum_gap(row):
    values = []
    for lane in row["pre"]["surroundings"].get("candidate_lanes", []):
        values.extend(value for value in (lane.get("front_gap_m"), lane.get("rear_gap_m"))
                      if value is not None)
    return min(values) if values else None


def _compact(segment):
    steps = []
    for row in segment["transitions"]:
        ego = row["pre"]["ego"]
        route = row["pre"]["route"]
        steps.append({
            "step": row["step"],
            "time_s": row["sim_time_pre_s"],
            "road": ego.get("road_id"),
            "lane": ego.get("lane_index"),
            "route_n": route.get("route_requirement_n_changes"),
            "distance_d3_m": route.get("distance_to_d3_start_m"),
            "speed_mps": ego.get("speed_mps"),
            "actor_request": row["sac"].get("lateral_direction"),
            "command": row["controller"]["lateral"].get("command_direction"),
            "completed_change": row["execution"].get("completed_direction"),
            "minimum_gap_m": _minimum_gap(row),
            "pending": row.get("pre", {}).get("pending", {}).get("active"),
        })
    return {"summary": segment["summary"], "steps": steps}


def _similar_start(left, right):
    a, b = left["transitions"][0]["pre"], right["transitions"][0]["pre"]
    ea, eb = a["ego"], b["ego"]
    if not ea.get("valid") or not eb.get("valid"):
        return False
    if ea.get("road_id") != eb.get("road_id") or ea.get("lane_index") != eb.get("lane_index"):
        return False
    if a["route"].get("route_requirement_n_changes") != b["route"].get("route_requirement_n_changes"):
        return False
    sa, sb = ea.get("speed_mps"), eb.get("speed_mps")
    return sa is not None and sb is not None and abs(sa - sb) <= 2.0


def _difference_score(left, right):
    score = 0.0
    for a, b in zip(left["transitions"], right["transitions"]):
        ea, eb = a["pre"]["ego"], b["pre"]["ego"]
        if ea.get("lane_index") != eb.get("lane_index"):
            score += 2.0
        if (a["pre"]["route"].get("route_requirement_n_changes") !=
                b["pre"]["route"].get("route_requirement_n_changes")):
            score += 2.0
        if ea.get("speed_mps") is not None and eb.get("speed_mps") is not None:
            score += min(abs(ea["speed_mps"] - eb["speed_mps"]), 10.0) / 10.0
        score += 0.5 * (a["sac"].get("lateral_direction") !=
                       b["sac"].get("lateral_direction"))
        score += 1.0 * (a["execution"].get("completed_direction") !=
                       b["execution"].get("completed_direction"))
        ga, gb = _minimum_gap(a), _minimum_gap(b)
        if ga is not None and gb is not None:
            score += min(abs(ga - gb), 30.0) / 30.0
    return score


def _slice_episode(rows, episode_id, start, length):
    ordered = {}
    for name, values in rows.items():
        by_step = {int(row["step"]): row for row in values}
        selected = [by_step.get(step) for step in range(start, start + length)]
        if any(row is None for row in selected):
            raise ValueError("Requested segment is not a complete contiguous interval")
        ordered[name] = selected
    return {
        "schema_version": ordered["transitions"][0]["schema_version"],
        "episode_id": episode_id, "start_step": start, "length_steps": length,
        "summary": summarize_segment(ordered["transitions"], ordered["derived"]),
        **ordered,
    }


def _best_window(dataset_a, episode_a, dataset_b, episode_b, length=100):
    steps_a = int(episode_a["steps"])
    steps_b = int(episode_b["steps"])
    last_start = min(steps_a, steps_b) - length
    if last_start < 0:
        raise ValueError("Episode pair is shorter than 100 steps")
    rows_a = load_episode(dataset_a, episode_a["episode_id"])
    rows_b = load_episode(dataset_b, episode_b["episode_id"])
    best = None
    # Ten-step stride bounds artifact generation while retaining 1 s resolution.
    starts = list(range(0, last_start + 1, 10))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        left = _slice_episode(rows_a, episode_a["episode_id"], start, length)
        right = _slice_episode(rows_b, episode_b["episode_id"], start, length)
        if not _similar_start(left, right):
            continue
        score = _difference_score(left, right)
        if best is None or score > best[0]:
            best = (score, start, left, right)
    if best is None:
        raise ValueError("No 100-step window has sufficiently similar initial state")
    return best


def build_pairs(dataset_a, dataset_b, seeds, output_dir, assets_dir, blind_salt):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "segments").mkdir()
    episodes_a, episodes_b = _episode_by_seed(dataset_a), _episode_by_seed(dataset_b)
    public, provenance = [], []
    for index, seed in enumerate(seeds, 1):
        if seed not in episodes_a or seed not in episodes_b:
            raise ValueError(f"Seed {seed} is missing from one dataset")
        score, start, source_a, source_b = _best_window(
            dataset_a, episodes_a[seed], dataset_b, episodes_b[seed])
        swap = int(hashlib.sha256(f"{blind_salt}:{seed}".encode()).hexdigest(), 16) % 2
        ordered = (source_b, source_a) if swap else (source_a, source_b)
        pair_id = f"pair-{index:02d}"
        segment_paths = []
        for side, segment in zip(("A", "B"), ordered):
            path = output / "segments" / f"{pair_id}-{side}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(segment, stream, ensure_ascii=False, allow_nan=False)
            segment_paths.append(str(path.relative_to(output)))
        public.append({
            "pair_id": pair_id,
            "initial_condition": {"eval_seed": seed, "start_step": start,
                                  "duration_s": 10.0},
            "A": _compact(ordered[0]),
            "B": _compact(ordered[1]),
        })
        provenance.append({
            "pair_id": pair_id, "eval_seed": seed, "start_step": start,
            "behavior_difference_score": score,
            "A_source": "dataset_b" if swap else "dataset_a",
            "B_source": "dataset_a" if swap else "dataset_b",
            "dataset_a_episode": episodes_a[seed]["episode_id"],
            "dataset_b_episode": episodes_b[seed]["episode_id"],
            "A_segment_file": segment_paths[0],
            "B_segment_file": segment_paths[1],
        })
    payload = {"version": PAIR_VERSION, "label_options": [
        "A_better", "B_better", "similar", "cannot_judge"], "pairs": public}
    (output / "pairs.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "pair_provenance.json").write_text(json.dumps({
        "version": PAIR_VERSION, "dataset_a": str(dataset_a),
        "dataset_b": str(dataset_b), "pairs": provenance,
        "selection_note": "No reward, terminal outcome, or source identity enters a label.",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copy2(Path(assets_dir) / name, output / name)
    return len(public)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_a", type=Path)
    parser.add_argument("dataset_b", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path,
                        default=Path("assets/p0_preference_labeler"))
    parser.add_argument("--blind-salt", required=True)
    args = parser.parse_args(argv)
    if len(args.seeds) > 20 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at most 20 unique paired seeds")
    build_pairs(args.dataset_a, args.dataset_b, args.seeds, args.output_dir,
                args.assets_dir, args.blind_salt)


if __name__ == "__main__":
    main()
