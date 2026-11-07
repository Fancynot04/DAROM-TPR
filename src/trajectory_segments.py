"""Read and summarize contiguous 10 s / 20 s PbRL comparison segments."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .trajectory_export import DERIVATION_VERSION, SCHEMA_VERSION


def _read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_episode(dataset_dir, episode_id):
    root = Path(dataset_dir)
    names = {
        "transitions": root / "trajectories" / f"{episode_id}.jsonl.gz",
        "observations": root / "observations" / f"{episode_id}.jsonl.gz",
        "derived": root / "derived" / f"{episode_id}.jsonl.gz",
    }
    missing = [str(path) for path in names.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing episode files: {missing}")
    rows = {name: _read_jsonl(path) for name, path in names.items()}
    step_sets = [{row["step_id"] for row in values} for values in rows.values()]
    if not step_sets or any(items != step_sets[0] for items in step_sets[1:]):
        raise ValueError("Episode files do not share exactly the same step_id set")
    if any(len(values) != len(step_sets[0]) for values in rows.values()):
        raise ValueError("Episode files contain duplicate step_id values")
    return rows


def summarize_segment(transitions, derived):
    if not transitions or len(transitions) != len(derived):
        raise ValueError("Segment transitions and derived rows must be nonempty and aligned")
    directions = [row["execution"]["completed_direction"] for row in transitions
                  if row["execution"]["completed_direction"] is not None]
    speeds = [row["pre"]["ego"]["speed_mps"] for row in transitions
              if row["pre"]["ego"]["speed_mps"] is not None]
    jerks = [row["jerk_mps3"] for row in derived if row["jerk_mps3"] is not None]
    gaps = []
    ttc = []
    for row in transitions:
        for lane in row["pre"]["surroundings"]["candidate_lanes"]:
            gaps.extend(value for value in (lane["front_gap_m"], lane["rear_gap_m"])
                        if value is not None)
            ttc.extend(value for value in (lane["front_ttc_s"], lane["rear_ttc_s"])
                       if value is not None)
    stable = [row["post_direct_route_stable"] for row in derived]
    longest = current = 0
    for value in stable:
        current = current + 1 if value else 0
        longest = max(longest, current)
    first_direct = next((row["post"]["route"]["distance_to_d3_start_m"]
                         for row in transitions
                         if row["post"]["route"]["has_direct_exit_route"] is True), None)
    terminal_events = [row["events"] for row in transitions
                       if row["events"]["terminated"] or row["events"]["truncated"]]
    return {
        "derivation_version": DERIVATION_VERSION,
        "steps": len(transitions),
        "duration_s": len(transitions) * 0.1,
        "start_route_requirement_n_changes": transitions[0]["pre"]["route"][
            "route_requirement_n_changes"],
        "end_route_requirement_n_changes": transitions[-1]["post"]["route"][
            "route_requirement_n_changes"],
        "first_direct_exit_route_distance_to_d3_start_m": first_direct,
        "longest_stable_direct_route_s": longest * 0.1,
        "completed_lane_changes": len(directions),
        "completed_direction_reversals": sum(
            left != right for left, right in zip(directions, directions[1:])),
        "repeated_actor_requests": sum(
            row["actor_request_repeated_from_previous_step"] for row in derived),
        "speed_mps": {"mean": float(np.mean(speeds)) if speeds else None,
                      "min": min(speeds) if speeds else None,
                      "max": max(speeds) if speeds else None},
        "absolute_jerk_mps3": {"mean": float(np.mean(np.abs(jerks))) if jerks else None,
                               "max": max(map(abs, jerks)) if jerks else None},
        "candidate_lane_gap_m": {"min": min(gaps) if gaps else None},
        "candidate_lane_ttc_s": {"min": min(ttc) if ttc else None},
        "terminal_events": terminal_events,
    }


def extract_segment(dataset_dir, episode_id, start_step, length):
    if length not in (100, 200):
        raise ValueError("PbRL P0 segment length must be 100 or 200 steps")
    rows = load_episode(dataset_dir, episode_id)
    ordered = {}
    for name, values in rows.items():
        selected = [row for row in values
                    if start_step <= int(row["step"]) < start_step + length]
        selected.sort(key=lambda row: int(row["step"]))
        if len(selected) != length or [row["step"] for row in selected] != list(
                range(start_step, start_step + length)):
            raise ValueError("Requested segment is not a complete contiguous interval")
        ordered[name] = selected
    return {
        "schema_version": SCHEMA_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "episode_id": episode_id,
        "start_step": start_step,
        "length_steps": length,
        "summary": summarize_segment(ordered["transitions"], ordered["derived"]),
        **ordered,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("episode_id")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--length", type=int, choices=(100, 200), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_step < 0 or args.output.exists():
        parser.error("start-step must be nonnegative and output must not exist")
    segment = extract_segment(args.dataset_dir, args.episode_id,
                              args.start_step, args.length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(segment, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
