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
GAP_DERIVATION_VERSION = "candidate-lane-signed-envelope-gap-v2"


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


def _gap_observations(row):
    ego = row["pre"]["ego"]
    time_s = row.get("sim_time_pre_s")
    observations = []
    for lane in row["pre"]["surroundings"].get("candidate_lanes", []):
        for relative_position in ("front", "rear"):
            vehicle = lane.get(f"{relative_position}_vehicle")
            gap = lane.get(f"{relative_position}_gap_m")
            ttc = lane.get(f"{relative_position}_ttc_s")
            if vehicle is None or gap is None:
                continue
            if relative_position == "front":
                closing = (ego.get("speed_mps") - vehicle.get("speed_mps")
                           if ego.get("speed_mps") is not None
                           and vehicle.get("speed_mps") is not None else None)
                formula = "other_front_position - other_length - ego_front_position"
            else:
                closing = (vehicle.get("speed_mps") - ego.get("speed_mps")
                           if ego.get("speed_mps") is not None
                           and vehicle.get("speed_mps") is not None else None)
                formula = "ego_front_position - ego_length - other_front_position"
            if gap < 0:
                ttc_valid, invalid_reason = False, "negative_gap_longitudinal_overlap"
                longitudinal_relation = "overlap"
            elif closing is None:
                ttc_valid, invalid_reason = False, "closing_speed_missing"
                longitudinal_relation = relative_position
            elif closing <= 0:
                ttc_valid, invalid_reason = False, "not_closing"
                longitudinal_relation = relative_position
            elif ttc is None:
                ttc_valid, invalid_reason = False, "ttc_missing"
                longitudinal_relation = relative_position
            else:
                ttc_valid, invalid_reason = True, None
                longitudinal_relation = relative_position
            observations.append({
                "time_s": time_s,
                "road": ego.get("road_id"),
                "ego_lane": ego.get("lane_index"),
                "candidate_lane_id": lane.get("lane_id"),
                "candidate_lane_index": lane.get("lane_index"),
                "candidate_lane_relation": lane.get("relation"),
                "other_vehicle_id": vehicle.get("vehicle_id"),
                "other_vehicle_lane_id": vehicle.get("lane_id"),
                "relative_position": relative_position,
                "longitudinal_relation": longitudinal_relation,
                "signed_gap_m": gap,
                "overlap_m": -gap if gap < 0 else None,
                "closing_speed_mps": closing,
                "ttc_s": ttc if ttc_valid else None,
                "ttc_valid": ttc_valid,
                "ttc_invalid_reason": invalid_reason,
                "gap_formula": formula,
                "ttc_formula": "signed_gap_m / closing_speed_mps",
            })
    return observations


def _gap_audit(segment):
    events = [event for row in segment["transitions"]
              for event in _gap_observations(row)]
    signed = min(events, key=lambda item: item["signed_gap_m"], default=None)
    nonnegative = min((item for item in events if item["signed_gap_m"] >= 0),
                      key=lambda item: item["signed_gap_m"], default=None)
    return {
        "derivation_version": GAP_DERIVATION_VERSION,
        "definition": ("Signed longitudinal envelope clearance on current and adjacent "
                       "candidate lanes; negative means longitudinal overlap, not a "
                       "negative Euclidean distance."),
        "minimum_signed_gap": signed,
        "minimum_nonnegative_gap": nonnegative,
        "negative_gap_event_count": sum(item["signed_gap_m"] < 0 for item in events),
        "ttc_validity": "gap >= 0 and closing_speed > 0; otherwise TTC is null",
    }


def _initial_conditions(segment):
    row = segment["transitions"][0]
    ego, route = row["pre"]["ego"], row["pre"]["route"]
    surroundings = []
    for lane in row["pre"]["surroundings"].get("candidate_lanes", []):
        surroundings.append({
            "lane_id": lane.get("lane_id"), "lane_index": lane.get("lane_index"),
            "relation": lane.get("relation"),
            "front_vehicle_id": (lane.get("front_vehicle") or {}).get("vehicle_id"),
            "front_gap_m": lane.get("front_gap_m"),
            "front_relative_speed_mps": lane.get("front_relative_speed_mps"),
            "front_ttc_s": lane.get("front_ttc_s"),
            "rear_vehicle_id": (lane.get("rear_vehicle") or {}).get("vehicle_id"),
            "rear_gap_m": lane.get("rear_gap_m"),
            "rear_relative_speed_mps": lane.get("rear_relative_speed_mps"),
            "rear_ttc_s": lane.get("rear_ttc_s"),
        })
    return {
        "eval_seed": row.get("eval_seed"), "sim_time_s": row.get("sim_time_pre_s"),
        "road": ego.get("road_id"), "lane_id": ego.get("lane_id"),
        "lane_index": ego.get("lane_index"),
        "distance_d3_m": route.get("distance_to_d3_start_m"),
        "speed_mps": ego.get("speed_mps"),
        "route_requirement_n_changes": route.get("route_requirement_n_changes"),
        "pending": row["pre"].get("pending"),
        "surroundings": surroundings,
    }


def _route_change_events(segment):
    events = []
    for row in segment["transitions"]:
        before, after = row["pre"], row["post"]
        before_n = before["route"].get("route_requirement_n_changes")
        after_n = after["route"].get("route_requirement_n_changes")
        if before_n is None or after_n is None or before_n == after_n:
            continue
        actual_change = bool(row["execution"].get("lane_change_completed"))
        road_connection = (before["ego"].get("road_id") != after["ego"].get("road_id"))
        if actual_change:
            cause = "actual_lane_change"
        elif road_connection:
            cause = "road_connection_transition"
        else:
            cause = "route_state_change_without_completion_event"
        events.append({
            "time_s": row.get("sim_time_post_s"), "before_n": before_n,
            "after_n": after_n,
            "before": {"road": before["ego"].get("road_id"),
                       "lane_id": before["ego"].get("lane_id"),
                       "lane_index": before["ego"].get("lane_index"),
                       "distance_d3_m": before["route"].get("distance_to_d3_start_m")},
            "after": {"road": after["ego"].get("road_id"),
                      "lane_id": after["ego"].get("lane_id"),
                      "lane_index": after["ego"].get("lane_index"),
                      "distance_d3_m": after["route"].get("distance_to_d3_start_m")},
            "actual_lane_change_completed": actual_change,
            "road_connection_transition": road_connection,
            "associated_change_direction": row["execution"].get("completed_direction"),
            "cause": cause,
        })
    return events


def _lane_change_events(segment):
    rows = segment["transitions"]
    events = []
    for row in rows:
        completion = row["execution"].get("lane_change_completion_event")
        if completion is None:
            continue
        request_time = completion.get("request_time_s")
        request_direction = completion.get("request_direction")
        completion_time = completion.get("completion_time_s")
        request_row = next((candidate for candidate in rows
                            if request_time is not None
                            and candidate.get("sim_time_pre_s") is not None
                            and abs(candidate["sim_time_pre_s"] - request_time) <= 0.11
                            and candidate["sac"].get("lateral_direction") == request_direction),
                           None)
        blocked = [candidate for candidate in rows
                   if request_time is not None
                   and completion_time is not None
                   and candidate.get("sim_time_pre_s") is not None
                   and request_time <= candidate["sim_time_pre_s"] <= completion_time
                   and candidate["controller"]["lateral"].get("blocked_by_pending")]
        events.append({
            "direction": completion.get("actual_direction"),
            "route_direction": completion.get("route_direction"),
            "request_time_s": request_time,
            "completion_time_s": completion_time,
            "request_distance_d3_m": completion.get("request_distance_to_d3_start_m"),
            "request_in_segment": request_row is not None,
            "request_actor_direction": (request_row["sac"].get("lateral_direction")
                                        if request_row is not None else None),
            "request_command_sent": (request_row["controller"]["lateral"].get(
                "traci_command_called") if request_row is not None else None),
            "request_blocked_by_pending": (request_row["controller"]["lateral"].get(
                "blocked_by_pending") if request_row is not None else None),
            "pending_start_time_s": completion.get("request_time_s"),
            "pending_duration_s": ((completion_time - request_time)
                                   if completion_time is not None
                                   and request_time is not None else None),
            "blocked_actor_requests_during_pending": len(blocked),
        })
    return events


def _comparability(left, right):
    a, b = left["initial"], right["initial"]
    differences, critical = [], []

    def compare(name, av, bv, threshold=None, exact=False):
        delta = abs(av - bv) if isinstance(av, (int, float)) and isinstance(
            bv, (int, float)) else None
        if exact:
            issue = av != bv
        elif av is None and bv is None:
            issue = False
        elif delta is None:
            issue = True
        else:
            issue = delta > threshold
        row = {"field": name, "A": av, "B": bv, "absolute_difference": delta,
               "threshold": threshold, "large_difference": issue}
        differences.append(row)
        if issue:
            critical.append(name)

    compare("eval_seed", a.get("eval_seed"), b.get("eval_seed"), exact=True)
    compare("sim_time_s", a.get("sim_time_s"), b.get("sim_time_s"), threshold=0.11)
    compare("road", a.get("road"), b.get("road"), exact=True)
    compare("lane_index", a.get("lane_index"), b.get("lane_index"), exact=True)
    compare("distance_d3_m", a.get("distance_d3_m"), b.get("distance_d3_m"),
            threshold=30.0)
    compare("speed_mps", a.get("speed_mps"), b.get("speed_mps"), threshold=2.0)
    compare("route_requirement_n_changes", a.get("route_requirement_n_changes"),
            b.get("route_requirement_n_changes"), exact=True)
    compare("pending_active", (a.get("pending") or {}).get("active"),
            (b.get("pending") or {}).get("active"), exact=True)
    lanes_a = {row["relation"]: row for row in a.get("surroundings", [])}
    lanes_b = {row["relation"]: row for row in b.get("surroundings", [])}
    for relation in sorted(lanes_a.keys() | lanes_b.keys()):
        for position in ("front", "rear"):
            aa, bb = lanes_a.get(relation, {}), lanes_b.get(relation, {})
            compare(f"{relation}_{position}_vehicle_id",
                    aa.get(f"{position}_vehicle_id"), bb.get(f"{position}_vehicle_id"),
                    exact=True)
            compare(f"{relation}_{position}_gap_m", aa.get(f"{position}_gap_m"),
                    bb.get(f"{position}_gap_m"), threshold=20.0)
    return {
        "status": "insufficient" if critical else "comparable",
        "message": ("可比性不足" if critical else "起始条件在当前阈值内可比"),
        "large_difference_fields": critical,
        "differences": differences,
        "note": "Comparability only; this does not rank A or B.",
    }


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
    audit = _gap_audit(segment)
    return {
        "summary": segment["summary"],
        "initial": _initial_conditions(segment),
        "gap_audit": audit,
        "route_change_events": _route_change_events(segment),
        "lane_change_events": _lane_change_events(segment),
        "distance_aligned_progress": [
            {"distance_d3_m": row["distance_d3_m"], "route_n": row["route_n"],
             "road": row["road"], "lane": row["lane"],
             "speed_mps": row["speed_mps"], "time_s": row["time_s"]}
            for row in steps],
        "steps": steps,
    }


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
    public, provenance, skipped = [], [], []
    for seed in seeds:
        if seed not in episodes_a or seed not in episodes_b:
            raise ValueError(f"Seed {seed} is missing from one dataset")
        short = [
            {"dataset": name, "episode_id": episode["episode_id"],
             "steps": int(episode["steps"]), "required_steps": 100}
            for name, episode in (("dataset_a", episodes_a[seed]),
                                  ("dataset_b", episodes_b[seed]))
            if int(episode["steps"]) < 100
        ]
        if short:
            skipped.append({"eval_seed": seed, "reason": "short_episode",
                            "short_trajectories": short})
            continue
        try:
            score, start, source_a, source_b = _best_window(
                dataset_a, episodes_a[seed], dataset_b, episodes_b[seed])
        except ValueError as error:
            if str(error) != "No 100-step window has sufficiently similar initial state":
                raise
            skipped.append({"eval_seed": seed,
                            "reason": "no_similar_initial_window"})
            continue
        swap = int(hashlib.sha256(f"{blind_salt}:{seed}".encode()).hexdigest(), 16) % 2
        ordered = (source_b, source_a) if swap else (source_a, source_b)
        pair_id = f"pair-{len(public) + 1:02d}"
        segment_paths = []
        for side, segment in zip(("A", "B"), ordered):
            path = output / "segments" / f"{pair_id}-{side}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(segment, stream, ensure_ascii=False, allow_nan=False)
            segment_paths.append(str(path.relative_to(output)))
        compact_a, compact_b = _compact(ordered[0]), _compact(ordered[1])
        public.append({
            "pair_id": pair_id,
            "initial_condition": {"eval_seed": seed, "start_step": start,
                                  "duration_s": 10.0},
            "comparability": _comparability(compact_a, compact_b),
            "A": compact_a,
            "B": compact_b,
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
        "dataset_b": str(dataset_b), "pairs": provenance, "skipped": skipped,
        "counts": {
            "requested_pairs": len(seeds),
            "candidate_pairs": len(public),
            "skipped_pairs": len(skipped),
            "short_trajectories": sum(
                len(row.get("short_trajectories", [])) for row in skipped),
        },
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
