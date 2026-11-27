"""Generate nested, per-observation Antwerp background traffic from the OD recovery artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import gzip
from hashlib import sha256
import io
from pathlib import Path
import random
from statistics import median
from xml.etree import ElementTree as ET

from .empirical_demand import largest_remainders, read_json, write_json, write_xml

SUMO = Path(__file__).resolve().parents[2] / "sumo_new"
BASE = SUMO / "empirical_observed_v1"
ORIGINS = ("O1", "O2", "O3")
DESTS = ("D1", "D2", "D3")
TYPE_KEYS = {"Motorbike": "motorbike", "Car": "car", "Van": "van",
             "Car/Van 待判定": "light_unresolved", "Rigid lorry": "rigid_lorry", "Trailer/Bus": "trailer_bus"}
BASELINE_RECOVERY = "baseline_lag_p50_exclude_3849"


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def check_entrances(network, entries, limit_kmh):
    lanes = {lane.get("id"): (edge.get("id"), int(lane.get("index")), float(lane.get("speed")))
             for edge in network.findall("edge") for lane in edge.findall("lane")}
    links = {(link.get("from") + "_" + link.get("fromLane"),
              link.get("to") + "_" + link.get("toLane")) for link in network.findall("connection")}
    if not all(abs(speed * 3.6 - limit_kmh) < .001 for _, _, speed in lanes.values()):
        raise ValueError("Road limit differs from the empirical traffic contract")
    for entry in entries.values():
        lane, next_lane = entry["lane"], entry["next"]
        edge, index, _ = lanes[lane]
        if (lane, next_lane) not in links or edge != entry["origin"]:
            raise ValueError(f"Invalid entry lane mapping: {entry}")
        if lane != f"{edge}_{index}":
            raise ValueError(f"Unexpected lane index: {lane}")
    return lanes


def targets(reference, source, config):
    day, period = config["date"], config["period"]
    if (reference["date"], reference["period"]) != (day, period):
        raise ValueError("Unexpected date or period in reference demand")
    if tuple(reference["origin_order"]) != ORIGINS or tuple(reference["destination_order"]) != DESTS:
        raise ValueError("OD order differs from the generation contract")
    recovered = source["days"][day]["periods"][period]
    if reference["observed_assignment"] != recovered["observed_match_od"]:
        raise ValueError("Recovered direct match counts disagree with reference")
    if reference["scenarios"]["Hard"]["od"] != recovered["variants"][BASELINE_RECOVERY]["integer_od"]:
        raise ValueError("Hard OD counts disagree with audited recovery")
    result = {}
    for scenario, total in config["scenarios"].items():
        rows = reference["scenarios"][scenario]["od"]
        counts = {(origin, dest): n for origin, row in zip(ORIGINS, rows, strict=True)
                  for dest, n in zip(DESTS, row, strict=True)}
        if sum(counts.values()) != total or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError(f"Bad OD counts in {scenario}")
        result[scenario] = counts
    return result


def read_entries(path, source, config):
    if digest(path) != source["entry_evidence_sha256"]:
        raise ValueError("Entry file hash does not match the preceding recovery")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as text:
        rows = [row for row in csv.DictReader(text) if (row["date"], row["period"]) ==
                (config["date"], config["period"])]
    end = config["demand_end_s"]
    start = datetime.fromisoformat(config["entry_reference"])
    by_key = set()
    rows.sort(key=lambda row: (row["entrance_time_source"], row["entrance_detector"], row["observation_id"]))
    for i, row in enumerate(rows):
        detector = row["entrance_detector"]
        entry = config["entrances"][detector]
        orig_lane = row["logical_entrance_lane"]
        if (row["origin"], orig_lane) != (entry["origin"], entry["logical_lane"]):
            raise ValueError(f"Entry detector does not match audited lane: {detector}")
        if row["matched_csv_lane"] and row["matched_csv_lane"] != orig_lane:
            raise ValueError(f"Matched lane differs from detector mapping: {detector}")
        key = (detector, row["observation_id"], row["entrance_time_source"])
        if key in by_key:
            raise ValueError(f"Duplicate observation: {key}")
        by_key.add(key)
        seconds = (datetime.fromisoformat(row["entrance_time_source"]) - start).total_seconds()
        if not 0 <= seconds < end or abs(seconds - round(seconds, 1)) > 1e-8:
            raise ValueError(f"Observation is outside the empirical 0.1 s clock: {key}")
        row["departure"] = round(seconds, 1)
        row["id"] = f"v{i:05d}"
        row["sumo_lane"] = entry["lane"]
        row["raw_speed"] = int(row["speed_raw_kmh"])
        row["raw_length"] = int(row["length_raw_dm"])
    if len(rows) != config["scenarios"]["Hard"]:
        raise ValueError("Hard must use the full set of reference-period observations")
    return rows


def assign_destinations(rows, quotas, expected_match, config, rng):
    count = Counter()
    pending = defaultdict(list)
    for row in rows:
        origin = row["origin"]
        dest = row["destination_status"]
        observed = row["match_status"] == "matched_od"
        row["dest_source"] = "observed_match" if observed else "furness_residual"
        if observed:
            if dest not in DESTS or int(row["matched_exit_detector"]) not in config["exit_detectors"][dest]:
                raise ValueError("Observed destination disagrees with the validated detector map")
            row["dest"] = dest
            count[origin, dest] += 1
        else:
            if dest in DESTS:
                raise ValueError("Only matched vehicles may have a definite destination")
            pending[origin].append(row)
    if [[count[o, d] for d in DESTS] for o in ORIGINS] != expected_match:
        raise ValueError("Direct matches do not reproduce the recovery seed matrix")
    for origin in ORIGINS:
        remaining = [quotas[origin, dest] - count[origin, dest] for dest in DESTS]
        if min(remaining) < 0 or sum(remaining) != len(pending[origin]):
            raise ValueError(f"Observed destinations are incompatible with Hard OD quotas at {origin}")
        destinations = [dest for dest, n in zip(DESTS, remaining) for _ in range(n)]
        rng.shuffle(destinations)
        for row, dest in zip(pending[origin], destinations, strict=True):
            row["dest"] = dest


def clean(rows, config, types, road_kmh):
    speed_border, length_border = config["speed_boundary_kmh"], config["length_boundary_dm"]
    bounds = config["physical_length_bounds_dm"]
    valid_lengths, valid_speeds = defaultdict(list), defaultdict(list)
    for row in rows:
        category = row["length_category"]
        if category not in TYPE_KEYS:
            raise ValueError(f"No physical vehicle policy for {category}")
        if category in bounds:
            low, high = bounds[category]
            if low <= row["raw_length"] <= high and row["raw_length"] != length_border and row["raw_speed"] != speed_border:
                valid_lengths[category].append(row["raw_length"])
        if 0 < row["raw_speed"] <= road_kmh:
            valid_speeds[row["entrance_detector"]].append(row["raw_speed"])
    size_replacement = {k: int(Decimal(str(median(v))).to_integral_value(rounding=ROUND_HALF_UP))
                        for k, v in valid_lengths.items()}
    speed_replacement = {k: median(v) for k, v in valid_speeds.items()}
    for row in rows:
        category = row["length_category"]
        kind = TYPE_KEYS[category]
        raw_length, raw_speed = row["raw_length"], row["raw_speed"]
        flags = []
        if category == "Motorbike":
            length = round(config["motorbike_physical_length_m"] * 10)
            flags.append("motorbike_detector_length_to_existing_physical_length")
        elif raw_length == length_border or raw_speed == speed_border or not bounds[category][0] <= raw_length <= bounds[category][1]:
            length = size_replacement[category]
            flags.append("replace_boundary_length" if raw_length == length_border else
                         "replace_length_on_boundary_speed_record" if raw_speed == speed_border else
                         "replace_outside_physical_length_bounds")
        else:
            length = raw_length
        if raw_speed == speed_border or raw_speed < 0:
            speed = speed_replacement[row["entrance_detector"]]
            flags.append("replace_speed_boundary_or_negative")
        else:
            speed = raw_speed
        cap = min(Decimal(types[kind]["maxSpeed"]), Decimal(road_kmh) / Decimal("3.6"))
        speed_mps = Decimal(str(speed)) / Decimal("3.6")
        if speed_mps > cap:
            flags.append("cap_speed_by_road_or_vehicle")
        row["length_dm_clean"] = length
        row["speed_mps_clean"] = format(min(cap, speed_mps).quantize(Decimal("0.000001"), rounding=ROUND_DOWN), "f")
        row["cleaning"] = "|".join(flags)
        row["kind"] = kind
        row["type_id"] = f"{kind}_l{length:03d}"
    return {"length_median_replacement_dm": size_replacement, "speed_median_replacement_kmh": speed_replacement}


def sample_group(rows, count, levels, rng):
    if not count:
        return []
    if count == len(rows):
        return rows
    if not levels:
        return rng.sample(rows, count)
    groups = defaultdict(list)
    for row in rows:
        groups[levels[0](row)].append(row)
    ordered = [groups[k] for k in sorted(groups)]
    sizes = largest_remainders(count, [len(g) for g in ordered])
    return [row for group, size in zip(ordered, sizes)
            for row in sample_group(group, size, levels[1:], rng)]


def rebalance(parent, picked, levels, rng):
    """Reduce marginal discrepancies with deterministic one-for-one swaps inside one OD."""
    count = len(picked)
    selected_ids = {r["id"] for r in picked}
    candidates = defaultdict(lambda: ([], []))
    for row in parent:
        key = tuple(level(row) for level in levels)
        candidates[key][0 if row["id"] in selected_ids else 1].append(row)
    targets, current = [], []
    for tag in levels:
        parent_counts = Counter(map(tag, parent))
        labels = sorted(parent_counts)
        targets.append(dict(zip(labels, largest_remainders(count, [parent_counts[k] for k in labels]))))
        current.append(Counter(map(tag, picked)))

    def gain(a, b):
        delta = 0
        for i, (ka, kb) in enumerate(zip(a, b)):
            if ka == kb:
                continue
            err_a = current[i][ka] - targets[i][ka]
            err_b = current[i][kb] - targets[i][kb]
            delta += 2 * (err_a - err_b - 1)
        return delta

    while any(current[i][k] != v for i, part in enumerate(targets) for k, v in part.items()):
        best_gain, pair = 0, None
        for a, (present, _) in candidates.items():
            if not present:
                continue
            for b, (_, available) in candidates.items():
                if not available:
                    continue
                improved = gain(a, b)
                if improved > best_gain:
                    best_gain, pair = improved, (a, b)
        if pair is None:
            break
        a, b = pair
        picked_a, other_a = candidates[a]
        picked_b, other_b = candidates[b]
        other_a.append(picked_a.pop(rng.randrange(len(picked_a))))
        picked_b.append(other_b.pop(rng.randrange(len(other_b))))
        for i, (ka, kb) in enumerate(zip(a, b)):
            current[i][ka] -= 1
            current[i][kb] += 1
    return [row for picked_rows, _ in candidates.values() for row in picked_rows]


def sample_scenario(parent, counts, bucket_s, rng):
    groups = defaultdict(list)
    for row in parent:
        groups[row["origin"], row["dest"]].append(row)
    levels = (lambda row: int(row["departure"] // bucket_s), lambda row: row["sumo_lane"],
              lambda row: row["length_category"])
    return [row for key, count in counts.items()
            for row in rebalance(groups[key], sample_group(groups[key], count, levels, rng), levels, rng)]


def write_route(path, rows, types):
    root = ET.Element("routes")
    used = {row["type_id"]: row for row in rows}
    for key, row in sorted(used.items()):
        details = types[row["kind"]]
        attrs = {k: v for k, v in details.items() if k not in ("pcu", "paper_type")}
        element = ET.SubElement(root, "vType", id=key, length=f"{row['length_dm_clean'] / 10:.1f}",
                                speedFactor="1", **attrs)
        ET.SubElement(element, "param", key="observedLengthCategory", value=row["length_category"])
        ET.SubElement(element, "param", key="pcu", value=str(details["pcu"]))
    for row in sorted(rows, key=lambda r: (r["departure"], r["id"])):
        ET.SubElement(root, "trip", id=row["id"], type=row["type_id"], depart=f"{row['departure']:.1f}",
                      departLane=row["sumo_lane"].rsplit("_", 1)[1], departPos="base",
                      departSpeed=row["speed_mps_clean"], **{"from": row["origin"], "to": row["dest"]})
    write_xml(path, root)


def write_sumocfg(path, route, net, config):
    import os
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=os.path.relpath(net, path.parent))
    ET.SubElement(inputs, "route-files", value=route.name)
    timing = ET.SubElement(root, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "step-length", value=str(config["step_s"]))
    ET.SubElement(timing, "end", value=str(config["drain_end_s"]))
    processing = ET.SubElement(root, "processing")
    for key, answer in (("lateral-resolution", config["lateral_resolution_m"]),
                        ("collision.check-junctions", "true"), ("collision.action", "remove"),
                        ("time-to-teleport", "-1"), ("time-to-teleport.highways", "-1"),
                        ("time-to-teleport.disconnected", "-1"), ("time-to-teleport.bidi", "-1"),
                        ("time-to-teleport.ride", "-1"), ("max-depart-delay", "-1"),
                        ("sloppy-insert", "false")):
        ET.SubElement(processing, key, value=str(answer))
    ET.SubElement(ET.SubElement(root, "random_number"), "seed", value=str(config["seed"]))
    write_xml(path, root)


def allocation_diagnostics(rows, baseline, bucket_s):
    features = {"time_bin": lambda r: str(int(r["departure"] // bucket_s)), "entry_lane": lambda r: r["sumo_lane"],
                "length_category": lambda r: r["length_category"]}
    result = {}
    for od in [(o, d) for o in ORIGINS for d in DESTS]:
        a, b = [[row for row in group if (row["origin"], row["dest"]) == od] for group in (rows, baseline)]
        result["".join(od)] = {}
        for k, key in features.items():
            child, parent = Counter(map(key, a)), Counter(map(key, b))
            delta = max((abs(child[x] / len(a) - parent[x] / len(b)) for x in parent), default=0)
            result["".join(od)][k] = {"counts": child, "hard_counts": parent, "max_abs_share_delta": delta}
    return result


def write_provenance(path, rows, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    source_fields = ("date", "period", "origin", "entrance_detector", "observation_id", "entrance_time_source",
                     "logical_entrance_lane", "matched_csv_lane", "speed_raw_kmh", "length_raw_dm",
                     "length_m_unmodified", "length_category", "match_status", "destination_status",
                     "matched_exit_detector", "matched_exit_time_source")
    derived_fields = ("id", "departure", "sumo_lane", "length_dm_clean", "speed_mps_clean", "type_id",
                      "dest", "dest_source", "cleaning")
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            writer = csv.writer(text)
            writer.writerow([*source_fields, *derived_fields, "in_easy", "in_medium", "in_hard"])
            for row in rows:
                writer.writerow([*(row[k] for k in (*source_fields, *derived_fields)),
                                 *(int(row["id"] in ids[k]) for k in ("Easy", "Medium", "Hard"))])


def generate(config_path, audit_dir, output, provenance):
    config = read_json(config_path)
    ref_path, recovery_path, observations_path = (audit_dir / name for name in
        ("reference-demand.json", "recovery.json", "entry-observations.csv.gz"))
    ref, recovery = read_json(ref_path), read_json(recovery_path)
    demand = targets(ref, recovery, config)
    if (config["sumo_version"], config["step_s"], config["road_limit_kmh"]) != ("1.18.0", .1, 120):
        raise ValueError("Incompatible SUMO version, step length or road speed limit")
    if not 0 < config["warmup_end_s"] < config["demand_end_s"] < config["drain_end_s"]:
        raise ValueError("Invalid experiment boundaries")
    net = SUMO / "Antwerp.net.xml"
    check_entrances(ET.parse(net).getroot(), config["entrances"], config["road_limit_kmh"])
    baseline = config_path.parent / config["vehicle_types_from"]
    types = read_json(baseline)["vehicle_types"]
    types["light_unresolved"] = {**types["car"], "paper_type": "Car/Van 待判定"}
    rows = read_entries(observations_path, recovery, config)
    rng = random.Random(config["seed"])
    assign_destinations(rows, demand["Hard"], ref["observed_assignment"], config, rng)
    repairs = clean(rows, config, types, config["road_limit_kmh"])
    medium = sample_scenario(rows, demand["Medium"], config["time_stratum_s"], rng)
    easy = sample_scenario(medium, demand["Easy"], config["time_stratum_s"], rng)
    cases = {"Easy": easy, "Medium": medium, "Hard": rows}
    ids = {key: {row["id"] for row in group} for key, group in cases.items()}
    assert ids["Easy"] < ids["Medium"] < ids["Hard"]
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, group in cases.items():
        observed = Counter((row["origin"], row["dest"]) for row in group)
        assert len(ids[name]) == len(group) == config["scenarios"][name] and observed == demand[name]
        route = output / f"Antwerp_observed_{name.lower()}.rou.xml"
        scene = output / f"Antwerp_observed_{name.lower()}.sumocfg"
        write_route(route, group, types)
        write_sumocfg(scene, route, net, config)
        result[name] = {"target": len(group), "od": [[observed[o, d] for d in DESTS] for o in ORIGINS],
                        "destination_sources": Counter(row["dest_source"] for row in group),
                        "length_categories": Counter(row["length_category"] for row in group),
                        "route": route.name, "route_sha256": digest(route),
                        "config": scene.name, "config_sha256": digest(scene),
                        "stratification": allocation_diagnostics(group, rows, config["time_stratum_s"])}
    write_provenance(provenance, rows, ids)
    write_json(output / "manifest.json", {
        "origin_order": ORIGINS, "destination_order": DESTS, "config_sha256": digest(config_path),
        "source_sha256": {path.name: digest(path) for path in (ref_path, recovery_path, observations_path)},
        "vehicle_setup_source_sha256": digest(baseline), "provenance_sha256": digest(provenance),
        "network_sha256": digest(net), "seed": config["seed"], "reference": config["entry_reference"],
        "entrance_mapping": config["entrances"], "unresolved_detector": config["unresolved_exit_detector"],
        "length_speed_repairs": repairs,
        "cleaning_flags_hard": Counter(flag for row in rows for flag in row["cleaning"].split("|") if flag),
        "statistics_windows_s": {"warmup": [0, config["warmup_end_s"]],
                                  "main": [config["warmup_end_s"], config["demand_end_s"]],
                                  "drain": [config["demand_end_s"], config["drain_end_s"]]},
        "scenarios": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    generate(args.config, args.audit_dir, args.output.resolve(), args.provenance)


if __name__ == "__main__":
    main()
