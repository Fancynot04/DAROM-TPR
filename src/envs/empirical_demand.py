"""Build and audit the three standalone Antwerp Empirical V1 scenarios."""

import argparse
from collections import Counter
from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random
from xml.etree import ElementTree as ET

SUMO = Path(__file__).resolve().parents[2] / "sumo_new"
DEFAULT_DIR = SUMO / "empirical_v1"
DESTS = ("D1", "D2", "D3")
KINDS = ("motorbike", "car", "van", "rigid_lorry", "trailer_bus")
LENGTH_GROUPS = ("motorbike", "car_confirmed", "light_overlap", "rigid_lorry", "trailer_bus")
COHORTS = ("all_days", "weekdays", "high_quality")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def ratio(a, b):
    return float(Fraction(a, b)) if b else None


def length_group(length):
    if length <= 0:
        return "invalid"
    for upper, group in zip((10, 40, 69, 120), LENGTH_GROUPS):
        if length <= upper:
            return group
    return "trailer_bus"


def weekday(day):
    return date.fromisoformat(day).weekday() < 5


def is_selected(cohort, window, threshold):
    if cohort == "weekdays":
        return weekday(window["date"])
    if cohort == "high_quality":
        return Fraction(window["assigned_od"], window["raw_unique_observations"]) >= threshold
    return True


def counts_from_audit(row, types):
    groups = {"motorbike": "motorbike", "car_confirmed": "car", "rigid_lorry": "rigid_lorry",
              "trailer_bus": "trailer_bus"}
    counts = {group: row["length_rule_type_counts"][types[kind]["paper_type"]]
              for group, kind in groups.items()}
    counts.update(light_overlap=row["undetermined_car_or_van"], invalid=row["invalid_length"])
    if sum(counts.values()) != row["matched_count"]:
        raise ValueError(f"Invalid match count for {row['date']} {row['origin']}→{row['destination']}")
    return counts


def extract_inputs(audit_dir, config):
    """Freeze only aggregate audit evidence. The generation path uses stdlib only."""
    import numpy as np

    audit_file = audit_dir / "audit-data.json"
    paper_file = audit_dir / "paper-tables.json"
    audit, paper = read_json(audit_file), read_json(paper_file)
    windows = []
    pools = {group: Counter() for group in LENGTH_GROUPS if group != "motorbike"}
    quality = {}
    policy = config["length_policy"]
    for day in sorted(audit["raw"]):
        day_path = audit_dir / f"raw-{day}.npz"
        with np.load(day_path, allow_pickle=False) as source:
            raw = dict(source)
        start = int(np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}T11:30", "ns").astype("int64"))
        hour = 3600 * 10**9
        mask = (start <= raw["time_ns"]) & (raw["time_ns"] < start + hour)
        quality[day] = Counter()
        for origin, detectors in audit["detectors"]["origin"].items():
            window = next(row for row in audit["origin_audit"]
                          if (row["date"], row["period"], row["origin"]) == (day, "P1", origin))
            matched = {
                row["destination"]: counts_from_audit(row, config["vehicle_types"])
                for row in audit["od_matrix"]
                if (row["date"], row["period"], row["origin"]) == (day, "P1", origin)
            }
            entrants, unique = Counter(), set()
            previous = {}
            at_origin = np.flatnonzero(mask & np.isin(raw["detector"], detectors))
            for i in at_origin:
                det, obs, time = (int(raw[column][i]) for column in ("detector", "obs", "time_ns"))
                key = (det, obs, time)
                if key in unique:
                    raise ValueError(f"Unexpected duplicate raw observation at {day} {origin}")
                unique.add(key)
                length, speed, headway = (int(raw[column][i]) for column in
                                          ("length_raw", "speed_raw", "headway_raw"))
                group = length_group(length)
                entrants[group] += 1
                q = quality[day]
                q["entry_observations"] += 1
                boundary_speed = speed == policy["exclude_speed_code_from_physical_length"]
                boundary_length = length == policy["length_boundary_raw"]
                q["speed_254"] += boundary_speed
                q["length_255"] += boundary_length
                q["joint_speed_254_length_255"] += boundary_speed and boundary_length
                prev = previous.get(det)
                if headway == policy["headway_censor_candidate_raw"]:
                    q["headway_3001"] += 1
                    if prev is not None and prev[0] + 1 == obs:
                        q["headway_3001_with_preceding_obs"] += 1
                        q["headway_3001_preceding_gap_ge_30s"] += time - prev[1] >= 30 * 10**9
                previous[det] = (obs, time)
                if weekday(day) and group in pools:
                    bottom, top = policy[group + "_dm"]
                    if bottom <= length <= top and not boundary_speed and not boundary_length:
                        pools[group][length] += 1
                    else:
                        q[f"excluded_from_physical_{group}"] += 1
            if len(unique) != window["raw_unique_observations"]:
                raise ValueError(f"Raw data does not reproduce the audit denominator for {day} {origin}")
            windows.append({**window, "raw_by_length_group": dict(entrants), "matched_by_destination": matched})
    return {
        "source": {
            "paper": audit["paper"], "archive": audit["archive"],
            "paper_tables_sha256": digest(paper_file), "audit_sha256": digest(audit_file),
            "raw_subset_sha256": {day: digest(audit_dir / f"raw-{day}.npz") for day in sorted(audit["raw"])},
            "raw_members": {day: info["member"] for day, info in audit["raw"].items()},
            "match_members": {day: {key: value["member"] for key, value in info["stage_sources"].items()}
                              for day, info in audit["date_evidence"].items()},
            "detectors": audit["detectors"]
        },
        "paper_tables": paper, "windows": windows,
        "physical_lengths_dm": {key: sorted(counts.items()) for key, counts in pools.items()},
        "p1_boundary_codes_by_day": quality
    }


def largest_remainders(total, weights):
    """Allocate an integer total with stable input order as the final tie-break."""
    denom = sum(weights)
    if total < 0 or denom <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("Apportionment requires a non-negative total and positive total weight")
    quotas = [Fraction(weight) * total / denom for weight in weights]
    counts = [quota.numerator // quota.denominator for quota in quotas]
    priority = sorted(range(len(counts)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in priority[:total - sum(counts)]:
        counts[i] += 1
    return counts


def aggregate(windows, dest):
    raw, matched = Counter(), Counter()
    for window in windows:
        raw.update(window["raw_by_length_group"])
        for key in (DESTS if dest is None else (dest,)):
            matched.update(window["matched_by_destination"][key])
    return raw, matched


def split_overlap(counts, van_share):
    return [Fraction(counts["motorbike"]),
            counts["car_confirmed"] + counts["light_overlap"] * (1 - van_share),
            counts["light_overlap"] * van_share,
            Fraction(counts["rigid_lorry"]), Fraction(counts["trailer_bus"])]


def sensitivity(inputs, config):
    period = config["paper_period"]
    index = ("P1", "P2").index(period)
    paper = inputs["paper_tables"]
    recorded = {origin + dest: counts[index] for origin, dest, counts, *_ in paper["table_2"]}
    t3 = {dest: [row[j][index] for row in paper["table_3"]] for j, dest in enumerate(DESTS, 1)}
    if any(window["period"] != period for window in inputs["windows"]):
        raise ValueError("The recorded P1 input windows do not match the configured period")
    threshold = Fraction(config["composition"]["high_quality_min_assignment_rate"])
    delta = Fraction(config["composition"]["van_prior_sensitivity_delta"])
    result, bias = {}, {}
    for cohort in COHORTS:
        chosen = [w for w in inputs["windows"] if is_selected(cohort, w, threshold)]
        result[cohort], bias[cohort] = {}, {}
        for origin in ("O1", "O2", "O3"):
            windows = [w for w in chosen if w["origin"] == origin]
            if not windows:
                raise ValueError(f"No windows for {cohort} {origin}")
            raw, assigned = aggregate(windows, None)
            bias[cohort][origin] = {
                "selected_dates": [w["date"] for w in windows],
                "raw": raw, "assigned": assigned,
                "assignment_rates_by_length_group": {g: ratio(assigned[g], raw[g]) for g in LENGTH_GROUPS},
                "raw_share_by_group": {g: ratio(raw[g], raw.total()) for g in LENGTH_GROUPS},
                "assigned_share_by_group": {g: ratio(assigned[g], assigned.total()) for g in LENGTH_GROUPS}
            }
            for dest, source in t3.items():
                prior = Fraction(source[2], source[1] + source[2])
                _, counts = aggregate(windows, dest)
                inferred = split_overlap(counts, prior)
                adjusted = [split_overlap(counts, max(Fraction(0), prior - delta)),
                            split_overlap(counts, min(Fraction(1), prior + delta))]
                token = origin + dest
                basis = source if origin == "O1" else inferred
                if not sum(basis):
                    raise ValueError(f"No source composition for {cohort} {token}")
                result[cohort][token] = {
                    "paper_od_total": recorded[token],
                    "paper_type_counts": dict(zip(KINDS, source)) if origin == "O1" else None,
                    "observed": counts, "matched_sample_size": counts.total(),
                    "van_prior_on_overlap": float(prior),
                    "estimated_overlap_car_count": float(counts["light_overlap"] * (1 - prior)),
                    "estimated_overlap_van_count": float(counts["light_overlap"] * prior),
                    "estimate_weights": dict(zip(KINDS, map(str, basis))),
                    "shares": dict(zip(KINDS, [ratio(v, sum(basis)) for v in basis])),
                    "overlap_prior_sensitivity": [
                        dict(zip(KINDS, [ratio(v, sum(weights)) for v in weights])) for weights in adjusted
                    ],
                    "composition_source": "Table 3" if origin == "O1" else "P1 matched entries + Table 3 Van prior",
                }
    return result, bias


def profile_for(kind, car_short_fraction, pools, defaults):
    if kind == "motorbike":
        return [(defaults["motorbike_assumed_m"], Fraction(1))]
    if kind == "car":
        sources = (("car_confirmed", car_short_fraction), ("light_overlap", 1 - car_short_fraction))
    elif kind == "van":
        sources = (("light_overlap", Fraction(1)),)
    else:
        sources = ((kind, Fraction(1)),)
    return [(dm / 10, mass * Fraction(freq, sum(item[1] for item in pools[group])))
            for group, mass in sources if mass for dm, freq in pools[group]]


def make_vtype(root, name, kind, length, parameters):
    attrs = {key: val for key, val in parameters.items() if key not in ("paper_type", "pcu")}
    node = ET.SubElement(root, "vType", id=name, length=f"{length:.1f}", speedFactor="1", **attrs)
    ET.SubElement(node, "param", key="empiricalVehicleClass", value=kind)
    ET.SubElement(node, "param", key="pcu", value=str(parameters["pcu"]))


def append_ego(root, config, template):
    ego = config["ego"]
    ET.SubElement(root, "trip", id="ego", type=ego["type"], depart=f"{ego['depart_s']:.1f}",
                  departLane=str(ego["depart_lane"]), departSpeed=str(ego["depart_speed_mps"]),
                  speedFactor="1.0", color="red", **{"from": template.get("from"), "to": template.get("to")})


def make_route(config, name, matrix, primary, inputs, net):
    rng = random.Random(config["seed"])
    historical = ET.parse(net.parent / "Antwerp_easy_o3_d3.rou.xml").getroot()
    root = ET.Element("routes")
    root.append(ET.Comment("Antwerp Empirical V1. background trips from Period 1; RNG seed " + str(config["seed"])))
    root.append(historical.find("./vType[@id='ego']"))
    records, used = [], {}
    for token, counts in matrix.items():
        observed = primary[token]["observed"]
        from_overlap = observed.get("light_overlap", 0) * (1 - Fraction(str(primary[token]["van_prior_on_overlap"])))
        short = Fraction(observed.get("car_confirmed", 0))
        short_fraction = short / (short + from_overlap) if short + from_overlap else Fraction(0)
        for kind, count in counts.items():
            if not count:
                continue
            profile = profile_for(kind, short_fraction, inputs["physical_lengths_dm"], config["length_policy"])
            lengths, weights = zip(*profile)
            for length in rng.choices(lengths, weights=weights, k=count):
                name_id = f"{kind}_l{round(length * 10):03d}"
                used[name_id] = (kind, length)
                records.append((token, kind, name_id))
    for name_id, (kind, length) in sorted(used.items()):
        make_vtype(root, name_id, kind, length, config["vehicle_types"][kind])
    rng.shuffle(records)
    duration_ticks = round(config["duration_s"] / config["step_length_s"])
    if len(records) > duration_ticks:
        raise ValueError("The explicit departure grid would contain duplicate ticks")
    for i, (token, kind, name_id) in enumerate(records):
        tick = (2 * i + 1) * duration_ticks // (2 * len(records))
        depart = tick * config["step_length_s"]
        ET.SubElement(root, "trip", id=f"{name}_{token}_{kind}_{i:05d}", type=name_id,
                      depart=f"{depart:.1f}", departLane="free", departSpeed="max",
                      **{"from": token[:2], "to": token[2:]})
    append_ego(root, config, historical.find("./trip[@id='ego']"))
    trips = [node for node in root if node.tag == "trip"]
    for node in trips:
        root.remove(node)
    for node in sorted(trips, key=lambda t: float(t.get("depart"))):
        root.append(node)
    ET.indent(root, space="  ")
    return root


def write_xml(path, element):
    ET.indent(element, space="  ")
    ET.ElementTree(element).write(path, encoding="utf-8", xml_declaration=True)


def generate(config_path, inputs_path, out):
    config, inputs = read_json(config_path), read_json(inputs_path)
    if (config["sumo_version"], config["step_length_s"], config["ego"]["speed_mode"],
            config["ego"]["lane_change_mode"]) != ("1.18.0", .1, 32, 0):
        raise ValueError("Empirical V1 requires SUMO 1.18.0, 0.1 s steps, SpeedMode=32 and LaneChangeMode=0")
    cap = Fraction(config["speed_cap_mps"])
    if not 0 < cap <= Fraction(100, 3) or any(
        not 0 < Fraction(vehicle["maxSpeed"]) <= cap for vehicle in config["vehicle_types"].values()
    ):
        raise ValueError("Empirical V1 background vehicle limits must be positive and at most 120 km/h")
    sensitivity_result, bias = sensitivity(inputs, config)
    primary = sensitivity_result[config["composition"]["primary"]]
    weights = [row["paper_od_total"] for row in primary.values()]
    out.mkdir(parents=True, exist_ok=True)
    net = SUMO / "Antwerp.net.xml"
    manifest = {"config_sha256": digest(config_path), "inputs_sha256": digest(inputs_path),
                "network_sha256": digest(net), "seed": config["seed"], "target_period": config["paper_period"],
                "cohorts": sensitivity_result, "matching_bias": bias,
                "p1_boundary_codes_by_day": inputs["p1_boundary_codes_by_day"], "scenarios": {}}
    for name, spec in config["scenarios"].items():
        scaled = sum(weights) * Fraction(spec["scale"])
        if int(scaled + Fraction(1, 2)) != spec["target"]:
            raise ValueError(f"{name}: target does not equal rounded scaled P1 demand")
        od_totals = largest_remainders(spec["target"], weights)
        matrix = {
            token: dict(zip(KINDS, largest_remainders(n, [Fraction(row["estimate_weights"][k]) for k in KINDS])))
            for n, (token, row) in zip(od_totals, primary.items())
        }
        filename = f"Antwerp_empirical_v1_{name}.rou.xml"
        write_xml(out / filename, make_route(config, name, matrix, primary, inputs, net))
        sumocfg = ET.Element("configuration")
        section = ET.SubElement(sumocfg, "input")
        ET.SubElement(section, "net-file", value="../Antwerp.net.xml")
        ET.SubElement(section, "route-files", value=filename)
        times = ET.SubElement(sumocfg, "time")
        ET.SubElement(times, "step-length", value=str(config["step_length_s"]))
        ET.SubElement(times, "end", value=str(config["duration_s"]))
        processing = ET.SubElement(sumocfg, "processing")
        ET.SubElement(processing, "lateral-resolution", value=str(config["lateral_resolution_m"]))
        ET.SubElement(processing, "collision.check-junctions", value="true")
        ET.SubElement(processing, "collision.action", value="remove")
        ET.SubElement(processing, "time-to-teleport", value="-1")
        random_number = ET.SubElement(sumocfg, "random_number")
        ET.SubElement(random_number, "seed", value=str(config["seed"]))
        write_xml(out / f"Antwerp_empirical_v1_{name}.sumocfg", sumocfg)
        old_route = SUMO / f"Antwerp_{name}_o3_d3.rou.xml"
        old_counts = Counter()
        for old in ET.parse(old_route).getroot().findall("flow"):
            old_counts[old.get("from") + old.get("to")] += int(old.get("number"))
        manifest["scenarios"][name] = {
            "scale": spec["scale"], "background_total": spec["target"], "od_type_counts": matrix,
            "rounding": {token: {
                "paper_target_scaled": float(primary[token]["paper_od_total"] * Fraction(spec["scale"])),
                "od_integer_total": sum(counts.values()),
                "type_expected": {k: sum(counts.values()) * primary[token]["shares"][k] for k in KINDS},
                "type_delta_from_expected": {k: counts[k] - sum(counts.values()) * primary[token]["shares"][k]
                                             for k in KINDS}
            } for token, counts in matrix.items()},
            "route": filename, "route_sha256": digest(out / filename),
            "historical": {"route": old_route.name, "background_total": old_counts.total(), "od_totals": old_counts}
        }
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_DIR / "config.json")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_DIR / "inputs.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--extract-audit", type=Path, help="Directory with the result of the raw traffic data audit")
    args = parser.parse_args()
    if args.extract_audit:
        if args.inputs.exists():
            parser.error("--extract-audit requires a fresh --inputs path")
        write_json(args.inputs, extract_inputs(args.extract_audit, read_json(args.config)))
    generate(args.config, args.inputs, args.output)


if __name__ == "__main__":
    main()
