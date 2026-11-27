"""Run the existing Antwerp E1 loops in an isolated background-validation directory."""

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


def prepare_detector_input(template, output_dir):
    """Copy the detector template, redirecting only the output file attributes."""
    output_dir = Path(output_dir).resolve()
    template = Path(template)
    tree = ET.parse(template)
    loops = tree.getroot().findall("inductionLoop")
    filenames = [Path(loop.attrib["file"]).name for loop in loops]
    if not filenames or len(filenames) != len(set(filenames)) or template.name in filenames:
        raise ValueError("Detector output file names must be distinct and separate from the template")

    # A repeated run must choose a new directory, even if its seed differs.
    output_dir.mkdir(parents=True, exist_ok=False)
    for loop, name in zip(loops, filenames, strict=True):
        loop.set("file", str(output_dir / name))
    run_input = output_dir / template.name
    tree.write(run_input, encoding="utf-8", xml_declaration=True)
    return run_input


def validate_background(route_file, detector_output_dir, seed=42, duration_s=600):
    from .sumo_world import SumoWorld

    if duration_s < 300 or duration_s % 300:
        raise ValueError("Detector validation must span one or more complete 300-second intervals")
    route_file = Path(route_file).resolve()
    traffic = ET.parse(route_file).getroot()
    if any(item.get("id") in ("ego", "ego_1", "ego_2") or item.get("type") == "ego"
           for tag in ("trip", "vehicle", "flow") for item in traffic.findall(tag)):
        raise ValueError("Detector validation requires background traffic without Ego")
    world = SumoWorld(route_file=route_file, seed=seed, detector_output_dir=detector_output_dir)
    try:
        conn = world.start()
        if not conn.getVersion()[1].endswith("1.18.0"):
            raise RuntimeError(f"Expected SUMO 1.18.0, got {conn.getVersion()[1]}")
        definitions = {loop.get("id"): loop for loop in ET.parse(world.detector_additional_file).getroot().findall(
            "inductionLoop")}
        if len(definitions) != 44 or set(definitions) != set(conn.inductionloop.getIDList()):
            raise RuntimeError("TraCI detector IDs do not match the 44 Antwerp E1 definitions")
        detectors = {}
        for key, loop in definitions.items():
            lane = conn.inductionloop.getLaneID(key)
            position = conn.inductionloop.getPosition(key)
            if lane != loop.get("lane") or not math.isclose(position, float(loop.get("pos")), abs_tol=1e-6):
                raise RuntimeError(f"TraCI detector location differs from the template for {key}")
            if float(loop.get("period")) != 300 or not 0 <= position <= conn.lane.getLength(lane):
                raise RuntimeError(f"Unexpected detector interval or position for {key}")
            detectors[key] = {"lane": lane, "pos_m": position, "file": Path(loop.get("file")).name}
        for _ in range(round(duration_s / world.step_length)):
            world.step()
    finally:
        world.close()

    expected_intervals = {(begin, begin+300) for begin in range(0, duration_s, 300)}
    total_passages = 0
    for key, detector in detectors.items():
        file = world.detector_additional_file.parent / detector["file"]
        intervals = ET.parse(file).getroot().findall("interval")
        if any(row.get("id") != key for row in intervals):
            raise RuntimeError(f"Detected output from another loop in {file}")
        actual = {(float(row.get("begin")), float(row.get("end"))) for row in intervals}
        if not expected_intervals.issubset(actual):
            raise RuntimeError(f"Missing complete 300-second interval in {file}")
        passages = sum(int(row.get("nVehContrib")) for row in intervals
                       if (float(row.get("begin")), float(row.get("end"))) in expected_intervals)
        detector.update(complete_intervals=len(expected_intervals), vehicle_passages=passages)
        total_passages += passages
    if total_passages == 0:
        raise RuntimeError("No observed detector passages in the validation window")
    report = {"sumo_version": "1.18.0", "seed": seed, "duration_s": duration_s,
              "period_s": 300, "detector_count": len(detectors), "total_detector_passages": total_passages,
              "detectors": detectors}
    (world.detector_additional_file.parent / "detector-validation.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-s", type=int, default=600)
    args = parser.parse_args()
    validate_background(args.route, args.output_dir, args.seed, args.duration_s)


if __name__ == "__main__":
    main()
