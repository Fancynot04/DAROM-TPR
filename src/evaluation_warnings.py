"""Evaluation diagnostics sourced from SUMO's warning log."""

from collections import defaultdict
from pathlib import Path
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NO_CONNECTION = "because there is no connection to the next edge"
_WRONG_LANE_STOP = re.compile(
    rf"Vehicle '(?P<vehicle_id>[^']+)' performs emergency stop at the end of lane "
    rf"'(?P<lane_id>[^']+)' {_NO_CONNECTION} "
    rf"\(decel=(?P<decel_mps2>{_NUMBER}), offset=(?P<offset_m>{_NUMBER})\), time=(?P<time_s>{_NUMBER})\."
)
_EMERGENCY_BRAKING = re.compile(
    rf"Vehicle '(?P<vehicle_id>[^']+)' performs emergency braking on lane "
    rf"'(?P<lane_id>[^']+)' with decel=(?P<decel_mps2>{_NUMBER}), "
    rf"wished=(?P<wished_decel_mps2>{_NUMBER}), severity=(?P<severity>{_NUMBER}), time=(?P<time_s>{_NUMBER})\."
)


def read_sumo_warnings(path, step_length):
    """Return per-vehicle warning categories, or None when the log is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    events = defaultdict(lambda: {"lane_end_blocks": [], "emergency_braking_warnings": []})
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = _WRONG_LANE_STOP.search(line)
            if match:
                warning = _warning_fields(match)
                blocks = events[warning["vehicle_id"]]["lane_end_blocks"]
                time = warning["time_s"]
                if (not blocks or blocks[-1]["lane_id"] != warning["lane_id"]
                        or not 0 <= time - blocks[-1]["last_time_s"] <= step_length + 1e-6):
                    blocks.append({
                        "vehicle_id": warning["vehicle_id"], "lane_id": warning["lane_id"],
                        "first_time_s": time, "last_time_s": time,
                        "warning_count": 0, "warnings": [],
                    })
                block = blocks[-1]
                block["last_time_s"] = time
                block["warning_count"] += 1
                block["warnings"].append(warning)
            elif match := _EMERGENCY_BRAKING.search(line):
                warning = _warning_fields(match)
                events[warning["vehicle_id"]]["emergency_braking_warnings"].append(warning)
            elif _NO_CONNECTION in line and "performs emergency stop" in line:
                # A different SUMO log format cannot establish a clean episode.
                return None
    return dict(events)


def _warning_fields(match):
    return {key: value if key.endswith("_id") else float(value)
            for key, value in match.groupdict().items()}
