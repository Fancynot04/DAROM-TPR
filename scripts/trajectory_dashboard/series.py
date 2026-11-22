"""Map recorded transitions to explicit pre-step, post-step and terminal samples."""

import math


def _xy(position):
    values = (position or {}).get("xy_m")
    if isinstance(values, list) and len(values) == 2 and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in values
    ):
        return values
    return None


def episode_series(records, ego, step_length_s):
    """Retain every episode step so a shared cursor also works before either Ego departs."""
    anchors = (
        (row["step"], obj["metrics"]["post_time"])
        for row in records for obj in row["egos"].values()
        if obj.get("metrics") and obj["metrics"].get("post_time") is not None
    )
    anchor = next(anchors, None)
    distance = 0.0
    points = []
    for row in records:
        obj = row["egos"][ego]
        metric = obj.get("metrics") or {}
        pre = metric.get("pre") or {}
        control = obj.get("control") or {}
        action = obj.get("action")
        acted = metric.get("agent_step") is True
        terminal = obj.get("terminal")
        pre_time = pre.get("time")
        clock = metric.get("post_time")
        if clock is None:
            clock = next((m["post_time"] for other in row["egos"].values()
                          if (m := other.get("metrics")) and m.get("post_time") is not None), None)
        inferred = clock is None and anchor is not None
        if inferred:
            clock = round(anchor[1] + (row["step"] - anchor[0]) * step_length_s, 6)
        step_start = pre_time if pre_time is not None else (
            round(clock - step_length_s, 6) if clock is not None else None
        )
        terminal_time = (clock if acted else step_start) if terminal else None
        pre_xy, post_xy = _xy(obj.get("pre_position")), _xy(obj.get("post_position"))
        has_post = acted and post_xy is not None and metric.get("post_road") is not None
        pre_distance = distance if pre_xy is not None else None
        previous = points[-1] if points else None
        pre_remaining = None
        if (previous is not None and pre_xy is not None and pre_time is not None
                and previous["position_source"] == "步末"
                and previous["position_time"] is not None
                and math.isclose(previous["position_time"], pre_time, rel_tol=0, abs_tol=1e-6)):
            pre_remaining = previous["remaining"]
        displacement = math.dist(pre_xy, post_xy) if acted and pre_xy is not None and post_xy is not None else None
        if displacement is not None:
            distance += displacement
        position_distance = distance if has_post or pre_xy is not None else None
        action_time = (pre_time if pre_time is not None else clock) if acted else None
        points.append({
            "step": row["step"], "clock": clock, "step_start": step_start, "action_time": action_time,
            "position_time": metric.get("post_time") if has_post else pre_time,
            "reward_time": terminal_time if terminal else clock, "terminal_time": terminal_time,
            "time_inferred": inferred or bool(terminal and not acted and pre_time is None),
            "terminal": terminal, "acted": acted,
            "road": metric.get("post_road") if has_post else pre.get("road"),
            "lane": metric.get("post_lane") if has_post else pre.get("lane"),
            "pre_road": pre.get("road"), "pre_lane": pre.get("lane"),
            "position_source": "步末" if has_post else "步首" if pre_xy is not None else None,
            "distance": position_distance, "displacement": displacement,
            "pre_distance": pre_distance, "pre_remaining": pre_remaining,
            "remaining": obj.get("remaining_goal_distance_m") if has_post else None,
            "pre_speed": pre.get("speed"), "post_speed": control.get("post_speed_mps") if acted else None,
            "pre_time": pre_time, "post_time": metric.get("post_time"),
            "raw_action": action, "requested_accel": control.get("requested_acceleration_mps2") if acted else None,
            "commanded_accel": control.get("commanded_acceleration_mps2") if acted else None,
            "actual_accel": metric.get("post_acceleration") if acted else None,
            "raw_lateral": action[1] if acted and isinstance(action, list) and len(action) == 2 else None,
            "submitted_direction": pre.get("request_direction") if acted else None,
            "leader_gap_m": control.get("leader_gap_m"),
            "signed_min_lane_changes": obj.get("signed_min_lane_changes"),
            "events": obj.get("lane_changes") or [], "preparation": obj.get("route_preparation"),
            "components": obj.get("components") or {}, "reward": obj.get("reward"),
            "team_reward": row["team_reward"],
        })
    return points


def cursor_point(point, at_step_start=False):
    """Select the time and state of the same transition without interpolating position."""
    at_step_start = at_step_start or bool(point["terminal"] and not point["acted"])
    if not at_step_start:
        return {**point, "cursor_time": point["clock"]}
    return {
        **point, "cursor_time": point["step_start"], "position_time": point["pre_time"],
        "road": point["pre_road"], "lane": point["pre_lane"],
        "distance": point["pre_distance"], "remaining": point["pre_remaining"],
        "position_source": "步首" if point["pre_distance"] is not None else None,
    }


def road_segments(points, x_key):
    """Yield contiguous, actually observed road spans, including internal lanes."""
    spans = []
    for point in points:
        road = point["road"]
        if road is None or point[x_key] is None or point["lane"] is None:
            continue
        if spans and spans[-1][-1]["road"] == road and spans[-1][-1]["step"] + 1 == point["step"]:
            spans[-1].append(point)
        else:
            spans.append([point])
    return spans


def cumulative_rewards(points, gamma=1.0):
    """Discount from this Ego's first recorded transition using shared step offsets."""
    observed = [point for point in points if point["acted"] or point["terminal"]]
    total = 0.0
    first = observed[0]["step"] if observed else 0
    series = []
    for point in observed:
        total += point["reward"] * gamma ** (point["step"] - first)
        series.append((point, total))
    return series
