from types import SimpleNamespace

import pytest

from src.envs.stalled import BLOCK_GRACE_SECONDS, StalledDetector


def samples(detector, speed, stop=101):
    return [detector.observe(i / 10, distance=speed * i / 10) for i in range(stop)]


@pytest.mark.parametrize("speed, expected", [(0, True), (0.6, True), (2, False), (3, False)])
def test_ten_second_window_and_progress_boundary(speed, expected):
    rows = samples(StalledDetector(), speed, 102)
    assert not any(row["would_stall"] for row in rows[:100])
    assert rows[100]["would_stall"] is expected
    assert rows[100]["window_progress_m"] == pytest.approx(speed * 10)
    assert rows[101]["trigger_time_s"] == (10.0 if expected else None)


def test_rolling_window_recovers_and_can_trigger_again():
    detector = StalledDetector()
    assert samples(detector, 0)[-1]["would_stall"]
    row = detector.observe(10.1, distance=20)
    assert not row["would_stall"]
    assert row["reason"] == "progress_sufficient"
    for tick in range(102, 202):
        row = detector.observe(tick / 10, distance=20)
    assert row["trigger_time_s"] == 20.1
    assert row["would_stall"]


@pytest.mark.parametrize("reason", ["inactive", "leader_constrained", "traffic_signal", "junction_conflict"])
def test_ineligible_state_restarts_window(reason):
    detector = StalledDetector()
    samples(detector, 0, stop=100)
    blocked = detector.observe(10, distance=0, unavailable_reason=reason)
    assert not blocked["would_stall"] and blocked["reason"] == reason
    assert blocked["valid_window_s"] == 0
    for tick in range(101, 201):
        assert not detector.observe(tick / 10, distance=0)["would_stall"]
    assert detector.observe(20.1, distance=0)["would_stall"]


def test_discontinuous_time_or_odometer_does_not_carry_stall():
    detector = StalledDetector()
    assert detector.observe(0, distance=float("nan"))["reason"] == "invalid_distance"
    detector.observe(0.1, distance=20)
    assert detector.observe(0.2, distance=19)["reason"] == "discontinuous_sample"
    row = detector.observe(1, distance=19)
    assert row["reason"] == "discontinuous_sample" and row["valid_window_s"] == 0


def test_road_and_junction_names_do_not_reset_odometer_window():
    detector = StalledDetector()
    roads = ["E76", ":J105_0", "E77", ":J106_0", "D3"]
    for i in range(101):
        road = roads[min(i // 20, 4)]
        row = detector.observe(i / 10, road=road, lane=road + "_0", distance=150 + i * .06)
    assert row["would_stall"] and row["road"] == "D3"
    assert row["window_progress_m"] == pytest.approx(6)


@pytest.mark.parametrize("speed", [0, 0.6, 2.0, 3.0])
def test_grace_matches_legacy_at_every_clear_sample(speed):
    legacy = samples(StalledDetector(block_grace_s=0), speed, 210)
    active = samples(StalledDetector(BLOCK_GRACE_SECONDS), speed, 210)
    for old, new in zip(legacy, active):
        assert new["window_progress_m"] == pytest.approx(old["window_progress_m"])
        assert {k: v for k, v in new.items() if k != "window_progress_m"} == {
            k: v for k, v in old.items() if k != "window_progress_m"
        }


def test_repeated_short_blocks_preserve_v4_progress_without_crediting_blocked_motion():
    legacy = StalledDetector(block_grace_s=0)
    active = StalledDetector(BLOCK_GRACE_SECONDS)
    distance = 0.0
    first_trigger = None
    for tick in range(150):
        blocked = tick % 50 < 10 and tick >= 50
        if blocked:
            distance += 1.0
        reason = "leader_constrained" if blocked else None
        assert not legacy.observe(tick / 10, distance=distance, unavailable_reason=reason)["would_stall"]
        row = active.observe(tick / 10, distance=distance, unavailable_reason=reason)
        if tick == 50:
            assert row["valid_window_s"] == 4.9
        if tick == 60:
            assert row["reason"] == "confirming_clear_passage"
        if row["would_stall"] and first_trigger is None:
            first_trigger = row["trigger_time_s"]
    assert first_trigger == 14.0
    assert row["window_progress_m"] == 0.0
    assert active.summary()["pause_count"] == 2
    assert active.summary()["block_duration_histogram_steps"] == {"10": 2}
    assert active.summary()["window_resets"] == 0


def test_long_block_clears_at_threshold_and_moving_recovery_does_not_stall():
    detector = StalledDetector(BLOCK_GRACE_SECONDS)
    samples(detector, 0, stop=50)
    for tick in range(50, 350):
        row = detector.observe(tick / 10, distance=0, unavailable_reason="traffic_signal")
        assert not row["would_stall"]
        if tick == 79:
            assert row["valid_window_s"] == 4.9
        if tick == 80:
            assert row["valid_window_s"] == 0.0
    for tick in range(350, 460):
        row = detector.observe(tick / 10, distance=(tick - 350) * 0.5)
        assert not row["would_stall"]
    assert row["window_progress_m"] == pytest.approx(50)
    summary = detector.summary()
    assert summary["window_resets"] == 1
    assert summary["window_resets_by_reason"] == {"long_block:traffic_signal": 1}
    assert summary["block_duration_histogram_steps"] == {"300": 1}


def test_one_clear_sample_per_second_does_not_accumulate_in_a_yield_queue():
    detector = StalledDetector(BLOCK_GRACE_SECONDS)
    samples(detector, 0, stop=50)
    for tick in range(50, 1600):
        reason = None if tick % 10 == 9 else "junction_conflict"
        assert not detector.observe(tick / 10, distance=0, unavailable_reason=reason)["would_stall"]
    summary = detector.summary()
    assert summary["pause_count"] == 1
    assert summary["window_resets_by_reason"] == {"long_block:junction_conflict": 1}
    assert summary["open_block_observed_s"] == 154.9
    assert summary["raw_block_duration_histogram_steps"] == {"9": 155}


@pytest.mark.parametrize("time,distance,reason,expected", [
    (5.1, 1, None, "discontinuous_sample"),
    (5.0, 0, "leader_constrained", "discontinuous_sample"),
    (5.0, float("nan"), "leader_constrained", "invalid_distance"),
    (5.0, 99, "teleport", "teleport"),
    (5.0, None, "traci_unavailable", "traci_unavailable"),
])
def test_grace_clears_invalid_or_discontinuous_samples(time, distance, reason, expected):
    detector = StalledDetector(BLOCK_GRACE_SECONDS)
    samples(detector, 0, stop=49)
    detector.observe(4.9, distance=1, unavailable_reason="leader_constrained")
    row = detector.observe(time, distance=distance, unavailable_reason=reason)
    assert row["reason"] == expected
    assert not row["would_stall"] and row["valid_window_s"] == 0
    summary = detector.summary()
    assert summary["window_resets_by_reason"] == {expected: 1}
    assert summary["censored_block_duration_histogram_steps"] == {"0": 1}


@pytest.mark.parametrize(
    "leader_gap, leader_speed, link, signal, expected",
    [
        (51, 0, [], [], "collecting_window"),
        (50, 0, [], [], "collecting_window"),
        (10, 0.6, [], [], "leader_constrained"),
        (10, 3, [], [], "collecting_window"),
        (None, 0, [("D3_0", False, False, True, "via", "m", "s", 5)], [], "junction_conflict"),
        (None, 0, [("D3_0", True, True, True, "via", "M", "s", 5)], [], "collecting_window"),
        (None, 0, [], [("signal", 0, 10, "r")], "traffic_signal"),
    ],
)
def test_fixed_leader_range_and_local_passage(leader_gap, leader_speed, link, signal, expected):
    lookaheads = []

    def get_leader(key, dist):
        lookaheads.append(dist)
        return None if leader_gap is None else ("leader", leader_gap - 2.5)

    vehicle = SimpleNamespace(
        getRoadID=lambda key: "D3",
        getLaneID=lambda key: "D3_0",
        getDistance=lambda key: 200.0,
        getMinGap=lambda key: 2.5,
        getLeader=get_leader,
        getSpeed=lambda key: leader_speed,
        getStopState=lambda key: 0,
        getLanePosition=lambda key: 195,
        getNextTLS=lambda key: signal,
        getNextLinks=lambda key: link,
        getJunctionFoes=lambda key, distance: (),
        getRoute=lambda key: ["O3", "D3"],
    )
    conn = SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLength=lambda key: 205))
    row = StalledDetector().sample(conn, "ego_1", 0)
    assert lookaheads == [50]
    assert row["reason"] == expected
    if leader_gap == 51:
        assert row["leader"] is None
    elif leader_gap is not None:
        assert row["leader"]["gap_m"] == leader_gap


@pytest.mark.parametrize(
    "other_dist, other_exit_dist, ego_yields, expected",
    [(3, 8, True, "junction_conflict"), (3, 8, False, None),
     (-1, 4, False, "junction_conflict"), (-6, -1, True, None)],
)
def test_internal_junction_only_blocks_for_relevant_foes(other_dist, other_exit_dist, ego_yields, expected):
    vehicle = SimpleNamespace(
        getStopState=lambda key: 0,
        getLanePosition=lambda key: 6,
        getNextTLS=lambda key: (),
        getJunctionFoes=lambda key, distance: [
            ("foe", 2, other_dist, 7, other_exit_dist, ":J_0_0", ":J_1_0", ego_yields, not ego_yields),
        ],
        getSpeed=lambda key: 4,
    )
    conn = SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLength=lambda key: 10))
    assert StalledDetector.passage_block(conn, "ego", ":J_0", ":J_0_0", None, 2.5) == expected
