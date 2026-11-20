from types import SimpleNamespace

import pytest

from src.envs.stalled import StalledDetector


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
