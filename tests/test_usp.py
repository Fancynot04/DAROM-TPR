from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rewards.usp import USPTracker


def make_conn():
    vehicle = MagicMock()
    lane = MagicMock()
    simulation = MagicMock()

    vehicle.getSpeed.return_value = 0.0
    vehicle.getRoadID.return_value = "O3"
    vehicle.getLaneID.return_value = "O3_0"
    vehicle.getLanePosition.return_value = 20.0
    vehicle.getLeader.return_value = None

    lane.getLength.return_value = 100.0
    simulation.getDeltaT.return_value = 0.1

    return SimpleNamespace(
        vehicle=vehicle,
        lane=lane,
        simulation=simulation,
    )


def test_usp_grace_period():
    conn = make_conn()
    usp = USPTracker(("ego_1", "ego_2"))

    for _ in range(50):
        assert usp.step(conn, "ego_1") == 0.0

    assert usp.step(conn, "ego_1") == -0.02

    # 两辆 Ego 的计数互相独立
    assert usp.streak["ego_2"] == 0


def test_usp_not_penalize_blocked_vehicle():
    conn = make_conn()
    usp = USPTracker(("ego_1", "ego_2"))

    conn.vehicle.getLeader.return_value = (
        "front", 1.0
    )
    conn.vehicle.getSecureGap.return_value = 3.0
    conn.vehicle.getMinGap.return_value = 2.5

    for _ in range(100):
        assert usp.step(conn, "ego_1") == 0.0


def test_usp_resets_after_moving():
    conn = make_conn()
    usp = USPTracker(("ego_1", "ego_2"))

    for _ in range(60):
        usp.step(conn, "ego_1")

    conn.vehicle.getSpeed.return_value = 2.0

    assert usp.step(conn, "ego_1") == 0.0
    assert usp.streak["ego_1"] == 0