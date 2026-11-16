
from unittest.mock import Mock

from src.envs.lane_change import ExitLaneChangeTracker
from src.envs.control import EgoController


def make_conn():
    conn = Mock()

    conn.simulation.getTime.return_value = 100.0
    conn.simulation.getDeltaT.return_value = 0.1

    conn.vehicle.getIDList.return_value = ["ego"]
    conn.vehicle.getRoadID.return_value = "E77"
    conn.vehicle.getLaneIndex.return_value = 1
    conn.vehicle.getLaneID.return_value = "E77_1"
    conn.vehicle.getSpeed.return_value = 20.0
    conn.vehicle.getLateralSpeed.return_value = 0.0

    conn.edge.getLaneNumber.return_value = 5

    conn.lane.getWidth.return_value = 3.2
    conn.vehicle.getWidth.return_value = 2.0
    conn.vehicle.getLateralLanePosition.return_value = 0.0

    return conn


def test_request_is_not_actual_motion():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    assert not tracker.active_to_zero(conn, "ego")


def test_rightward_motion_is_active():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    conn.simulation.getTime.return_value = 100.1
    conn.vehicle.getLateralSpeed.return_value = -0.3

    tracker.observe(conn, "ego")

    assert tracker.active_to_zero(conn, "ego")


def test_wrong_direction_not_active():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    conn.simulation.getTime.return_value = 100.1
    conn.vehicle.getLateralSpeed.return_value = 0.3

    tracker.observe(conn, "ego")

    assert not tracker.active_to_zero(conn, "ego")


def test_repeated_request_keeps_start_time():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    conn.simulation.getTime.return_value = 101.0
    tracker.request(conn)

    assert tracker.state.start_time == 100.0


def test_stalled_maneuver_expires():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    conn.simulation.getTime.return_value = 100.1
    conn.vehicle.getLateralSpeed.return_value = -0.3
    tracker.observe(conn, "ego")

    conn.simulation.getTime.return_value = 102.2
    conn.vehicle.getLateralSpeed.return_value = 0.0
    tracker.observe(conn, "ego")

    assert tracker.state is None
    assert not tracker.active_to_zero(conn, "ego")


def test_stable_target_lane_completes():
    conn = make_conn()
    tracker = ExitLaneChangeTracker()

    tracker.request(conn)

    conn.vehicle.getLaneIndex.return_value = 0
    conn.vehicle.getLaneID.return_value = "E77_0"

    for t in (100.1, 100.2, 100.3):
        conn.simulation.getTime.return_value = t
        tracker.observe(conn, "ego")

    assert tracker.state is None


def test_controller_records_e77_request():
    conn = make_conn()
    controller = EgoController(conn, "ego")

    result = controller.apply_action([0.0, -5.0])

    assert result.target_lane == 0
    assert controller.exit_change.state is not None

    conn.simulation.getTime.return_value = 100.1
    conn.vehicle.getLateralSpeed.return_value = -0.3

    controller.observe_after_step()

    assert controller.active_change_to_zero

    conn.simulationStep.assert_not_called()
