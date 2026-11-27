
"""测试 DAROM 动作解析与车辆控制。"""

from argparse import ArgumentParser
from unittest.mock import Mock

import numpy as np
import pytest

from src.envs.control import (
    EgoController,
    LaneChangeRequest,
    add_lane_change_request_arguments,
    resolve_lane_change_request,
    MAX_SPEED,
    parse_action,
    limit_acceleration,
)
from src.envs.lane_change_gate import LaneChangeGate


@pytest.mark.parametrize(
    "action,expected",
    [
        ([-2.0, -5.0], (-2.0, -1)),
        ([0.0, 0.0], (0.0, 0)),
        ([3.0, 5.0], (3.0, 1)),
        ([10.0, 0.0], (5.0, 0)),
    ],
)
def test_parse_action(action, expected):
    assert parse_action(action) == expected


def test_acceleration_limit():
    assert limit_acceleration(
        3.0, MAX_SPEED, 0.1
    ) == 0.0

    assert limit_acceleration(
        -3.0, MAX_SPEED, 0.1
    ) == -3.0


def test_controller():
    conn = Mock()

    conn.vehicle.getIDList.return_value = ["ego"]
    conn.vehicle.getRoadID.return_value = "E76"
    conn.vehicle.getLaneIndex.return_value = 2
    conn.vehicle.getSpeed.return_value = 20.0
    conn.edge.getLaneNumber.return_value = 5
    conn.simulation.getDeltaT.return_value = 0.1

    controller = EgoController(conn, "ego")
    controller.configure()

    result = controller.apply_action([2.0, -5.0])

    assert result.lane_direction == -1
    assert result.target_lane == 1
    assert result.executed_accel == 2.0
    assert result.lane_change_gate is None
    conn.vehicle.getRightLeaders.assert_not_called()

    conn.vehicle.setLaneChangeMode.assert_called_once_with(
        "ego", 0
    )
    conn.vehicle.setSpeedMode.assert_called_once_with(
        "ego", 32
    )
    conn.vehicle.changeLane.assert_called_once_with(
        "ego", 1, 4.0
    )
    conn.vehicle.setAcceleration.assert_called_once_with(
        "ego", 2.0, 1
    )

    # EgoController 不允许推进 SUMO
    conn.simulationStep.assert_not_called()


@pytest.mark.parametrize(
    "front_speed,rear_speed,gap,blocked_by",
    [(20.0, 20.0, 24.0, []), (13.0, 20.0, 24.0, ["front"]),
     (20.0, 27.0, 24.0, ["rear"]), (13.0, 27.0, 24.0, ["front", "rear"]),
     (22.0, 18.0, -1.0, ["front", "rear"])],
)
def test_gate_checks_projected_front_and_rear_gaps(front_speed, rear_speed, gap, blocked_by):
    conn = Mock()
    vehicle = conn.vehicle
    vehicle.getIDList.return_value = ["ego", "front", "rear"]
    vehicle.getRoadID.return_value = "E77"
    vehicle.getLaneIndex.return_value = 1
    vehicle.getSpeed.side_effect = {"ego": 20.0, "front": front_speed, "rear": rear_speed}.__getitem__
    vehicle.getMinGap.side_effect = {"ego": 2.0, "rear": 3.0}.__getitem__
    vehicle.getRightLeaders.return_value = [("front", gap - 2)]
    vehicle.getRightFollowers.return_value = [("rear", gap - 3)]
    conn.edge.getLaneNumber.return_value = 3
    conn.simulation.getDeltaT.return_value = 0.1
    conn.simulation.getTime.return_value = 12.0
    controller = EgoController(conn, "ego", lane_change_gate=LaneChangeGate())

    result = controller.apply_action([-2.0, -5.0])

    gate = result.lane_change_gate
    assert gate["blocked_by"] == blocked_by
    assert gate["front"]["gap_m"] == gate["rear"]["gap_m"] == gap
    vehicle.setAcceleration.assert_called_once_with("ego", -2.0, 1)
    if blocked_by:
        assert result.target_lane is None
        assert controller.exit_change.state is None
        vehicle.changeLane.assert_not_called()
    else:
        assert result.target_lane == 0
        vehicle.changeLane.assert_called_once_with("ego", 0, 4.0)


def test_gate_leaves_ongoing_same_direction_change_in_place():
    conn = Mock()
    vehicle = conn.vehicle
    vehicle.getIDList.return_value = ["ego"]
    vehicle.getRoadID.return_value = "E76"
    vehicle.getLaneIndex.return_value = 2
    vehicle.getSpeed.return_value = 20.0
    conn.edge.getLaneNumber.return_value = 5
    conn.simulation.getDeltaT.return_value = 0.1
    controller = EgoController(conn, "ego", lane_change_gate=LaneChangeGate())

    assert controller.apply_action([2.0, -5.0], active_lane_change=(1, -1)).lane_change_gate is None

    vehicle.getRightLeaders.assert_not_called()
    vehicle.getRightFollowers.assert_not_called()
    vehicle.changeLane.assert_called_once_with("ego", 1, 4.0)


@pytest.mark.parametrize("road,legacy_duration", [("E75", 0.0), ("E76", 4.0), ("E77", 4.0)])
@pytest.mark.parametrize("uniform_duration", [None, 0.0, 0.1, 0.5, 1.0, 4.0])
def test_lane_request_duration_and_repeated_direction(road, legacy_duration, uniform_duration):
    conn = Mock()
    vehicle = conn.vehicle
    vehicle.getIDList.return_value = ["ego"]
    vehicle.getRoadID.return_value = road
    vehicle.getLaneIndex.return_value = 2
    vehicle.getSpeed.return_value = 10.0
    conn.edge.getLaneNumber.return_value = 5
    conn.simulation.getDeltaT.return_value = 0.1
    request = (LaneChangeRequest() if uniform_duration is None
               else LaneChangeRequest("uniform", uniform_duration))
    controller = EgoController(conn, "ego", lane_change_request=request)

    command = controller.apply_action([0.0, 5.0])
    duration = legacy_duration if uniform_duration is None else uniform_duration
    assert command.lane_duration == duration
    assert command.target_lane == 3

    vehicle.getLaneIndex.return_value = 3
    assert controller.apply_action([0.0, 0.0], active_lane_change=(3, 1)).target_lane is None
    assert controller.apply_action([0.0, 5.0], active_lane_change=(3, 1)).target_lane is None
    vehicle.changeLane.assert_called_once_with("ego", 3, duration)


def test_request_cli_and_checkpoint_defaults():
    parser = ArgumentParser()
    add_lane_change_request_arguments(parser)
    legacy = LaneChangeRequest.from_checkpoint({"environment": {"exit_lane_change_duration_s": 4.0}})
    saved = LaneChangeRequest.from_checkpoint({
        "lane_change_request_mode": "uniform", "lane_change_request_duration": 0.5,
    })
    assert legacy == LaneChangeRequest()
    assert resolve_lane_change_request(parser.parse_args([]), saved) == saved
    assert resolve_lane_change_request(parser.parse_args([])) == legacy
    assert resolve_lane_change_request(
        parser.parse_args(["--lane-change-request-mode", "uniform"]),
    ) == LaneChangeRequest("uniform", 0.1)
    assert resolve_lane_change_request(
        parser.parse_args(["--lane-change-request-duration", "0"]), saved,
    ) == LaneChangeRequest("uniform", 0.0)
    assert resolve_lane_change_request(
        parser.parse_args(["--lane-change-request-mode", "legacy"]), saved,
    ) == legacy
    with pytest.raises(ValueError):
        resolve_lane_change_request(parser.parse_args([
            "--lane-change-request-mode", "legacy", "--lane-change-request-duration", "1",
        ]))
