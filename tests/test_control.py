
"""测试 DAROM 动作解析与车辆控制。"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.envs.control import (
    EgoController,
    MAX_SPEED,
    parse_action,
    limit_acceleration,
)


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

    conn.vehicle.setLaneChangeMode.assert_called_once_with(
        "ego", 0
    )
    conn.vehicle.setSpeedMode.assert_called_once_with(
        "ego", 96
    )
    conn.vehicle.changeLane.assert_called_once_with(
        "ego", 1, 4.0
    )
    conn.vehicle.setAcceleration.assert_called_once_with(
        "ego", 2.0, 1
    )

    # EgoController 不允许推进 SUMO
    conn.simulationStep.assert_not_called()
