
"""验证 E77 错过出口判定。"""

from unittest.mock import Mock

import pytest

from src.envs.exit_rules import missed_exit_before_step


@pytest.mark.parametrize(
    "road,lane,remaining,speed,pending,expected",
    [
        ("E76", 2, 5.0, 20.0, False, False),
        ("E77", 0, 5.0, 20.0, False, False),
        ("E77", 1, 15.0, 20.0, False, False),
        ("E77", 1, 9.0, 5.0, False, False),
        ("E77", 1, 9.0, 20.0, True, False),
        ("E77", 1, 9.0, 20.0, False, True),
    ],
)
def test_missed_exit(
    road, lane, remaining, speed, pending, expected
):
    conn = Mock()

    conn.vehicle.getIDList.return_value = ["ego"]
    conn.vehicle.getRoadID.return_value = road
    conn.vehicle.getLaneIndex.return_value = lane
    conn.vehicle.getLaneID.return_value = f"{road}_{lane}"

    conn.lane.getLength.return_value = 100.0
    conn.vehicle.getLanePosition.return_value = (
        100.0 - remaining
    )

    conn.vehicle.getSpeed.return_value = speed
    conn.vehicle.getEmergencyDecel.return_value = 7.0

    result = missed_exit_before_step(
        conn,
        "ego",
        active_change_to_zero=pending,
    )

    assert result is expected

    # 纯判定函数不能推进 SUMO
    conn.simulationStep.assert_not_called()
