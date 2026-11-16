
"""比较新旧 DAROM 基础观测是否完全一致。"""

from types import SimpleNamespace

import numpy as np

from src.envs.sumo_world import SumoWorld
from src.envs.observation import build_observation
from src.merging import Merging


def test_observation_parity():
    world = SumoWorld(scenario="easy", seed=42)

    checked = 0
    sample_times = {120.1, 125.0, 142.6, 170.7, 175.9}

    try:
        conn = world.start()

        for _ in range(1800):
            t = round(world.step(), 1)

            if t not in sample_times:
                continue

            if "ego" not in conn.vehicle.getIDList():
                raise AssertionError(f"Ego missing at {t}s")

            # 模拟旧 Merging._get_obs() 需要的属性
            legacy = SimpleNamespace(
                traci=conn,
                ego_id="ego",
                radius=50,
                num_entities=31,
                observation_snapshot=None,
                _ahead_state=None,
                _behind_state=None,
            )

            old_obs = Merging._get_obs(legacy)

            new = build_observation(
                conn,
                ego_id="ego",
                radius=50,
                num_entities=31,
            )

            np.testing.assert_array_equal(
                old_obs,
                new.observation,
            )

            np.testing.assert_array_equal(
                legacy._ahead_state,
                new.ahead_state,
            )

            np.testing.assert_array_equal(
                legacy._behind_state,
                new.behind_state,
            )

            old_snap = legacy.observation_snapshot
            new_snap = new.snapshot

            assert old_snap.time == new_snap.time
            assert old_snap.ego_angle == new_snap.ego_angle

            for field in (
                "ego_position",
                "positions",
                "velocities",
                "valid",
            ):
                np.testing.assert_array_equal(
                    getattr(old_snap, field),
                    getattr(new_snap, field),
                )

            assert new.observation.shape == (31, 3)
            assert new.observation.dtype == np.float32

            checked += 1
            print(f"PASS: {t:.1f}s observation parity")

        assert checked == len(sample_times)

    finally:
        world.close()


if __name__ == "__main__":
    test_observation_parity()
