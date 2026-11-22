import gzip
import json

import pytest

from scripts.trajectory_dashboard.series import cumulative_rewards, cursor_point, episode_series, road_segments
from scripts.trajectory_dashboard.trace import build_index, file_sha256


def vehicle(step, road, x, reward, *, acted=True, terminal=None, post_road=None):
    t = 100.0 + (step - 1) * 0.1
    return {
        "action": [1.0, 5.0],
        "pre_position": {"xy_m": [x, 0.0]},
        "post_position": {"xy_m": [x + 1, 0.0]} if acted else None,
        "metrics": {
            "pre": {"time": round(t, 2), "road": road, "lane": 1,
                    "speed": 10.0, "request_direction": -1},
            "post_time": round(t + 0.1, 2), "post_road": post_road or road,
            "post_lane": 1, "post_acceleration": 0.0, "agent_step": acted,
        },
        "remaining_goal_distance_m": 90.0 - x,
        "reward": reward, "components": {"task": reward if terminal else 0.0,
                                          "efficiency": 0.0 if terminal else reward},
        "terminal": terminal,
    }


@pytest.fixture
def published(tmp_path):
    seed = 200002
    reset = dict(event="reset", seed=seed, episode=1, global_step=0)
    rows = [
        (vehicle(1, "E76", 0.0, 0.01), vehicle(1, "E76", 0.0, 0.01)),
        (vehicle(2, "E76", 1.0, 0.01, post_road=":J105_0"), vehicle(2, "E77", 1.0, 0.01)),
        (vehicle(3, "E77", 2.0, -400.0, acted=False, terminal="Missed Exit"),
         vehicle(3, "E77", 2.0, 50.0, terminal="Arrived")),
    ]
    steps = [
        dict(event="step", step=i, global_step=i, episode=1, seed=seed,
             egos=dict(zip(("ego_1", "ego_2"), egos)), team_reward=0.0)
        for i, egos in enumerate(rows, 1)
    ]
    trace = tmp_path / "eval_trajectory.jsonl.gz"
    with gzip.open(trace, "wt") as stream:
        for row in [reset, *steps]:
            stream.write(json.dumps(row) + "\n")
    report = {
        "trajectory": {"format": "dual-ego-eval-jsonl-gzip-v1", "sha256": file_sha256(trace),
                       "step_records": 3, "episodes": 1},
        "training_step": 1_000_000, "checkpoint_sha256": "abc",
        "environment": {"agent_ids": ["ego_1", "ego_2"], "step_length_s": 0.1},
        "trainer_config": {"gamma": 0.99}, "global_steps": 3, "episodes": 1,
        "records": [{
            "seed": seed, "global_steps": 3, "team_reward": 0.0,
            "outcomes": {"ego_1": "Missed Exit", "ego_2": "Arrived"},
            "rewards": {"ego_1": -399.98, "ego_2": 50.02},
            "reward_components": {"ego_1": {"task": -400.0, "efficiency": 0.02},
                                  "ego_2": {"task": 50.0, "efficiency": 0.02}},
        }],
    }
    summary = trace.with_name("summary.json")
    summary.write_text(json.dumps(report))
    return trace, summary, tmp_path / "cache"


def test_index_and_pre_step_terminal_alignment(published):
    archive = build_index(*published)
    assert archive.metadata["step_records"] == 3
    assert build_index(*published).path == archive.path
    with pytest.raises(KeyError):
        archive.episode(999)
    points = episode_series(archive.episode(200002), "ego_1", 0.1)
    terminal = points[-1]
    assert terminal["terminal_time"] == terminal["pre_time"] == 100.2
    assert terminal["post_time"] == 100.3
    assert terminal["raw_action"] == [1.0, 5.0] and terminal["submitted_direction"] is None
    assert terminal["remaining"] is None and terminal["distance"] == 2.0
    first = cursor_point(points[0], at_step_start=True)
    assert first["cursor_time"] == first["action_time"] == 100.0
    assert first["distance"] == 0.0 and first["remaining"] is None
    assert cursor_point(terminal)["cursor_time"] == 100.2
    assert [span[0]["road"] for span in road_segments(points, "distance")] == ["E76", ":J105_0", "E77"]
    assert cumulative_rewards(points)[-1][1] == pytest.approx(-399.98)
    assert cumulative_rewards(points, 0.99)[-1][1] == pytest.approx(0.01 + 0.99 * 0.01 - 400 * 0.99**2)


def test_wrong_hash_is_rejected(published):
    trace, summary, cache = published
    with trace.open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(ValueError, match="SHA-256"):
        build_index(trace, summary, cache)


def test_semantic_mismatch_does_not_publish_cache(published):
    trace, summary, cache = published
    report = json.loads(summary.read_text())
    report["records"][0]["reward_components"]["ego_1"]["efficiency"] = 7.0
    summary.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="efficiency"):
        build_index(trace, summary, cache)
    assert not list(cache.glob("*.sqlite3"))
