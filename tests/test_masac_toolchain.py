"""O3->D3 homogeneous dual-Ego MASAC toolchain tests.

No formal training or 100-seed evaluation is started by these tests.
The short SUMO test exercises the training/checkpoint/evaluation pipeline.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.experiment_tracking import (
    ActionWindow,
    EpisodeWindow,
    TrainingLog,
    atomic_json,
    atomic_write,
    checkpoint_due,
    evaluation_fingerprint,
    git_metadata,
    runtime_versions,
    sha256,
)
from src.evaluation_metrics import (
    AgentMetrics,
    REFERENCE_COMMIT,
    aggregate,
    joint_record,
    lane_end_arrival_fields,
    wilson,
)
from src.evaluation_results import (
    EVALUATION_PROTOCOL_VERSION,
    validate_results,
    write_results,
)
from train_masac import AGENT_IDS


METRIC_VERSION = "homogeneous-dual-ego-v1"


@pytest.fixture
def work(tmp_path):
    """Use pytest's isolated temporary directory, not historical tmp/ai data."""
    return tmp_path


def test_checkpoint_schedule_and_atomic_write(work):
    steps = (99_999, 100_000, 100_001, 200_000, 250_000)
    assert [
        s for s in steps if checkpoint_due(s, 250_000, 100_000)
    ] == [100_000, 200_000, 250_000]
    assert sum(
        checkpoint_due(s, 200_000, 100_000)
        for s in (100_000, 200_000)
    ) == 2

    path = work / "atomic"
    atomic_write(path, lambda f: f.write(b"valid"))

    def interrupted(stream):
        stream.write(b"broken")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write(path, interrupted)
    assert path.read_bytes() == b"valid"


def test_training_windows_and_log(work):
    assert AGENT_IDS == ("ego_1", "ego_2")
    window = ActionWindow(AGENT_IDS)
    window.observe(np.array([[1., -5.], [2., 0.]]), [1, 0])
    window.observe(np.array([[3., 0.], [2., 5.]]), [1, 1])
    row = window.result()

    assert row["ego_1_accel_mean"] == 2
    assert row["ego_1_accel_std"] == 1
    assert row["ego_1_lane_request_fraction"] == 0.5
    assert row["ego_2_valid_action_fraction"] == 0.5

    agents = {
        key: AgentMetrics().finish("Arrived", 50)
        for key in AGENT_IDS
    }
    rolling = EpisodeWindow()
    assert rolling.add(agents)["rolling_episodes"] == 1
    for _ in range(120):
        result = rolling.add(agents)
    assert result["rolling_episodes"] == 100
    assert result["joint_success_rate"] == 100

    log = TrainingLog(work / "train", tensorboard=True)
    try:
        log.write("episode", 42, **result)
        with pytest.raises(ValueError):
            log.write("update", 43, critic_loss=float("nan"))
    finally:
        log.close()

    events = [
        json.loads(line)
        for line in (work / "train/metrics.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["global_step"] == 42
    assert events[0]["policy"] == "exploration_training"
    assert list((work / "train/tb").glob("events*"))


def test_efficiency_and_wilson():
    m = AgentMetrics()
    for road, lane, post_road, post_lane in [
        ("E76", 2, "E76", 1),
        ("E76", 1, "E77", 0),
    ]:
        m.observe(
            dict(
                pre=dict(
                    speed=0.4,
                    acceleration=0,
                    time=1,
                    road=road,
                    lane=lane,
                    speed_mode=32,
                    request_direction=-1,
                ),
                post_time=1.1,
                post_acceleration=0,
                on_road=True,
                post_road=post_road,
                post_lane=post_lane,
                agent_step=True,
                emergency_stopping=False,
                depart_time=0.1,
            ),
            False,
        )

    row = m.finish("Arrived", 50)
    assert row["lane_change_requests"] == row["toward_requests"] == 2
    assert row["physical_lane_changes"] == 1
    assert row["physical_lane_change_valid_pairs"] == 1
    assert row["first_e77_lane"] == 0
    assert row["stopped_step_fraction"] == 1
    assert row["success_exit_time_s"] == 1
    assert wilson(0, 0)["ci95"] is None
    assert wilson(50, 100)["ci95"] == pytest.approx(
        [40.383153, 59.616847]
    )

    result = aggregate([row])
    assert result["success_rate_stats"]["denominator"] == 1
    assert result["first_e77_lane_distribution"]["0"]["denominator"] == 1


def fixture_report(checkpoint, seed=42, step=100_000, episodes=100,
                   protocol_version=EVALUATION_PROTOCOL_VERSION):
    """Synthetic but internally consistent evaluator output."""
    from scripts.evaluate_masac_checkpoints import expected_environment

    include_stalled = protocol_version >= 3
    records = []
    for index in range(episodes):
        agents = {}
        for key in AGENT_IDS:
            m = AgentMetrics()
            m.speed_modes.add(32)
            m.speeds = [10.0 + index % 2]
            m.jerks = [1.0]
            m.behavior_available = True
            m.first_e77_lane = 1
            m.steps = 2
            m.event_samples = 2
            m.braking_samples = 1
            agents[key] = m.finish(
                "Arrived" if index % 2 else "Collision", 50
            )
            if protocol_version >= 4:
                agents[key].update(lane_end_arrival_fields(agents[key]["raw_arrival"], False))
                agents[key]["lane_end_blocks"] = []

        records.append(
            dict(
                seed=200001 + index,
                global_steps=3,
                agent_steps_sum=4,
                per_agent=agents,
                joint=joint_record(agents, include_stalled=include_stalled),
            )
        )

    return dict(
        episodes=episodes,
        records=records,
        per_agent={
            key: aggregate([r["per_agent"][key] for r in records], include_stalled=include_stalled)
            for key in AGENT_IDS
        },
        joint=aggregate([r["joint"] for r in records], include_stalled=include_stalled),
        stochastic=False,
        training_step=step,
        training_seed=seed,
        seed_start=200001,
        checkpoint_sha256=sha256(checkpoint),
        reference_commit=REFERENCE_COMMIT,
        metric_version=METRIC_VERSION,
        evaluation_protocol_version=protocol_version,
        git=git_metadata(),
        evaluation_code_sha256=evaluation_fingerprint(),
        runtime_versions=runtime_versions(),
        environment={
            **expected_environment(4500),
            "actual_speed_modes": {key: [32] for key in AGENT_IDS},
        },
    )


def test_batch_fixed_protocol_skip_failure_and_recompute(work):
    from scripts.evaluate_masac_checkpoints import run

    (work / "checkpoints").mkdir()
    for step in (200_000, 100_000):
        torch.save(
            {"global_step": step},
            work / "checkpoints" / f"model_{step}.pt",
        )

    calls = []

    def fake(args):
        assert args.episodes == 100
        assert args.seed_start == 200001
        assert not args.stochastic
        assert not args.trace
        step = int(Path(args.output).parent.name.removeprefix("step_"))
        calls.append(step)
        write_results(args.output, fixture_report(args.checkpoint, step=step))

    assert [
        r["status"] for r in run(work, evaluator=fake)
    ] == ["complete", "complete"]
    assert calls == [100_000, 200_000]
    assert [
        r["status"] for r in run(work, evaluator=fake)
    ] == ["skipped_complete", "skipped_complete"]

    path = work / "eval/step_100000/summary.json"
    report = validate_results(path)
    assert report["joint"]["collision_free_rate"] == 50
    assert report["metric_version"] == METRIC_VERSION

    csv_path = path.with_name("episodes.csv")
    csv_path.write_text(
        csv_path.read_text().replace("200001", "999999", 1)
    )
    with pytest.raises(ValueError, match="mismatch"):
        run(work, evaluator=fake)
    failed = json.loads((work / "eval/index.json").read_text())
    assert failed[0]["status"] == "failed"

    run(work, rerun=True, evaluator=fake)
    report = json.loads(path.read_text())
    report["stochastic"] = True
    atomic_json(path, report)
    with pytest.raises(ValueError, match="protocol mismatch"):
        run(work, evaluator=fake)


@pytest.mark.parametrize("protocol_version", [2, 3])
def test_legacy_evaluation_report_remains_readable(work, protocol_version):
    checkpoint = work / "model.pt"
    torch.save({"global_step": 100_000}, checkpoint)
    summary = work / "summary.json"
    write_results(summary, fixture_report(checkpoint, episodes=2, protocol_version=protocol_version))
    joint = validate_results(summary)["joint"]
    assert ("stalled_rate" in joint) is (protocol_version == 3)
    assert joint["raw_arrival_rate"] == 50
    assert "lane_end_block_free_arrival_rate" not in joint


def test_plot_single_multi_seed_and_algorithm(work, monkeypatch):
    from matplotlib.axes import Axes
    import pandas as pd
    from scripts.plot_masac_experiments import plot

    experiments = []
    for algorithm, seed in [
        ("MASAC", 42),
        ("MASAC", 43),
        ("baseline", 42),
    ]:
        directory = work / f"{algorithm}_{seed}"
        checkpoint = directory / "checkpoints/model.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"global_step": 100_000}, checkpoint)
        write_results(
            directory / "eval/step_100000/summary.json",
            fixture_report(checkpoint, seed),
        )

        (directory / "train").mkdir()
        record = dict(
            event="update",
            global_step=10,
            critic_loss=1,
            mean_q=2,
            mean_target=3,
            ego_1_actor_loss=1,
            ego_2_actor_loss=1,
            ego_1_alpha=0.01,
            ego_2_alpha=0.02,
        )
        episode = dict(event="episode", global_step=40, rolling_episodes=1, ego_1_stalled_rate=100)
        (directory / "train/metrics.jsonl").write_text(
            json.dumps(record) + "\n" + json.dumps(episode) + "\n"
        )
        experiments.append((algorithm, seed, str(directory)))

    original_fill = Axes.fill_between
    fills = []

    def observed_fill(self, *args, **kwargs):
        fills.append(True)
        return original_fill(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", observed_fill)
    plot(experiments[:1], work / "single")
    assert not fills
    plot(experiments, work / "multi")
    assert fills

    table = pd.read_csv(work / "multi/final_metrics.csv")
    available = table[
        (table.algorithm == "MASAC") & table["mean"].notna()
    ]
    assert available.training_seeds.eq(2).all()
    missing = table[table["mean"].isna()]
    assert missing.training_seeds.eq(0).all()
    assert missing.valid_samples_sum.eq(0).all()
    assert table[table.algorithm == "baseline"]["std"].isna().all()
    assert len(list((work / "multi").glob("*.png"))) == 7
    assert len(list((work / "multi").glob("*.pdf"))) == 7
    failures = pd.read_csv(work / "single/02_failures_source.csv")
    stalls = failures[failures.metric.eq("stalled_rate")]
    assert set(stalls.source.unique()) == {"exploration_training_rolling100", "checkpoint_evaluation"}
    assert (work / "single/plot_source.csv").exists()


@pytest.mark.parametrize("actor_leader_features,env_version,gamma,shared_actor", [
    (False, 3, 0.99, False), (True, 2, 0.995, False), (False, 5, 0.99, True),
])
def test_short_training_and_checkpoint_loading(
    work, monkeypatch, capsys, actor_leader_features, env_version, gamma, shared_actor,
):
    """Exercise real SUMO->replay->SAC updates->checkpoint->evaluation."""
    import train_masac
    import evaluate_masac
    from src.agents.masac import MultiAgentGaussianActor
    from src.envs.multi_ego_env import MultiEgoEnv

    class ShortEpisodeEnv(MultiEgoEnv):
        def step(self, actions):
            # Allow the second O3 ego to actually depart before completing
            # the synthetic episode. The first ego must not be terminated
            # before a shared-active window has been possible.
            if self.step_count >= 180:
                for key in self.AGENTS:
                    if (
                        self.egos[key].state.status == "active"
                        and not self.terminations[key]
                    ):
                        self.active_steps[key] = self.max_active_steps - 1
            return super().step(actions)

    def safe_fixed_actions(
        actors, policy_obs, active_mask, rng, random_exploration
    ):
        # The separate actor unit tests cover actor inference; this smoke
        # test isolates the training pipeline from untrained random driving.
        joint_actions = np.zeros((len(AGENT_IDS), 2), dtype=np.float32)
        action_dict = {
            key: joint_actions[i].copy()
            for i, key in enumerate(AGENT_IDS)
            if active_mask[i] == 1
        }
        return joint_actions, action_dict

    monkeypatch.setattr(train_masac, "MultiEgoEnv", ShortEpisodeEnv)
    monkeypatch.setattr(train_masac, "choose_actions", safe_fixed_actions)
    monkeypatch.setattr(evaluate_masac, "MultiEgoEnv", ShortEpisodeEnv)

    args = SimpleNamespace(
        total_steps=220,
        seed=42,
        learning_starts=20,
        batch_size=160,
        buffer_capacity=400,
        train_every=20,
        log_every=20,
        update_log_every=40,
        trace_controls=True,
        stalled_shadow=True,
        stalled_terminate=False,
        torch_threads=1,
        device="cpu",
        gamma=gamma,
        reward_ttc_threshold=2.75,
        reward_route_preparation_bonus_cap=0.007 if actor_leader_features else None,
        actor_leader_features=actor_leader_features,
        shared_actor=shared_actor,
        masac_env_version=env_version,
        learning_rate=3e-5,
        initial_alpha=0.01,
        output_dir=str(work),
        checkpoint_every=100,
        replay_final_only=actor_leader_features,
        tensorboard=False,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_masac.main(args)
        checkpoint = work / "checkpoints/masac_seed42_step_220.pt"
        assert checkpoint.is_file()

        report = evaluate_masac.evaluate(
            SimpleNamespace(
                checkpoint=str(checkpoint),
                episodes=1,
                seed_start=200001,
                stochastic=False,
                trace=False,
                max_steps=4500,
                output=str(work / "short_evaluation/summary.json"),
            )
        )
        assert report["metric_version"] == METRIC_VERSION
        assert report["trainer_config"]["gamma"] == gamma
        assert report["training_config"]["masac_env_version"] == env_version
        assert report["training_config"]["shared_actor"] is shared_actor
        assert report["training_config"]["safety_reward"] == {
            "ttc_threshold": 2.75, "following_lookahead": 120.0, "lane_change_weight": 0.5,
        }
        assert report["route_preparation"]["settings"]["mode"] == "first_progress_staircase"
        assert report["route_preparation"]["settings"]["reference_step_m"] == 3.333
        assert report["route_preparation"]["settings"]["bonus_cap"] == (0.007 if actor_leader_features else 0.005)
        assert report["environment"]["requested_speed_mode"] == 96
        for key in AGENT_IDS:
            modes = report["environment"]["actual_speed_modes"][key]
            assert modes and all(isinstance(mode, int) for mode in modes)
            assert modes == report["records"][0]["per_agent"][key]["speed_modes"]
        assert report["sumo_version"]
        assert report["environment"]["max_active_steps"] == 2500
        assert report["environment"]["stalled_terminate"] is False
        record = report["records"][0]
        assert record["team_reward"] == 0.0
        assert "teleport_events" not in record
        for key in AGENT_IDS:
            components = record["reward_components"][key]
            assert components["task"] == -400.0
            assert sum(components.values()) == pytest.approx(record["rewards"][key])
            assert "first_ttc_risk_to_requested_brake_s" in record["per_agent"][key]
        validate_results(work / "short_evaluation/summary.json")
        assert "TRACE step=" not in capsys.readouterr().out
    finally:
        torch.set_num_threads(previous_threads)

    checkpoints = sorted((work / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 3
    index = json.loads((work / "checkpoints/index.json").read_text())
    assert [r["global_step"] for r in index] == [100, 200, 220]
    for row in index:
        path = Path(row["path"])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        resumable = not actor_leader_features or row["global_step"] == 220
        assert (saved["resume_state"] is not None) == resumable
        assert ("critic_optimizer" in saved) == resumable
        assert (row["replay"] is not None) == resumable
        assert path.with_suffix(".replay.npz").exists() == resumable

    final = torch.load(
        work / "checkpoints/masac_seed42_step_220.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert tuple(final["agent_ids"]) == AGENT_IDS
    assert final["config"]["device"] == "cpu"
    assert final["config"]["effective_torch_threads"] == 1
    assert final["config"]["team_q_weight"] == 1.0
    assert final["config"]["reward_components"]["rear_following"] is True
    assert final["config"]["reward_components"]["route_guidance"] is False
    assert final["config"]["reward_protocol"] == {
        "merge_on_success": True, "efficiency_weight": 0.01, "route_guidance_weight": 5.0,
        "terminal_penalty_scale": 1.0,
        "comfort_weight": 0.0025,
    }
    assert final["config"]["route_guidance_rules"][0]["accepted_connections"] == [["E76_0", "E77_0"]]
    assert final["team_critic"] and final["target_team_critic"]
    assert final["team_critic_optimizer"]["state"]
    assert final["config"]["actor_leader_features"] is actor_leader_features
    actor = MultiAgentGaussianActor(ego_ids=AGENT_IDS, nav_dim=2,
                                    actor_leader_features=actor_leader_features, shared=shared_actor)
    assert actor.obs_dim == (139 if actor_leader_features else 136)
    actor.load_state_dict(final["actors"], strict=True)
    assert all(sha256(r["path"]) == r["sha256"] for r in index)

    records = [
        json.loads(line)
        for line in (work / "train/metrics.jsonl").read_text().splitlines()
    ]
    shadows = [r for r in records if r["event"] == "stalled_shadow"]
    assert len(shadows) == 220
    assert shadows[0]["egos"]["ego_1"]["reason"] == "collecting_window"
    assert shadows[0]["egos"]["ego_2"]["reason"] == "inactive"
    controls = [r for r in records if r["event"] == "control_trace"]
    first = controls[0]["egos"]["ego_1"]
    assert first["reward_components"]["route_guidance"] == 0
    assert first["road"] == "O3"
    assert first["speed_mps"] == pytest.approx(10.0)
    assert first["requested_acceleration_mps2"] == first["commanded_acceleration_mps2"] == 0.0
    assert first["actual_acceleration_mps2"] is not None
    assert "leader_gap_m" in first and first["lane_remaining_m"] > 0
    assert not first["sumo_arrived"]
    updates = [r for r in records if r["event"] == "update"]
    assert len(updates) == 2
    assert updates[0]["update_index"] == 1
    runtime = next(r for r in reversed(records) if r["event"] == "runtime")
    assert runtime["update_count"] == updates[-1]["update_index"] == 3
    assert all("mean_q" in r and "valid_agents" in r for r in updates)
    assert any(r.get("team_valid_samples", 0) > 0 for r in updates)
    assert all(r["global_step"] % 20 == 0 for r in updates)
    # 检查两套独立 Actor 都实际进行过参数更新
    actor_updates = {
        ego_id: [
            r["global_step"]
            for r in updates
            if r.get(f"{ego_id}_actor_updates") == 1
        ]
        for ego_id in AGENT_IDS
    }

    print("Actor update steps:", actor_updates)

    assert actor_updates["ego_1"], (
        "ego_1 Actor never updated"
    )

    assert actor_updates["ego_2"], (
        "ego_2 Actor never updated"
    )

    episodes = [r for r in records if r["event"] == "episode"]
    assert episodes
    assert 1 <= episodes[0]["episode_global_steps"] <= 220
    assert episodes[0]["rolling_episodes"] == 1


def test_training_cli_defaults():
    from train_masac import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert parser.parse_args(["--no-stalled-terminate"]).stalled_terminate is False
    assert args.masac_env_version == 5
    assert parser.parse_args(["--masac-env-version", "4"]).masac_env_version == 4
    assert parser.parse_args(["--masac-env-version", "3"]).masac_env_version == 3
    assert parser.parse_args(["--masac-env-version", "2"]).masac_env_version == 2
    assert not args.actor_leader_features
    assert parser.parse_args(["--actor-leader-features"]).actor_leader_features
    assert (
        args.total_steps,
        args.buffer_capacity,
        args.batch_size,
        args.learning_starts,
    ) == (1_000_000, 1_000_000, 512, 100)
    assert args.learning_rate == 3e-5
    assert args.gamma == 0.99
    assert parser.parse_args(["--gamma", "0.995"]).gamma == 0.995
    assert args.tau == 0.005
    assert args.train_every == 1
    assert args.checkpoint_every == 500_000
    assert args.resume
    assert not parser.parse_args(["--no-resume"]).resume
    assert args.device == "auto"
    assert args.torch_threads is None
    assert args.update_log_every == 40
    assert args.max_active_steps == 2500
    assert args.reward_rear_following
    assert args.reward_route_preparation is None
    assert parser.parse_args(["--no-reward-route-preparation"]).reward_route_preparation is False
    from train_masac import preparation_settings
    assert preparation_settings(args)["mode"] == "first_progress_staircase"
    assert preparation_settings(args)["bonus_cap"] == 0.005
    fixed = {"mode": "continuous_bonus", "stages": [], "bonus_cap": 0.007}
    fixed_args = parser.parse_args(["--reward-route-preparation-bonus-cap", "0.007"])
    assert preparation_settings(fixed_args, fixed, resuming=True) == fixed
    progress = {"mode": "progress_bonus", "stages": [], "bonus_cap": 0.005, "reference_step_m": 10 / 3}
    assert preparation_settings(args, progress, resuming=True) == progress
    legacy = parser.parse_args(["--reward-route-preparation-exit-weight", "0.003"])
    assert preparation_settings(legacy)["weights"] == {"weaving": 0.001, "exit": 0.003}
    assert not parser.parse_args(["--no-reward-rear-following"]).reward_rear_following
    assert not args.reward_route_guidance
    assert parser.parse_args(["--reward-route-guidance"]).reward_route_guidance
    assert args.reward_route_guidance_weight == 5.0


@pytest.mark.parametrize("shared_actor", [False, True])
def test_training_resumes_replay_and_updates_after_interruption(work, shared_actor):
    from scripts.evaluate_masac_checkpoints import discover
    from train_masac import build_parser, main

    parser = build_parser()
    flags = ["--output-dir", str(work), "--device", "cpu", "--torch-threads", "1",
             "--batch-size", "4", "--learning-starts", "4", "--buffer-capacity", "16",
             "--checkpoint-every", "12", "--log-every", "4", "--no-tensorboard"]
    if shared_actor:
        flags.append("--shared-actor")
    main(parser.parse_args(flags + ["--total-steps", "20", "--reward-comfort-weight", "0.01"]))
    directory = work / "checkpoints"
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    assert [row["global_step"] for row in index] == [12, 20]
    model = directory / "masac_seed42_step_20.pt"
    before = torch.load(model, weights_only=True, map_location="cpu")
    replay = before["resume_state"]["replay"]
    assert replay["size"] == 16
    with np.load(directory / replay["path"], allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
    assert metadata["position"] != 0

    # Simulate a published model whose index write was interrupted, with stale log tail.
    # Omit the comfort field to exercise a checkpoint written before it was recorded.
    del before["config"]["reward_protocol"]["comfort_weight"]
    if not shared_actor:
        del before["config"]["shared_actor"]
    old_stages = [{
        "route": ["O3", "E75", "E76", "E77", "D3"],
        "phases": [
            {"kind": "weaving", "critical_connection": ["E76", "E77"],
             "accepted_entry_lanes": ["E77_0"], "preparation_distance_m": 300.0},
            {"kind": "exit", "critical_connection": ["E77", "D3"],
             "accepted_entry_lanes": ["D3_0"], "preparation_distance_m": 172.35},
        ],
    }]
    before["config"]["route_preparation_settings"] = {
        "stages": old_stages, "weights": {"weaving": 0.001, "exit": 0.005},
    }
    before["config"]["safety_reward"]["lane_change_weight"] = 1.0
    torch.save(before, model)
    index_path.write_text(json.dumps(index[:-1]))
    metrics_path = work / "train/metrics.jsonl"
    with metrics_path.open("a") as stream:
        stream.write('{"event":"interrupted","global_step":21}\n')
    main(parser.parse_args(flags + ["--total-steps", "28"]))
    final = torch.load(directory / "masac_seed42_step_28.pt", weights_only=True, map_location="cpu")
    assert final["resume_state"]["update_count"] == before["resume_state"]["update_count"] + 8
    assert final["resume_state"]["replay"]["size"] == 16
    assert final["config"]["reward_protocol"]["comfort_weight"] == 0.01
    assert final["config"]["route_preparation_settings"] == {
        "mode": "legacy_penalty", "stages": old_stages, "weights": {"weaving": 0.001, "exit": 0.005},
    }
    assert final["config"]["safety_reward"]["lane_change_weight"] == 1.0
    for parameter, original in before["critic_optimizer"]["state"].items():
        assert final["critic_optimizer"]["state"][parameter]["step"] == original["step"] + 8
    if shared_actor:
        for key in ("actor_optimizers", "alpha_optimizers"):
            assert set(final[key]) == {"shared"}
            for parameter, original in before[key]["shared"]["state"].items():
                assert final[key]["shared"]["state"][parameter]["step"] == original["step"] + 8
    replay_rng = np.random.default_rng()
    replay_rng.bit_generator.state = metadata["rng"]
    for _ in range(8):
        replay_rng.choice(16, size=4, replace=False)
    with np.load(directory / final["resume_state"]["replay"]["path"], allow_pickle=False) as data:
        assert json.loads(data["metadata"].item())["rng"] == replay_rng.bit_generator.state
    assert [row["global_step"] for row in discover(work)] == [12, 20, 24, 28]

    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert not any(row["event"] == "interrupted" for row in metrics)
    assert len(list(metrics_path.parent.glob("metrics.before-resume-20-*.jsonl"))) == 1
    resume = next(row for row in metrics if row["event"] == "resume")
    assert resume["global_step"] == 20 and resume["replay_size"] == 16 and resume["sumo_seed"] == 43
    with pytest.raises(ValueError, match="configuration mismatch: batch_size"):
        main(parser.parse_args(flags + ["--total-steps", "32", "--batch-size", "8"]))
    with pytest.raises(ValueError, match="configuration mismatch: reward_protocol"):
        main(parser.parse_args(flags + ["--total-steps", "32", "--reward-comfort-weight", "0.0025"]))
    with pytest.raises(ValueError, match="configuration mismatch: route_preparation_settings"):
        main(parser.parse_args(flags + ["--total-steps", "32", "--reward-route-preparation-bonus-cap", "0.005"]))
    opposite_mode = flags[:-1] if shared_actor else flags + ["--shared-actor"]
    with pytest.raises(ValueError, match="configuration mismatch: shared_actor"):
        main(parser.parse_args(opposite_mode + ["--total-steps", "32"]))


@pytest.mark.parametrize("env_version,legacy,saved_grace,effective_grace", [
    (2, True, None, 0.0),
    (3, True, 3.0, 0.0),
    (3, False, None, 0.0),
    (4, False, None, 3.0),
    (4, False, 0.0, 0.0),
    (4, False, 2.0, 2.0),
    (5, False, None, 3.0),
    (5, False, 3.0, 3.0),
])
def test_checkpoint_evaluation_uses_saved_reward_protocol(
    work, monkeypatch, env_version, legacy, saved_grace, effective_grace,
):
    import evaluate_masac
    from src.agents.masac import MultiAgentGaussianActor

    env_class = evaluate_masac.MultiEgoEnv
    with_preparation = env_version == 5 and saved_grace is not None

    def check_env_configuration(**kwargs):
        assert kwargs["stalled_match_route_links"] is (env_version == 5)
        assert kwargs["reward_config"].route_preparation is with_preparation
        assert kwargs["reward_config"].lane_change_weight == 1.0
        return env_class(**kwargs)

    monkeypatch.setattr(evaluate_masac, "MultiEgoEnv", check_env_configuration)
    actors = MultiAgentGaussianActor(ego_ids=AGENT_IDS, nav_dim=2)
    config = {
        "masac_env_version": env_version,
        "max_active_steps": 1,
        "reward_components": {},
        "stalled_terminate": True,
    }
    if saved_grace is not None:
        config["stalled_block_grace_s"] = saved_grace
    if with_preparation:
        from src.envs.route_preparation import DEFAULT_ROUTE_PREPARATION_STAGES

        config["reward_components"]["route_preparation"] = True
        config["route_preparation_settings"] = {
            "weights": {"weaving": 0.001, "exit": 0.005}, "stages": DEFAULT_ROUTE_PREPARATION_STAGES,
        }
    if not legacy:
        config["reward_components"]["route_guidance"] = True
        config["reward_protocol"] = {
            "merge_on_success": True, "efficiency_weight": 0.01, "route_guidance_weight": 7.0,
        }
        config["route_guidance_rules"] = [{
            "route": ["O3", "E75", "E76", "E77", "D3"],
            "critical_connection": ["E76", "E77"],
            "accepted_connections": [["E76_0", "E77_1"]],
        }]
        if with_preparation:
            config["reward_protocol"]["terminal_penalty_scale"] = 0.2
            config["reward_protocol"]["collision_penalty"] = 1000
            config["reward_protocol"]["comfort_weight"] = 0.0025
    checkpoint = work / "untrained.pt"
    torch.save({"agent_ids": AGENT_IDS, "actors": actors.state_dict(), "config": config,
                "global_step": 0}, checkpoint)
    report = evaluate_masac.evaluate(SimpleNamespace(
        checkpoint=str(checkpoint), episodes=1, seed_start=200001, stochastic=False,
        max_steps=150, output=str(work / "summary.json"),
        **({"trajectory": False} if env_version == 2 else {}),
    ))
    if env_version == 2:
        assert report["trajectory"] is None
        assert not (work / "eval_trajectory.jsonl.gz").exists()
    else:
        assert report["trajectory"]["step_records"] == report["global_steps"]
    protocol = report["reward_protocol"]
    assert report["environment"]["masac_env_version"] == env_version
    assert report["environment"]["stalled_block_grace_s"] == effective_grace
    assert ("stalled_diagnostics" in report["records"][0]) is bool(effective_grace)
    assert protocol["merge_on_success"] is not legacy
    assert report["route_preparation"]["enabled_in_checkpoint"] is with_preparation
    assert report["route_preparation"]["settings"] == (
        {"mode": "legacy_penalty", **config["route_preparation_settings"]} if with_preparation else None
    )
    assert report["records"][0]["per_agent"]["ego_1"]["route_preparation"]["enabled"] is with_preparation
    assert protocol["route_guidance_weight"] == (0.0 if legacy else 7.0)
    assert protocol["terminal_penalty_scale"] == (0.2 if with_preparation else 1.0)
    assert protocol["collision_penalty"] == (1000 if with_preparation else None)
    assert protocol["comfort_weight"] == (0.0025 if with_preparation else 0.01)
    assert report["route_guidance_rules"] == config.get("route_guidance_rules", [])
    for values in report["records"][0]["reward_components"].values():
        assert values["rear_following"] == 0
        assert values["route_preparation"] == 0
        assert values["efficiency"] == pytest.approx(0.025 if legacy else 0.005)
        assert values["route_guidance"] == 0


@pytest.mark.parametrize(
    "cuda_available, previous_threads, override, expected_device, expected_threads",
    [(True, 23, None, "cuda:0", 2), (False, 23, None, "cpu", 8),
     (False, 4, None, "cpu", 4), (True, 23, 4, "cuda:0", 4)],
)
def test_training_resolves_performance_defaults(
    monkeypatch, cuda_available, previous_threads, override, expected_device, expected_threads,
):
    import train_masac

    thread_changes = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch, "get_num_threads", lambda: previous_threads)
    monkeypatch.setattr(torch, "set_num_threads", thread_changes.append)

    def validate(args, device):
        assert args.requested_device == "auto"
        assert args.device == str(device) == expected_device
        assert args.effective_torch_threads == expected_threads

    monkeypatch.setattr(train_masac, "train", validate)
    args = train_masac.build_parser().parse_args([])
    args.torch_threads = override
    train_masac.main(args)
    assert thread_changes == [expected_threads, previous_threads]
