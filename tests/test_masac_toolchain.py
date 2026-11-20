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
                teleport_started=False,
                teleport_ended=False,
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


def fixture_report(checkpoint, seed=42, step=100_000, episodes=100):
    """Synthetic but internally consistent evaluator output."""
    from scripts.evaluate_masac_checkpoints import expected_environment

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

        records.append(
            dict(
                seed=200001 + index,
                global_steps=3,
                agent_steps_sum=4,
                per_agent=agents,
                joint=joint_record(agents),
            )
        )

    return dict(
        episodes=episodes,
        records=records,
        per_agent={
            key: aggregate([r["per_agent"][key] for r in records])
            for key in AGENT_IDS
        },
        joint=aggregate([r["joint"] for r in records]),
        stochastic=False,
        training_step=step,
        training_seed=seed,
        seed_start=200001,
        checkpoint_sha256=sha256(checkpoint),
        reference_commit=REFERENCE_COMMIT,
        metric_version=METRIC_VERSION,
        evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
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
        (directory / "train/metrics.jsonl").write_text(
            json.dumps(record) + "\n"
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
    assert (work / "single/plot_source.csv").exists()


def test_short_training_and_checkpoint_loading(work, monkeypatch, capsys):
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
        torch_threads=1,
        device="cpu",
        gamma=0.999,
        learning_rate=3e-5,
        initial_alpha=0.01,
        output_dir=str(work),
        checkpoint_every=100,
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
        assert report["environment"]["requested_speed_mode"] == 96
        for key in AGENT_IDS:
            modes = report["environment"]["actual_speed_modes"][key]
            assert modes and all(isinstance(mode, int) for mode in modes)
            assert modes == report["records"][0]["per_agent"][key]["speed_modes"]
        assert report["sumo_version"]
        assert report["environment"]["max_active_steps"] == 2500
        record = report["records"][0]
        assert record["team_reward"] == 0.0
        assert record["teleport_events"] == []
        for key in AGENT_IDS:
            components = record["reward_components"][key]
            assert components["task"] == -400.0
            assert sum(components.values()) == pytest.approx(record["rewards"][key])
        validate_results(work / "short_evaluation/summary.json")
        assert "TRACE step=" not in capsys.readouterr().out
    finally:
        torch.set_num_threads(previous_threads)

    checkpoints = sorted((work / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 3
    index = json.loads((work / "checkpoints/index.json").read_text())
    assert [r["global_step"] for r in index] == [100, 200, 220]

    final = torch.load(
        work / "checkpoints/masac_seed42_step_220.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert tuple(final["agent_ids"]) == AGENT_IDS
    assert final["config"]["device"] == "cpu"
    assert final["config"]["effective_torch_threads"] == 1
    assert final["config"]["team_q_weight"] == 1.0
    assert final["team_critic"] and final["target_team_critic"]
    assert final["team_critic_optimizer"]["state"]
    actor = MultiAgentGaussianActor(ego_ids=AGENT_IDS, nav_dim=2)
    actor.load_state_dict(final["actors"], strict=True)
    assert all(sha256(r["path"]) == r["sha256"] for r in index)

    records = [
        json.loads(line)
        for line in (work / "train/metrics.jsonl").read_text().splitlines()
    ]
    controls = [r for r in records if r["event"] == "control_trace"]
    first = controls[0]["egos"]["ego_1"]
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

    args = build_parser().parse_args([])
    assert (
        args.total_steps,
        args.buffer_capacity,
        args.batch_size,
        args.learning_starts,
    ) == (1_000_000, 1_000_000, 512, 100)
    assert args.learning_rate == 3e-5
    assert args.gamma == 0.99
    assert args.tau == 0.005
    assert args.train_every == 1
    assert args.checkpoint_every == 500_000
    assert args.device == "auto"
    assert args.torch_threads is None
    assert args.update_log_every == 40
    assert args.max_active_steps == 2500


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
