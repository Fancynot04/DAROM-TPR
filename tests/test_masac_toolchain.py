"""Small model and synthetic-counter toolchain checks; no formal evaluation run."""
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch

from src.experiment_tracking import (ActionWindow, EpisodeWindow, TrainingLog, checkpoint_due,
    atomic_write, atomic_json, sha256, git_metadata, evaluation_fingerprint, runtime_versions)
from src.evaluation_metrics import AgentMetrics, aggregate, joint_record, wilson, REFERENCE_COMMIT
from src.evaluation_results import write_results, validate_results, EVALUATION_PROTOCOL_VERSION
from train_masac import AGENT_IDS


@pytest.fixture
def work():
    path = Path('tmp/ai/masac_toolchain') / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_checkpoint_schedule_and_atomic_write(work):
    assert [s for s in (99999, 100000, 100001, 200000, 250000) if checkpoint_due(s, 250000, 100000)] == [100000, 200000, 250000]
    assert sum(checkpoint_due(s, 200000, 100000) for s in (100000, 200000)) == 2
    path = work / 'atomic'
    atomic_write(path, lambda f: f.write(b'valid'))
    def interrupted(stream):
        stream.write(b'broken'); raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): atomic_write(path, interrupted)
    assert path.read_bytes() == b'valid'


def test_training_windows_and_log(work):
    window = ActionWindow(AGENT_IDS)
    window.observe(np.array([[1., -5.], [2., 0.]]), [1, 0])
    window.observe(np.array([[3., 0.], [2., 5.]]), [1, 1])
    row = window.result()
    assert row['ego_o3_accel_mean'] == 2 and row['ego_o3_accel_std'] == 1
    assert row['ego_o3_lane_request_fraction'] == .5
    assert row['ego_e74_valid_action_fraction'] == .5
    agents = {key: AgentMetrics().finish('Arrived', 50) for key in AGENT_IDS}
    rolling = EpisodeWindow()
    assert rolling.add(agents)['rolling_episodes'] == 1
    for _ in range(120): r = rolling.add(agents)
    assert r['rolling_episodes'] == 100 and r['joint_success_rate'] == 100
    log = TrainingLog(work / 'train', tensorboard=True)
    log.write('episode', 42, **r)
    with pytest.raises(ValueError): log.write('update', 43, critic_loss=float('nan'))
    log.close()
    recorded = json.loads((work / 'train/metrics.jsonl').read_text())
    assert recorded['global_step'] == 42 and recorded['policy'] == 'exploration_training'
    assert list((work / 'train/tb').glob('events*'))


def test_efficiency_and_wilson():
    m = AgentMetrics()
    for road, lane, post_road, post_lane in [('E76', 2, 'E76', 1), ('E76', 1, 'E77', 0)]:
        m.observe(dict(pre=dict(speed=.4, acceleration=0, time=1, road=road, lane=lane,
                               speed_mode=96, request_direction=-1), post_time=1.1,
                       post_acceleration=0, on_road=True, post_road=post_road, post_lane=post_lane,
                       agent_step=True, teleport_started=False, teleport_ended=False,
                       emergency_stopping=False, depart_time=.1), False)
    row = m.finish('Arrived', 50)
    assert row['lane_change_requests'] == row['toward_requests'] == 2
    assert row['physical_lane_changes'] == 1 and row['physical_lane_change_valid_pairs'] == 1
    assert row['first_e77_lane'] == 0 and row['stopped_step_fraction'] == 1
    assert row['success_exit_time_s'] == 1
    assert wilson(0, 0)['ci95'] is None
    assert wilson(50, 100)['ci95'] == pytest.approx([40.383153, 59.616847])
    result = aggregate([row])
    assert result['success_rate_stats']['denominator'] == 1
    assert result['first_e77_lane_distribution']['0']['denominator'] == 1


def fixture_report(checkpoint, seed=42, step=100000, episodes=100):
    from scripts.evaluate_masac_checkpoints import expected_environment
    records = []
    for index in range(episodes):
        agents = {}
        for key in AGENT_IDS:
            m = AgentMetrics(); m.speed_modes.add(96)
            m.speeds = [10. + index % 2]; m.jerks = [1.]
            m.behavior_available = True; m.first_e77_lane = 1
            m.steps = 2; m.event_samples = 2; m.braking_samples = 1
            agents[key] = m.finish('Arrived' if index % 2 else 'Collision', 50)
        records.append(dict(seed=200001 + index, global_steps=3, agent_steps_sum=4,
                            per_agent=agents, joint=joint_record(agents)))
    return dict(episodes=episodes, records=records,
                per_agent={k: aggregate([r['per_agent'][k] for r in records]) for k in AGENT_IDS},
                joint=aggregate([r['joint'] for r in records]), stochastic=False,
                training_step=step, training_seed=seed, seed_start=200001,
                checkpoint_sha256=sha256(checkpoint), reference_commit=REFERENCE_COMMIT,
                metric_version='dual-ego-v1', evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
                git=git_metadata(), evaluation_code_sha256=evaluation_fingerprint(), runtime_versions=runtime_versions(),
                environment={**expected_environment(4500), 'actual_speed_modes': {k: [96] for k in AGENT_IDS}})


def test_batch_fixed_protocol_skip_failure_and_recompute(work):
    from scripts.evaluate_masac_checkpoints import run
    (work / 'checkpoints').mkdir()
    for step in (200000, 100000):
        torch.save(dict(global_step=step), work / 'checkpoints' / f'model_{step}.pt')
    calls = []
    def fake(args):
        assert args.episodes == 100 and args.seed_start == 200001 and not args.stochastic
        assert not args.trace
        step = int(Path(args.output).parent.name.removeprefix('step_'))
        calls.append(step)
        write_results(args.output, fixture_report(args.checkpoint, step=step))
    assert [r['status'] for r in run(work, evaluator=fake)] == ['complete', 'complete']
    assert calls == [100000, 200000]
    assert [r['status'] for r in run(work, evaluator=fake)] == ['skipped_complete'] * 2
    path = work / 'eval/step_100000/summary.json'
    report = validate_results(path)
    assert report['joint']['collision_free_rate'] == 50
    csv_path = path.with_name('episodes.csv')
    csv_path.write_text(csv_path.read_text().replace('200001', '999999', 1))
    with pytest.raises(ValueError, match='mismatch'): run(work, evaluator=fake)
    assert json.loads((work / 'eval/index.json').read_text())[0]['status'] == 'failed'
    run(work, rerun=True, evaluator=fake)
    report = json.loads(path.read_text()); report['stochastic'] = True
    atomic_json(path, report)
    with pytest.raises(ValueError, match='protocol mismatch'): run(work, evaluator=fake)


def test_plot_single_multi_seed_and_algorithm(work, monkeypatch):
    from scripts.plot_masac_experiments import plot
    experiments = []
    for algorithm, seed in [('MASAC', 42), ('MASAC', 43), ('baseline', 42)]:
        directory = work / f'{algorithm}_{seed}'
        checkpoint = directory / 'checkpoints/model.pt'
        checkpoint.parent.mkdir(parents=True)
        torch.save({'global_step': 100000}, checkpoint)
        write_results(directory / 'eval/step_100000/summary.json', fixture_report(checkpoint, seed))
        directory.joinpath('train').mkdir()
        directory.joinpath('train/metrics.jsonl').write_text(json.dumps(dict(event='update', global_step=10,
            critic_loss=1, mean_q=2, mean_target=3, ego_o3_actor_loss=1, ego_e74_actor_loss=1,
            ego_o3_alpha=.01, ego_e74_alpha=.02)) + '\n')
        experiments.append((algorithm, seed, str(directory)))
    from matplotlib.axes import Axes
    original_fill = Axes.fill_between
    fills = []
    def observed_fill(self, *args, **kwargs):
        fills.append(True)
        return original_fill(self, *args, **kwargs)
    monkeypatch.setattr(Axes, 'fill_between', observed_fill)
    plot(experiments[:1], work / 'single')
    assert not fills
    plot(experiments, work / 'multi')
    assert fills
    import pandas as pd
    table = pd.read_csv(work / 'multi/final_metrics.csv')
    available = table[(table.algorithm == 'MASAC') & table['mean'].notna()]
    assert available.training_seeds.eq(2).all()
    missing = table[table['mean'].isna()]
    assert missing.training_seeds.eq(0).all()
    assert missing.valid_samples_sum.eq(0).all()
    assert table[table.algorithm == 'baseline']['std'].isna().all()
    assert len(list((work / 'multi').glob('*.png'))) == 7
    assert len(list((work / 'multi').glob('*.pdf'))) == 7
    assert (work / 'single/plot_source.csv').exists()


def test_short_training_and_checkpoint_loading(work, monkeypatch, capsys):
    import train_masac
    from src.envs.multi_ego_env import MultiEgoEnv
    class ShortEpisodeEnv(MultiEgoEnv):
        def step(self, actions):
            if self.step_count == 30:
                for key in self.AGENTS:
                    if self.egos[key].state.status == 'active': self.active_steps[key] = 3999
            return super().step(actions)
    monkeypatch.setattr(train_masac, 'MultiEgoEnv', ShortEpisodeEnv)
    args = SimpleNamespace(total_steps=35, seed=42, learning_starts=2, batch_size=2,
        buffer_capacity=100, train_every=5, log_every=10, gamma=.999,
        learning_rate=3e-5, initial_alpha=.01, output_dir=str(work),
        checkpoint_every=10, tensorboard=False)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        train_masac.main(args)
        import evaluate_masac
        monkeypatch.setattr(evaluate_masac, 'MultiEgoEnv', ShortEpisodeEnv)
        report = evaluate_masac.evaluate(SimpleNamespace(
            checkpoint=str(work / 'checkpoints/masac_seed42_step_35.pt'), episodes=1,
            seed_start=200001, stochastic=False, max_steps=35,
            output=str(work / 'short_evaluation/summary.json')))
        assert report['environment']['requested_speed_mode'] == 96
        assert all(values and all(isinstance(v, int) for v in values) for values in report['environment']['actual_speed_modes'].values())
        for key in AGENT_IDS:
            assert report['environment']['actual_speed_modes'][key] == report['records'][0]['per_agent'][key]['speed_modes']
        assert report['sumo_version']
        validate_results(work / 'short_evaluation/summary.json')
        assert 'TRACE step=' not in capsys.readouterr().out
    finally: torch.set_num_threads(previous)
    checkpoints = sorted((work / 'checkpoints').glob('*.pt'))
    assert len(checkpoints) == 4
    index = json.loads((work / 'checkpoints/index.json').read_text())
    assert [r['global_step'] for r in index] == [10, 20, 30, 35]
    final = torch.load(work / 'checkpoints/masac_seed42_step_35.pt', weights_only=True)
    from src.agents.masac import MultiAgentGaussianActor
    actor = MultiAgentGaussianActor(); actor.load_state_dict(final['actors'], strict=True)
    assert all(sha256(r['path']) == r['sha256'] for r in index)
    records = [json.loads(line) for line in (work / 'train/metrics.jsonl').read_text().splitlines()]
    updates = [r for r in records if r['event'] == 'update']
    assert updates and all('mean_q' in r and 'valid_agents' in r for r in updates)
    assert all(r['global_step'] % 5 == 0 for r in updates)
    episodes = [r for r in records if r['event'] == 'episode']
    assert episodes and episodes[0]['episode_global_steps'] == 31
    assert episodes[0]['rolling_episodes'] == 1


def test_formal_training_defaults():
    from train_masac import build_parser
    args = build_parser().parse_args([])
    assert (args.total_steps, args.buffer_capacity, args.batch_size, args.learning_starts) == (1000000, 1000000, 512, 1000)
    assert args.learning_rate == 3e-5 and args.gamma == .999 and args.tau == .005
    assert args.checkpoint_every == 500000
