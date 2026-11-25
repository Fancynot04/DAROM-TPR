
"""DAROM 双 Ego CTDE-MASAC 训练入口。

结构化训练记录与周期 checkpoint；正式评估由独立入口执行。
不提供精确断点恢复或 W&B。
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, fields
import random
from pathlib import Path
import time

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
from src.envs import control, stalled
from src.envs.route_lane_plan import DEFAULT_ROUTE_GUIDANCE_RULES
from src.envs.route_preparation import (
    DEFAULT_LINEAR_PREPARATION_SETTINGS, DEFAULT_REFERENCE_STEP_M,
    DEFAULT_ROUTE_PREPARATION_BONUS, DEFAULT_ROUTE_PREPARATION_STAGES,
    DEFAULT_ROUTE_PREPARATION_WEIGHTS, PREPARATION_BONUS_MODES, PREPARATION_PENALTY,
    PREPARATION_DISTANCE_MODES, PREPARATION_LINEAR_V1, PREPARATION_V1, V1_REFERENCE_STEP_M,
    checkpoint_preparation_settings,
)
from src.envs.stalled import stalled_block_grace_for_version
from src.rewards import masac as reward_parameters
from src.rewards.events import EMERGENCY_BRAKING_EVENT, EVENTS
from src.rewards.masac import (
    LEGACY_LANE_CHANGE_WEIGHT, LEGACY_MASAC_ENV_VERSION, MASACRewardConfig, MASACRewardProtocol, MASAC_ENV_VERSION,
    SUPPORTED_MASAC_ENV_VERSIONS,
    TARGET_EXIT_DIM,
)
from src.agents.masac import MultiAgentGaussianActor
from src.agents.centralized_critic import CentralizedTwinQ
from src.agents.replay_buffer import JointReplayBuffer
from src.agents.transition_collector import JointTransitionCollector
from src.agents.masac_trainer import MASACTrainer


from src.experiment_tracking import (TrainingLog, ActionWindow, EpisodeWindow,
    checkpoint_due, atomic_write, atomic_json, sha256, git_metadata, evaluation_fingerprint,
    runtime_versions)
from src.evaluation_metrics import AgentMetrics

AGENT_IDS = JointReplayBuffer.AGENT_IDS
DEFAULT_CHECKPOINT_EVERY = 500_000
DEFAULT_GAMMA = 0.99
SUPPORTED_GAMMAS = (DEFAULT_GAMMA, 0.995)
EXPLORATION_BOUNDS = (-5.0, 5.0)

RESUME_CONFIG_KEYS = (
    "seed", "batch_size", "buffer_capacity", "learning_starts", "train_every",
    "gamma", "tau", "learning_rate", "initial_alpha", "team_q_weight",
    "masac_env_version", "actor_leader_features", "max_active_steps", "usp",
    "usp_penalty", "stalled_terminate", "emergency_braking_terminate", "stalled_block_grace_s",
    "route_guidance_rules",
)


def snapshot(env, observations):
    """读取同一 SUMO 时刻的联合状态。"""

    policy_obs = np.stack([
        env.policy_observations[ego_id]
        for ego_id in AGENT_IDS
    ]).astype(np.float32)

    critic_obs = np.stack([
        observations[ego_id].reshape(-1)
        for ego_id in AGENT_IDS
    ]).astype(np.float32)

    if env.target_exit_dim:
        targets = np.stack([env.target_exit_features(ego_id) for ego_id in AGENT_IDS])
        critic_obs = np.concatenate([critic_obs, targets], axis=1)

    active_mask = np.asarray([
        float(
            env.egos[ego_id].state.status == "active"
            and not env.terminations[ego_id]
        )
        for ego_id in AGENT_IDS
    ], dtype=np.float32)

    return policy_obs, critic_obs, active_mask


def choose_actions(
    actors,
    policy_obs,
    active_mask,
    rng,
    random_exploration,
):
    """随机探索或通过两辆独立 Actor 采样动作。"""

    if random_exploration:
        joint_actions = rng.uniform(
            *EXPLORATION_BOUNDS, size=(2, 2)
        ).astype(np.float32)

    else:
        device = next(actors.parameters()).device
        with torch.inference_mode():
            action_tensor, _ = actors(
                torch.as_tensor(policy_obs[None], device=device),
                torch.as_tensor(active_mask[None], device=device),
                deterministic=False,
            )

        joint_actions = (
            action_tensor[0].cpu().numpy().copy()
        )

    joint_actions[active_mask == 0] = 0.0

    action_dict = {
        ego_id: joint_actions[i].copy()
        for i, ego_id in enumerate(AGENT_IDS)
        if active_mask[i] == 1
    }

    return joint_actions, action_dict


def resolve_run_config(args, env, trainer, buffer, episode_seed_index):
    seed = args.seed + episode_seed_index
    world = env.make_world(seed)
    reward = env.reward_tracker
    actors = {
        ego_id: dict(
            observation_dim=actor.obs_dim, action_dim=actor.action_dim,
            action_scale=actor.action_scale, log_std_bounds=actor.LOG_STD_BOUNDS,
            encoder_input_mode=actor.encoder.input_mode, nav_dim=actor.encoder.nav_dim,
            action_history_steps=actor.encoder.hist_len, delay_dim=actor.encoder.delay_dim,
            observation_scale=actor.encoder.observation_scale.tolist(),
            architecture=repr(actor),
        )
        for ego_id, actor in trainer.actors.actors.items()
    }
    critics = {
        name: dict(
            num_agents=model.num_agents, observation_dim=model.obs_dim,
            action_dim=model.action_dim, action_scale=model.action_scale,
            observation_scale=model.observation_scale.tolist(), architecture=repr(model),
        ) if model is not None else None
        for name, model in (("critic", trainer.critic), ("team_critic", trainer.team_critic))
    }
    optimizers = {
        "critic": trainer.critic_optimizer, "team_critic": trainer.team_critic_optimizer,
        **{f"actor_{ego}": opt for ego, opt in trainer.actor_optimizers.items()},
        **{f"alpha_{ego}": opt for ego, opt in trainer.alpha_optimizers.items()},
    }
    config = {
        **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "resolved_config_version": 1,
        "runtime_versions": runtime_versions(),
        "environment": dict(
            ego_ids=env.AGENTS, scenario=env.scenario, speed_mode=env.speed_mode, gui=env.gui,
            collect_metrics=env.collect_metrics, collect_control_trace=env.collect_control_trace,
            observation_shape=env.OBS_SHAPE, observation_radius_m=env.OBS_RADIUS_M,
            target_exit_dim=env.target_exit_dim, legacy_route_semantics=env.legacy_route_semantics,
            actor_leader_features=env.actor_leader_features, max_active_steps=env.max_active_steps,
            action_history_steps=env.action_history.hist_len, max_delay_steps=env.delay_buffer.max_delay,
            delayed_observation_shape=env.delay_buffer.observation_shape,
            delay_constructor_seed=env.delay_buffer.seed, episode_seed_base=args.seed,
            initial_episode_seed=seed, initial_episode_seed_index=episode_seed_index,
            stalled_shadow=env.stalled_shadow, stalled_terminate=env.stalled_terminate,
            stalled_block_grace_s=env.stalled_block_grace_s, stalled_match_route_links=env.stalled_match_route_links,
            stalled_window_s=stalled.WINDOW_SECONDS, stalled_min_progress_m=stalled.MIN_PROGRESS_M,
            stalled_leader_range_m=stalled.LEADER_RANGE_M, stalled_junction_approach_m=stalled.JUNCTION_APPROACH_M,
            acceleration_bounds_mps2=[control.MIN_ACCEL, control.MAX_ACCEL], max_speed_mps=control.MAX_SPEED,
            lane_change_edges=sorted(control.LANE_CHANGE_EDGES), exit_edges=sorted(control.EXIT_EDGES),
            exit_lane_change_duration_s=control.EXIT_LANE_CHANGE_DURATION,
            sumo=dict(command=world.launch_command(), step_length_s=world.step_length,
                      net_file=str(world.net_file), net_sha256=sha256(world.net_file),
                      route_file=str(world.route_file), route_sha256=sha256(world.route_file)),
        ),
        "reward": dict(
            config=asdict(reward.config), protocol=asdict(reward.protocol),
            route_guidance_rules=args.route_guidance_rules if reward.config.route_guidance else [],
            route_preparation_settings=(
                args.route_preparation_settings or DEFAULT_LINEAR_PREPARATION_SETTINGS
            ) if env.route_preparation is not None else None,
            terminal_values={
                event: reward.terminal(event)
                for event, key in EVENTS.items()
                if key in reward_parameters.TASK_EVENTS or event == EMERGENCY_BRAKING_EVENT
            },
            following_weight=reward_parameters.FOLLOWING_WEIGHT,
            reaction_time_s=reward_parameters.REACTION_TIME_S,
            lane_change_min_gap_m=reward_parameters.LANE_CHANGE_MIN_GAP_M,
            lane_change_reference_decel_mps2=reward_parameters.LANE_CHANGE_REFERENCE_DECEL,
            efficiency_reference_speed_mps=reward.protocol.efficiency_reference_speed_mps,
            comfort_delta_cap_mps2=reward_parameters.COMFORT_DELTA_CAP_MPS2,
            team_success_reward=reward_parameters.TEAM_SUCCESS_REWARD,
            usp=dict(enabled=env.usp_enabled, penalty=env.usp.penalty, stop_speed_mps=env.usp.STOP_SPEED,
                     grace_s=env.usp.GRACE_SECONDS, junction_margin_m=env.usp.JUNCTION_MARGIN,
                     road_edges=sorted(env.usp.ROAD_EDGES)),
        ),
        "networks": {"actors": actors, **critics},
        "trainer": dict(
            gamma=trainer.gamma, tau=trainer.tau, target_entropy=trainer.target_entropy,
            team_q_weight=trainer.team_q_weight, initial_alpha=args.initial_alpha,
            exploration_action_bounds=EXPLORATION_BOUNDS,
            optimizers={
                name: dict(type=type(opt).__name__, param_groups=[
                    {key: value for key, value in group.items() if key != "params"}
                    for group in opt.param_groups
                ])
                for name, opt in optimizers.items() if opt is not None
            },
        ),
        "replay_buffer": dict(capacity=buffer.capacity, shapes=buffer.shapes, mask_fields=buffer.mask_fields),
    }
    # Normalize tuples to JSON lists so checkpoint metadata matches the run file.
    return json.loads(json.dumps(config, allow_nan=False))


def save_checkpoint(
    path,
    actors,
    critic,
    trainer,
    global_step,
    episodes,
    config,
    resume_state,
):
    """Save model weights, with optimizer and replay metadata for resumable checkpoints."""

    team_state = {}
    if trainer.team_critic is not None:
        team_state = {
            "team_critic": trainer.team_critic.state_dict(),
            "target_team_critic": trainer.target_team_critic.state_dict(),
        }
    state = {
        **team_state,
        "git": git_metadata(),
        "source_sha256": evaluation_fingerprint(),
        "global_step": global_step,
        "episodes": episodes,
        "agent_ids": AGENT_IDS,
        "config": config,
        "resume_state": resume_state,
        "trainer_config": {"gamma": trainer.gamma, "tau": trainer.tau, "target_entropy": trainer.target_entropy},
        "actors": actors.state_dict(),
        "critic": critic.state_dict(),
        "target_critic":
            trainer.target_critic.state_dict(),
        "log_alpha": trainer.log_alpha.state_dict(),
    }
    if resume_state is not None:
        state.update(
            critic_optimizer=trainer.critic_optimizer.state_dict(),
            actor_optimizers={k: v.state_dict() for k, v in trainer.actor_optimizers.items()},
            alpha_optimizers={k: v.state_dict() for k, v in trainer.alpha_optimizers.items()},
        )
        if trainer.team_critic is not None:
            state["team_critic_optimizer"] = trainer.team_critic_optimizer.state_dict()
    atomic_write(path, lambda stream: torch.save(state, stream))
    return checkpoint_index_row(path, state, sha256(path))


def checkpoint_index_row(path, state, digest):
    return dict(training_seed=state["config"]["seed"], global_step=state["global_step"],
                path=str(path.resolve()), sha256=digest, git=state["git"],
                source_sha256=state["source_sha256"], config=state["config"],
                replay=state["resume_state"]["replay"] if state["resume_state"] is not None else None)


def preparation_settings(args, saved=None, *, resuming=False):
    stage_file = getattr(args, "route_preparation_stages", None)
    bonus = getattr(args, "reward_route_preparation_bonus_cap", None)
    penalty_weights = {
        kind: getattr(args, "reward_route_preparation_" + kind + "_weight", None)
        for kind in DEFAULT_ROUTE_PREPARATION_WEIGHTS
    }
    legacy_override = any(value is not None for value in penalty_weights.values())
    mode = getattr(args, "reward_route_preparation_mode", None)
    if mode is None:
        if legacy_override:
            mode = PREPARATION_PENALTY
        elif (bonus is not None and (saved is None or saved["mode"] == PREPARATION_PENALTY)
              or stage_file and saved is None):
            mode = PREPARATION_V1
    if resuming and saved is None and mode is None and stage_file is None:
        return None
    saved = saved or {}
    mode = mode or saved.get("mode", PREPARATION_LINEAR_V1)
    if mode == PREPARATION_LINEAR_V1:
        if stage_file or bonus is not None or legacy_override:
            raise ValueError("linear_progress_v1 uses fixed parameters and the planned E77 to D3 exit")
        return DEFAULT_LINEAR_PREPARATION_SETTINGS.copy()
    stages = saved.get("stages", DEFAULT_ROUTE_PREPARATION_STAGES)
    if stage_file:
        stages = json.loads(Path(stage_file).read_text(encoding="utf-8"))
    if mode in PREPARATION_BONUS_MODES:
        if legacy_override:
            raise ValueError("Legacy preparation weights require legacy_penalty mode")
        if bonus is None:
            bonus = saved.get("bonus_cap", DEFAULT_ROUTE_PREPARATION_BONUS)
        settings = {"mode": mode, "stages": stages, "bonus_cap": bonus}
        if mode in PREPARATION_DISTANCE_MODES:
            default_step = V1_REFERENCE_STEP_M if mode == PREPARATION_V1 else DEFAULT_REFERENCE_STEP_M
            settings["reference_step_m"] = (
                saved.get("reference_step_m", default_step) if mode == saved.get("mode") else default_step
            )
        return settings
    if bonus is not None:
        raise ValueError("Preparation bonus cap requires a bonus mode")
    weights = saved.get("weights", DEFAULT_ROUTE_PREPARATION_WEIGHTS)
    return {"mode": mode, "stages": stages, "weights": {
        kind: weights[kind] if value is None else value for kind, value in penalty_weights.items()
    }}


def find_resume_checkpoint(directory, args):
    paths = {path.name: path for path in directory.glob("*.pt")}
    index_path = directory / "index.json"
    if not paths and not index_path.exists():
        return None, None, []
    if not args.resume:
        raise FileExistsError("Experiment already has checkpoints; use a new --output-dir with --no-resume")

    index = json.loads(index_path.read_text()) if index_path.exists() else []
    indexed = {Path(row["path"]).name: row for row in index}
    if len(indexed) != len(index) or indexed.keys() - paths.keys():
        raise ValueError("Checkpoint index contains missing or duplicate checkpoints")
    # A model published just before an interruption is complete even if its index was not written.
    orphaned = paths.keys() - indexed.keys()
    states = {}
    for name in sorted(orphaned):
        path = paths[name]
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not state.get("resume_state"):
            raise ValueError(f"{path} has no resumable Replay Buffer; use a new --output-dir")
        row = checkpoint_index_row(path, state, sha256(path))
        index.append(row)
        states[name] = state
    if not index or len({row["global_step"] for row in index}) != len(index):
        raise ValueError("Checkpoint index has no entries or has duplicate training steps")
    index.sort(key=lambda row: row["global_step"])
    latest = index[-1]
    name = Path(latest["path"]).name
    path = paths[name]
    if name not in states and sha256(path) != latest["sha256"]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    state = states[name] if name in states else torch.load(path, map_location="cpu", weights_only=True)
    resume = state.get("resume_state")
    if not resume:
        raise ValueError(f"{path} has no resumable Replay Buffer; use a new --output-dir")
    if (resume["version"] != 1 or tuple(state["agent_ids"]) != AGENT_IDS
            or state["global_step"] != latest["global_step"]):
        raise ValueError(f"Invalid resume metadata: {path}")
    changed = [key for key in RESUME_CONFIG_KEYS
               if state["config"].get(key)
               != getattr(args, key)]
    if state["config"].get("decision_interval_steps", 1) != getattr(args, "decision_interval_steps", 1):
        changed.append("decision_interval_steps")
    saved_route_settings = checkpoint_preparation_settings(state["config"].get("route_preparation_settings"))
    route_settings = preparation_settings(args, saved_route_settings, resuming=True)
    if saved_route_settings != route_settings:
        changed.append("route_preparation_settings")
    saved_components = {
        "rear_following": False,
        "exit_retention": False,
        **state["config"].get("reward_components", {}),
    }
    saved_components["route_preparation"] = (
        saved_route_settings is not None and saved_components.get("route_preparation", False)
    )
    components = args.reward_components.copy()
    for component in ("route_preparation", "exit_retention"):
        if getattr(args, "reward_" + component, None) is None:
            components[component] = saved_components[component]
    if components != saved_components:
        changed.append("reward_components")
    saved_safety = {
        "ttc_threshold": MASACRewardConfig.ttc_threshold,
        "following_lookahead": MASACRewardConfig.following_lookahead,
        "lane_change_weight": LEGACY_LANE_CHANGE_WEIGHT,
        **state["config"].get("safety_reward", {}),
    }
    safety = args.safety_reward.copy()
    if getattr(args, "reward_lane_change_weight", None) is None:
        safety["lane_change_weight"] = saved_safety["lane_change_weight"]
    if safety != saved_safety:
        changed.append("safety_reward")
    reward_protocol = args.reward_protocol.copy()
    saved_settings = state["config"].get("reward_protocol")
    if saved_settings is None:
        changed.append("reward_protocol")
    else:
        saved_protocol = MASACRewardProtocol.from_checkpoint(saved_settings)
        reward_protocol["merge_on_success"] = saved_protocol.merge_on_success
        reward_protocol["merge_reward"] = saved_protocol.merge_reward
        reward_protocol["efficiency_reference_speed_mps"] = saved_protocol.efficiency_reference_speed_mps
        if getattr(args, "reward_comfort_weight", None) is None:
            reward_protocol["comfort_weight"] = saved_protocol.comfort_weight
        if saved_protocol != MASACRewardProtocol(**reward_protocol):
            changed.append("reward_protocol")
    if changed:
        raise ValueError(f"Resume configuration mismatch: {', '.join(changed)}")
    replay_path = path.with_suffix(".replay.npz")
    if not replay_path.is_file() or sha256(replay_path) != resume["replay"]["sha256"]:
        raise ValueError(f"Replay Buffer missing or SHA256 mismatch: {replay_path}")
    if orphaned:
        atomic_json(index_path, index)
    args.reward_protocol = reward_protocol
    args.route_preparation_settings = route_settings
    args.reward_components = components
    args.safety_reward = safety
    return state, replay_path, index


def restore_training_state(state, trainer):
    for key in ("actors", "critic", "target_critic", "log_alpha", "critic_optimizer"):
        getattr(trainer, key).load_state_dict(state[key])
    if trainer.team_critic is not None:
        for key in ("team_critic", "target_team_critic", "team_critic_optimizer"):
            getattr(trainer, key).load_state_dict(state[key])
    for key in ("actor_optimizers", "alpha_optimizers"):
        for ego_id, optimizer in getattr(trainer, key).items():
            optimizer.load_state_dict(state[key][ego_id])


def initialize_actor_weights(actors, path, expected_sha256, actor_leader_features):
    """Start an independent reward run with only the verified source Actor weights."""
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError(f"Actor initialization checkpoint SHA-256 mismatch: {path}")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    config = state.get("config", {})
    if (
        tuple(state.get("agent_ids", ())) != AGENT_IDS
        or config.get("masac_env_version") not in SUPPORTED_MASAC_ENV_VERSIONS[1:]
        or config.get("actor_leader_features", False) != actor_leader_features
    ):
        raise ValueError("Actor initialization requires matching route-aware observation semantics and Ego IDs")
    actors.load_state_dict(state["actors"], strict=True)
    return {"path": str(Path(path).resolve()), "sha256": digest, "source_global_step": state["global_step"],
            "mode": "actors_only_fresh_critics_optimizers_alpha_and_replay"}


def resolve_training_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested, but PyTorch cannot access a CUDA device")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {index}")
        return torch.device("cuda", index)
    if device != torch.device("cpu"):
        raise ValueError("Device must be auto, cpu, cuda, or cuda:N")
    return device


def main(args):
    requested = getattr(args, "device", "auto")
    device = resolve_training_device(requested)
    previous_threads = torch.get_num_threads()
    threads = getattr(args, "torch_threads", None)
    if threads is None:
        threads = 2 if device.type == "cuda" else min(previous_threads, 8)
    if threads <= 0:
        raise ValueError("torch_threads must be positive")

    args.requested_device = requested
    args.device = str(device)
    args.effective_torch_threads = threads
    torch.set_num_threads(threads)
    try:
        print(f"Training device: {device} (requested: {requested}, PyTorch CPU threads: {threads})", flush=True)
        return train(args, device)
    finally:
        torch.set_num_threads(previous_threads)


def train(args, device):
    import fcntl

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training already active in {output_dir}") from error
        return train_locked(args, device, output_dir)


def train_locked(args, device, output_dir):
    if args.total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.buffer_capacity < args.batch_size:
        raise ValueError("Buffer smaller than batch")

    if args.learning_starts < 0:
        raise ValueError("Invalid learning_starts")

    if args.train_every <= 0 or args.log_every <= 0:
        raise ValueError("Invalid interval")
    args.decision_interval_steps = getattr(args, "decision_interval_steps", 1)
    if args.decision_interval_steps < 1:
        raise ValueError("decision_interval_steps must be positive")
    decision_transitions = args.decision_interval_steps > 1

    args.trace_controls = getattr(args, "trace_controls", False)
    args.stalled_shadow = getattr(args, "stalled_shadow", False)
    args.usp = getattr(args, "usp", False)
    args.stalled_terminate = getattr(args, "stalled_terminate", None)
    args.emergency_braking_terminate = True
    if args.stalled_terminate is None:
        args.stalled_terminate = not args.usp
    if args.stalled_terminate and args.usp:
        raise ValueError("Stalled termination and USP cannot both be enabled")
    args.max_active_steps = getattr(args, "max_active_steps", MultiEgoEnv.DEFAULT_MAX_ACTIVE_STEPS)
    if args.max_active_steps <= 0:
        raise ValueError("max_active_steps must be positive")
    args.update_log_every = getattr(args, "update_log_every", 40)
    if args.update_log_every <= 0:
        raise ValueError("update_log_every must be positive")

    args.gamma = getattr(args, "gamma", DEFAULT_GAMMA)
    if args.gamma not in SUPPORTED_GAMMAS:
        raise ValueError(f"O3-D3 MASAC gamma must be one of {SUPPORTED_GAMMAS}")
    args.masac_env_version = getattr(args, "masac_env_version", MASAC_ENV_VERSION)
    if args.masac_env_version not in SUPPORTED_MASAC_ENV_VERSIONS:
        raise ValueError(f"Unsupported MASAC environment version: {args.masac_env_version}")
    args.stalled_block_grace_s = stalled_block_grace_for_version(args.masac_env_version)
    args.actor_leader_features = getattr(args, "actor_leader_features", False)
    args.tau = getattr(args, "tau", 0.005)
    args.checkpoint_every = getattr(args, "checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
    args.resume = getattr(args, "resume", True)
    args.replay_final_only = getattr(args, "replay_final_only", False)
    args.tensorboard = getattr(args, "tensorboard", True)
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.usp_penalty = getattr(
        args, "usp_penalty", 0.02
    )

    args.reward_components = {}
    args.safety_reward = {}
    for field in fields(MASACRewardConfig):
        value = getattr(args, "reward_" + field.name, None)
        target = args.reward_components if isinstance(field.default, bool) else args.safety_reward
        target[field.name] = field.default if value is None else value
    comfort_weight = getattr(args, "reward_comfort_weight", None)
    reward_protocol = MASACRewardProtocol(
        route_guidance_weight=getattr(
            args, "reward_route_guidance_weight", MASACRewardProtocol.route_guidance_weight,
        ),
        terminal_penalty_scale=getattr(
            args, "terminal_penalty_scale", MASACRewardProtocol.terminal_penalty_scale,
        ),
        collision_penalty=getattr(args, "collision_penalty", None),
        comfort_weight=MASACRewardProtocol.comfort_weight if comfort_weight is None else comfort_weight,
    )
    args.reward_protocol = asdict(reward_protocol)
    if reward_protocol.collision_penalty is None:
        del args.reward_protocol["collision_penalty"]
    args.route_guidance_rules = getattr(args, "route_guidance_rules", DEFAULT_ROUTE_GUIDANCE_RULES)
    args.team_q_weight = getattr(args, "team_q_weight", 1.0)

    checkpoint_dir = output_dir / "checkpoints"
    state, replay_path, checkpoint_index = find_resume_checkpoint(checkpoint_dir, args)
    if state is not None:
        reward_protocol = MASACRewardProtocol(**args.reward_protocol)
    else:
        args.route_preparation_settings = preparation_settings(args)
    reward_config = MASACRewardConfig(**args.reward_components, **args.safety_reward)
    args.reward_lane_change_weight = reward_config.lane_change_weight
    args.reward_route_preparation = reward_config.route_preparation
    if reward_config.team and (not math.isfinite(args.team_q_weight) or args.team_q_weight <= 0):
        raise ValueError("team_q_weight must be finite and positive when team reward is enabled")
    start_step = state["global_step"] if state is not None else 0
    if state is None and (output_dir / "train/metrics.jsonl").exists():
        raise FileExistsError("Existing training log has no resumable checkpoint; use a new --output-dir")
    if args.total_steps < start_step:
        raise ValueError(f"--total-steps must be at least the saved step ({start_step})")
    if args.total_steps == start_step:
        config_path = output_dir / "resolved_config.json"
        if not config_path.exists() and state["config"].get("resolved_config_version"):
            atomic_json(config_path, state["config"])
        print(f"Training already reached {start_step} global steps", flush=True)
        return

    # 1. 创建网络及优化器
    actors = MultiAgentGaussianActor(
        ego_ids=AGENT_IDS, nav_dim=TARGET_EXIT_DIM,
        actor_leader_features=args.actor_leader_features,
    ).to(device)
    init_checkpoint = getattr(args, "init_actors_from", None)
    init_sha = getattr(args, "init_actors_sha256", None)
    if init_sha is not None and init_checkpoint is None:
        raise ValueError("--init-actors-sha256 requires --init-actors-from")
    if state is not None:
        args.actor_initialization = state["config"].get("actor_initialization")
    else:
        args.actor_initialization = (
            initialize_actor_weights(actors, init_checkpoint, init_sha, args.actor_leader_features)
            if init_checkpoint else None
        )
    critic = CentralizedTwinQ(
        num_agents=len(AGENT_IDS), nav_dim=TARGET_EXIT_DIM,
    ).to(device)

    team_critic = CentralizedTwinQ(nav_dim=TARGET_EXIT_DIM, team=True).to(device) if reward_config.team else None

    trainer = MASACTrainer(
        actors=actors,
        critic=critic,
        gamma=args.gamma,
        tau=args.tau,
        learning_rate=args.learning_rate,
        alpha=(args.initial_alpha,) * len(AGENT_IDS),
        team_critic=team_critic,
        team_q_weight=args.team_q_weight,
    )

    # 2. 创建联合经验池
    buffer = JointReplayBuffer(
        capacity=args.buffer_capacity,
        seed=args.seed,
        agent_ids=AGENT_IDS,
        nav_dim=TARGET_EXIT_DIM,
        team=reward_config.team,
        policy_obs_dim=actors.obs_dim,
        decision_transitions=decision_transitions,
    )
    resume = None if state is None else state["resume_state"]
    if state is not None:
        buffer.load(replay_path)
        if len(buffer) != resume["replay"]["size"]:
            raise ValueError("Replay Buffer size does not match the checkpoint")
        restore_training_state(state, trainer)
    completed_episodes = state["episodes"] if state is not None else 0
    update_count = resume["update_count"] if resume is not None else 0
    episode_seed_index = resume["next_episode_seed_index"] if resume is not None else 0
    last_losses = resume["last_losses"] if resume is not None else {}
    del state

    # 3. 创建双 Ego 环境
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
        collect_metrics=True,
        usp_enabled=args.usp,
        usp_penalty=args.usp_penalty,
        reward_config=reward_config,
        reward_protocol=reward_protocol,
        route_guidance_rules=args.route_guidance_rules,
        route_preparation_settings=args.route_preparation_settings,
        max_active_steps=args.max_active_steps,
        collect_control_trace=args.trace_controls,
        stalled_shadow=args.stalled_shadow,
        stalled_terminate=args.stalled_terminate,
        stalled_block_grace_s=args.stalled_block_grace_s,
        stalled_match_route_links=args.masac_env_version == MASAC_ENV_VERSION,
        legacy_route_semantics=args.masac_env_version == LEGACY_MASAC_ENV_VERSION,
        actor_leader_features=args.actor_leader_features,
    )

    resolved_config = resolve_run_config(args, env, trainer, buffer, episode_seed_index)
    atomic_json(output_dir / "resolved_config.json", resolved_config)
    provenance = git_metadata()
    episode_window = EpisodeWindow()
    if resume is not None:
        episode_window.rows.extend(resume["episode_window"])
    log = TrainingLog(output_dir / "train", args.tensorboard,
                      resume_step=start_step if resume is not None else None)
    started = time.monotonic()

    try:
        if resume is None:
            log.write("config", 0, config=vars(args), git=provenance, policy="exploration_training")
        else:
            log.write("resume", start_step, config=vars(args), git=provenance,
                      replay_size=len(buffer), update_count=update_count,
                      sumo_seed=args.seed + episode_seed_index,
                      replay_sha256=resume["replay"]["sha256"],
                      discarded_pending_transition=resume["pending_transition"])
            print(f"Resuming at step {start_step} with {len(buffer)} replay transitions", flush=True)
        action_window = ActionWindow(AGENT_IDS)
        observations, _ = env.reset(seed=args.seed + episode_seed_index)
        if resume is not None:
            random.setstate(resume["python_rng"])
            rng.bit_generator.state = resume["exploration_rng"]
            torch.set_rng_state(resume["torch_rng"])
            if device.type == "cuda" and resume["cuda_rng"] is not None:
                torch.cuda.set_rng_state(resume["cuda_rng"], device=device)
        collector = JointTransitionCollector(buffer)
        decision_transition = None
        decision_age = 0

        episode_rewards = np.zeros(
            len(AGENT_IDS), dtype=np.float64
        )
        outcomes = {}
        episode_start = start_step
        episode_metrics = {key: AgentMetrics() for key in AGENT_IDS}

        episode_usp_steps = {
            ego_id: 0 for ego_id in AGENT_IDS
        }

        episode_usp_total = {
            ego_id: 0.0 for ego_id in AGENT_IDS
        }

        for global_step in range(start_step + 1, args.total_steps + 1):

            # A. 推进前快照
            policy_obs, critic_obs, active_mask = (
                snapshot(env, observations)
            )

            # B. 随机探索或 Actor 决策
            if decision_transition is None:
                joint_actions, action_dict = choose_actions(
                    actors, policy_obs, active_mask, rng,
                    random_exploration=global_step <= args.learning_starts,
                )
            else:
                joint_actions = decision_transition["actions"].copy()
                joint_actions[:, 1] = 0.0
                action_dict = {
                    ego_id: joint_actions[i]
                    for i, ego_id in enumerate(AGENT_IDS)
                    if active_mask[i]
                }

            # C. SUMO 只推进一次
            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(action_dict)
            
            if args.stalled_shadow:
                log.write("stalled_shadow", global_step, episode_index=completed_episodes + 1,
                          egos={key: infos[key]["stalled_shadow"] for key in AGENT_IDS})
            if args.trace_controls:
                traces = {
                    ego_id: {**(info["control_trace"] or {}),
                             "terminal_reason": info["terminal_reason"],
                             "reward_components": info["reward_components"]}
                    for ego_id, info in infos.items()
                    if info["control_trace"] is not None or info["pre_step_terminal"]
                }
                if traces:
                    log.write("control_trace", global_step, episode_index=completed_episodes + 1, egos=traces)

            # 统计本步每辆 Ego 实际支付的 USP 奖励
            for ego_id in AGENT_IDS:
                usp_value = float(
                    infos[ego_id].get("usp_reward", 0.0)
                )

                episode_usp_total[ego_id] += usp_value

                if usp_value < 0.0:
                    episode_usp_steps[ego_id] += 1

            # D. 推进后快照
            (
                next_policy,
                next_critic,
                next_mask,
            ) = snapshot(env, next_observations)

            rewards_array = np.asarray([
                rewards[ego_id]
                for ego_id in AGENT_IDS
            ], dtype=np.float32)

            pre_step = np.asarray([
                float(
                    active_mask[i] == 1
                    and infos[ego_id].get(
                        "pre_step_terminal", False
                    )
                )
                for i, ego_id in enumerate(AGENT_IDS)
            ], dtype=np.float32)

            acted_mask = active_mask * (1.0 - pre_step)
            action_window.observe(joint_actions, acted_mask)
            for key in AGENT_IDS:
                if key not in outcomes:
                    episode_metrics[key].observe(infos[key]["evaluation_metrics"],
                        terminations[key] or truncations[key])

            dones = np.asarray([
                float(
                    active_mask[i] == 1
                    and (
                        terminations[ego_id]
                        or truncations[ego_id]
                    )
                )
                for i, ego_id in enumerate(AGENT_IDS)
            ], dtype=np.float32)

            transition = {
                "policy_obs": policy_obs,
                "critic_obs": critic_obs,
                "actions": joint_actions,
                "rewards": rewards_array,
                "dones": dones,
                "active_mask": active_mask,
                "acted_mask": acted_mask,
                "next_policy_obs": next_policy,
                "next_critic_obs": next_critic,
                "next_active_mask": next_mask,
            }

            if reward_config.team:
                reasons = [env.terminal_reasons[key] for key in AGENT_IDS]
                transition.update(
                    team_reward=env.team_reward,
                    team_active=float(all(reason == "Arrived" for reason in outcomes.values())),
                    team_done=float(all(terminations.values()) or any(
                        reason not in (None, "Arrived") for reason in reasons
                    )),
                    success_mask=[outcomes.get(key) == "Arrived" for key in AGENT_IDS],
                    next_success_mask=[reason == "Arrived" for reason in reasons],
                )

            # E. 由采集器处理 pre-step 奖励归因
            if decision_transitions:
                if decision_transition is None:
                    decision_transition = transition
                    decision_pre_step = pre_step
                else:
                    weight = args.gamma ** decision_age
                    decision_transition["rewards"] += weight * transition["rewards"]
                    np.maximum(decision_transition["dones"], dones, out=decision_transition["dones"])
                    for key in ("next_policy_obs", "next_critic_obs", "next_active_mask"):
                        decision_transition[key] = transition[key]
                    if reward_config.team:
                        decision_transition["team_reward"] += weight * transition["team_reward"]
                        decision_transition["team_done"] = transition["team_done"]
                        decision_transition["next_success_mask"] = transition["next_success_mask"]
                decision_age += 1
                # Newly active Egos need a first action immediately on the next SUMO step.
                if (decision_age == args.decision_interval_steps
                        or np.any(next_mask > active_mask)
                        or all(terminations.values())
                        or global_step == args.total_steps):
                    decision_transition["bootstrap_discount"] = args.gamma ** decision_age
                    collector.push(decision_transition, pre_step_terminal=decision_pre_step)
                    decision_transition = None
                    decision_age = 0
            else:
                collector.push(transition, pre_step_terminal=pre_step)

            episode_rewards += rewards_array

            for ego_id in AGENT_IDS:
                if ego_id in outcomes:
                    continue

                if (
                    terminations[ego_id]
                    or truncations[ego_id]
                ):
                    outcomes[ego_id] = infos[
                        ego_id
                    ]["terminal_reason"]

            # F. 经验足够时更新网络
            if (
                global_step >= args.learning_starts
                and len(buffer) >= args.batch_size
                and global_step % args.train_every == 0
            ):
                update_count += 1
                record_update = (
                    update_count == 1
                    or update_count % args.update_log_every == 0
                    or global_step + args.train_every > args.total_steps
                )
                batch = buffer.sample(args.batch_size, device=device)

                critic_stats = trainer.update_critic(batch, collect_stats=record_update)
                actor_stats = trainer.update_actors(batch, collect_stats=record_update)
                alpha_stats = trainer.update_alpha(batch, collect_stats=record_update)

                if record_update:
                    last_losses = {**critic_stats, **actor_stats, **alpha_stats}
                    alphas = trainer.alpha_values().cpu().tolist()
                    for i, key in enumerate(AGENT_IDS):
                        last_losses.setdefault(key + "_alpha", alphas[i])
                        last_losses.setdefault(key + "_actor_loss", None)
                        last_losses.setdefault(key + "_mean_log_prob", None)
                        last_losses.setdefault(key + "_valid_samples", 0)
                    log.write("update", global_step, update_index=update_count, **last_losses)

            if global_step % args.log_every == 0 or global_step == args.total_steps:
                log.write("runtime", global_step, replay_size=len(buffer), update_count=update_count,
                          **action_window.result())
                action_window.clear()

            # G. 打印训练进度
            if global_step % args.log_every == 0:
                alpha = trainer.alpha_values().cpu().numpy()

                print(
                    f"step={global_step} "
                    f"replay={len(buffer)} "
                    f"updates={update_count} "
                    f"episodes={completed_episodes} "
                    f"alpha={np.round(alpha, 8)} "
                    f"critic_loss="
                    f"{last_losses.get('critic_loss', 'N/A')}",
                    flush=True,
                )

            observations = next_observations

            # H. 两辆车都终止后，开始新回合
            episode_finished = len(outcomes) == len(AGENT_IDS)
            if episode_finished:
                collector.flush()

                completed_episodes += 1
                agents = {key: episode_metrics[key].finish(outcomes[key], episode_rewards[i])
                    for i, key in enumerate(AGENT_IDS)}   

                episode_values = {}
                for key, row in agents.items():
                    # 原有训练指标
                    for name in (
                        "terminal_reason",
                        "success",
                        "agent_steps",
                        "mean_policy_reward_return",
                    ):
                        episode_values[key + "_" + name] = row[name]

                    # 新增 USP 统计
                    episode_values[key + "_usp_steps"] = (
                        episode_usp_steps[key]
                    )

                    episode_values[key + "_usp_reward_total"] = (
                        episode_usp_total[key]
                    )        
                    
                log.write("episode", global_step, episode_index=completed_episodes,
                    episode_global_steps=global_step - episode_start,
                    joint_success=all(r["success"] for r in agents.values()),
                    **env.reward_summary(), **episode_values, **episode_window.add(agents))

                print(
                    f"episode={completed_episodes} "
                    f"outcomes={outcomes} "
                    f"rewards={episode_rewards.tolist()}",
                    flush=True,
                )

            if checkpoint_due(global_step, args.total_steps, args.checkpoint_every):
                checkpoint = checkpoint_dir / f"masac_seed{args.seed}_step_{global_step}.pt"
                resume_state = None
                if not args.replay_final_only or global_step == args.total_steps:
                    replay = checkpoint.with_suffix(".replay.npz")
                    atomic_write(replay, buffer.save)
                    resume_state = dict(
                        version=1,
                        replay=dict(path=replay.name, sha256=sha256(replay), size=len(buffer)),
                        update_count=update_count,
                        next_episode_seed_index=episode_seed_index + 1,
                        pending_transition=collector.pending is not None,
                        last_losses=last_losses,
                        episode_window=list(episode_window.rows),
                        python_rng=random.getstate(),
                        exploration_rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    )
                checkpoint_index.append(save_checkpoint(
                    checkpoint, actors, critic, trainer, global_step, completed_episodes,
                    resolved_config, resume_state,
                ))
                atomic_json(checkpoint_dir / "index.json", checkpoint_index)

            if episode_finished and global_step < args.total_steps:
                episode_seed_index += 1
                observations, _ = env.reset(seed=args.seed + episode_seed_index)
                collector = JointTransitionCollector(buffer)
                outcomes = {}
                episode_start = global_step
                episode_metrics = {key: AgentMetrics() for key in AGENT_IDS}
                episode_rewards = np.zeros(len(AGENT_IDS), dtype=np.float64)
                episode_usp_steps = {ego_id: 0 for ego_id in AGENT_IDS}
                episode_usp_total = {ego_id: 0.0 for ego_id in AGENT_IDS}

        print("\n===== Training smoke summary =====")
        print("Global steps:", args.total_steps)
        print("Stored transitions:", len(buffer))
        print("Gradient updates:", update_count)
        print("Completed episodes:", completed_episodes)
        print("Elapsed seconds:", round(
            time.monotonic() - started, 2
        ))
        print("Checkpoint:", checkpoint)

        assert tuple(env.AGENTS) == tuple(AGENT_IDS)
        assert tuple(actors.ego_ids) == tuple(AGENT_IDS)
        assert critic.num_agents == len(AGENT_IDS)

    finally:
        env.close()
        log.close()


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-steps", type=int, default=1000000
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--learning-starts", type=int, default=100
    )
    parser.add_argument(
        "--batch-size", type=int, default=512
    )
    parser.add_argument(
        "--buffer-capacity", type=int, default=1000000
    )
    parser.add_argument(
        "--train-every", type=int, default=1
    )
    parser.add_argument(
        "--decision-interval-steps", type=int, default=1,
        help="SUMO steps per Actor decision interval (total-steps still counts SUMO steps)",
    )
    parser.add_argument(
        "--log-every", type=int, default=40
    )
    parser.add_argument(
        "--gamma", type=float, choices=SUPPORTED_GAMMAS, default=DEFAULT_GAMMA
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--initial-alpha", type=float, default=0.01
    )
    parser.add_argument(
        "--output-dir",
        default="runs/masac",
    )
    
    parser.add_argument(
        "--usp",
        action="store_true",
        help="Enable USP-v1 reward",
    )

    parser.add_argument(
        "--usp-penalty",
        type=float,
        default=0.02,
    )

    for field in fields(MASACRewardConfig):
        parser.add_argument(
            "--reward-" + field.name.replace("_", "-"),
            default=None if field.name in {"lane_change_weight", "route_preparation", "exit_retention"}
            else field.default,
            **({"action": argparse.BooleanOptionalAction} if isinstance(field.default, bool)
               else {"type": float}),
        )
    parser.add_argument("--reward-route-guidance-weight", type=float,
                        default=MASACRewardProtocol().route_guidance_weight)
    parser.add_argument("--reward-comfort-weight", type=float,
                        help=f"Comfort weight (default: {MASACRewardProtocol.comfort_weight} for fresh training; "
                             "saved weight when resuming)")
    parser.add_argument("--terminal-penalty-scale", type=float,
                        default=MASACRewardProtocol.terminal_penalty_scale,
                        help="Multiply negative MASAC terminal task rewards by this factor")
    parser.add_argument("--collision-penalty", type=float,
                        help="Override the final MASAC collision penalty with this nonnegative magnitude")
    parser.add_argument("--reward-route-preparation-mode", choices=(PREPARATION_LINEAR_V1, *PREPARATION_BONUS_MODES, PREPARATION_PENALTY),
                        help="Default: linear_progress_v1 for fresh training, saved mode when resuming")
    parser.add_argument("--reward-route-preparation-bonus-cap", type=float,
                        help=f"Per-step cruise bonus cap for older bonus modes (default: {DEFAULT_ROUTE_PREPARATION_BONUS})")
    for kind in DEFAULT_ROUTE_PREPARATION_WEIGHTS:
        parser.add_argument(f"--reward-route-preparation-{kind}-weight", type=float,
                            help="Select legacy penalty mode and set its phase weight")
    parser.add_argument("--route-preparation-stages", help="JSON file containing route preparation stage rules")
    parser.add_argument("--init-actors-from", help="Load only Actor weights from a route-aware MASAC checkpoint")
    parser.add_argument("--init-actors-sha256", help="Expected SHA-256 of the Actor initialization checkpoint")

    parser.add_argument(
        "--device", default="auto",
        help="Training device: auto (default: cuda:0 if available, otherwise cpu), cpu, cuda, or cuda:N",
    )
    parser.add_argument("--torch-threads", type=int, default=None, help="PyTorch intra-op threads (default: 2 on CUDA, up to 8 on CPU)")
    parser.add_argument("--update-log-every", type=int, default=40,
                        help="Write update metrics every N optimizer rounds (1 logs every round)")
    parser.add_argument("--team-q-weight", type=float, default=1.0,
                        help="Positive team Q weight in each Actor objective")
    parser.add_argument("--stalled-terminate", action=argparse.BooleanOptionalAction, default=None,
                        help="End a stalled Ego episode with a -400 penalty (default: enabled unless using USP)")
    parser.add_argument("--stalled-shadow", action="store_true", help="Log per-step stalled diagnostics")
    parser.add_argument("--masac-env-version", type=int,
                        choices=SUPPORTED_MASAC_ENV_VERSIONS, default=MASAC_ENV_VERSION,
                        help="v2: legacy route rules; v3: route-aware with Stalled-v1; "
                             "v4: stalled block grace; v5: match stalled links to the actual lane")
    parser.add_argument("--actor-leader-features", action="store_true",
                        help="Add delayed leader bumper gap, closing speed and TTC to each Actor")
    parser.add_argument("--trace-controls", action="store_true", help="Log per-Ego TraCI gaps, motion, and control commands")
    parser.add_argument("--max-active-steps", type=int, default=MultiEgoEnv.DEFAULT_MAX_ACTIVE_STEPS,
                        help="Per-Ego timeout in active control steps")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="Save model and replay every N global steps (default: 500000)")
    parser.add_argument("--replay-final-only", action="store_true",
                        help="Save model weights only at intermediate checkpoints and save Replay at the final step")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Continue from the latest checkpoint in --output-dir (default: enabled)")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
