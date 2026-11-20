
"""DAROM 双 Ego CTDE-MASAC 训练入口。

结构化训练记录与周期 checkpoint；正式评估由独立入口执行。
不提供精确断点恢复或 W&B。
"""

import argparse
import math
from dataclasses import asdict, fields
import random
from pathlib import Path
import time

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
from src.envs.route_lane_plan import DEFAULT_ROUTE_GUIDANCE_RULES
from src.rewards.masac import (
    LEGACY_MASAC_ENV_VERSION, MASACRewardConfig, MASACRewardProtocol, MASAC_ENV_VERSION,
    TARGET_EXIT_DIM,
)
from src.agents.masac import MultiAgentGaussianActor
from src.agents.centralized_critic import CentralizedTwinQ
from src.agents.replay_buffer import JointReplayBuffer
from src.agents.transition_collector import JointTransitionCollector
from src.agents.masac_trainer import MASACTrainer


from src.experiment_tracking import (TrainingLog, ActionWindow, EpisodeWindow,
    checkpoint_due, atomic_write, atomic_json, sha256, git_metadata, evaluation_fingerprint)
from src.evaluation_metrics import AgentMetrics

AGENT_IDS = JointReplayBuffer.AGENT_IDS


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
            -5.0, 5.0, size=(2, 2)
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


def save_checkpoint(
    path,
    actors,
    critic,
    trainer,
    global_step,
    episodes,
    config,
):
    """保存模型和优化器状态。"""

    team_state = {}
    if trainer.team_critic is not None:
        team_state = {
            "team_critic": trainer.team_critic.state_dict(),
            "target_team_critic": trainer.target_team_critic.state_dict(),
            "team_critic_optimizer": trainer.team_critic_optimizer.state_dict(),
        }
    atomic_write(path, lambda stream: torch.save({
        **team_state,
        "git": git_metadata(),
        "source_sha256": evaluation_fingerprint(),
        "global_step": global_step,
        "episodes": episodes,
        "agent_ids": AGENT_IDS,
        "config": vars(config),
        "trainer_config": {"gamma": trainer.gamma, "tau": trainer.tau, "target_entropy": trainer.target_entropy},
        "actors": actors.state_dict(),
        "critic": critic.state_dict(),
        "target_critic":
            trainer.target_critic.state_dict(),
        "log_alpha": trainer.log_alpha.state_dict(),
        "critic_optimizer":
            trainer.critic_optimizer.state_dict(),
        "actor_optimizers": {
            k: v.state_dict()
            for k, v in trainer.actor_optimizers.items()
        },
        "alpha_optimizers": {
            k: v.state_dict()
            for k, v in trainer.alpha_optimizers.items()
        },
    }, stream))


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
        return train(args, device)
    finally:
        torch.set_num_threads(previous_threads)


def train(args, device):
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

    args.trace_controls = getattr(args, "trace_controls", False)
    args.stalled_shadow = getattr(args, "stalled_shadow", False)
    args.usp = getattr(args, "usp", False)
    args.stalled_terminate = getattr(args, "stalled_terminate", None)
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

    if getattr(args, "gamma", 0.99) != 0.99:
        raise ValueError("O3-D3 MASAC requires gamma=0.99")
    args.gamma = 0.99
    args.masac_env_version = getattr(args, "masac_env_version", MASAC_ENV_VERSION)
    if args.masac_env_version not in (LEGACY_MASAC_ENV_VERSION, MASAC_ENV_VERSION):
        raise ValueError(f"Unsupported MASAC environment version: {args.masac_env_version}")
    args.actor_leader_features = getattr(args, "actor_leader_features", False)
    args.tau = getattr(args, "tau", 0.005)
    args.checkpoint_every = getattr(args, "checkpoint_every", 100000)
    args.tensorboard = getattr(args, "tensorboard", True)
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if list((output_dir / "checkpoints").glob("*.pt")):
        raise FileExistsError("Experiment directory already contains checkpoints")
    
    args.usp_penalty = getattr(
        args, "usp_penalty", 0.02
    )

    args.reward_components = {
        field.name: getattr(args, "reward_" + field.name, field.default)
        for field in fields(MASACRewardConfig)
        if isinstance(field.default, bool)
    }
    args.safety_reward = {
        field.name: getattr(args, "reward_" + field.name, field.default)
        for field in fields(MASACRewardConfig)
        if not isinstance(field.default, bool)
    }
    reward_config = MASACRewardConfig(**args.reward_components, **args.safety_reward)
    reward_protocol = MASACRewardProtocol(
        route_guidance_weight=getattr(
            args, "reward_route_guidance_weight", MASACRewardProtocol.route_guidance_weight,
        ),
    )
    args.reward_protocol = asdict(reward_protocol)
    args.route_guidance_rules = getattr(args, "route_guidance_rules", DEFAULT_ROUTE_GUIDANCE_RULES)
    args.team_q_weight = getattr(args, "team_q_weight", 1.0)
    if reward_config.team and (not math.isfinite(args.team_q_weight) or args.team_q_weight <= 0):
        raise ValueError("team_q_weight must be finite and positive when team reward is enabled")


    # 1. 创建网络及优化器
    actors = MultiAgentGaussianActor(
        ego_ids=AGENT_IDS, nav_dim=TARGET_EXIT_DIM,
        actor_leader_features=args.actor_leader_features,
    ).to(device)
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
    )

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
        max_active_steps=args.max_active_steps,
        collect_control_trace=args.trace_controls,
        stalled_shadow=args.stalled_shadow,
        stalled_terminate=args.stalled_terminate,
        legacy_route_semantics=args.masac_env_version == LEGACY_MASAC_ENV_VERSION,
        actor_leader_features=args.actor_leader_features,
    )

    log = TrainingLog(output_dir / "train", args.tensorboard)
    provenance = git_metadata()
    log.write("config", 0, config=vars(args), git=provenance, policy="exploration_training")
    action_window = ActionWindow(AGENT_IDS)
    episode_window = EpisodeWindow()
    checkpoint_index = []

    def checkpoint_at(step, episodes):
        path = output_dir / "checkpoints" / f"masac_seed{args.seed}_step_{step}.pt"
        save_checkpoint(path, actors, critic, trainer, step, episodes, args)
        checkpoint_index.append(dict(training_seed=args.seed, global_step=step,
            path=str(path.resolve()), sha256=sha256(path), git=provenance, source_sha256=evaluation_fingerprint(), config=vars(args)))
        atomic_json(output_dir / "checkpoints" / "index.json", checkpoint_index)
        return path

    completed_episodes = 0
    update_count = 0
    last_losses = {}
    started = time.monotonic()

    try:
        observations, _ = env.reset(seed=args.seed)
        collector = JointTransitionCollector(buffer)

        episode_rewards = np.zeros(
            len(AGENT_IDS), dtype=np.float64
        )
        outcomes = {}
        episode_start = 0
        episode_metrics = {key: AgentMetrics() for key in AGENT_IDS}

        episode_usp_steps = {
            ego_id: 0 for ego_id in AGENT_IDS
        }

        episode_usp_total = {
            ego_id: 0.0 for ego_id in AGENT_IDS
        }

        for global_step in range(1, args.total_steps + 1):

            # A. 推进前快照
            policy_obs, critic_obs, active_mask = (
                snapshot(env, observations)
            )

            # B. 随机探索或 Actor 决策
            joint_actions, action_dict = choose_actions(
                actors,
                policy_obs,
                active_mask,
                rng,
                random_exploration=(
                    global_step <= args.learning_starts
                ),
            )

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
            collector.push(
                transition,
                pre_step_terminal=pre_step,
            )

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
            if len(outcomes) == len(AGENT_IDS):
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

                if global_step < args.total_steps:
                    observations, _ = env.reset(
                        seed=args.seed + completed_episodes
                    )

                    collector = JointTransitionCollector(
                        buffer
                    )

                    outcomes = {}
                    episode_start = global_step
                    episode_metrics = {key: AgentMetrics() for key in AGENT_IDS}
                    episode_rewards = np.zeros(
                        len(AGENT_IDS), dtype=np.float64
                    )
                    
                    # 新 Episode，重置 USP 统计
                    episode_usp_steps = {
                        ego_id: 0 for ego_id in AGENT_IDS
                    }

                    episode_usp_total = {
                        ego_id: 0.0 for ego_id in AGENT_IDS
                    }

            if checkpoint_due(global_step, args.total_steps, args.checkpoint_every):
                checkpoint = checkpoint_at(global_step, completed_episodes)

        print("\n===== Training smoke summary =====")
        print("Global steps:", args.total_steps)
        print("Stored transitions:", len(buffer))
        print("Gradient updates:", update_count)
        print("Completed episodes:", completed_episodes)
        print("Elapsed seconds:", round(
            time.monotonic() - started, 2
        ))
        print("Checkpoint:", checkpoint)

        assert update_count > 0, (
            "No gradient update was performed"
        )
        
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
        "--log-every", type=int, default=40
    )
    parser.add_argument(
        "--gamma", type=float, choices=(0.99,), default=0.99
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
            default=field.default,
            **({"action": argparse.BooleanOptionalAction} if isinstance(field.default, bool)
               else {"type": float}),
        )
    parser.add_argument("--reward-route-guidance-weight", type=float,
                        default=MASACRewardProtocol().route_guidance_weight)

    parser.add_argument("--device", default="auto", help="Training device: auto (CUDA first), cpu, cuda, or cuda:N")
    parser.add_argument("--torch-threads", type=int, default=None, help="PyTorch intra-op threads (default: 2 on CUDA, up to 8 on CPU)")
    parser.add_argument("--update-log-every", type=int, default=40,
                        help="Write update metrics every N optimizer rounds (1 logs every round)")
    parser.add_argument("--team-q-weight", type=float, default=1.0,
                        help="Positive team Q weight in each Actor objective")
    parser.add_argument("--stalled-terminate", action=argparse.BooleanOptionalAction, default=None,
                        help="End a stalled Ego episode with a -400 penalty (default: enabled unless using USP)")
    parser.add_argument("--stalled-shadow", action="store_true", help="Log Stalled-v1 diagnostics without changing the task")
    parser.add_argument("--masac-env-version", type=int,
                        choices=(LEGACY_MASAC_ENV_VERSION, MASAC_ENV_VERSION), default=MASAC_ENV_VERSION,
                        help="Use version 2 to match legacy checkpoints' lane observation and exit rules")
    parser.add_argument("--actor-leader-features", action="store_true",
                        help="Add delayed leader bumper gap, closing speed and TTC to each Actor")
    parser.add_argument("--trace-controls", action="store_true", help="Log per-Ego TraCI gaps, motion, and control commands")
    parser.add_argument("--max-active-steps", type=int, default=MultiEgoEnv.DEFAULT_MAX_ACTIVE_STEPS,
                        help="Per-Ego timeout in active control steps")
    parser.add_argument("--checkpoint-every", type=int, default=200000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
