
"""DAROM 双 Ego CTDE-MASAC 训练入口。

结构化训练记录与周期 checkpoint；正式评估由独立入口执行。
不提供精确断点恢复或 W&B。
"""

import argparse
import random
from pathlib import Path
import time

import numpy as np
import torch

from src.envs.multi_ego_env import MultiEgoEnv
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
        with torch.no_grad():
            action_tensor, _ = actors(
                torch.from_numpy(
                    policy_obs[None]
                ),
                torch.from_numpy(
                    active_mask[None]
                ),
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

    atomic_write(path, lambda stream: torch.save({
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


def main(args):
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
    
    args.usp = getattr(args, "usp", False)
    args.usp_penalty = getattr(
        args, "usp_penalty", 0.02
    )

    # 1. 创建网络及优化器
    actors = MultiAgentGaussianActor(
        ego_ids=AGENT_IDS
    )
    critic = CentralizedTwinQ(
        num_agents=len(AGENT_IDS)
    )

    trainer = MASACTrainer(
        actors=actors,
        critic=critic,
        gamma=args.gamma,
        tau=args.tau,
        learning_rate=args.learning_rate,
        alpha=(args.initial_alpha,) * len(AGENT_IDS),
    )

    # 2. 创建联合经验池
    buffer = JointReplayBuffer(
        capacity=args.buffer_capacity,
        seed=args.seed,
        agent_ids=AGENT_IDS,
    )

    # 3. 创建双 Ego 环境
    env = MultiEgoEnv(
        scenario="easy",
        speed_mode=96,
        collect_metrics=True,
        usp_enabled=args.usp,
        usp_penalty=args.usp_penalty,
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
                batch = buffer.sample(args.batch_size)

                critic_stats = trainer.update_critic(batch)
                actor_stats = trainer.update_actors(batch)
                alpha_stats = trainer.update_alpha(batch)

                last_losses = {
                    **critic_stats,
                    **actor_stats,
                    **alpha_stats,
                }

                for i, key in enumerate(AGENT_IDS):
                    last_losses.setdefault(key + "_alpha", float(trainer.alpha_values()[i].detach()))
                    last_losses.setdefault(key + "_actor_loss", None)
                    last_losses.setdefault(key + "_alpha_loss", None)
                    last_losses.setdefault(key + "_mean_log_prob", None)
                    last_losses.setdefault(key + "_valid_samples", 0)
                update_count += 1
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
                    **episode_values, **episode_window.add(agents))

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
        "--learning-starts", type=int, default=1000
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
        "--gamma", type=float, default=0.999
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

    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
