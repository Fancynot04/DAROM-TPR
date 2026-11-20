
"""CTDE-MASAC 的个人、团队和熵系数更新。"""

from copy import deepcopy

import torch
from torch import nn

import math

def compute_td_target(
    rewards,
    dones,
    next_active_mask,
    next_q1,
    next_q2,
    next_log_probs,
    alpha,
    gamma,
):
    """逐 Ego 计算 SAC Bellman Target。"""

    bootstrap_mask = (
        (1.0 - dones) * next_active_mask
    )

    next_q = torch.minimum(next_q1, next_q2)

    next_value = (
        next_q
        - alpha.reshape(1, -1) * next_log_probs
    )

    return (
        rewards
        + gamma * bootstrap_mask * next_value
    )


class MASACTrainer:
    """更新集中 Critic 和分散 Actor。"""

    def __init__(
        self,
        actors,
        critic,
        gamma=0.999,
        tau=0.005,
        learning_rate=3e-5,
        alpha=(0.01, 0.01),
        team_critic=None,
        team_q_weight=1.0,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Invalid gamma")

        if not 0.0 < tau <= 1.0:
            raise ValueError("Invalid tau")

        if learning_rate <= 0:
            raise ValueError("Invalid learning rate")

        if len(alpha) != critic.num_agents:
            raise ValueError("Alpha count mismatch")

        if any(a <= 0 for a in alpha):
            raise ValueError("Initial alpha must be positive")

        self.actors = actors
        self.critic = critic

        self.target_critic = deepcopy(critic)
        self.target_critic.eval()
        self.target_critic.requires_grad_(False)

        if team_critic is not None and (not math.isfinite(team_q_weight) or team_q_weight <= 0):
            raise ValueError("team_q_weight must be finite and positive when team reward is enabled")
        self.team_critic = team_critic
        self.team_q_weight = team_q_weight
        self.target_team_critic = None
        self.team_critic_optimizer = None
        if self.team_critic is not None:
            self.target_team_critic = deepcopy(self.team_critic).eval().requires_grad_(False)
            self.team_critic_optimizer = torch.optim.Adam(self.team_critic.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.tau = tau

        
        # 每辆 Ego 独立学习 log(alpha)，确保 alpha 始终为正
        device = next(self.critic.parameters()).device

        self.log_alpha = nn.ParameterDict({
            ego_id: nn.Parameter(
                torch.tensor(
                    math.log(float(alpha[i])),
                    dtype=torch.float32,
                    device=device,
                )
            )
            for i, ego_id in enumerate(self.actors.ego_ids)
        })

        self.alpha_optimizers = {
            ego_id: torch.optim.Adam(
                [self.log_alpha[ego_id]],
                lr=learning_rate,
            )
            for ego_id in self.actors.ego_ids
        }

        self.target_entropy = -float(self.critic.action_dim)


        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate,
        )
        
        if len(actors.ego_ids) != critic.num_agents:
            raise ValueError("Actor/Critic agent count mismatch")

        if any(
            abs(actor.action_scale - critic.action_scale) > 1e-8
            for actor in actors.actors.values()
        ):
            raise ValueError("Actor/Critic action scale mismatch")

        self.actor_optimizers = {
            ego_id: torch.optim.Adam(
                actors.actors[ego_id].parameters(),
                lr=learning_rate,
            )
            for ego_id in actors.ego_ids
        }

    @torch.no_grad()
    def soft_update_target(self):
        """Polyak 平滑更新目标 Critic。"""

        pairs = [(self.critic, self.target_critic)]
        if self.team_critic is not None:
            pairs.append((self.team_critic, self.target_team_critic))
        for online, target in pairs:
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.mul_(1.0 - self.tau)
                destination.add_(source, alpha=self.tau)

    def update_team_critic(self, batch, next_actions, *, collect_stats):
        # A failure ends the team objective. Samples after that event have no team target.
        eligible = batch["team_active"].bool()
        valid = int(eligible.sum().item())
        if not valid:
            return {}

        with torch.no_grad():
            target_q1, target_q2 = self.target_team_critic(
                batch["next_critic_obs"][eligible],
                next_actions[eligible],
                batch["next_active_mask"][eligible],
                success_mask=batch["next_success_mask"][eligible],
            )
            targets = batch["team_reward"][eligible] + self.gamma * (
                1.0 - batch["team_done"][eligible]
            ) * torch.minimum(target_q1, target_q2)

        q1, q2 = self.team_critic(
            batch["critic_obs"][eligible],
            batch["actions"][eligible],
            batch["active_mask"][eligible],
            success_mask=batch["success_mask"][eligible],
        )
        loss = ((q1 - targets).square() + (q2 - targets).square()).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite team Critic loss")

        self.team_critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.team_critic.parameters(), max_norm=10.0)
        self.team_critic_optimizer.step()
        if not collect_stats:
            return {}
        return {
            "team_critic_loss": loss.item(),
            "team_valid_samples": valid,
            "team_mean_target": targets.mean().item(),
            "team_mean_q": ((q1.detach() + q2.detach()) / 2).mean().item(),
        }

    def update_critic(self, batch, *, collect_stats=True):
        """使用一批联合经验更新一次 Twin-Q。"""

        policy_next = batch["next_policy_obs"]
        critic_next = batch["next_critic_obs"]

        next_mask = batch["next_active_mask"]

        rewards = batch["rewards"]
        dones = batch["dones"]

        acted_mask = batch["acted_mask"]

        valid_count = acted_mask.sum()

        count = int(valid_count.item())
        if count == 0 and self.team_critic is None:
            raise ValueError(
                "Batch contains no executed actions"
            )

        # 1. 计算目标 Q，不更新 Actor
        with torch.no_grad():
            next_actions, next_log_probs = (
                self.actors(
                    policy_next,
                    next_mask,
                    deterministic=False,
                )
            )
            
            # 转换为归一化动作空间下的 log probability
            next_log_probs = (
                next_log_probs
                + self.critic.action_dim
                * math.log(self.critic.action_scale)
            )

            target_q1, target_q2 = (
                self.target_critic(
                    critic_next,
                    next_actions,
                    next_mask,
                )
            )
            
            alpha = self.alpha_values().to(
                device=rewards.device,
                dtype=rewards.dtype,
            )

            targets = compute_td_target(
                rewards=rewards,
                dones=dones,
                next_active_mask=next_mask,
                next_q1=target_q1,
                next_q2=target_q2,
                next_log_probs=next_log_probs,
                alpha=alpha,
                gamma=self.gamma,
            )

        stats = {}
        if count:
            q1, q2 = self.critic(
                batch["critic_obs"],
                batch["actions"],
                batch["active_mask"],
            )
            errors = (q1 - targets).square() + (q2 - targets).square()
            loss = (errors * acted_mask).sum() / valid_count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Critic loss")

            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic_optimizer.step()

            if collect_stats:
                stats.update(
                    critic_loss=loss.item(),
                    valid_agents=count,
                    mean_q=((((q1.detach() + q2.detach()) / 2) * acted_mask).sum().item() / count),
                    mean_target=(targets * acted_mask).sum().item() / count,
                )

        if self.team_critic is not None:
            stats.update(self.update_team_critic(batch, next_actions, collect_stats=collect_stats))
        self.soft_update_target()
        return stats

    def update_actors(self, batch, *, collect_stats=True):
        """逐个更新 Actor，阻止跨 Agent 梯度传播。"""

        policy_obs = batch["policy_obs"]
        critic_obs = batch["critic_obs"]
        active_mask = batch["active_mask"]
        acted_mask = batch["acted_mask"]

        # 只有真正执行过动作的 Ego 参与 Actor 更新
        valid_mask = active_mask * acted_mask

        # 其他 Actor 的动作不参与当前 Actor 的梯度
        with torch.no_grad():
            base_actions, _ = self.actors(
                policy_obs,
                active_mask,
                deterministic=False,
            )

        results = {}

        # Actor 更新时不修改 Critic 参数，
        # 但必须保留 Q 对当前 Actor 动作的梯度。
        critic_params = list(self.critic.parameters())
        if self.team_critic is not None:
            critic_params.extend(self.team_critic.parameters())
        old_flags = [
            p.requires_grad for p in critic_params
        ]

        try:
            for parameter in critic_params:
                parameter.requires_grad_(False)

            for i, ego_id in enumerate(self.actors.ego_ids):
                valid = valid_mask[:, i]
                count = int(valid.sum().item())

                if count == 0:
                    if collect_stats:
                        results[f"{ego_id}_actor_updates"] = 0
                    continue

                actor = self.actors.actors[ego_id]

                # 只重新计算当前 Ego 的动作，保留其梯度
                action_i, log_prob_i = actor(
                    policy_obs[:, i, :],
                    deterministic=False,
                )

                # 其他 Ego 的动作来自无梯度采样
                joint_actions = base_actions.clone()
                joint_actions[:, i, :] = (
                    action_i
                    * active_mask[:, i].unsqueeze(-1)
                )

                # 计算当前 Ego 对应的 Q
                q1, q2 = self.critic(
                    critic_obs,
                    joint_actions,
                    active_mask,
                )

                q_i = torch.minimum(
                    q1[:, i],
                    q2[:, i],
                )

                if self.team_critic is not None:
                    team_q1, team_q2 = self.team_critic(
                        critic_obs, joint_actions, active_mask,
                        success_mask=batch["success_mask"],
                    )
                    team_q = torch.minimum(team_q1, team_q2) * batch["team_active"]
                    q_i = q_i + self.team_q_weight * team_q
                    if collect_stats:
                        results[f"{ego_id}_team_q"] = (team_q.detach() * valid).sum().item() / count

                # 实际动作概率密度 -> 归一化动作密度
                log_prob_norm = (
                    log_prob_i
                    + self.critic.action_dim
                    * math.log(self.critic.action_scale)
                )
                
                alpha_i = self.log_alpha[ego_id].detach().exp()

                # SAC Actor Loss
                objective = (
                    alpha_i * log_prob_norm - q_i
                )

                actor_loss = (
                    objective * valid
                ).sum() / count

                if not torch.isfinite(actor_loss):
                    raise RuntimeError(
                        f"Non-finite Actor loss: {ego_id}"
                    )

                optimizer = self.actor_optimizers[ego_id]
                optimizer.zero_grad(set_to_none=True)

                actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=10.0,
                )

                optimizer.step()

                if collect_stats:
                    results[f"{ego_id}_mean_log_prob"] = float(
                        (log_prob_i.detach() * valid).sum().item() / count
                    )
                    results[f"{ego_id}_actor_loss"] = float(actor_loss.item())
                    results[f"{ego_id}_actor_updates"] = 1
                    results[f"{ego_id}_valid_samples"] = count

        finally:
            # 恢复 Critic 原本的梯度开关
            for parameter, flag in zip(
                critic_params, old_flags
            ):
                parameter.requires_grad_(flag)

        return results

    @torch.no_grad()
    def alpha_values(self):
        return torch.stack([
            self.log_alpha[ego_id].exp()
            for ego_id in self.actors.ego_ids
        ])
        
    
    def update_alpha_from_log_probs(
        self,
        log_probs,
        valid_mask,
        *,
        collect_stats=True,
    ):
        """使用归一化动作概率密度更新各自的 alpha。"""

        n = len(self.actors.ego_ids)

        if (
            log_probs.ndim != 2
            or log_probs.shape[1] != n
            or valid_mask.shape != log_probs.shape
        ):
            raise ValueError("Invalid alpha batch shape")

        if (
            not torch.isfinite(log_probs).all()
            or not torch.isfinite(valid_mask).all()
        ):
            raise ValueError("Non-finite alpha input")

        if not torch.all(
            (valid_mask == 0) | (valid_mask == 1)
        ):
            raise ValueError("Invalid alpha mask")

        results = {}

        for i, ego_id in enumerate(self.actors.ego_ids):
            valid = valid_mask[:, i]
            count = valid.sum()

            if count.item() == 0:
                if collect_stats:
                    results[f"{ego_id}_alpha_updates"] = 0
                continue

            # SAC 自动熵系数损失
            entropy_error = (
                log_probs[:, i].detach()
                + self.target_entropy
            )

            loss = -(
                self.log_alpha[ego_id]
                * entropy_error
                * valid
            ).sum() / count

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite alpha loss")

            optimizer = self.alpha_optimizers[ego_id]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if collect_stats:
                results[f"{ego_id}_alpha_loss"] = float(loss.item())
                results[f"{ego_id}_alpha"] = float(self.log_alpha[ego_id].detach().exp().item())
                results[f"{ego_id}_alpha_updates"] = 1

        return results

    def update_alpha(self, batch, *, collect_stats=True):
        """从 Replay Batch 中采样策略，自动更新 alpha。"""

        policy_obs = batch["policy_obs"]
        active_mask = batch["active_mask"]
        acted_mask = batch["acted_mask"]

        with torch.no_grad():
            _, log_probs = self.actors(
                policy_obs,
                active_mask,
                deterministic=False,
            )

            # 实际 [-5,5] 动作密度转换为 [-1,1]
            log_probs_norm = (
                log_probs
                + self.critic.action_dim
                * math.log(self.critic.action_scale)
            )

        return self.update_alpha_from_log_probs(
            log_probs_norm,
            active_mask * acted_mask,
            collect_stats=collect_stats,
        )
