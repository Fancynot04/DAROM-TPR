
"""CTDE-MASAC 更新器：第一阶段仅实现 Critic 更新。"""

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
    """当前仅负责集中式 Critic 的学习。"""

    def __init__(
        self,
        actors,
        critic,
        gamma=0.999,
        tau=0.005,
        learning_rate=3e-5,
        alpha=(0.01, 0.01),
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

        for online, target in zip(
            self.critic.parameters(),
            self.target_critic.parameters(),
        ):
            target.mul_(1.0 - self.tau)
            target.add_(online, alpha=self.tau)

    def update_critic(self, batch):
        """使用一批联合经验更新一次 Twin-Q。"""

        policy_next = batch["next_policy_obs"]
        critic_next = batch["next_critic_obs"]

        next_mask = batch["next_active_mask"]

        rewards = batch["rewards"]
        dones = batch["dones"]

        acted_mask = batch["acted_mask"]

        valid_count = acted_mask.sum()

        if valid_count.item() == 0:
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

        # 2. 计算当前 Twin-Q
        q1, q2 = self.critic(
            batch["critic_obs"],
            batch["actions"],
            batch["active_mask"],
        )

        # 3. 对两个 Q 网络分别计算 TD 误差
        errors = (
            (q1 - targets).square()
            + (q2 - targets).square()
        )

        # 仅使用真正执行动作的 Ego 样本
        loss = (
            errors * acted_mask
        ).sum() / valid_count

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Critic loss")

        # 4. 更新当前 Critic
        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=10.0,
        )

        self.critic_optimizer.step()

        # 5. 更新目标 Critic
        self.soft_update_target()

        return {
            "critic_loss": float(loss.item()),
            "valid_agents": int(valid_count.item()),
            "mean_q": float((((q1.detach() + q2.detach()) / 2) * acted_mask).sum().item() / valid_count.item()),
            "mean_target": float(
                (
                    targets * acted_mask
                ).sum().item() / valid_count.item()
            ),
        }

    
    def update_actors(self, batch):
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
        old_flags = [
            p.requires_grad for p in critic_params
        ]

        try:
            self.critic.requires_grad_(False)

            for i, ego_id in enumerate(self.actors.ego_ids):
                valid = valid_mask[:, i]
                count = valid.sum()

                if count.item() == 0:
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

                results[f"{ego_id}_mean_log_prob"] = float((log_prob_i.detach() * valid).sum().item() / count.item())
                results[f"{ego_id}_actor_loss"] = (
                    float(actor_loss.item())
                )
                results[f"{ego_id}_actor_updates"] = 1
                results[f"{ego_id}_valid_samples"] = (
                    int(count.item())
                )

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

            results[f"{ego_id}_alpha_loss"] = (
                float(loss.item())
            )

            results[f"{ego_id}_alpha"] = float(
                self.log_alpha[ego_id].detach().exp().item()
            )

            results[f"{ego_id}_alpha_updates"] = 1

        return results

    def update_alpha(self, batch):
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
        )
