"""Two-agent SAC learners for the shared-clock Antwerp environment.

Independent mode has two separate replay buffers and twin Q networks. CTDE
mode has one aligned joint replay buffer and a shared twin Q network with one
head per agent. Both modes use two decentralized DAROM-GRU Gaussian actors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .delay_encoder import DelayAwareGRUEncoder
from .dual_merging import AGENT_TASKS


OBS_DIM = 134
ACTION_DIM = 2
AGENTS = tuple(AGENT_TASKS)


def encoder():
    space = gym.spaces.Box(-5000, 5000, (OBS_DIM,), dtype=np.float32)
    return DelayAwareGRUEncoder(space, act_dim=2, hist_len=20, delay_dim=1,
                                features_dim=256, gru_hidden=64, input_mode="paper_raw")


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim))


class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = encoder()
        self.policy = mlp(256, 2 * ACTION_DIM)

    def sample(self, obs, deterministic=False):
        mean, log_std = self.policy(self.features(obs)).chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        # SAC optimizes in normalized [-1, 1] action coordinates. The fixed
        # 5x SUMO scale does not appear in the entropy or replay action.
        log_prob = (distribution.log_prob(raw) -
                    2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1, keepdim=True)
        return action, log_prob


class IndependentTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = encoder()
        self.q1 = mlp(256 + ACTION_DIM, 1)
        self.q2 = mlp(256 + ACTION_DIM, 1)

    def forward(self, obs, action):
        state_action = torch.cat((self.features(obs), action), dim=-1)
        return self.q1(state_action), self.q2(state_action)


class CentralTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList((encoder(), encoder()))
        self.q1 = mlp(2 * 256 + 2 * ACTION_DIM + 2, 2)
        self.q2 = mlp(2 * 256 + 2 * ACTION_DIM + 2, 2)

    def forward(self, obs, action, mask):
        # Shapes: obs [B,2,134], action [B,2,2], mask [B,2].
        features = [self.features[i](obs[:, i]) * mask[:, i:i + 1]
                    for i in range(2)]
        joined = torch.cat((*features, action.flatten(1), mask), dim=-1)
        return self.q1(joined), self.q2(joined)


class IndependentReplay:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_DIM), np.float32)
        self.actions = np.zeros((capacity, ACTION_DIM), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.done = np.zeros((capacity, 1), np.float32)
        self.size = self.write = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self.write
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.done[i] = next_obs, done
        self.write = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = self.rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(array[indices], device=device)
                     for array in (self.obs, self.actions, self.rewards,
                                   self.next_obs, self.done))


class JointReplay:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 2, OBS_DIM), np.float32)
        self.actions = np.zeros((capacity, 2, ACTION_DIM), np.float32)
        self.rewards = np.zeros((capacity, 2), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.done = np.zeros((capacity, 2), np.float32)
        self.mask = np.zeros((capacity, 2), np.float32)
        self.next_mask = np.zeros((capacity, 2), np.float32)
        self.size = self.write = 0
        self.rng = np.random.default_rng(seed)
        self.valid_indices = ([], [])
        self.positions = ({}, {})

    def add(self, obs, action, reward, next_obs, done, mask, next_mask):
        i = self.write
        for agent in range(2):
            position = self.positions[agent].pop(i, None)
            if position is not None:
                last = self.valid_indices[agent].pop()
                if position < len(self.valid_indices[agent]):
                    self.valid_indices[agent][position] = last
                    self.positions[agent][last] = position
            if mask[agent]:
                self.positions[agent][i] = len(self.valid_indices[agent])
                self.valid_indices[agent].append(i)
        for name, value in (("obs", obs), ("actions", action), ("rewards", reward),
                            ("next_obs", next_obs), ("done", done), ("mask", mask),
                            ("next_mask", next_mask)):
            getattr(self, name)[i] = value
        self.write = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device, agent=None):
        if agent is None:
            indices = self.rng.integers(self.size, size=batch_size)
        else:
            choices = self.valid_indices[agent]
            indices = [choices[i] for i in self.rng.integers(len(choices), size=batch_size)]
        return tuple(torch.as_tensor(getattr(self, name)[indices], device=device)
                     for name in ("obs", "actions", "rewards", "next_obs",
                                  "done", "mask", "next_mask"))


@dataclass(frozen=True)
class SACConfig:
    mode: str
    seed: int = 42
    learning_rate: float = 3e-5
    batch_size: int = 512
    replay_capacity: int = 1_000_000
    learning_starts: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -2.0
    device: str = "cpu"

    def __post_init__(self):
        if self.mode not in ("independent", "ctde"):
            raise ValueError("mode must be independent or ctde")
        if min(self.batch_size, self.replay_capacity, self.learning_starts) < 1:
            raise ValueError("batch_size, replay_capacity and learning_starts must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must lie in (0, 1]")


class TwoAgentSAC:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.actors = nn.ModuleList((GaussianActor(), GaussianActor())).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
                                 for actor in self.actors]
        self.log_alpha = [torch.zeros((), device=self.device, requires_grad=True)
                          for _ in AGENTS]
        self.alpha_optimizers = [torch.optim.Adam([value], lr=config.learning_rate)
                                 for value in self.log_alpha]
        if config.mode == "independent":
            self.critics = nn.ModuleList((IndependentTwinQ(), IndependentTwinQ())).to(self.device)
            self.targets = deepcopy(self.critics)
            self.critic_optimizers = [torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                                      for model in self.critics]
            self.replays = [IndependentReplay(config.replay_capacity, config.seed + i + 1)
                            for i in range(2)]
        else:
            self.critic = CentralTwinQ().to(self.device)
            self.target = deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
            self.replay = JointReplay(config.replay_capacity, config.seed + 1)
        self.experiences = [0, 0]
        self.actor_updates = [0, 0]
        self.critic_updates = [0, 0]
        self.last_losses = [{}, {}]

    def actions(self, observations):
        result = {}
        for i, key in enumerate(AGENTS):
            if key not in observations:
                continue
            if self.experiences[i] < self.config.learning_starts:
                action = self.rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
            else:
                with torch.no_grad():
                    tensor = torch.as_tensor(observations[key][None], device=self.device)
                    action = self.actors[i].sample(tensor)[0][0].cpu().numpy()
            result[key] = action
        return result

    def observe(self, before, actions, next_observations, rewards, terminated, truncated,
                active_after, experience_target):
        """Store only valid pre-step transitions, then perform eligible updates."""
        present = np.array([key in before for key in AGENTS], np.float32)
        next_mask = np.array([key in active_after for key in AGENTS], np.float32)
        eligible = [bool(present[i] and self.experiences[i] < experience_target)
                    for i in range(2)]
        if self.config.mode == "ctde" and any(eligible):
            current = np.stack([before.get(key, np.zeros(OBS_DIM, np.float32))
                                for key in AGENTS])
            following = np.stack([next_observations.get(key, np.zeros(OBS_DIM, np.float32))
                                  for key in AGENTS])
            chosen = np.stack([actions.get(key, np.zeros(ACTION_DIM, np.float32))
                               for key in AGENTS])
            payoffs = np.array([rewards.get(key, 0) for key in AGENTS], np.float32)
            done = np.array([terminated.get(key, False) or truncated.get(key, False)
                             for key in AGENTS], np.float32)
            self.replay.add(current, chosen, payoffs, following, done, present, next_mask)
        for i, key in enumerate(AGENTS):
            if not eligible[i]:
                continue
            if self.config.mode == "independent":
                self.replays[i].add(before[key], actions[key], rewards[key],
                                    next_observations[key],
                                    terminated[key] or truncated[key])
            self.experiences[i] += 1
        train = [eligible[i] and self.experiences[i] >= self.config.learning_starts
                 for i in range(2)]
        if self.config.mode == "independent":
            for i in range(2):
                if train[i]:
                    self._update_independent(i)
        elif any(train):
            self._update_ctde(train)

    def _next_joint_actions(self, obs, mask):
        chosen, log_probs = [], []
        for i in range(2):
            action, log_prob = self.actors[i].sample(obs[:, i])
            chosen.append(action * mask[:, i:i + 1])
            log_probs.append(log_prob.squeeze(-1))
        return torch.stack(chosen, dim=1), torch.stack(log_probs, dim=1)

    def _update_independent(self, i):
        cfg = self.config
        obs, action, reward, following, done = self.replays[i].sample(cfg.batch_size, self.device)
        alpha = self.log_alpha[i].exp().detach()
        with torch.no_grad():
            next_action, next_logp = self.actors[i].sample(following)
            q1_next, q2_next = self.targets[i](following, next_action)
            target = reward + cfg.gamma * (1 - done) * (
                torch.minimum(q1_next, q2_next) - alpha * next_logp)
        q1, q2 = self.critics[i](obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError(f"Nonfinite independent critic loss for {AGENTS[i]}")
        self.critic_optimizers[i].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizers[i].step()
        self.critic_updates[i] += 1

        self.critics[i].requires_grad_(False)
        new_action, logp = self.actors[i].sample(obs)
        q1, q2 = self.critics[i](obs, new_action)
        actor_loss = (alpha * logp - torch.minimum(q1, q2)).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError(f"Nonfinite independent actor loss for {AGENTS[i]}")
        self.actor_optimizers[i].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizers[i].step()
        self.critics[i].requires_grad_(True)
        self.actor_updates[i] += 1
        alpha_loss = self._update_alpha(i, logp)
        self.last_losses[i] = {"critic": float(critic_loss.detach()),
                               "actor": float(actor_loss.detach()),
                               "alpha": alpha_loss,
                               "max_abs_target": float(target.abs().max())}
        self._polyak(self.critics[i], self.targets[i])

    def _update_ctde(self, eligible):
        cfg = self.config
        obs, action, reward, following, done, mask, next_mask = self.replay.sample(
            cfg.batch_size, self.device)
        active_heads = torch.as_tensor(eligible, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action, next_logp = self._next_joint_actions(following, next_mask)
            q1_next, q2_next = self.target(following, next_action, next_mask)
            alpha = torch.stack([x.exp() for x in self.log_alpha])
            target = reward + cfg.gamma * (1 - done) * (
                torch.minimum(q1_next, q2_next) - alpha * next_logp)
        q1, q2 = self.critic(obs, action, mask)
        weights = mask * active_heads
        critic_loss = (((q1 - target).square() + (q2 - target).square()) * weights).sum() / weights.sum().clamp_min(1)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite centralized critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        for i in range(2):
            if eligible[i]:
                self.critic_updates[i] += 1

        self.critic.requires_grad_(False)
        for i in range(2):
            if not eligible[i]:
                continue
            actor_obs, _, _, _, _, actor_mask, _ = self.replay.sample(
                cfg.batch_size, self.device, agent=i)
            current_actions = []
            selected_logp = None
            for j in range(2):
                if j == i:
                    chosen, selected_logp = self.actors[j].sample(actor_obs[:, j])
                else:
                    with torch.no_grad():
                        chosen, _ = self.actors[j].sample(actor_obs[:, j])
                current_actions.append(chosen * actor_mask[:, j:j + 1])
            current_actions = torch.stack(current_actions, dim=1)
            q1_policy, q2_policy = self.critic(actor_obs, current_actions, actor_mask)
            actor_loss = (self.log_alpha[i].exp().detach() * selected_logp.squeeze(-1)
                          - torch.minimum(q1_policy[:, i], q2_policy[:, i])).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError(f"Nonfinite centralized actor loss for {AGENTS[i]}")
            self.actor_optimizers[i].zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizers[i].step()
            self.actor_updates[i] += 1
            alpha_loss = self._update_alpha(i, selected_logp)
            self.last_losses[i] = {"critic": float(critic_loss.detach()),
                                   "actor": float(actor_loss.detach()),
                                   "alpha": alpha_loss,
                                   "max_abs_target": float(target[:, i].abs().max())}
        self.critic.requires_grad_(True)
        self._polyak(self.critic, self.target)

    def _update_alpha(self, i, logp):
        loss = -(self.log_alpha[i] * (logp.detach() + self.config.target_entropy)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite entropy loss for {AGENTS[i]}")
        self.alpha_optimizers[i].zero_grad(set_to_none=True)
        loss.backward()
        self.alpha_optimizers[i].step()
        return float(loss.detach())

    def _polyak(self, source, target):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.lerp_(source_param, self.config.tau)
            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(source_buffer)

    def checkpoint(self, metadata):
        data = {"config": asdict(self.config), "metadata": metadata,
                "actors": self.actors.state_dict(),
                "actor_optimizers": [x.state_dict() for x in self.actor_optimizers],
                "log_alpha": [x.detach().cpu() for x in self.log_alpha],
                "alpha_optimizers": [x.state_dict() for x in self.alpha_optimizers],
                "experiences": self.experiences.copy(),
                "actor_updates": self.actor_updates.copy(),
                "critic_updates": self.critic_updates.copy()}
        if self.config.mode == "independent":
            data.update(critics=self.critics.state_dict(), targets=self.targets.state_dict(),
                        critic_optimizers=[x.state_dict() for x in self.critic_optimizers])
        else:
            data.update(critic=self.critic.state_dict(), target=self.target.state_dict(),
                        critic_optimizer=self.critic_optimizer.state_dict())
        return data

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=False)
        config = SACConfig(**{**saved["config"], "device": device})
        # Evaluation needs the decentralized actors only. In particular, do
        # not allocate two million-slot training replay buffers for a 200-scene
        # evaluation of a full-budget checkpoint.
        learner = cls.__new__(cls)
        learner.config = config
        learner.device = torch.device(device)
        learner.actors = nn.ModuleList((GaussianActor(), GaussianActor())).to(learner.device)
        learner.actors.load_state_dict(saved["actors"])
        learner.actors.eval()
        return learner, saved["metadata"]
