"""Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent for Unitree Go1 Quadruped.

Includes:
  - ReplayBuffer: Fast NumPy circular replay buffer storing off-policy transitions.
  - DeterministicActor: Multi-layer perceptron mapping state -> deterministic continuous action [-1, 1].
  - TwinCritic: Twin Q-networks (Q1, Q2) to mitigate value overestimation.
  - TD3Agent: Agent managing networks, target smoothing, delayed policy updates, Polyak soft updates, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer:
    """Experience replay buffer for storing off-policy transitions."""
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store a new transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store(self, obs, act, rew, next_obs, done):
        """Backward-compatible alias for add()."""
        self.add(obs, act, rew, next_obs, done)

    def sample(self, batch_size: int, device: torch.device = None):
        """Sample a batch of transitions uniformly at random as PyTorch tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)
        device = device or torch.device("cpu")
        return (
            torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def sample_batch(self, batch_size: int = 32):
        """Sample a batch returning a dictionary of numpy arrays."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.states[idxs],
            act=self.actions[idxs],
            rew=self.rewards[idxs],
            next_obs=self.next_states[idxs],
            done=self.dones[idxs],
        )

    def __len__(self):
        return self.size


class TwinCritic(nn.Module):
    """Twin Q-network: maps (state, action) -> (Q1, Q2) to prevent value overestimation."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TwinCritic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class DeterministicActor(nn.Module):
    """Deterministic policy network mapping state -> continuous action in [-1, 1]."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Direct [-1, 1] output
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent."""
    def __init__(
        self,
        state_dim: int = 56,
        action_dim: int = 12,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.device = torch.device(device)

        # Actor and Target Actor
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critic and Target Twin Critic
        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_it = 0
        self.last_actor_loss = 0.0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action. If evaluate=False, add exploration noise clipped to [-1, 1]."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if not evaluate and self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action.astype(np.float32)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """
        Perform a single TD3 gradient update:
          1. Sample transitions from buffer
          2. Compute target Q with Target Policy Smoothing
          3. Update Twin Critic
          4. If policy_delay steps reached, update Actor and soft-update targets
        """
        self.total_it += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

        # ----------------------------
        # 1. Target Policy Smoothing
        # ----------------------------
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            q1_target, q2_target = self.critic_target(next_states, next_actions)
            target_q = torch.min(q1_target, q2_target)
            y = rewards + self.gamma * (1.0 - dones) * target_q

        # ----------------------------
        # 2. Update Twin Critic
        # ----------------------------
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 3. Delayed Actor & Target Updates
        # ----------------------------
        actor_loss_val = self.last_actor_loss
        if self.total_it % self.policy_delay == 0:
            # Policy gradient: maximize Q1(s, pi(s))
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()
            self.last_actor_loss = actor_loss_val

            # Polyak soft updates for BOTH actor and critic targets
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
            "mean_q": q1.mean().item(),
        }

    def soft_update(self, net: nn.Module, target_net: nn.Module):
        """Polyak soft update: theta_target = tau * theta + (1 - tau) * theta_target."""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath: str, extra_state: dict = None):
        """Save network weights, optimizer states, and metadata."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
        }
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """Load network weights, optimizer states, and metadata."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
        return checkpoint