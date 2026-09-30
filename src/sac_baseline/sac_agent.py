import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
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

    def sample(self, batch_size: int, device: torch.device):
        """Sample a batch of transitions uniformly at random as PyTorch tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.actions[indices], dtype=torch.float32, device=device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size


class TwinCritic(nn.Module):
    """Twin Q-network: maps (state, action) -> (Q1, Q2) to prevent value overestimation."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 architecture: [state + action] -> hidden -> hidden -> 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture: [state + action] -> hidden -> hidden -> 1
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Returns Q1(s, a) and Q2(s, a)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with tanh squashing for continuous [-1, 1] action bounds."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Heads for Gaussian parameters
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """Returns mean and log_std given state."""
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor, epsilon: float = 1e-6):
        """
        Samples an action using the reparameterization trick, applies tanh squashing,
        and computes the analytical change-of-variables log-probability:
            log pi(a|s) = log N(u; mu, sigma) - sum(log(1 - tanh^2(u) + eps))
        Returns:
            action: Squashed action in [-1, 1]
            log_prob: Scalar log probability per batch element
            mean_action: Deterministic squashed mean (for greedy evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick: u = mu + sigma * eps
        dist = Normal(mean, std)
        u = dist.rsample()

        # Squashing: a = tanh(u)
        action = torch.tanh(u)

        # Log-probability with change-of-variables correction
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class SACAgent:
    """Soft Actor-Critic (SAC) Agent for continuous quadruped control."""
    def __init__(
        self,
        state_dim: int = 56,
        action_dim: int = 12,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float = None,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # 1. Critic and Target Critic
        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 2. Actor
        self.actor = SquashedGaussianActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 3. Entropy Temperature (Alpha)
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            # Target entropy heuristic: -dim(Action)
            self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(alpha)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action for environment step. If evaluate=True, use deterministic mean."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(state_tensor)
        selected = mean_action if evaluate else action
        return selected.cpu().numpy().flatten()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """
        Perform a single SAC gradient update:
          1. Sample transitions from buffer
          2. Compute Bellman regression target and update Twin Critic
          3. Compute entropy-regularized Q-value and update Actor
          4. Update temperature parameter alpha
          5. Polyak soft-update target critics
        Returns dictionary of training metrics.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

        # ----------------------------
        # 1. Update Twin Critic
        # ----------------------------
        with torch.no_grad():
            # Sample next actions from current policy: a' ~ pi(·|s')
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            # Clipped double-Q trick: target_Q = min(Q1_targ, Q2_targ) - alpha * log_prob
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            # Bellman target: y = r + gamma * (1 - done) * min_q_target
            y = rewards + self.gamma * (1.0 - dones) * min_q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 2. Update Actor
        # ----------------------------
        sampled_actions, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # J_pi = alpha * log_prob - min(Q1, Q2)
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # 3. Update Temperature (Alpha)
        # ----------------------------
        alpha_loss = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ----------------------------
        # 4. Soft Update Target Critic (Polyak)
        # ----------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            "mean_q": min_q_pi.mean().item(),
        }

    def save_checkpoint(self, filepath: str, extra_state: dict = None):
        """Save network weights and optimizer states, plus optional extra metadata."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
        }
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """Load network weights and optimizer states. Returns the loaded checkpoint dict."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.alpha = checkpoint["alpha"]
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        return checkpoint
