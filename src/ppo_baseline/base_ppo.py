import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import os
import sys

# Repository path configuration (3 levels up from src/ppo_baseline/base_ppo.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch.distributions import Normal
import gymnasium as gym

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig


def _make_go1_env(**kwargs):
    return go1_env(**kwargs)

try:
    gym.register(
        id='Go1Env-v0',
        entry_point=_make_go1_env,
    )
except Exception:  # re-import (e.g. renderers) must not crash
    pass

# --- Curriculum / CPG config (override via env vars) ---
# CPG_MODE: off | fixed_residual | parametric | hopf (fixed at start; action dim follows)
CPG_MODE = os.environ.get("GO1_CPG_MODE", "fixed_residual")
RESIDUAL_SCALE = float(os.environ.get("GO1_RESIDUAL_SCALE", "0.10"))
CPG_K_FB = float(os.environ.get("GO1_CPG_K_FB", "0.5"))
CPG_COUPLING = float(os.environ.get("GO1_CPG_COUPLING", "2.0"))
LANDING_BONUS_W = float(os.environ.get("GO1_LANDING_BONUS_W", "1.0"))
HURDLE_HIT_TERMINATE = os.environ.get("GO1_HURDLE_HIT_TERMINATE", "0") == "1"
STATE_DIM = 56          # 52 proprio + 4 hurdle-relative extras
RESIDUAL_DIM = 12
MOD_DIM = 6 if CPG_MODE in ("parametric", "hopf") else 0
ACTION_DIM = RESIDUAL_DIM + MOD_DIM  # 12 for off/fixed, 18 for parametric/hopf
MENAGERIE_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1")
FLAT_XML = os.path.join(MENAGERIE_DIR, "scene.xml")
OBSTACLE_XML = os.path.join(MENAGERIE_DIR, "scene_obstacles.xml")
HURDLE_FLAT_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle_flat.xml")
HURDLE_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle.xml")
STAGE0_END = int(os.environ.get("GO1_STAGE0_END", "1500000"))  # flat
STAGE1_END = int(os.environ.get("GO1_STAGE1_END", "3000000"))  # rough


def stage_for_step(global_step):
    if global_step < STAGE0_END:
        return 0
    if global_step < STAGE1_END:
        return 1
    return 2


def build_env(stage):
    """Stage 0: flat trot. Stage 1: bumps. Stage 2: hurdle jump."""
    cpg_kwargs = {}
    if CPG_MODE == "hopf":
        # Schedule: weak feedback on flat (open-loop trot), full on rough/hurdle.
        cpg_kwargs = {"cpg_k_fb": CPG_K_FB if stage >= 1 else 0.1,
                      "cpg_coupling": CPG_COUPLING}
    # Plan Step-5: residual authority grows with the curriculum so the flat
    # stage leans on the CPG trot and rough/hurdle stages give the policy
    # room to correct and jump. GO1_RESIDUAL_SCALE overrides only cpg off.
    if CPG_MODE == "off":
        residual_scale = RESIDUAL_SCALE
    else:
        residual_scale = 0.05 if stage == 0 else 0.15
    if stage == 0:
        return gym.make('Go1Env-v0', xml_file=FLAT_XML,
                        terrain_config=TerrainConfig(mode="flat"),
                        cpg_mode=CPG_MODE, residual_scale=residual_scale,
                        landing_bonus_w=LANDING_BONUS_W,
                        hurdle_hit_terminate=HURDLE_HIT_TERMINATE,
                        **cpg_kwargs)
    if stage == 1:
        stage1_mode = os.environ.get("GO1_STAGE1_MODE", "hurdle_flat" if os.path.exists(HURDLE_FLAT_XML) else "rough")
        if stage1_mode == "hurdle":
            stage1_xml = HURDLE_XML if os.path.exists(HURDLE_XML) else OBSTACLE_XML
        elif stage1_mode == "hurdle_flat":
            stage1_xml = HURDLE_FLAT_XML if os.path.exists(HURDLE_FLAT_XML) else OBSTACLE_XML
        else:
            stage1_xml = OBSTACLE_XML
        return gym.make('Go1Env-v0', xml_file=stage1_xml,
                        terrain_config=TerrainConfig(mode=stage1_mode),
                        cpg_mode=CPG_MODE, residual_scale=residual_scale,
                        landing_bonus_w=LANDING_BONUS_W,
                        hurdle_hit_terminate=HURDLE_HIT_TERMINATE,
                        **cpg_kwargs)
    stage2_xml = HURDLE_XML if os.path.exists(HURDLE_XML) else OBSTACLE_XML
    return gym.make('Go1Env-v0', xml_file=stage2_xml,
                    terrain_config=TerrainConfig(mode="hurdle"),
                    cpg_mode=CPG_MODE, residual_scale=residual_scale,
                    landing_bonus_w=LANDING_BONUS_W,
                    hurdle_hit_terminate=HURDLE_HIT_TERMINATE,
                    **cpg_kwargs)

CHECKPOINT_DIR = os.environ.get("GO1_CHECKPOINT_DIR", REPO_ROOT)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint_latest.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define the neural network architecture for the actor and critic
class ActorNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, residual_dim=RESIDUAL_DIM,
                 mod_dim=MOD_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.residual_dim = residual_dim
        self.mod_dim = mod_dim
        self.action_dim = residual_dim + mod_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),

        )
        self.residual_mean = nn.Linear(256, residual_dim)
        self.residual_log_std = nn.Parameter(torch.zeros(residual_dim))
        if mod_dim > 0:
            self.mod_mean = nn.Linear(256, mod_dim)
            self.mod_log_std = nn.Parameter(torch.zeros(mod_dim))
        else:
            self.mod_mean = None
            self.mod_log_std = None

    # Back-compat: old checkpoints used actor_mean / actor_log_std (12-dim).
    @property
    def actor_mean(self):
        return self.residual_mean

    @property
    def actor_log_std(self):
        return self.residual_log_std

    def forward(self, state, action=None):
        features = self.actor(state)
        mean_parts = [self.residual_mean(features)]
        std_parts = [torch.clamp(self.residual_log_std, min=-20, max=2).expand(
            features.shape[0], -1)]
        if self.mod_mean is not None:
            mean_parts.append(self.mod_mean(features))
            std_parts.append(torch.clamp(self.mod_log_std, min=-20, max=2).expand(
                features.shape[0], -1))
        action_mean = torch.cat(mean_parts, dim=-1)
        std = torch.cat(std_parts, dim=-1)
        probs = Normal(action_mean, torch.exp(std))
        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(axis=-1)
        entropy = probs.entropy().sum(axis=-1)

        return action, log_prob, entropy

    def load_legacy_state_dict(self, legacy, strict_shapes=True):
        """Load a 52/12 checkpoint into a 56/12+ actor.

        Shared trunk + residual head copy where shapes match; 56-dim input
        row extras and (optional) mod head stay randomly initialized.
        Returns list of skipped keys.
        """
        own = self.state_dict()
        skipped = []
        for key, value in legacy.items():
            if key not in own:
                skipped.append(key)
                continue
            if own[key].shape != value.shape:
                skipped.append(key)
                continue
            own[key] = value
        # Map legacy actor_mean/actor_log_std names onto residual head.
        for legacy_key, own_key in (
                ("actor_mean.weight", "residual_mean.weight"),
                ("actor_mean.bias", "residual_mean.bias"),
                ("actor_log_std", "residual_log_std")):
            if (legacy_key in legacy and own_key in own
                    and legacy[legacy_key].shape == own[own_key].shape
                    and own_key not in legacy):
                own[own_key] = legacy[legacy_key]
        self.load_state_dict(own, strict=True)
        return skipped

class CriticNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, priv_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.priv_dim = priv_dim

        self.critic = nn.Sequential(
            nn.Linear(state_dim + priv_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    @classmethod
    def asymmetric(cls, state_dim=STATE_DIM, priv_dim=9):
        """Critic with privileged obstacle extras (ext obs + contacts + z)."""
        return cls(state_dim=state_dim, priv_dim=priv_dim)

    def forward(self, state, priv=None):
        if self.priv_dim > 0:
            assert priv is not None, "asymmetric critic needs priv obs"
            state = torch.cat([state, priv], dim=-1)
        value = self.critic(state)
        return value
# Rollout buffer to store experiences for PPO updates
class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def store(self, state, action, reward, log_prob, value, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done 
        self.ptr = (self.ptr + 1) % self.buffer_size

    def clear(self):
        self.ptr = 0
# Compute advantages and returns using Generalized Advantage Estimation (GAE)
    def compute_advantage(self, last_value, gamma=0.99, lam=0.95):
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size -1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values
        return self.advantages, self.returns

# PPO Agent class to handle the update of actor and critic networks
class PPOAgent:
    def update(self, 
               buffer, 
               actor, 
               critic, 
               last_value,
               actor_optimizer, 
               critic_optimizer, 
               epochs=5, 
               minibatch_size=64, 
               clip_epsilon=0.2, 
               value_coef=0.5, 
               entropy_coef=0.01
               ):
        advantages, returns = buffer.compute_advantage(last_value = last_value) # Compute advantages and returns
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.tensor(buffer.states, dtype=torch.float32)
        actions = torch.tensor(buffer.actions, dtype=torch.float32)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)

        
        small_batch_size = minibatch_size
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            indices = torch.randperm(buffer.buffer_size)
            batches = torch.split(indices, small_batch_size)
            for batch_indice in batches:
                batch_states = states[batch_indice]
                batch_actions = actions[batch_indice]
                batch_old_log_probs = old_log_probs[batch_indice]
                batch_returns = returns[batch_indice]
                batch_advantages = advantages[batch_indice]
                

                new_actions, new_log_probs, entropy = actor(batch_states, batch_actions)
                new_values = critic(batch_states).squeeze()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages # Unclipped surrogate
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages # Clipped surrogate
                actor_loss = -torch.min(surrogate1, surrogate2).mean() - entropy_coef * entropy.mean() # Actor loss with entropy regularization

                critic_loss = value_coef * (batch_returns - new_values).pow(2).mean()

                if torch.any(torch.isnan(ratio)):
                    print("WARNING: NaN detected in ratio. Skipping this update.")
                

                # Update actor and critic networks
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                critic_optimizer.step()  

def resolve_checkpoint_path():
    # Priority 1: Environment variable override
    env_path = os.environ.get("GO1_CHECKPOINT_PATH", None)
    if env_path and env_path.lower() in ("none", "fresh", "scratch", "new"):
        return None
    if env_path and os.path.exists(env_path):
        return env_path

    candidate_dirs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs.extend([
        REPO_ROOT,
        os.path.join(REPO_ROOT, "checkpoints"),
        os.path.join(REPO_ROOT, "flat_checkpoints"),
        script_dir,
        os.path.join(script_dir, "checkpoints"),
        os.getcwd(),
    ])

    seen_dirs = set()
    for directory in candidate_dirs:
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)

        if not os.path.isdir(directory):
            continue

        latest_path = os.path.join(directory, "ppo_checkpoint_latest.pth")
        if os.path.exists(latest_path):
            return latest_path

        checkpoint_files = [
            name for name in os.listdir(directory)
            if name.startswith("ppo_checkpoint_") and name.endswith(".pth")
        ]
        if not checkpoint_files:
            continue

        numbered_checkpoints = []
        for name in checkpoint_files:
            stem = os.path.splitext(name)[0]
            suffix = stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                numbered_checkpoints.append((int(suffix), os.path.join(directory, name)))

        if numbered_checkpoints:
            return max(numbered_checkpoints, key=lambda item: item[0])[1]

    return None


def main():
    #initializtion of the actor and critic networks, optimizers, and the PPO agent
    agent = PPOAgent()
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    STAGE_START_STEP = int(os.environ.get("GO1_STAGE_START_STEP", "0"))
    TARGET_MAX_STEPS = int(os.environ.get("GO1_MAX_STEPS", "12000000"))
    LR_BASE = float(os.environ.get("GO1_LR_BASE", "1e-4"))
    max_training_timesteps = TARGET_MAX_STEPS
    buffer_size = 2048
    actor = ActorNetwork(state_dim, RESIDUAL_DIM, MOD_DIM)
    critic = CriticNetwork(state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_BASE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_BASE)
    buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
    checkpoint_interval = 10
    update_count = 0
    prev_step = 0
    stage = stage_for_step(0)
    env = build_env(stage)

    state, info = env.reset()
    global_step = 0
    accumulated_reward = 0
    episode_count = 0

    # Random policy sanity check
    print(f"Running random policy sanity check (CPG_MODE={CPG_MODE}, stage={stage})...")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}, mean: {obs.mean():.4f}, min: {obs.min():.4f}, max: {obs.max():.4f}")
    assert obs.shape == (state_dim,), f"expected obs {(state_dim,)}, got {obs.shape}"
    total_reward = 0
    for i in range(200):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i}, total reward: {total_reward:.4f}")
            obs, _ = env.reset()
            total_reward = 0
    print("Sanity check complete. Starting training...")

    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
        except RuntimeError as exc:
            # Legacy 52/12 checkpoint -> warm-start trunk + residual head.
            print(f"Shape mismatch loading checkpoint ({exc}); warm-starting trunk + residual head.")
            skipped = actor.load_legacy_state_dict(checkpoint['actor_state_dict'])
            print(f"Skipped keys (kept random init): {skipped}")
            try:
                critic.load_state_dict(checkpoint['critic_state_dict'])
            except RuntimeError:
                print("Critic shape mismatch too; keeping fresh critic.")
        try:
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        except (ValueError, KeyError) as exc:
            print(f"Optimizer parameter group mismatch ({exc}). Initializing fresh Adam optimizer for expanded action space.")
            actor_optimizer = optim.Adam(actor.parameters(), lr=LR_BASE)
        try:
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        except (ValueError, KeyError) as exc:
            print(f"Critic optimizer parameter group mismatch ({exc}). Initializing fresh Adam optimizer.")
            critic_optimizer = optim.Adam(critic.parameters(), lr=LR_BASE)

        RESET_EXPLORATION = os.environ.get("GO1_RESET_EXPLORATION", "0") == "1"
        if RESET_EXPLORATION:
            with torch.no_grad():
                actor.residual_log_std.clamp_(min=-0.7)  # sigma >= 0.5 rad
                if actor.mod_log_std is not None:
                    actor.mod_log_std.fill_(0.0)        # sigma = 1.0 for modulation
            print("Exploration noise re-inflated (residual_log_std clamped >= -0.7, mod_log_std = 0.0)")

        global_step = checkpoint['global_step']
        episode_count = checkpoint.get('episode_count', 0)
        stage = stage_for_step(global_step)
        env = build_env(stage)
        state, info = env.reset()
        print(f"Resuming training from checkpoint at global step {global_step} from {checkpoint_path}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    jump_hits = 0
    jump_successes = 0
    jump_episodes = 0
    while global_step < max_training_timesteps:
        new_stage = stage_for_step(global_step)
        if new_stage != stage:
            stage = new_stage
            env = build_env(stage)
            state, info = env.reset()
            print(f"=== Curriculum stage -> {stage} (CPG_MODE={CPG_MODE}) at step {global_step} ===")
        with torch.no_grad():
            for _ in range(buffer_size):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, log_prob, _ = actor(state_tensor)
                action_np = action.detach().numpy().flatten()

                # Kinematic floor: keep posture/trot pressure from step 0 so the
                # policy cannot entrench slide/fall habits during the early
                # explore phase; ramp to full weight by 2M steps.
                if global_step < 1000000:
                    form_weight = 0.2
                elif global_step > 2000000:
                    form_weight = 1.0
                else:
                    form_weight = 0.2 + 0.8 * (global_step - 1000000) / 1000000.0

                env.unwrapped.form_weight = form_weight

                next_state, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                value = critic(state_tensor).item()

                buffer.store(state, action_np, reward, log_prob.item(), value, done)

                state = next_state
                accumulated_reward += reward
                global_step += 1
                done = terminated or truncated
                if done:
                    if terminated:
                        last_value = 0.0
                    elif truncated:
                        with torch.no_grad():
                            last_value = critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item()

                    print(f"Episode: {episode_count}, Total Reward: {accumulated_reward}, Global Step: {global_step}")
                    episode_length = global_step - prev_step
                    print(f"Episode Length: {episode_length} steps")
                    print(f"velocity_reward: {info['velocity_reward']:.4f}, smoothness_penalty: {info['action_smoothness_penalty']:.4f}")
                    if stage >= 1 and 'hurdle_hit' in info:
                        jump_episodes += 1
                        jump_hits += int(bool(info.get('hurdle_hit', False)))
                        jump_successes += int(bool(info.get('jump_success', False)))
                        print(f"jump: clearance={info.get('clearance_m', 0.0):.3f}m "
                              f"hit_rate={jump_hits/max(1,jump_episodes):.2f} "
                              f"success_rate={jump_successes/max(1,jump_episodes):.2f}")

                    prev_step = global_step

                    state, info = env.reset()
                    episode_count += 1
                    accumulated_reward = 0
                if not done :
                    with torch.no_grad():
                        last_value = critic(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).item()
                # Calculate the remaining fraction of training
                progress = (global_step - STAGE_START_STEP) / max(1, TARGET_MAX_STEPS - STAGE_START_STEP)
                frac = max(0.05, 1.0 - progress)  # Floor at 5% LR
                lr_now = LR_BASE * frac

                # Apply the decayed learning rate to both optimizers
                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = lr_now
                for param_group in critic_optimizer.param_groups:
                    param_group['lr'] = lr_now
        # Tighter clip + lower entropy once the jump stage starts.
        clip_eps = 0.15 if stage == 2 else 0.2
        ent_coef = 0.005 if stage >= 1 else 0.01
        agent.update(buffer, actor, critic, last_value, actor_optimizer, critic_optimizer,
                     clip_epsilon=clip_eps, entropy_coef=ent_coef)
        buffer.clear()
        update_count +=1
        if update_count % checkpoint_interval == 0:
            checkpoint = {
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'global_step': global_step,
                'episode_count': episode_count,
                'cpg_mode': CPG_MODE,
                'state_dim': state_dim,
                'action_dim': action_dim,
                }
            torch.save(checkpoint, CHECKPOINT_PATH)
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"ppo_checkpoint_{global_step}.pth"))

if __name__ == "__main__": 
    main()
    