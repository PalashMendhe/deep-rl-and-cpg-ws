"""Twin Delayed Deep Deterministic Policy Gradient (TD3) Training Pipeline for Unitree Go1 Quadruped.

Implements Sections A through F:
  - Section A: Configuration & Hyperparameters (device, seeds, paths, TD3 hyperparams, schedule, logging)
  - Section B: Environment Construction & Curriculum Wrapper (stages 0, 1, 2, CPG selection: parametric -> hopf)
  - Section C: Warmup Phase (Buffer Seeding without network updates)
  - Section D: Main Environment & Training Loop (action query with exploration noise, step, done logic, buffer, TD3 updates)
  - Section E: Curriculum Progression Logic (stage transitions at 2M and 4M thresholds, buffer preservation)
  - Section F: Periodic Evaluation Protocol (deterministic evaluation, metrics logging, best/latest checkpoints)
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit

# ------------------------------------------------------------------------------
# Repository path configuration
# ------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig
from src.td3_baseline.td3_agent import TD3Agent, ReplayBuffer


# ==============================================================================
# Section A: Configuration & Hyperparameters
# ==============================================================================

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint and Log directories inside td3_baseline
TD3_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(TD3_DIR, "checkpoints")
DEFAULT_LOG_DIR = os.path.join(TD3_DIR, "logs")

# Environment XML paths
MENAGERIE_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1")
FLAT_XML = os.path.join(MENAGERIE_DIR, "scene.xml")
OBSTACLE_XML = os.path.join(MENAGERIE_DIR, "scene_obstacles.xml")
HURDLE_FLAT_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle_flat.xml")
HURDLE_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle.xml")

# Default TD3 Hyperparameters
BATCH_SIZE = 256
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005          # Target smoothing coefficient (Polyak tau)
POLICY_NOISE = 0.2   # Target policy smoothing noise std
NOISE_CLIP = 0.5     # Target policy noise clip limit
EXPLORATION_NOISE = 0.1  # Exploration noise std added to actions during training
POLICY_DELAY = 2     # Frequency of delayed policy / target updates
BUFFER_SIZE = 1_000_000

# Default Training schedule: 3.5M timesteps per stage over 3 curriculum stages (10.5M total)
TOTAL_TIMESTEPS = 10_500_000
WARMUP_STEPS = 10_000
EVAL_FREQ = 20_000
EVAL_EPISODES = 15

# Curriculum stage boundaries (steps): 3.5M per stage
STAGE0_END = 3_500_000  # Stage 0 (Flat, Parametric CPG) -> Stage 1 (Rough, Hopf CPG)
STAGE1_END = 7_000_000  # Stage 1 (Rough, Hopf CPG)       -> Stage 2 (Hurdles, Hopf CPG)

# CPG mode: "parametric_then_hopf" (Stage 0: parametric -> Stage 1 & 2: hopf; both 18-dim)
CPG_MODE = os.environ.get("GO1_CPG_MODE", "parametric_then_hopf")


def seed_all(seed: int):
    """Sets random seeds for NumPy, PyTorch, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_curriculum_stage(step: int, stage0_end: int = STAGE0_END, stage1_end: int = STAGE1_END) -> int:
    """Returns curriculum stage (0, 1, or 2) corresponding to the step count."""
    if step < stage0_end:
        return 0
    elif step < stage1_end:
        return 1
    return 2


def resolve_stage_cpg(stage: int, base_cpg_mode: str) -> str:
    """
    Resolves the active CPG mode for the specified stage.
    When base_cpg_mode is 'parametric_then_hopf', Stage 0 uses 'parametric'
    and Stages 1 & 2 use 'hopf'. Both modes share the 18-dim action space.
    """
    if base_cpg_mode == "parametric_then_hopf":
        return "parametric" if stage == 0 else "hopf"
    return base_cpg_mode


# ==============================================================================
# Section B: Environment Construction & Curriculum Wrapper
# ==============================================================================

def build_env(stage: int, cpg_mode: str = CPG_MODE, max_episode_steps: int = 1000):
    """
    Constructs the Go1 gymnasium environment for the specified curriculum stage.

    Stage 0: Flat ground (mode="flat"), Parametric CPG.
    Stage 1: Rough bumps (mode="hurdle_flat"), Hopf CPG.
    Stage 2: Hurdles (mode="hurdle"), Hopf CPG.
    """
    effective_cpg = resolve_stage_cpg(stage, cpg_mode)

    if stage == 0:
        xml_file = FLAT_XML
        terrain_mode = "flat"
    elif stage == 1:
        xml_file = HURDLE_FLAT_XML if os.path.exists(HURDLE_FLAT_XML) else OBSTACLE_XML
        terrain_mode = "hurdle_flat" if os.path.exists(HURDLE_FLAT_XML) else "rough"
    elif stage == 2:
        xml_file = HURDLE_XML if os.path.exists(HURDLE_XML) else OBSTACLE_XML
        terrain_mode = "hurdle"
    else:
        raise ValueError(f"Unknown curriculum stage: {stage}. Expected 0, 1, or 2.")

    terrain_config = TerrainConfig(mode=terrain_mode)
    raw_env = go1_env(
        xml_file=xml_file,
        terrain_config=terrain_config,
        cpg_mode=effective_cpg,
    )
    # Wrap with TimeLimit to handle maximum episode duration (truncation)
    env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)
    return env


# ==============================================================================
# Section F (Helper): Periodic Evaluation Protocol
# ==============================================================================

def evaluate_policy(
    agent: TD3Agent,
    stage: int,
    cpg_mode: str = CPG_MODE,
    num_episodes: int = EVAL_EPISODES,
    max_episode_steps: int = 1000,
):
    """
    Runs deterministic evaluation episodes without exploration noise.
    Computes mean evaluation reward, mean forward velocity, and fall rate.
    """
    effective_cpg = resolve_stage_cpg(stage, cpg_mode)
    eval_env = build_env(stage=stage, cpg_mode=effective_cpg, max_episode_steps=max_episode_steps)
    eval_returns = []
    eval_velocities = []
    falls = 0

    for ep in range(num_episodes):
        state, _ = eval_env.reset()
        ep_return = 0.0
        ep_vels = []
        done = False
        fell = False

        while not done:
            # Deterministic action selection (evaluate=True)
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += reward

            # Forward velocity extraction
            fwd_vel = info.get("local_x_velocity", 0.0)
            ep_vels.append(fwd_vel)

            # Robot fell / terminated
            if terminated:
                fell = True

            done = terminated or truncated
            state = next_state

        eval_returns.append(ep_return)
        eval_velocities.append(float(np.mean(ep_vels)) if ep_vels else 0.0)
        if fell:
            falls += 1

    eval_env.close()

    mean_eval_reward = float(np.mean(eval_returns)) if eval_returns else 0.0
    mean_forward_velocity = float(np.mean(eval_velocities)) if eval_velocities else 0.0
    fall_rate = float(falls / max(1, num_episodes))

    return mean_eval_reward, mean_forward_velocity, fall_rate


# ==============================================================================
# Training Orchestration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TD3 Curriculum Training on Unitree Go1")
    parser.add_argument("--run-name", type=str, default="td3_go1_curriculum", help="Run name for logging")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total training timesteps (default: 10.5M)")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS, help="Warmup exploration steps")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ, help="Evaluation frequency in timesteps")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="Evaluation episodes count")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="TD3 batch size")
    parser.add_argument("--actor-lr", type=float, default=ACTOR_LR, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=CRITIC_LR, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor")
    parser.add_argument("--tau", type=float, default=TAU, help="Target smoothing coefficient (Polyak tau)")
    parser.add_argument("--policy-noise", type=float, default=POLICY_NOISE, help="Target policy smoothing noise std")
    parser.add_argument("--noise-clip", type=float, default=NOISE_CLIP, help="Target policy noise clip limit")
    parser.add_argument("--exploration-noise", type=float, default=EXPLORATION_NOISE, help="Action exploration noise std")
    parser.add_argument("--policy-delay", type=int, default=POLICY_DELAY, help="Delayed policy update frequency")
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE, help="Replay buffer capacity")
    parser.add_argument("--cpg-mode", type=str, default=CPG_MODE,
                        choices=["parametric_then_hopf", "parametric", "hopf", "fixed_residual", "off"],
                        help="CPG mode: parametric_then_hopf (Stage 0: parametric -> Stage 1&2: hopf, 18-dim), parametric, hopf, fixed_residual, or off")
    parser.add_argument("--stage0-end", type=int, default=STAGE0_END, help="Curriculum threshold for Stage 0 -> 1 (default: 3.5M)")
    parser.add_argument("--stage1-end", type=int, default=STAGE1_END, help="Curriculum threshold for Stage 1 -> 2 (default: 7.0M)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Computation device (cuda/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (defaults to src/td3_baseline/checkpoints/<run_name>)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory (defaults to src/td3_baseline/logs/<run_name>)")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto",
                        help="Resume from checkpoint ('auto' or path to .pth checkpoint file)")
    parser.add_argument("--start-step", type=int, default=None,
                        help="Step count to resume training from (auto-detected if None)")
    return parser.parse_args()


def train():
    args = parse_args()

    # Section A: Configuration & Logging Directories
    seed_all(args.seed)
    device = torch.device(args.device)

    checkpoint_dir = args.checkpoint_dir or (
        os.path.join(DEFAULT_CHECKPOINT_DIR, args.run_name) if args.run_name else DEFAULT_CHECKPOINT_DIR
    )
    log_dir = args.log_dir or (
        os.path.join(DEFAULT_LOG_DIR, args.run_name) if args.run_name else DEFAULT_LOG_DIR
    )
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    print("=== Starting TD3 Training on Go1 ===")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"CPG Mode: {args.cpg_mode}")
    print(f"Total Timesteps: {args.total_timesteps:,} (Stage 0: {args.stage0_end:,}, Stage 1: {args.stage1_end:,})")
    print(f"Warmup Steps: {args.warmup_steps:,}")
    print(f"Eval Frequency: {args.eval_freq:,}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

    # Resolve resume checkpoint if requested
    resume_path = None
    start_step = args.warmup_steps
    best_eval_reward = -float("inf")

    if args.resume is not None:
        if args.resume == "auto":
            candidates = [
                os.path.join(checkpoint_dir, "latest_checkpoint.pth"),
                os.path.join(DEFAULT_CHECKPOINT_DIR, "latest_checkpoint.pth"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resume_path = cand
                    break
            if resume_path is None:
                print(f"Warning: --resume auto specified, but no latest_checkpoint.pth found in {checkpoint_dir} or {DEFAULT_CHECKPOINT_DIR}. Starting from scratch.")
        elif os.path.exists(args.resume):
            resume_path = args.resume
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    # Determine initial curriculum stage
    initial_step = 0
    if resume_path:
        try:
            probe_dict = torch.load(resume_path, map_location="cpu", weights_only=False)
            if "step" in probe_dict:
                initial_step = int(probe_dict["step"])
            elif args.start_step is not None:
                initial_step = args.start_step
            else:
                initial_step = 0
                log_file = os.path.join(DEFAULT_LOG_DIR, "train_td3.log")
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r") as f:
                            for line in reversed(f.readlines()):
                                if "Periodic Evaluation at Step" in line:
                                    parts = line.split("Step")[1].split("(")[0].strip().replace(",", "")
                                    initial_step = int(parts)
                                    break
                    except Exception:
                        pass
            if "best_eval_reward" in probe_dict:
                best_eval_reward = float(probe_dict["best_eval_reward"])
            del probe_dict
        except Exception as exc:
            print(f"Warning probing checkpoint: {exc}")
            initial_step = args.start_step or 0

        start_step = initial_step

    current_stage = get_curriculum_stage(start_step, args.stage0_end, args.stage1_end)
    effective_cpg = resolve_stage_cpg(current_stage, args.cpg_mode)
    env = build_env(stage=current_stage, cpg_mode=effective_cpg)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Obs Dimension: {state_dim}, Action Dimension: {action_dim} (Initial CPG: {effective_cpg})")

    # Initialize Replay Buffer & TD3 Agent
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size)
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        exploration_noise=args.exploration_noise,
        policy_delay=args.policy_delay,
        device=str(device),
    )

    if resume_path:
        print(f"Loading checkpoint weights from: {resume_path}")
        agent.load_checkpoint(resume_path)
        print(f"Resuming training from Step {start_step:,} (Stage {current_stage}, CPG {effective_cpg}, best eval reward: {best_eval_reward:.2f})")

        # Seed replay buffer with on-policy transitions
        seed_steps = min(2000, args.warmup_steps)
        print(f"\n--- Seeding replay buffer with {seed_steps:,} on-policy transitions ---")
        state, _ = env.reset(seed=args.seed)
        for seed_step in range(seed_steps):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            replay_buffer.add(state, action, reward, next_state, done)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
        print(f"Buffer seeded with {len(replay_buffer):,} on-policy transitions.\n")
    else:
        # ==========================================================================
        # Section C: Warmup Phase (Buffer Seeding with random actions)
        # ==========================================================================
        print(f"\n--- Starting Warmup Phase ({args.warmup_steps:,} steps) ---")
        state, _ = env.reset(seed=args.seed)

        for warmup_step in range(args.warmup_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            replay_buffer.add(state, action, reward, next_state, done)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

            if (warmup_step + 1) % max(1, args.warmup_steps // 5) == 0 or (warmup_step + 1) == args.warmup_steps:
                print(f"Warmup progress: {warmup_step + 1:,} / {args.warmup_steps:,} steps seeded.")

        print(f"Warmup completed. Replay buffer size: {len(replay_buffer):,}\n")

    # ==========================================================================
    # Section D: Main Environment & Training Loop
    # ==========================================================================
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    state, _ = env.reset()

    for step in range(start_step, args.total_timesteps):
        # 1. Action querying: action with exploration noise
        action = agent.select_action(state, evaluate=False)

        # 2. Environment step
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # 3. Done Flag Logic:
        # If truncated (time limit reached), store done = False.
        # If terminated (robot fell/died), store done = True.
        done = terminated

        # 4. Buffer storage
        replay_buffer.add(state, action, reward, next_state, done)

        # 5. Gradient step (1 update per environment step)
        metrics = agent.update(replay_buffer, batch_size=args.batch_size)

        # 6. State advancement
        state = next_state

        # Periodic training loss logging
        if (step + 1) % 100 == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, step + 1)

        # 7. Episode boundary handling
        if terminated or truncated:
            writer.add_scalar("train/episode_return", episode_reward, step + 1)
            writer.add_scalar("train/episode_length", episode_length, step + 1)
            writer.add_scalar("curriculum/stage", current_stage, step + 1)

            if episode_count % 10 == 0 or (step + 1) % 5000 == 0:
                print(
                    f"Step {step + 1:,} | Ep {episode_count:,} (Stage {current_stage}, CPG: {effective_cpg}) | "
                    f"Return: {episode_reward:7.2f} | Length: {episode_length:4d} | "
                    f"{'Fell' if terminated else 'Timeout'}"
                )

            state, _ = env.reset()
            episode_count += 1
            episode_reward = 0.0
            episode_length = 0

        # ======================================================================
        # Section E: Curriculum Progression Logic
        # ======================================================================
        new_stage = get_curriculum_stage(step + 1, args.stage0_end, args.stage1_end)
        if new_stage != current_stage:
            old_cpg = resolve_stage_cpg(current_stage, args.cpg_mode)
            new_cpg = resolve_stage_cpg(new_stage, args.cpg_mode)
            print(f"\n{'='*70}")
            print(f"=== Curriculum / CPG Transition at Step {step + 1:,} ===")
            print(f"Stage: {current_stage} -> {new_stage} | CPG Mode: {old_cpg} -> {new_cpg}")
            print(f"Preserving existing replay buffer ({len(replay_buffer):,} transitions).")
            print(f"{'='*70}\n")
            current_stage = new_stage
            effective_cpg = new_cpg
            env.close()
            # Rebuild environment with new terrain configuration and CPG mode
            env = build_env(stage=current_stage, cpg_mode=effective_cpg)
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            writer.add_scalar("curriculum/stage", current_stage, step + 1)

        # ======================================================================
        # Section F: Periodic Evaluation Protocol
        # ======================================================================
        if (step + 1) % args.eval_freq == 0:
            eval_cpg = resolve_stage_cpg(current_stage, args.cpg_mode)
            print(f"\n--- Periodic Evaluation at Step {step + 1:,} (Stage {current_stage}, CPG: {eval_cpg}) ---")
            mean_eval_reward, mean_fwd_vel, fall_rate = evaluate_policy(
                agent=agent,
                stage=current_stage,
                cpg_mode=eval_cpg,
                num_episodes=args.eval_episodes,
            )
            print(
                f"Eval @ Step {step + 1:,} | Mean Reward: {mean_eval_reward:.2f} | "
                f"Fwd Vel: {mean_fwd_vel:.3f} m/s | Fall Rate: {fall_rate:.2%}"
            )

            # Log evaluation metrics to TensorBoard
            writer.add_scalar("eval/mean_reward", mean_eval_reward, step + 1)
            writer.add_scalar("eval/mean_forward_velocity", mean_fwd_vel, step + 1)
            writer.add_scalar("eval/fall_rate", fall_rate, step + 1)
            writer.add_scalar("curriculum/stage", current_stage, step + 1)

            # Rolling latest checkpoint
            checkpoint_metadata = {
                "step": step + 1,
                "best_eval_reward": best_eval_reward,
                "stage": current_stage,
                "cpg_mode": args.cpg_mode,
                "effective_cpg": eval_cpg,
            }
            latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            agent.save_checkpoint(latest_ckpt_path, extra_state=checkpoint_metadata)
            if checkpoint_dir != DEFAULT_CHECKPOINT_DIR:
                agent.save_checkpoint(os.path.join(DEFAULT_CHECKPOINT_DIR, "latest_checkpoint.pth"), extra_state=checkpoint_metadata)
            print(f"Saved latest checkpoint -> {latest_ckpt_path}")

            # Best checkpoint on new record return
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                checkpoint_metadata["best_eval_reward"] = best_eval_reward
                best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
                agent.save_checkpoint(best_ckpt_path, extra_state=checkpoint_metadata)
                if checkpoint_dir != DEFAULT_CHECKPOINT_DIR:
                    agent.save_checkpoint(os.path.join(DEFAULT_CHECKPOINT_DIR, "best_checkpoint.pth"), extra_state=checkpoint_metadata)
                print(f"*** New best evaluation record ({best_eval_reward:.2f})! Saved -> {best_ckpt_path} ***\n")

    # Cleanup and final save
    final_metadata = {
        "step": args.total_timesteps,
        "best_eval_reward": best_eval_reward,
        "stage": current_stage,
        "cpg_mode": args.cpg_mode,
        "effective_cpg": effective_cpg,
    }
    final_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    agent.save_checkpoint(final_ckpt_path, extra_state=final_metadata)
    if checkpoint_dir != DEFAULT_CHECKPOINT_DIR:
        agent.save_checkpoint(os.path.join(DEFAULT_CHECKPOINT_DIR, "latest_checkpoint.pth"), extra_state=final_metadata)
    print(f"\nTraining complete. Final checkpoint saved to {final_ckpt_path}")
    writer.close()
    env.close()


if __name__ == "__main__":
    train()