"""Simulation and Rollout Renderer for SAC Policy on Unitree Go1.

Supports all curriculum stages (flat, hurdle_flat / flat_hurdle, hurdle),
tracks real-time locomotion metrics, and generates annotated rollout GIFs
with CPG phase and leg state overlays.
"""

import os
import sys
import time
import argparse

# Avoid Wayland/GLFW compositor crashes on GNOME/Ubuntu
os.environ.setdefault("GLFW_PLATFORM", "x11")

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import imageio.v2 as imageio

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import glfw
except ImportError:
    glfw = None

# Repository path configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig
from src.sac_baseline.sac_agent import SACAgent

MENAGERIE_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1")
FLAT_XML = os.path.join(MENAGERIE_DIR, "scene.xml")
OBSTACLE_XML = os.path.join(MENAGERIE_DIR, "scene_obstacles.xml")
HURDLE_FLAT_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle_flat.xml")
HURDLE_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle.xml")

LEG_NAMES = ("FR", "FL", "RR", "RL")


def resolve_stage_config(stage_str: str):
    """Normalize stage string and return corresponding XML and TerrainConfig."""
    s = stage_str.lower().strip()
    if s in ("flat", "stage0", "0"):
        return FLAT_XML, TerrainConfig(mode="flat"), "flat"
    elif s in ("flat_hurdle", "hurdle_flat", "rough", "bumps", "stage1", "1"):
        xml = HURDLE_FLAT_XML if os.path.exists(HURDLE_FLAT_XML) else OBSTACLE_XML
        return xml, TerrainConfig(mode="hurdle_flat"), "hurdle_flat"
    elif s in ("hurdle", "jump", "stage2", "2"):
        xml = HURDLE_XML if os.path.exists(HURDLE_XML) else OBSTACLE_XML
        return xml, TerrainConfig(mode="hurdle"), "hurdle"
    else:
        xml = OBSTACLE_XML if os.path.exists(OBSTACLE_XML) else FLAT_XML
        return xml, TerrainConfig(mode="mixed"), "mixed"


def resolve_checkpoint(ckpt_arg=None):
    """Find best or latest SAC checkpoint."""
    if ckpt_arg and os.path.exists(ckpt_arg):
        return ckpt_arg

    candidates = [
        os.path.join(REPO_ROOT, "src", "sac_baseline", "checkpoints", "sac_go1_curriculum", "best_checkpoint.pth"),
        os.path.join(REPO_ROOT, "src", "sac_baseline", "checkpoints", "best_checkpoint.pth"),
        os.path.join(REPO_ROOT, "src", "sac_baseline", "checkpoints", "sac_go1_curriculum", "latest_checkpoint.pth"),
        os.path.join(REPO_ROOT, "src", "sac_baseline", "checkpoints", "latest_checkpoint.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find a valid SAC checkpoint. Provide one via --checkpoint.")


def overlay_diagnostics(frame, info, step_num, ep_return, stage_name):
    """Overlay HUD diagnostics (CPG bars, phase, velocities, clearances) on video frame."""
    if cv2 is None or not isinstance(info, dict):
        return frame

    try:
        h, w = frame.shape[0], frame.shape[1]
        img = frame.copy()

        # Top banner with semi-transparent background
        cv2.rectangle(img, (0, 0), (w, 42), (20, 20, 20), -1)
        banner_text = f"SAC Simulation | Stage: {stage_name.upper()} | Step: {step_num} | Return: {ep_return:.1f}"
        cv2.putText(img, banner_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry metrics
        fwd_vel = float(info.get("local_x_velocity", 0.0))
        height = float(info.get("current_height", 0.3))
        phase = float(info.get("cpg_phase", 0.0))
        boost = float(info.get("jump_boost", 0.0))

        # Bottom info card
        y_bot = h - 110
        cv2.rectangle(img, (8, y_bot - 10), (220, h - 8), (25, 25, 25), -1)
        cv2.rectangle(img, (8, y_bot - 10), (220, h - 8), (80, 80, 80), 1)

        cv2.putText(img, f"Fwd Vel: {fwd_vel:.2f} m/s", (16, y_bot + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)
        cv2.putText(img, f"Height:  {height:.3f} m", (16, y_bot + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)
        cv2.putText(img, f"Phase:   {phase % 6.283:.2f} rad", (16, y_bot + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        if boost > 0.01:
            cv2.putText(img, f"Boost:   {boost:.2f}", (16, y_bot + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1)

        # CPG Swing bars for each leg (bottom-right)
        bar_w, bar_h = 75, 10
        x0 = w - bar_w - 60
        y0 = h - 4 * 20 - 15
        cv2.rectangle(img, (x0 - 8, y0 - 16), (w - 6, h - 8), (25, 25, 25), -1)
        cv2.putText(img, "CPG Swing", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        for i, leg in enumerate(LEG_NAMES):
            s = float(np.sin(phase + i * np.pi))
            fill = int(bar_w * (0.5 + 0.5 * s))
            y = y0 + i * 20 + 8
            cv2.rectangle(img, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(img, (x0, y), (x0 + fill, y + bar_h), (50, 220, 80), -1)
            cv2.putText(img, leg, (x0 + bar_w + 6, y + bar_h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        return img
    except Exception:
        return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate and visualize trained SAC policy on Unitree Go1")
    parser.add_argument("--stage", type=str, default="hurdle_flat",
                        help="Curriculum stage: flat, hurdle_flat (flat_hurdle), hurdle")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAC checkpoint file (.pth)")
    parser.add_argument("--cpg-mode", type=str, default="auto",
                        choices=["auto", "fixed_residual", "parametric", "hopf"],
                        help="CPG mode: auto (detected from checkpoint/stage), fixed_residual, parametric, or hopf")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Number of simulation episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--render-mode", type=str, default="human",
                        choices=["human", "rgb_array"], help="Rendering mode (human for real-time GUI, rgb_array for GIF)")
    parser.add_argument("--viewer", type=str, default="opencv",
                        choices=["opencv", "mujoco"], help="GUI viewer backend: opencv (rock-solid, Wayland-safe, with HUD) or mujoco (native GLFW 3D window)")
    parser.add_argument("--adapt-ext-obs", action="store_true", default=True,
                        help="Align exteroceptive observation on stages without hurdle to prevent policy saturation")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier for real-time simulation (1.0 = real-time, 0.5 = half speed, 2.0 = 2x speed)")
    parser.add_argument("--gif-path", type=str, default=None,
                        help="Custom path to output GIF (rgb_array mode only)")
    parser.add_argument("--render-interval", type=int, default=2,
                        help="Frame capture interval for GIF (2 = every 2nd step for smooth 30fps)")
    return parser.parse_args()


def simulate():
    args = parse_args()

    # 1. Resolve Environment & Stage Config
    xml_file, terrain_config, stage_name = resolve_stage_config(args.stage)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    # Resolve CPG mode (support auto-detection from checkpoint)
    cpg_mode = args.cpg_mode
    if cpg_mode == "auto":
        cpg_mode = "parametric"
        if os.path.exists(checkpoint_path):
            try:
                probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(probe, dict) and "cpg_mode" in probe:
                    cpg_mode = probe["cpg_mode"]
                elif stage_name in ("hurdle", "hurdle_flat", "rough"):
                    cpg_mode = "hopf"
            except Exception:
                pass

    # Determine internal render_mode
    use_opencv_viewer = (args.render_mode == "human" and args.viewer == "opencv")
    env_render_mode = "rgb_array" if (use_opencv_viewer or args.render_mode == "rgb_array") else "human"

    print("=================================================================")
    print("=== SAC Quadruped Simulation & Rollout ===")
    print("=================================================================")
    print(f"Stage:           {stage_name} ({args.stage})")
    print(f"XML Scene:       {os.path.basename(xml_file)}")
    print(f"CPG Mode:        {cpg_mode} {'(Auto-detected)' if args.cpg_mode == 'auto' else ''}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Episodes:        {args.num_episodes}")
    print(f"Max Steps/Ep:    {args.max_steps}")
    print(f"Render Mode:     {args.render_mode}")
    if args.render_mode == "human":
        print(f"Viewer Backend:  {args.viewer.upper()} {'(Offscreen + OpenCV Window, Wayland Safe)' if use_opencv_viewer else '(MuJoCo Native GLFW via X11)'}")
        print(f"Speed:           {args.speed:.1f}x real-time")
    if args.adapt_ext_obs and stage_name != "hurdle":
        print("Ext-Obs Align:   Enabled (prevents policy saturation on hurdle-free terrain)")
    print("=================================================================\n")

    # 2. Build Environment
    raw_env = go1_env(
        xml_file=xml_file,
        terrain_config=terrain_config,
        cpg_mode=cpg_mode,
        render_mode=env_render_mode,
    )
    env = TimeLimit(raw_env, max_episode_steps=args.max_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 3. Load SAC Agent
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    loaded_meta = agent.load_checkpoint(checkpoint_path)
    if isinstance(loaded_meta, dict):
        step_trained = loaded_meta.get("step", "N/A")
        eval_rew = loaded_meta.get("best_eval_reward", "N/A")
        step_str = f"{step_trained:,}" if isinstance(step_trained, int) else str(step_trained)
        print(f"Checkpoint loaded successfully! (Trained steps: {step_str}, Record return: {eval_rew})")

    # Configure Tracking Camera locked on robot trunk
    mujoco_env = env.unwrapped
    try:
        # Pre-render once to instantiate viewer
        _ = env.render()
    except Exception:
        pass

    viewer = getattr(mujoco_env.mujoco_renderer, "viewer", None)
    try:
        body_id = mujoco_env.model.body("trunk").id
        viewers_to_tune = []
        if hasattr(mujoco_env.mujoco_renderer, "_viewers"):
            viewers_to_tune.extend(mujoco_env.mujoco_renderer._viewers.values())
        if viewer is not None and viewer not in viewers_to_tune:
            viewers_to_tune.append(viewer)

        for v in viewers_to_tune:
            if hasattr(v, "cam"):
                v.cam.type = 1  # mjCAMERA_TRACKING
                v.cam.trackbodyid = body_id
                v.cam.distance = 2.5
                v.cam.elevation = -15.0
                v.cam.azimuth = 90.0
        print("Configured MuJoCo camera locked on robot trunk.")
    except Exception as e:
        print(f"Note: Camera tuning skipped ({e})")

    # 4. Simulation Rollout
    frames = []
    episode_stats = []
    dt = float(getattr(mujoco_env, "dt", 0.01))
    target_step_time = dt / max(0.05, args.speed) if args.render_mode == "human" else 0.0
    user_exited = False

    if use_opencv_viewer and cv2 is not None:
        cv2.namedWindow("Go1 SAC Simulation", cv2.WINDOW_AUTOSIZE)

    for ep in range(args.num_episodes):
        if user_exited:
            break
        state, _ = env.reset(seed=ep * 100 + 42)
        ep_reward = 0.0
        step_count = 0
        done = False
        fell = False
        velocities = []

        print(f"\n--- Running Episode {ep + 1}/{args.num_episodes} ---")

        while not done:
            step_t0 = time.time()

            # Align exteroceptive observation if running on stage without physical hurdle
            if args.adapt_ext_obs and stage_name != "hurdle" and state.shape[0] >= 56:
                policy_state = state.copy()
                policy_state[52] = 0.60
                policy_state[53] = 0.80
                policy_state[54] = policy_state[2] - 0.12
            else:
                policy_state = state

            # Greedy deterministic action for evaluation
            action = agent.select_action(policy_state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            step_count += 1
            vel = float(info.get("local_x_velocity", 0.0))
            velocities.append(vel)

            # Option A: OpenCV Window Viewer (Wayland/VS Code safe, live HUD)
            if use_opencv_viewer and cv2 is not None:
                frame = env.render()
                if frame is not None:
                    annotated = overlay_diagnostics(frame, info, step_count, ep_reward, stage_name)
                    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Go1 SAC Simulation", bgr)
                    elapsed = time.time() - step_t0
                    wait_ms = max(1, int((target_step_time - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (27, ord('q')):
                        print("\nUser pressed exit key. Stopping simulation.")
                        user_exited = True
                        break

            # Option B: Native MuJoCo GLFW Viewer (via X11)
            elif args.render_mode == "human" and args.viewer == "mujoco":
                if viewer is not None and glfw is not None and hasattr(viewer, "window"):
                    if glfw.window_should_close(viewer.window):
                        print("\nViewer window closed by user. Exiting simulation.")
                        user_exited = True
                        break

                # Live HUD overlays on MuJoCo window
                if viewer is not None and hasattr(viewer, "add_overlay"):
                    try:
                        viewer.add_overlay(0, "Stage", stage_name.upper())
                        viewer.add_overlay(0, "Step", f"{step_count} / {args.max_steps}")
                        viewer.add_overlay(0, "Return", f"{ep_reward:.1f}")
                        viewer.add_overlay(2, "Fwd Vel", f"{vel:.2f} m/s")
                        viewer.add_overlay(2, "Height", f"{float(info.get('current_height', 0.3)):.3f} m")
                        viewer.add_overlay(2, "CPG Phase", f"{float(info.get('cpg_phase', 0.0)) % 6.283:.2f} rad")
                    except Exception:
                        pass

                # Update MuJoCo window display
                env.render()

                # Real-time pacing
                elapsed = time.time() - step_t0
                sleep_dur = target_step_time - elapsed
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

            # Option C: Capture video frame (rgb_array mode for GIF)
            elif args.render_mode == "rgb_array" and step_count % args.render_interval == 0:
                frame = env.render()
                if frame is not None:
                    annotated = overlay_diagnostics(frame, info, step_count, ep_reward, stage_name)
                    frames.append(annotated)

            if terminated:
                fell = True
            done = terminated or truncated
            state = next_state

        mean_v = float(np.mean(velocities)) if velocities else 0.0
        status = "Timeout (Completed Full Horizon)" if truncated else ("Fell" if fell else "Done")
        print(f"Episode {ep + 1} Result: {status}")
        print(f"  Return:        {ep_reward:.2f}")
        print(f"  Duration:      {step_count} steps")
        print(f"  Mean Fwd Vel:  {mean_v:.3f} m/s")

        episode_stats.append({
            "episode": ep + 1,
            "return": ep_reward,
            "steps": step_count,
            "mean_velocity": mean_v,
            "fell": fell,
            "completed": truncated,
        })

    env.close()
    if use_opencv_viewer and cv2 is not None:
        cv2.destroyAllWindows()

    # 5. Output Summary
    returns = [s["return"] for s in episode_stats]
    steps_list = [s["steps"] for s in episode_stats]
    vels = [s["mean_velocity"] for s in episode_stats]
    falls = sum(1 for s in episode_stats if s["fell"])

    print("\n=================================================================")
    print("=== Simulation Results Summary ===")
    print("=================================================================")
    print(f"Stage Tested:          {stage_name.upper()}")
    print(f"Episodes Run:          {args.num_episodes}")
    print(f"Mean Return:           {np.mean(returns):.2f} (Max: {np.max(returns):.2f}, Min: {np.min(returns):.2f})")
    print(f"Mean Episode Length:   {np.mean(steps_list):.1f} / {args.max_steps} steps")
    print(f"Mean Forward Velocity: {np.mean(vels):.3f} m/s")
    print(f"Fall Rate:             {falls / args.num_episodes:.1%} ({args.num_episodes - falls}/{args.num_episodes} clean traversals)")
    print("=================================================================")

    # 6. Save Animated GIF
    if frames and args.render_mode == "rgb_array":
        gif_dir = os.path.join(REPO_ROOT, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        default_gif_name = f"sac_go1_{stage_name}_rollout.gif"
        out_gif = args.gif_path or os.path.join(gif_dir, default_gif_name)

        print(f"\nEncoding {len(frames)} frames into animated GIF (30 fps)...")
        imageio.mimsave(out_gif, frames, fps=30, loop=0)
        file_size_mb = os.path.getsize(out_gif) / (1024 * 1024)
        print(f"Saved simulation rollout GIF -> {out_gif} ({file_size_mb:.2f} MB)")

        # Also copy to visualizations directory
        vis_gif = os.path.join(REPO_ROOT, "src", "visualizations", default_gif_name)
        try:
            imageio.mimsave(vis_gif, frames, fps=30, loop=0)
            print(f"Saved visualization copy     -> {vis_gif}")
        except Exception:
            pass


if __name__ == "__main__":
    simulate()
