import os
import sys
import glob
import time
import numpy as np
import torch
import gymnasium as gym

# Avoid Wayland/GLFW/GNOME compositor and GLib-GIO antialiasing schema crashes
local_share = os.path.expanduser("~/.local/share")
if local_share not in os.environ.get("XDG_DATA_DIRS", ""):
    os.environ["XDG_DATA_DIRS"] = f"{local_share}:{os.environ.get('XDG_DATA_DIRS', '')}"
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"
os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
os.environ["GDK_BACKEND"] = "x11"
os.environ.setdefault("GLFW_PLATFORM", "x11")

try:
    import glfw
except Exception:
    glfw = None

try:
    import cv2
except Exception:
    cv2 = None

import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
gif_dir = os.path.join(REPO_ROOT, "gifs")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.ppo_baseline.base_ppo import ActorNetwork, CHECKPOINT_PATH
from src.terrain.config import TerrainConfig


def resolve_checkpoint_path():
    env_path = os.environ.get("GO1_CHECKPOINT_PATH", None)
    if env_path:
        for p in [env_path, os.path.join(REPO_ROOT, env_path), os.path.join(SCRIPT_DIR, env_path)]:
            if os.path.exists(p):
                return p

    # 1. Search for .pth checkpoints
    pth_candidates = [
        os.path.join(REPO_ROOT, "ppo_checkpoint_latest.pth"),
        CHECKPOINT_PATH,
        os.path.join(REPO_ROOT, "flat_checkpoints", "ppo_checkpoint_3010560.pth"),
        os.path.join(REPO_ROOT, "flat_checkpoints", "ppo_checkpoint_1515520.pth"),
        os.path.join(REPO_ROOT, "ppo_checkpoint_6000640.pth"),
        os.path.join(REPO_ROOT, "checkpoints", "ppo_checkpoint_latest.pth"),
    ]
    for path in pth_candidates:
        if path and os.path.exists(path):
            return path

    found_pths = [p for p in glob.glob(os.path.join(REPO_ROOT, "**", "ppo_checkpoint_*.pth"), recursive=True) if os.path.exists(p)]
    if found_pths:
        found_pths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found_pths[0]

    # 2. Search for SB3 .zip checkpoints
    zip_candidates = [
        os.path.join(REPO_ROOT, "src", "checkpoints", "flat_trot_ppo_v1", "final.zip"),
    ]
    for path in zip_candidates:
        if path and os.path.exists(path):
            return path

    found_zips = [p for p in glob.glob(os.path.join(REPO_ROOT, "src", "checkpoints", "**", "*.zip"), recursive=True) if os.path.exists(p)]
    if found_zips:
        found_zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found_zips[0]

    raise FileNotFoundError("Could not find a PPO checkpoint (.pth or .zip) in the workspace.")


checkpoint_path = resolve_checkpoint_path()
is_sb3 = checkpoint_path.endswith(".zip")
print(f"Loaded checkpoint: {checkpoint_path} (Format: {'Stable-Baselines3' if is_sb3 else 'PyTorch'})")

STAGE = os.environ.get("GO1_EVAL_STAGE", "flat")
MENAGERIE_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1")

if STAGE == "flat":
    EVAL_XML = os.path.join(MENAGERIE_DIR, "scene.xml")
    terrain = TerrainConfig(mode="flat")
elif STAGE in ("hurdle_flat", "rough"):
    EVAL_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle_flat.xml")
    terrain = TerrainConfig(mode="hurdle_flat")
elif STAGE == "hurdle":
    EVAL_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle.xml")
    terrain = TerrainConfig(mode="hurdle")
else:
    EVAL_XML = os.path.join(MENAGERIE_DIR, "scene_obstacles.xml")
    terrain = TerrainConfig(mode="mixed")

if not os.path.exists(EVAL_XML):
    EVAL_XML = os.path.join(MENAGERIE_DIR, "scene.xml")

render_mode_env = os.environ.get("GO1_RENDER_MODE", "human")
is_human = (render_mode_env == "human")
viewer_backend = os.environ.get("GO1_VIEWER", "mujoco").lower()
speed = float(os.environ.get("GO1_SPEED", "1.0"))
target_step_dt = 0.02 / max(0.1, speed)  # 50 Hz physics pacing

internal_render_mode = "rgb_array" if (is_human and viewer_backend == "opencv") else render_mode_env

# ── Load Model & Setup Predictor ─────────────────────────────────────────────
if is_sb3:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cpg_mode = "fixed_residual"

    def _make():
        from src.go1_env import go1_env
        e = go1_env(xml_file=EVAL_XML, terrain_config=terrain, cpg_mode=cpg_mode, render_mode=internal_render_mode)
        e.form_weight = 1.0
        return e

    raw_vec_env = DummyVecEnv([_make])
    vecnorm_path = os.path.join(os.path.dirname(checkpoint_path), "latest_vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_vec_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_vec_env

    sb3_model = PPO.load(checkpoint_path, env=env)

    def predict_action(obs):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return action[0] if isinstance(env, (DummyVecEnv, VecNormalize)) else action

    state = env.reset()
    mujoco_env = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped

else:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cpg_mode = checkpoint.get("cpg_mode", "fixed_residual") if isinstance(checkpoint, dict) else "fixed_residual"

    def _infer_dims(state_dict, default_state=56, default_residual=12):
        get = state_dict.get
        if "state_dim" in state_dict and "action_dim" in state_dict:
            sd, ad = int(get("state_dim")), int(get("action_dim"))
            return sd, min(ad, 12), max(0, ad - 12)
        w = get("actor.0.weight", get("residual_mean.weight", None))
        sd = int(w.shape[1]) if w is not None and hasattr(w, "shape") else default_state
        mw = get("mod_mean.weight", None)
        md = int(mw.shape[0]) if mw is not None and hasattr(mw, "shape") else 0
        return sd, default_residual, md

    ckpt_dims = _infer_dims(checkpoint["actor_state_dict"]) if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint else (56, 12, 0)
    state_dim, residual_dim, mod_dim = ckpt_dims

    actor = ActorNetwork(state_dim, residual_dim, mod_dim)
    try:
        actor.load_state_dict(checkpoint["actor_state_dict"])
    except RuntimeError:
        actor.load_legacy_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    def predict_action(obs):
        st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act, _, _ = actor(st)
        return act.detach().numpy().flatten()

    env = gym.make("Go1Env-v0", xml_file=EVAL_XML, terrain_config=terrain, cpg_mode=cpg_mode,
                   residual_scale=0.10, render_mode=internal_render_mode)
    env.unwrapped.form_weight = 1.0
    state, _ = env.reset()
    mujoco_env = env.unwrapped

print(f"Eval: stage={STAGE} xml={os.path.basename(EVAL_XML)} cpg_mode={cpg_mode} viewer={viewer_backend.upper()}")
if is_human:
    print(f"Realtime Mode: Enabled ({speed:.1f}x speed)")
    print("Controls: Press 'q' or 'ESC' in the window to exit.")

# ── Camera Tracking Setup ───────────────────────────────────────────────────
try:
    _ = env.render()
except Exception:
    pass

viewers_to_tune = []
if hasattr(mujoco_env.mujoco_renderer, "_viewers"):
    viewers_to_tune.extend(mujoco_env.mujoco_renderer._viewers.values())
if hasattr(mujoco_env.mujoco_renderer, "viewer") and mujoco_env.mujoco_renderer.viewer is not None:
    if mujoco_env.mujoco_renderer.viewer not in viewers_to_tune:
        viewers_to_tune.append(mujoco_env.mujoco_renderer.viewer)

if viewers_to_tune:
    try:
        body_id = mujoco_env.model.body("trunk").id
        for v in viewers_to_tune:
            if hasattr(v, "cam"):
                v.cam.type = 2  # mjCAMERA_TRACKING
                v.cam.trackbodyid = body_id
                v.cam.distance = 2.5
                v.cam.elevation = -18.0
                v.cam.azimuth = 90
    except Exception as exc:
        print(f"Camera setting info: {exc}")

frames = []
os.makedirs(gif_dir, exist_ok=True)
gif_path = os.path.join(gif_dir, f"go1_{STAGE}_{cpg_mode}_rollout.gif")

LEG_NAMES = ("FR", "FL", "RR", "RL")


def _overlay_cpg(frame, info, pitch=0.0):
    if cv2 is None or not isinstance(info, dict):
        return frame
    try:
        h, w = frame.shape[0], frame.shape[1]
        img = frame.copy()
        phase = float(info.get("cpg_phase", 0.0))
        height = float(info.get("current_height", 0.0))
        bar_w, bar_h, x0, y0 = 130, 12, 12, h - 4 * 26 - 44
        for i, name in enumerate(LEG_NAMES):
            s = float(np.sin(phase + i * np.pi))
            fill = int(bar_w * (0.5 + 0.5 * s))
            y = y0 + i * 26
            cv2.rectangle(img, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
            cv2.rectangle(img, (x0, y), (x0 + fill, y + bar_h), (90, 200, 90), -1)
            cv2.putText(img, name, (x0 + bar_w + 8, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        txt = f"phase {phase % 6.283:.2f}  h {height:.3f}m  pitch {pitch:.3f}rad"
        cv2.putText(img, txt, (12, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return img
    except Exception:
        return frame


episodes = 0
trunk_heights, pitch_angles, height_penalties, trot_flags, x_velocities = [], [], [], [], []

user_exited = False
for step_idx in range(2000):
    step_t0 = time.time()
    action_np = predict_action(state)

    step_res = env.step(action_np)
    if len(step_res) == 5:
        state, reward, terminated, truncated, info = step_res
        done = terminated or truncated
    else:
        state, reward, done, info = step_res
        done = done[0] if isinstance(done, (list, np.ndarray)) else done

    raw_info = info[0] if isinstance(info, (list, tuple)) else info
    current_pitch = 0.0
    if hasattr(mujoco_env, "data") and hasattr(mujoco_env.data, "qpos"):
        q = mujoco_env.data.qpos[3:7]
        current_pitch = float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[1])

    if isinstance(raw_info, dict):
        trunk_heights.append(float(raw_info.get("current_height", 0.0)))
        height_penalties.append(float(raw_info.get("raw_body_height_penalty", 0.0)))
        trot_flags.append(float(raw_info.get("raw_trot_flag", 0.0)))
        x_velocities.append(float(raw_info.get("local_x_velocity", 0.0)))
        pitch_angles.append(current_pitch)

    if is_human and viewer_backend == "mujoco":
        try:
            env.render()
            h_viewer = mujoco_env.mujoco_renderer._viewers.get("human") if hasattr(mujoco_env.mujoco_renderer, "_viewers") else None
            if h_viewer is not None and glfw is not None and hasattr(h_viewer, "window"):
                if glfw.window_should_close(h_viewer.window):
                    print("\nUser closed simulation window. Stopping simulation.")
                    user_exited = True
                    break
        except Exception as exc:
            print(f"Render error: {exc}")

        elapsed = time.time() - step_t0
        if target_step_dt > elapsed:
            time.sleep(target_step_dt - elapsed)
    else:
        try:
            frame = env.render()
            if frame is not None:
                raw_frame = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
                annotated = _overlay_cpg(raw_frame.astype(np.uint8), raw_info, pitch=current_pitch)
                frames.append(annotated)

                if is_human and viewer_backend == "opencv" and cv2 is not None:
                    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Go1 PPO Simulation ({STAGE.upper()})", bgr)
                    elapsed = time.time() - step_t0
                    wait_ms = max(1, int((target_step_dt - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (27, ord('q')):
                        print("\nUser pressed exit key. Stopping simulation.")
                        user_exited = True
                        break
        except Exception as exc:
            print(f"Render skipped: {exc}")

    if user_exited:
        break

    if done:
        episodes += 1
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        if episodes >= 20:
            break

def _fmt_stat(arr, unit=""):
    if not arr:
        return "n/a"
    return f"mean={np.mean(arr):.4f}{unit} std={np.std(arr):.4f}{unit} max={np.max(np.abs(arr)):.4f}{unit}"

print(f"\n{'─'*60}")
print(f"  Flat-Body Trot PPO Eval  |  {episodes} episodes")
print(f"{'─'*60}")
print(f"  Trunk height  : {_fmt_stat(trunk_heights, 'm')}  [target: 0.30 - 0.35 m]")
print(f"  Pitch angle   : {_fmt_stat(pitch_angles, 'rad')}  [target: ~0]")
print(f"  Height penalty: {_fmt_stat(height_penalties)}")
print(f"  Trot validity : valid_frac={1.0 - np.mean(trot_flags):.3f}  [target: 1.0]" if trot_flags else "  Trot validity : n/a")
print(f"  X velocity    : {_fmt_stat(x_velocities, 'm/s')}  [target: 0.6 m/s]")
print(f"{'─'*60}")

if is_human and cv2 is not None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if frames:
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"Saved GIF to {gif_path}")

env.close()
