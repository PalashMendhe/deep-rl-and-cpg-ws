"""Evaluation renderer for the flat-body-trot PPO policy (SB3).

Loads the trained flat_trot_ppo_v1 model and runs a visual rollout.
Tracks gait-quality metrics specific to the flat-body trot:
  - trunk height (should stay near 0.30 m)
  - pitch angle (should stay near 0 rad)
  - body_height_penalty
  - trot rhythm validity

Usage:
    # Headless — saves a GIF
    GO1_SAVE_GIF=1 python src/eval_render.py

    # Live viewer
    python src/eval_render.py
"""
import os
import sys

# Avoid Wayland/GLFW/GNOME compositor and GLib-GIO antialiasing schema crashes
local_share = os.path.expanduser("~/.local/share")
if local_share not in os.environ.get("XDG_DATA_DIRS", ""):
    os.environ["XDG_DATA_DIRS"] = f"{local_share}:{os.environ.get('XDG_DATA_DIRS', '')}"
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"
os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
os.environ["GDK_BACKEND"] = "x11"
os.environ.setdefault("GLFW_PLATFORM", "x11")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env

# ── Paths ────────────────────────────────────────────────────────────────────
RUN_NAME = os.environ.get("GO1_RUN_NAME", "flat_trot_ppo_v1")
CHECKPOINT_DIR = os.environ.get(
    "GO1_CHECKPOINT_DIR",
    os.path.join(REPO_ROOT, "src", "checkpoints", RUN_NAME),
)

# Pick the latest checkpoint zip automatically, or use a specific one.
_zips = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")] if os.path.isdir(CHECKPOINT_DIR) else []
_default_model = (
    os.path.join(CHECKPOINT_DIR, max(_zips, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f))))
    if _zips else os.path.join(CHECKPOINT_DIR, "final.zip")
)
model_path    = os.environ.get("GO1_MODEL_PATH", _default_model)
vecnorm_path  = os.path.join(CHECKPOINT_DIR, "latest_vecnormalize.pkl")

FLAT_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene.xml")
SAVE_GIF = os.environ.get("GO1_SAVE_GIF", "0") == "1"
output_gif = os.path.join(CHECKPOINT_DIR, "flat_trot_eval.gif")

print(f"[eval] model     : {model_path}")
print(f"[eval] vecnorm   : {vecnorm_path}")
print(f"[eval] cpg_mode  : fixed_residual | form_weight=1.0")

# ── Environment ──────────────────────────────────────────────────────────────
render_mode = "human" if not SAVE_GIF else "rgb_array"


def _make_env():
    env = go1_env(
        xml_file=FLAT_XML,
        cpg_mode="fixed_residual",      # flat-body trot CPG
        render_mode=render_mode,
    )
    env.form_weight = 1.0               # full kinematic penalties (no curriculum)
    return env


eval_env = DummyVecEnv([_make_env])
eval_env = VecNormalize.load(vecnorm_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(model_path, env=eval_env)

# ── Camera setup ─────────────────────────────────────────────────────────────
obs = eval_env.reset()
mujoco_env = eval_env.envs[0].unwrapped
eval_env.render()

viewer = getattr(mujoco_env.mujoco_renderer, "viewer", None)
if viewer is not None and hasattr(viewer, "cam"):
    viewer.cam.trackbodyid = mujoco_env.model.body("trunk").id
    viewer.cam.type = 2         # mjCAMERA_TRACKING
    viewer.cam.distance = 2.5   # closer — flat trot is compact
    viewer.cam.elevation = -18  # slight downward angle to see the flat body
    viewer.cam.azimuth = 90     # side-on view shows body level best

# ── GIF writer ───────────────────────────────────────────────────────────────
if SAVE_GIF:
    writer = imageio.get_writer(output_gif, fps=30)

# ── Rollout ──────────────────────────────────────────────────────────────────
episodes = 0

# Gait-quality accumulators (flat-body specific)
trunk_heights   = []
pitch_angles    = []
height_penalties = []
trot_flags      = []
x_velocities    = []

import time

speed = float(os.environ.get("GO1_SPEED", "1.0"))
target_step_dt = 0.02 / max(0.1, speed)  # 50 Hz base physics step

if not SAVE_GIF:
    print(f"Realtime Mode: Enabled ({speed:.1f}x speed)")

for _ in range(1000):
    step_t0 = time.time()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

    frame = eval_env.render()
    if SAVE_GIF and frame is not None:
        writer.append_data(frame)
    elif not SAVE_GIF:
        elapsed = time.time() - step_t0
        if target_step_dt > elapsed:
            time.sleep(target_step_dt - elapsed)

    raw_info = info[0] if isinstance(info, (list, tuple)) else info
    if isinstance(raw_info, dict):
        trunk_heights.append(float(raw_info.get("current_height", 0.0)))
        height_penalties.append(float(raw_info.get("raw_body_height_penalty", 0.0)))
        trot_flags.append(float(raw_info.get("raw_trot_flag", 0.0)))
        x_velocities.append(float(raw_info.get("local_x_velocity", 0.0)))

        # Compute pitch from raw env state
        mj = eval_env.envs[0].unwrapped
        q = mj.data.qpos[3:7]
        euler = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        pitch_angles.append(float(euler[1]))

    if done[0]:
        episodes += 1
        if episodes >= 10:
            break

# ── Summary ──────────────────────────────────────────────────────────────────
def _fmt(arr, unit=""):
    if not arr:
        return "n/a"
    return f"mean={np.mean(arr):.4f}{unit} std={np.std(arr):.4f}{unit} max={np.max(np.abs(arr)):.4f}{unit}"

print(f"\n{'─'*60}")
print(f"  Flat-Body Trot Eval  |  {episodes} episodes")
print(f"{'─'*60}")
print(f"  Trunk height  : {_fmt(trunk_heights, 'm')}  [target: 0.30 - 0.35 m]")
print(f"  Pitch angle   : {_fmt(pitch_angles, 'rad')}  [target: ~0]")
print(f"  Height penalty: {_fmt(height_penalties)}")
print(f"  Trot validity : valid_frac={1.0 - np.mean(trot_flags):.3f}  [target: 1.0]")
print(f"  X velocity    : {_fmt(x_velocities, 'm/s')}  [target: 0.6 m/s]")
print(f"{'─'*60}")

if SAVE_GIF:
    writer.close()
    print(f"\nGIF saved → {output_gif}")

eval_env.close()