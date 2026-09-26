import os
import sys
import imageio.v2 as imageio
import numpy as np
import torch
import gymnasium as gym
import time

try:
    import cv2  # CPG overlay bars; falls back to raw frames when missing
except Exception:
    cv2 = None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
gif_dir = os.path.expanduser("~/rl_env2/gifs")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_ppo import ActorNetwork, CHECKPOINT_PATH


def resolve_checkpoint_path():
    candidates = [
        CHECKPOINT_PATH,  # same file base_ppo.py writes -> no path drift
        os.path.join(ROOT_DIR, "ppo_checkpoint_6000640.pth"),
        os.path.join(ROOT_DIR, "checkpoints", "ppo_checkpoint_latest.pth"),
        os.path.join(ROOT_DIR, "checkpoints", "ppo_checkpoint_6000640.pth"),
        os.path.join(os.getcwd(), "ppo_checkpoint_6000640.pth"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find a PPO checkpoint in the workspace.")


checkpoint_path = resolve_checkpoint_path()
checkpoint = torch.load(checkpoint_path, map_location="cpu")


def _infer_dims(state_dict, default_state=56, default_residual=12):
    """Infer (state_dim, residual_dim, mod_dim) from a checkpoint.

    New checkpoints store state_dim/action_dim; legacy 52/12 ones are
    detected from weight shapes.
    """
    get = state_dict.get
    if "state_dim" in state_dict and "action_dim" in state_dict:
        sd, ad = int(get("state_dim")), int(get("action_dim"))
        return sd, min(ad, 12), max(0, ad - 12)
    w = get("actor.0.weight", get("residual_mean.weight", None))
    sd = default_state
    if w is not None and hasattr(w, "shape"):
        sd = int(w.shape[1])
    mw = get("mod_mean.weight", None)
    md = int(mw.shape[0]) if mw is not None and hasattr(mw, "shape") else 0
    return sd, default_residual, md


ckpt_dims = None
if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
    ckpt_dims = _infer_dims(checkpoint["actor_state_dict"])
    print(f"Checkpoint dims: state={ckpt_dims[0]}, residual={ckpt_dims[1]}, mod={ckpt_dims[2]}")

state_dim = ckpt_dims[0] if ckpt_dims else 56
residual_dim = ckpt_dims[1] if ckpt_dims else 12
mod_dim = ckpt_dims[2] if ckpt_dims else 0

actor = ActorNetwork(state_dim, residual_dim, mod_dim)
try:
    actor.load_state_dict(checkpoint["actor_state_dict"])
except RuntimeError as exc:
    print(f"Shape mismatch ({exc}); warm-starting trunk + residual head.")
    skipped = actor.load_legacy_state_dict(checkpoint["actor_state_dict"])
    print(f"Skipped keys: {skipped}")
actor.eval()

STAGE = os.environ.get("GO1_EVAL_STAGE", "hurdle")
EVAL_XML = os.path.join(ROOT_DIR, "mujoco_menagerie", "unitree_go1",
                        "scene_obstacles.xml")
if STAGE == "flat" or not os.path.exists(EVAL_XML):
    EVAL_XML = os.path.join(ROOT_DIR, "mujoco_menagerie", "unitree_go1", "scene.xml")
cpg_mode = checkpoint.get("cpg_mode", "off") if isinstance(checkpoint, dict) else "off"
print(f"Eval: stage={STAGE} xml={os.path.basename(EVAL_XML)} cpg_mode={cpg_mode} "
      f"actor=({state_dim},{residual_dim}+{mod_dim})")

env = gym.make("Go1Env-v0", xml_file=EVAL_XML, cpg_mode=cpg_mode,
               residual_scale=0.10, render_mode="human")
state, _ = env.reset()
assert state.shape == (state_dim,), f"obs {state.shape} != actor ({state_dim},)"

frames = []
gif_dir = os.path.join(ROOT_DIR, "gifs")
os.makedirs(gif_dir, exist_ok=True)
gif_path = os.path.join(gif_dir, f"go1_{STAGE}_{cpg_mode}_rollout.gif")
# Ensure the renderer/viewer is initialized before touching camera params.
mujoco_env = env.unwrapped
try:
    # First render call often creates the viewer; ignore return value.
    _ = env.render()
except Exception:
    pass

viewer = getattr(mujoco_env.mujoco_renderer, "viewer", None)
if viewer is None:
    # wait a short moment for the renderer to initialize
    for _ in range(10):
        time.sleep(0.05)
        viewer = getattr(mujoco_env.mujoco_renderer, "viewer", None)
        if viewer is not None:
            break
mujoco_env = env.unwrapped

# Force the renderer to initialize its offscreen viewer
mujoco_env.render()

# Access the offscreen viewer directly
viewer = mujoco_env.mujoco_renderer.viewer

if viewer is not None:
    try:
        # Get the body ID for 'trunk' (MuJoCo 3.x style via model)
        body_id = mujoco_env.model.body('trunk').id
        
        # Configure smooth tracking
        viewer.cam.type = 1  # mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        
        # 2.5 to 3.5 meters is optimal zoom for a Go1. 12.0 is way too far.
        viewer.cam.distance = 2.0  
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 90  # Gives an angled 3/4 view
    except Exception as exc:
        print(f"Couldn't set camera parameters: {exc}")
else:
    print("WARNING: Mujoco viewer is None. Camera offsets will not be applied.")
episodes = 0
jump_episodes = 0
jump_hits = 0
jump_successes = 0
jump_landings = 0.0
clearances = []
LEG_NAMES = ("FR", "FL", "RR", "RL")


def _overlay_cpg(frame, info):
    """Plan Step-9 CPG overlay: per-leg swing bars + phase/clearance text.

    Skipped (raw frame) when cv2 is unavailable so GIF saving never breaks.
    """
    if cv2 is None or not isinstance(info, dict):
        return frame
    try:
        h, w = frame.shape[0], frame.shape[1]
        img = frame.copy()
        phase = float(info.get("cpg_phase", 0.0))
        boost = float(info.get("jump_boost", 0.0))
        clear = float(info.get("clearance_m", 0.0))
        dist = float(info.get("dist_to_hurdle", 0.0))
        bar_w, bar_h, x0, y0 = 130, 12, 12, h - 4 * 26 - 44
        for i, name in enumerate(LEG_NAMES):
            s = float(np.sin(phase + i * np.pi))  # trot: diagonal pairs oppose
            fill = int(bar_w * (0.5 + 0.5 * s))
            y = y0 + i * 26
            cv2.rectangle(img, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
            cv2.rectangle(img, (x0, y), (x0 + fill, y + bar_h), (90, 200, 90), -1)
            cv2.putText(img, name, (x0 + bar_w + 8, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        txt = f"phase {phase % 6.283:.2f}  boost {boost:.2f}  dist {dist:.2f}m  clear {clear:.2f}m"
        cv2.putText(img, txt, (12, y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return img
    except Exception:
        return frame


for _ in range(2000):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = actor(state_tensor)
    state, reward, terminated, truncated, info = env.step(action.numpy().flatten())
    if isinstance(info, dict) and "dist_to_hurdle" in info:
        clearances.append(float(info.get("clearance_m", 0.0)))

    try:
        frame = env.render()
        if frame is not None:
            if isinstance(frame, np.ndarray):
                frames.append(_overlay_cpg(frame.astype(np.uint8), info))
            else:
                frames.append(_overlay_cpg(np.asarray(frame, dtype=np.uint8), info))
    except Exception as exc:
        print(f"Render skipped: {exc}")

    if terminated or truncated:
        episodes += 1
        if isinstance(info, dict) and "jump_success" in info:
            jump_episodes += 1
            jump_hits += int(bool(info.get("hurdle_hit", False)))
            jump_successes += int(bool(info.get("jump_success", False)))
        if isinstance(info, dict) and "landing_bonus" in info:
            jump_landings += float(info.get("landing_bonus", 0.0))
        state, _ = env.reset()
        if episodes >= 20:
            break

print(f"Episodes: {episodes} | jump_eps: {jump_episodes} "
      f"hit_rate={jump_hits/max(1,jump_episodes):.2f} "
      f"success_rate={jump_successes/max(1,jump_episodes):.2f} "
      f"mean_clearance={float(np.mean(clearances)) if clearances else 0.0:.3f}m "
      f"mean_landing_bonus={jump_landings/max(1,jump_episodes):.3f}")

if frames:
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"Saved GIF to {gif_path}")
else:
    print("No frames were captured; GIF was not saved.")

env.close()
