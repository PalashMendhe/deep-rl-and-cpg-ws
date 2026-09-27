import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import imageio
import numpy as np
import os
from src.go1_env import go1_env

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.environ.get("GO1_CHECKPOINT_DIR", os.path.join(REPO_ROOT, "src", "checkpoints", "your_run_name"))
model_path = os.path.join(CHECKPOINT_DIR, "rl_model_120000_steps.zip")              # Path to the trained model file.
vecnorm_path = os.path.join(CHECKPOINT_DIR, "latest_vecnormalize.pkl")          # Path to the vector normalization file.
EVAL_STAGE = os.environ.get("GO1_EVAL_STAGE", "flat")  # flat | hurdle
EVAL_XML = os.environ.get(
    "GO1_EVAL_XML",
    os.path.join(
        REPO_ROOT,
        "mujoco_menagerie",
        "unitree_go1",
        "scene_obstacles.xml" if EVAL_STAGE == "hurdle" else "scene.xml",
    ),
)
output_gif = os.path.join(CHECKPOINT_DIR, f"output_{EVAL_STAGE}.gif")                  # Path to save the output GIF.

'''
Create the evaluation environment using DummyVecEnv and load the vector normalization parameters.
human rendering mode is enabled for visualization
change to rgb_array if you want to save the gif
'''
eval_env = DummyVecEnv([lambda: go1_env(xml_file=EVAL_XML, render_mode="human")]) # Replace with the actual path to your XML file for the Go1 environment.
eval_env = VecNormalize.load(vecnorm_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(model_path, env=eval_env)

obs = eval_env.reset()
mujoco_env = eval_env.envs[0].unwrapped
eval_env.render()

viewer = getattr(mujoco_env.mujoco_renderer, "viewer", None)
if viewer is not None:
    viewer.cam.trackbodyid = mujoco_env.model.body('trunk').id  # check actual body name, may not be "trunk"
    viewer.cam.type = 2  # mjCAMERA_TRACKING
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20

SAVE_GIF = os.environ.get("GO1_SAVE_GIF", "0") == "1"  # GIF slows rendering; opt in per run.
if SAVE_GIF:
    writer = imageio.get_writer(output_gif, fps=30)

episodes = 0
jump_hits = 0
jump_successes = 0
jump_landings = 0.0
clearances = []
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if SAVE_GIF:
        writer.append_data(eval_env.render())
    raw_info = info[0] if isinstance(info, (list, tuple)) else info
    if isinstance(raw_info, dict):
        if "clearance_m" in raw_info:
            clearances.append(float(raw_info.get("clearance_m", 0.0)))
        if "landing_bonus" in raw_info:
            jump_landings += float(raw_info.get("landing_bonus", 0.0))
    if done[0]:
        episodes += 1
        if isinstance(raw_info, dict) and "jump_success" in raw_info:
            jump_hits += int(bool(raw_info.get("hurdle_hit", False)))
            jump_successes += int(bool(raw_info.get("jump_success", False)))
        if episodes >= 20:
            break

print(f"Eval stage={EVAL_STAGE} episodes={episodes} "
      f"hit_rate={jump_hits/max(1,episodes):.2f} "
      f"success_rate={jump_successes/max(1,episodes):.2f} "
      f"mean_clearance={float(np.mean(clearances)) if clearances else 0.0:.3f}m "
      f"mean_landing_bonus={jump_landings/max(1,episodes):.3f}")

if SAVE_GIF:
    writer.close()  