''' 
This model is trained on google colab, so there may be some dependencies that are not installed in your local environment.
And there might be some things that are not compatible with your local environment and you want to change it.
'''

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
from src.go1_env import go1_env
from src.terrain.config import TerrainConfig

RUN_NAME = "your_run_name"  # Replace with your desired run name
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "src", "checkpoints", RUN_NAME)
LOG_DIR = os.path.join(REPO_ROOT, "src", "logs", RUN_NAME)
FLAT_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene.xml")
OBSTACLE_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene_obstacles.xml")
# Curriculum stage: 0 = flat trot, 1 = bumps, 2 = hurdle. Override via env.
CURRICULUM_STAGE = int(os.environ.get("GO1_STAGE", "0"))
CPG_MODE = os.environ.get("GO1_CPG_MODE", "off")  # SB3 path stays cpg off for now
STAGE_XML = {0: FLAT_XML, 1: OBSTACLE_XML, 2: OBSTACLE_XML}
STAGE_TERRAIN = {0: "flat", 1: "rough", 2: "hurdle"}
XML_PATH = STAGE_XML.get(CURRICULUM_STAGE, FLAT_XML)

# --- Config ---


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Custom callback: save VecNormalize stats alongside model checkpoints ---
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.get_vec_normalize_env().save(
                os.path.join(self.save_path, "latest_vecnormalize.pkl")
            )
        return True

# --- Env constructor (not registered — passed as a callable, not a string id) ---
def make_go1_env(curriculum_stage=CURRICULUM_STAGE, cpg_mode=CPG_MODE):
    xml = STAGE_XML.get(curriculum_stage, FLAT_XML)
    terrain = TerrainConfig(mode=STAGE_TERRAIN.get(curriculum_stage, "flat"))
    return go1_env(xml_file=xml, terrain_config=terrain, cpg_mode=cpg_mode)

# --- Resume detection ---
latest_checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
        print(f"Loading latest checkpoint: {latest_checkpoint}")

# Plan hygiene: old latest_vecnormalize.pkl was fit on 52-dim obs; the env is
# now 56-dim. GO1_FRESH_VECNORM=1 forces a fresh VecNormalize fit (required
# before any 56-dim SB3 training). Without it we still try the saved stats
# but fall back to fresh on obs-dim mismatch instead of crashing mid-run.
FRESH_VECNORM = os.environ.get("GO1_FRESH_VECNORM", "0") == "1"


def _fresh_vec_env():
    return VecNormalize(make_vec_env(make_go1_env, n_envs=4), norm_obs=True, norm_reward=True)


if latest_checkpoint and not FRESH_VECNORM:
    vecnorm_path = os.path.join(CHECKPOINT_DIR, "latest_vecnormalize.pkl")
    try:
        env = VecNormalize.load(vecnorm_path, make_vec_env(make_go1_env, n_envs=4))
    except Exception as exc:
        print(f"VecNormalize load failed ({exc}); starting fresh 56-dim stats.")
        env = _fresh_vec_env()
        latest_checkpoint = None
    else:
        model = PPO.load(
            os.path.join(CHECKPOINT_DIR, latest_checkpoint),
            env=env,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0001   # plain float, no closure, no custom_objects
        )
        # Guard: stale checkpoints trained on 52-dim obs cannot drive a 56-dim env.
        if int(model.observation_space.shape[0]) != 56:
            print("Stale 52-dim SB3 checkpoint; starting fresh 56-dim policy.")
            env = _fresh_vec_env()
            latest_checkpoint = None
if not latest_checkpoint:
    env = _fresh_vec_env()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=0.0001,   # plain float
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

# --- Checkpointing ---
checkpoint_callback = CheckpointCallback(
    save_freq=15000,
    save_path=CHECKPOINT_DIR,
    name_prefix="rl_model",
)


model.learn(
    total_timesteps=2500000,
    callback=[checkpoint_callback, SaveVecNormalizeCallback(save_freq=15000, save_path=CHECKPOINT_DIR)],
    reset_num_timesteps=(latest_checkpoint is None),
    tb_log_name=RUN_NAME
)

model.save(os.path.join(CHECKPOINT_DIR, "final"))
model.env.save(os.path.join(CHECKPOINT_DIR, "final_vecnormalize.pkl"))