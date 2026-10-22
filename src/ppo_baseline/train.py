from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import sys

# Repository path configuration (3 levels up from src/ppo_baseline/train.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig

# -----------------------------------------------------------------------
# Flat-Body Trot — PPO training (feature/flat-body-trot branch)
#
# Goals:
#   • Biological trot gait: trunk stays flat and parallel to the floor.
#   • CPG mode: fixed_residual — RL policy learns small residual corrections
#     on top of the open-loop FixedTrotCPG waveform.
#   • No curriculum ramp: form_weight = 1.0 from step 0 (the CPG already
#     provides a warm gait baseline, so full kinematic penalties are safe
#     from the start).
# -----------------------------------------------------------------------

RUN_NAME = "flat_trot_ppo_v1"
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "src", "checkpoints", RUN_NAME)
LOG_DIR = os.path.join(REPO_ROOT, "src", "logs", RUN_NAME)
FLAT_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene.xml")
OBSTACLE_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene_obstacles.xml")

# CPG mode is fixed_residual for this branch; override via env var if needed.
CPG_MODE = os.environ.get("GO1_CPG_MODE", "fixed_residual")

# Curriculum stage: 0 = flat trot (only flat supported on this branch).
CURRICULUM_STAGE = int(os.environ.get("GO1_STAGE", "0"))
STAGE_XML = {0: FLAT_XML, 1: OBSTACLE_XML}
STAGE_TERRAIN = {0: "flat", 1: "rough"}
XML_PATH = STAGE_XML.get(CURRICULUM_STAGE, FLAT_XML)

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


# --- Env constructor ---
def make_go1_env(curriculum_stage=CURRICULUM_STAGE, cpg_mode=CPG_MODE):
    xml = STAGE_XML.get(curriculum_stage, FLAT_XML)
    terrain = TerrainConfig(mode=STAGE_TERRAIN.get(curriculum_stage, "flat"))
    env = go1_env(xml_file=xml, terrain_config=terrain, cpg_mode=cpg_mode)
    # Flat-body trot: full kinematic penalties from step 0 — no curriculum ramp.
    env.form_weight = 1.0
    return env


# --- Resume detection ---
latest_checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
        print(f"Loading latest checkpoint: {latest_checkpoint}")

FRESH_VECNORM = os.environ.get("GO1_FRESH_VECNORM", "0") == "1"


def _fresh_vec_env():
    return VecNormalize(make_vec_env(make_go1_env, n_envs=4), norm_obs=True, norm_reward=True)


if latest_checkpoint and not FRESH_VECNORM:
    vecnorm_path = os.path.join(CHECKPOINT_DIR, "latest_vecnormalize.pkl")
    try:
        env = VecNormalize.load(vecnorm_path, make_vec_env(make_go1_env, n_envs=4))
    except Exception as exc:
        print(f"VecNormalize load failed ({exc}); starting fresh.")
        env = _fresh_vec_env()
        latest_checkpoint = None
    else:
        model = PPO.load(
            os.path.join(CHECKPOINT_DIR, latest_checkpoint),
            env=env,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003
        )
        if int(model.observation_space.shape[0]) != 56:
            print("Stale checkpoint; starting fresh.")
            env = _fresh_vec_env()
            latest_checkpoint = None
if not latest_checkpoint:
    env = _fresh_vec_env()
    model = PPO(
        "MlpPolicy", env,
        # --- Hyperparameters tuned for flat-body trot ---
        # Larger rollouts (1024 steps) improve credit assignment for gait quality.
        # Small entropy bonus (0.005) prevents premature gait collapse early in training.
        # Slightly higher learning rate (0.0003) for faster initial convergence.
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
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