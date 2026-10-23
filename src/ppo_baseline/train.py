from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
import os
import sys

# Repository path configuration (3 levels up from src/ppo_baseline/train.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig

# -----------------------------------------------------------------------
# Flat-Body Trot — PPO Parallel Training (feature/flat-body-trot branch)
#
# Compute & Environment optimizations:
#   • SubprocVecEnv with 8 parallel workers (full utilization of Ryzen cores)
#   • 1,400+ steps/s throughput (~45-55 mins for 4.5M steps)
#   • Relaxed healthy ceiling (0.44m) avoids premature death during trot swing
#   • Dynamic target height centered at 0.325 ± 0.025m (settled rest height)
# -----------------------------------------------------------------------

RUN_NAME = "flat_trot_ppo_v1"
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "src", "checkpoints", RUN_NAME)
LOG_DIR = os.path.join(REPO_ROOT, "src", "logs", RUN_NAME)
FLAT_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene.xml")
OBSTACLE_XML = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1", "scene_obstacles.xml")

CPG_MODE = os.environ.get("GO1_CPG_MODE", "fixed_residual")
CURRICULUM_STAGE = int(os.environ.get("GO1_STAGE", "0"))
STAGE_XML = {0: FLAT_XML, 1: OBSTACLE_XML}
STAGE_TERRAIN = {0: "flat", 1: "rough"}
XML_PATH = STAGE_XML.get(CURRICULUM_STAGE, FLAT_XML)

# Parallel workers: default 8 (utilizes the 6 physical cores + SMT on Ryzen 5 7530U)
N_ENVS = int(os.environ.get("GO1_N_ENVS", "8"))
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
    # Flat-body trot: full kinematic penalties from step 0
    env.form_weight = 1.0
    return env


def make_parallel_vec_env(n_envs=N_ENVS):
    return make_vec_env(make_go1_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


def fresh_vec_env(n_envs=N_ENVS):
    return VecNormalize(make_parallel_vec_env(n_envs=n_envs), norm_obs=True, norm_reward=True)


def train():
    latest_checkpoint = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
            print(f"Loading latest checkpoint: {latest_checkpoint}")

    fresh_vecnorm = os.environ.get("GO1_FRESH_VECNORM", "0") == "1"

    if latest_checkpoint and not fresh_vecnorm:
        vecnorm_path = os.path.join(CHECKPOINT_DIR, "latest_vecnormalize.pkl")
        try:
            env = VecNormalize.load(vecnorm_path, make_parallel_vec_env(n_envs=N_ENVS))
        except Exception as exc:
            print(f"VecNormalize load failed ({exc}); starting fresh with {N_ENVS} workers.")
            env = fresh_vec_env(n_envs=N_ENVS)
            latest_checkpoint = None
        else:
            model = PPO.load(
                os.path.join(CHECKPOINT_DIR, latest_checkpoint),
                env=env,
                tensorboard_log=LOG_DIR,
                learning_rate=0.0003
            )
            if int(model.observation_space.shape[0]) != 56:
                print("Stale checkpoint dimension; starting fresh policy.")
                env = fresh_vec_env(n_envs=N_ENVS)
                latest_checkpoint = None
    if not latest_checkpoint:
        env = fresh_vec_env(n_envs=N_ENVS)
        model = PPO(
            "MlpPolicy", env,
            # Hyperparameters tuned for 8 parallel environments:
            # 8 envs * 512 = 4,096 rollout buffer size
            learning_rate=0.0003,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=LOG_DIR
        )

    # Checkpointing every 15,000 steps per environment (120,000 total steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=CHECKPOINT_DIR,
        name_prefix="rl_model",
    )

    total_timesteps = int(os.environ.get("GO1_TOTAL_TIMESTEPS", "4500000"))
    print(f"Starting PPO training for {total_timesteps:,} steps with {N_ENVS} parallel environments...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, SaveVecNormalizeCallback(save_freq=15000, save_path=CHECKPOINT_DIR)],
        reset_num_timesteps=(latest_checkpoint is None),
        tb_log_name=RUN_NAME
    )

    model.save(os.path.join(CHECKPOINT_DIR, "final"))
    model.env.save(os.path.join(CHECKPOINT_DIR, "final_vecnormalize.pkl"))
    print("Training complete! Model and normalization stats saved.")


if __name__ == "__main__":
    train()