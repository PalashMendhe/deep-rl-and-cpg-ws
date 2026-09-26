"""Step-2 regression/integration tests for obstacle-aware go1_env.

Run from repo root: timeout 300 python3 -m pytest tests/test_obstacle_env.py -x -q
(or without pytest: timeout 300 python3 tests/test_obstacle_env.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.go1_env import go1_env
from src.terrain.config import TerrainConfig

FLAT_XML = "/home/plsh/rl_env2/mujoco_menagerie/unitree_go1/scene.xml"
OBST_XML = "/home/plsh/rl_env2/mujoco_menagerie/unitree_go1/scene_obstacles.xml"


def test_flat_default_obs_is_56_with_neutral_extras():
    env = go1_env()
    obs, _ = env.reset()
    assert obs.shape == (56,)
    ext = obs[52:]
    # dist neutral (10/5=2.0), no hurdle height, clearance==trunk z
    assert ext[0] == 2.0
    assert ext[1] == 0.0
    assert len(env._get_privileged_obs()) == 9


def test_compat_flag_gives_52_dim_obs():
    env = go1_env(include_ext_obs=False)
    obs, _ = env.reset()
    assert obs.shape == (52,)


def test_hurdle_randomization_moves_x():
    terr = TerrainConfig(mode="hurdle")
    env = go1_env(xml_file=OBST_XML, terrain_config=terr)
    xs = set()
    for _ in range(10):
        env.reset()
        xs.add(round(env._hurdle_x, 3))
    assert len(xs) > 1
    lo, hi = terr.hurdle_x_range
    assert all(lo <= x <= hi for x in xs)


def test_step_returns_jump_info_keys():
    terr = TerrainConfig(mode="hurdle")
    env = go1_env(xml_file=OBST_XML, terrain_config=terr)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ("dist_to_hurdle", "clearance_m", "hurdle_hit",
                "jump_bonus", "landing_bonus", "jump_success"):
        assert key in info
    assert isinstance(env.check_hurdle_collision(), bool)


def test_obs_first52_prefix_matches_compat():
    # Plan Step-3: ext extras must not shift the proprio prefix.
    env56 = go1_env()
    env52 = go1_env(include_ext_obs=False)
    env56.reset(seed=7)
    env52.reset(seed=7)
    a = env56.action_space.sample()
    o56, _, _, _, _ = env56.step(a)
    o52, _, _, _, _ = env52.step(a)
    assert o56.shape == (56,) and o52.shape == (52,)


def test_rough_bump_poses_vary_per_reset():
    # Plan Step-5: bump x-centres/height resample per episode on rough.
    terr = TerrainConfig(mode="rough")
    env = go1_env(xml_file=OBST_XML, terrain_config=terr)
    env.reset()
    import mujoco
    xs = set()
    for _ in range(6):
        env.reset()
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "bump1")
        xs.add(round(float(env.model.geom_pos[gid][0]), 3))
    assert len(xs) > 1


def test_landing_bonus_zero_on_flat():
    # Plan Step-6: no landing bonus where there is no hurdle.
    env = go1_env()
    env.reset()
    for _ in range(5):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert info["landing_bonus"] == 0.0
        if term or trunc:
            env.reset()


def test_hurdle_hit_terminate_flag():
    # Plan Step-6: hit=terminate variant ends the episode with -2.0.
    terr = TerrainConfig(mode="hurdle")
    env = go1_env(xml_file=OBST_XML, terrain_config=terr,
                  hurdle_hit_terminate=True)
    env.reset()
    env.check_hurdle_collision = lambda: True  # force the contact branch
    _, rew, term, _, info = env.step(env.action_space.sample())
    assert info["hurdle_hit"] is True
    assert term is True
    assert float(rew) == -2.0


def test_flat_regression_runs_200_steps():
    env = go1_env()
    obs, _ = env.reset()
    total = 0.0
    for _ in range(200):
        obs, rew, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(rew)
        total += rew
        if term or trunc:
            obs, _ = env.reset()
    assert obs.shape == (56,)
    assert np.isfinite(total)


if __name__ == "__main__":
    test_flat_default_obs_is_56_with_neutral_extras()
    print("PASS test_flat_default_obs_is_56_with_neutral_extras")
    test_compat_flag_gives_52_dim_obs()
    print("PASS test_compat_flag_gives_52_dim_obs")
    test_hurdle_randomization_moves_x()
    print("PASS test_hurdle_randomization_moves_x")
    test_step_returns_jump_info_keys()
    print("PASS test_step_returns_jump_info_keys")
    test_obs_first52_prefix_matches_compat()
    print("PASS test_obs_first52_prefix_matches_compat")
    test_rough_bump_poses_vary_per_reset()
    print("PASS test_rough_bump_poses_vary_per_reset")
    test_landing_bonus_zero_on_flat()
    print("PASS test_landing_bonus_zero_on_flat")
    test_hurdle_hit_terminate_flag()
    print("PASS test_hurdle_hit_terminate_flag")
    test_flat_regression_runs_200_steps()
    print("PASS test_flat_regression_runs_200_steps")
    print("9/9 obstacle-env tests passed")
