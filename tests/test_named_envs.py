"""Unit and integration tests for hurdle_flat and hurdle environments."""
import os
import sys
import numpy as np
import gymnasium as gym
import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import hurdle_flat, hurdle, go1_env
from src.terrain.config import TerrainConfig


def test_hurdle_flat_geoms():
    """Verify hurdle_flat has three bump boxes only and no hurdle box."""
    env = hurdle_flat()
    geom_names = [env.model.geom(i).name for i in range(env.model.ngeom)]
    for bump in ("bump1", "bump2", "bump3"):
        assert bump in geom_names, f"Missing {bump} in hurdle_flat"
    assert "hurdle" not in geom_names, "hurdle should NOT be in hurdle_flat"
    assert env._has_hurdle_geom() is False
    env.close()


def test_hurdle_geoms():
    """Verify hurdle has larger hurdle box only and no bump boxes."""
    env = hurdle()
    geom_names = [env.model.geom(i).name for i in range(env.model.ngeom)]
    assert "hurdle" in geom_names, "hurdle must be in hurdle env"
    for bump in ("bump1", "bump2", "bump3"):
        assert bump not in geom_names, f"{bump} should NOT be in hurdle env"
    assert env._has_hurdle_geom() is True
    env.close()


def test_hurdle_flat_reset_and_rollout():
    """Verify hurdle_flat runs 100 steps, bumps reposition, obs is 56-dim."""
    env = hurdle_flat()
    obs, info = env.reset(seed=42)
    assert obs.shape == (56,)
    # Bump repositioning across resets
    bump1_xs = set()
    for _ in range(6):
        env.reset()
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "bump1")
        bump1_xs.add(round(float(env.model.geom_pos[gid][0]), 3))
    assert len(bump1_xs) > 1, "Bump poses should vary across resets"

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (56,)
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_hurdle_reset_and_rollout():
    """Verify hurdle runs 100 steps, hurdle repositions, jump keys in info."""
    env = hurdle()
    obs, info = env.reset(seed=42)
    assert obs.shape == (56,)
    # Hurdle repositioning across resets
    hurdle_xs = set()
    for _ in range(6):
        env.reset()
        hurdle_xs.add(round(float(env._hurdle_x), 3))
    assert len(hurdle_xs) > 1, "Hurdle x should vary across resets"

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (56,)
        assert np.isfinite(reward)
        for k in ("dist_to_hurdle", "clearance_m", "hurdle_hit", "jump_bonus", "landing_bonus", "jump_success"):
            assert k in info, f"Missing info key {k}"
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_gym_make_hurdle_flat():
    """Verify gym.make('hurdle_flat') works."""
    env = gym.make("hurdle_flat")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (56,)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(rew)
    env.close()


def test_gym_make_hurdle():
    """Verify gym.make('hurdle') works."""
    env = gym.make("hurdle")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (56,)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(rew)
    env.close()


def test_cpg_modes_with_named_envs():
    """Verify all CPG modes initialize and step properly in both environments."""
    for cpg_mode, act_dim in [("off", 12), ("fixed_residual", 12), ("parametric", 18), ("hopf", 18)]:
        env1 = hurdle_flat(cpg_mode=cpg_mode)
        assert env1.action_space.shape == (act_dim,)
        obs1, _ = env1.reset(seed=10)
        assert obs1.shape == (56,)
        _, r1, _, _, _ = env1.step(env1.action_space.sample())
        assert np.isfinite(r1)
        env1.close()

        env2 = hurdle(cpg_mode=cpg_mode)
        assert env2.action_space.shape == (act_dim,)
        obs2, _ = env2.reset(seed=10)
        assert obs2.shape == (56,)
        _, r2, _, _, _ = env2.step(env2.action_space.sample())
        assert np.isfinite(r2)
        env2.close()


if __name__ == "__main__":
    test_hurdle_flat_geoms()
    print("PASS test_hurdle_flat_geoms")
    test_hurdle_geoms()
    print("PASS test_hurdle_geoms")
    test_hurdle_flat_reset_and_rollout()
    print("PASS test_hurdle_flat_reset_and_rollout")
    test_hurdle_reset_and_rollout()
    print("PASS test_hurdle_reset_and_rollout")
    test_gym_make_hurdle_flat()
    print("PASS test_gym_make_hurdle_flat")
    test_gym_make_hurdle()
    print("PASS test_gym_make_hurdle")
    test_cpg_modes_with_named_envs()
    print("PASS test_cpg_modes_with_named_envs")
    print("All named env tests passed!")
