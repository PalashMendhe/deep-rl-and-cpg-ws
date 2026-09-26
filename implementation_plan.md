# Implementation Plan — Obstacles/Irregular Terrain + Staged CPG for Go1

## Overview
Enable the flat-ground Go1 trot to handle rough terrain and jump a hurdle, then migrate from pure joint-residual RL to staged CPG (fixed trot + residuals, then policy-modulated params, then coupled oscillators with feedback) across `base_ppo.py`, `src/train.py`, and both renderers.

Scope: (a) new MuJoCo obstacle scene preserving flat `mujoco_menagerie/unitree_go1/scene.xml` for regression, (b) env cleanup + obstacle-aware obs/reward in the effective `go1_env`, (c) decoupled CPG modules, (d) actor/critic + loop updates, (e) renderer upgrades. Approach is curriculum-first (flat, rough, hurdle) and residuals-first (CPG gives open-loop trot, RL corrects), then progressively hand CPG control to the policy.

Ground truth: effective env is the SECOND `go1_env` in `src/go1_env.py` (line 312; first class lines 23-289 is dead shadowed code). Obs 52-dim. Control `home_qpos[7:]+action*0.25`, `frame_skip=5`. Reward `3.0*vel - 0.005*smooth - nav - form_weight*kinematic`, clip [-10,10]. Prototype `_get_cpg_targets` lives only in dead code. Primary stack is custom PPO `base_ppo.py` (52->256->256->12 actor, `form_weight` 0->1 over steps 1M-2M, 6.01M steps, gym id `Go1Env-v0`); secondary SB3 `src/train.py` (2.5M steps, VecNormalize). World is flat plane only. MuJoCo 3.10, Gymnasium 1.3, SB3 2.9, torch 2.13 CPU-only.

## Types
- `TerrainConfig` (new dataclass, `src/terrain/config.py`): `mode: flat|rough|hurdle|mixed`, `rough_seed: int`, `rough_amplitude_m=0.02`, `rough_length_m=0.4`, `hurdle_x=3.0`, `hurdle_height_m=0.12`, `hurdle_thickness_m=0.08`, `hurdle_width_m=2.0`, `randomize_hurdle_x=True`, `hurdle_x_range=(2.5,5.0)`.
- `CPGParams` (new, `src/cpg/types.py`): `phase, frequency_hz` (1.5-3.0), `thigh_amp, calf_amp`, `phase_offsets[4]` default trot `[0,pi,pi,0]`, `duty_factor=0.5`, `jump_boost` (0 normal, 1 tuck).
- `OscillatorState` (new, `src/cpg/types.py`): `r[4]` amplitudes, `phi[4]` phases, coupling `W[4,4]` trot topology.
- `JumpEvent` (NamedTuple, in `src/go1_env.py`): `apogee_z, clearance_m, hurdle_hit, success` logged to `info_dict`.
- Obs extension: keep 52-dim proprio prefix; append `ext_obs[4]` = `[dist_to_hurdle_x, hurdle_height_norm, trunk_z - hurdle_top, vertical_vel_z]` -> new `obs_dim=56`. Privileged critic vector `priv[8]` = ext_obs + 4 contacts + terrain height (optional asymmetric critic).
## Files
- NEW `mujoco_menagerie/unitree_go1/scene_obstacles.xml`: includes `go1.xml`, flat floor + 3-5 low bump boxes (h 0.02-0.04, x 1.0-2.0) + hurdle box at x~3.0 (bright material, contype/conaffinity=1). Never edit vendored `scene.xml`/`go1.xml` in place.
- NEW `src/terrain/config.py`: `TerrainConfig` + `sample_terrain(rng, stage)` + hurdle pose rewrite via `model.geom(...).pos` on reset.
- NEW `src/cpg/types.py`: `CPGParams`, `OscillatorState`, trot coupling constants.
- NEW `src/cpg/fixed_trot.py`: `FixedTrotCPG` (port `_get_cpg_targets` from dead code; `reset/step/targets_at`).
- NEW `src/cpg/parametric.py`: `ParametricCPG` (freq/amp/offset/duty + `jump_boost` tuck; `params_from_action(mod: Box(6))`).
- NEW `src/cpg/hopf.py`: `HopfCPGNetwork` (4 Hopf oscillators, Euler-integrated, Kuramoto trot coupling + contact feedback).
- NEW `tests/test_cpg.py`, `tests/test_obstacle_env.py` (see Testing).
- MODIFY `src/go1_env.py`: delete dead first class (lines 1-289 + stray quotes + unused `from sympy import euler`); extend effective class with `terrain_config`, `cpg_mode: off|fixed_residual|parametric|hopf`, `residual_scale`, obstacle reset/reward/obs; bump `obs_shape` 52->56 with `include_ext_obs=True` compat flag.
- MODIFY `base_ppo.py`: actor/critic dims + optional two-head actor; `main()` curriculum + CPG-stage schedule.
- MODIFY `src/train.py`: `make_go1_env(curriculum_stage, xml_path, cpg_mode)` passthrough; SB3 hyperparams for 56-dim obs.
- MODIFY `customs_eval_render.py` + `src/eval_render.py`: obstacle scene default, CPG-phase overlay + jump metrics, per-stage GIF naming.
- DELETE none; keep flat scene + old checkpoints for regression.

## Functions
- NEW `FixedTrotCPG.targets_at(phase)->[12]` (`src/cpg/fixed_trot.py`): pure trot (tall stance `[0,0.8,-1.5]` + diagonal sine/cosine + ease-in ramp). NEW `step(dt)->[12]`, `reset()`.
- NEW `ParametricCPG.params_from_action(mod: [-1,1]^6)->CPGParams` (`src/cpg/parametric.py`): maps to `[d_freq, d_thigh, d_calf, d_offset, duty, jump_boost]` clipped to safe ranges.
- NEW `HopfCPGNetwork.step(dt, feedback:[4])->[12]` (`src/cpg/hopf.py`): Hopf amplitude + Kuramoto phase update with foot-contact feedback, then joint mapping.
- MODIFIED `go1_env.__init__ (src/go1_env.py:312)`: add `cpg_mode="off"`, `residual_scale=0.25`, `terrain_config`, `jump_reward_w=2.0`, `hurdle_hit_penalty=5.0`; `obs_shape=56` when ext obs on.
- MODIFIED `go1_env.reset_model (:394)`: resample hurdle x/height, reposition hurdle geom + bumps (seeded `np_random`), reset CPG phase/state, zero `_last_action`.
- MODIFIED `go1_env._get_obs (:371)`: keep 52-dim prefix identical; append `ext_obs[4]`; add `_get_privileged_obs()`.
- MODIFIED `go1_env.step (:415)`: branch on `cpg_mode`: `off` = legacy `home+action*0.25`; `fixed_residual` = `FixedTrot.step + action*residual_scale` (0.05->0.15 schedule); `parametric` = split `[12 residuals + 6 mod]`; `hopf` = mod -> `HopfCPG.step(contact_fb)`. Add jump reward (clearance + airtime + landing stability) + hurdle-contact penalty via `data.contact` geom-id check; extend `info_dict` (`dist_to_hurdle, clearance_m, hurdle_hit, jump_success, cpg_phase, cpg_freq`).
- MODIFIED `go1_env._get_foot_contacts`: keep `>1.0N` threshold; add `check_hurdle_collision()->bool` scanning `data.contact` for hurdle geom id.
- MODIFIED `base_ppo.main (:218)`: Stage 0 flat + `off->fixed_residual`, Stage 1 rough (`scene_obstacles.xml`, `mode=rough`), Stage 2 hurdle (`mode=hurdle`, enable jump weights); set `env.unwrapped.form_weight/cpg_mode/residual_scale` per global step; log jump metrics.
- MODIFIED `make_go1_env (src/train.py:39)`: accept `(curriculum_stage, cpg_mode)`.
- MODIFIED renderers: detect `ActorNetwork(56,18)` vs legacy (52/12) checkpoints, track `trunk` camera, overlay `cpg_phase/dist_to_hurdle`, save `gifs/go1_<stage>_rollout_N.gif`.
## Classes
- NEW `FixedTrotCPG (src/cpg/fixed_trot.py)`: attrs `freq_hz, thigh_amp, calf_amp, stance[12]`; methods `reset/step/targets_at`. No learning; test diagonal antisymmetry (`targets(p+pi)` swaps pairs).
- NEW `ParametricCPG (src/cpg/parametric.py)`: extends fixed mapping with `CPGParams`; `rollout_targets(params, phase)`.
- NEW `HopfCPGNetwork (src/cpg/hopf.py)`: attrs `mu, alpha, omega, W[4,4], dt, k_fb`; `reset/step/targets`; feedback = foot-contact error.
- MODIFIED `go1_env (src/go1_env.py:312)`: add `_cpg, _terrain: TerrainConfig, _residual_scale, _cpg_mode`; same bases (`MujocoEnv, EzPickle`).
- MODIFIED `ActorNetwork (base_ppo.py:25)`: `__init__(state_dim=56, residual_dim=12, mod_dim=6)`; shared trunk `Linear(56,256)->Tanh->Linear(256,256)->Tanh`; heads `residual_mean: Linear(256,12)` + `mod_mean: Linear(256,6)` (+`mod_log_std`); `forward` returns `[residual|mod]` 18-dim (slice 12 in fixed mode). Keep 52/12 loader with zero-pad fallback.
- MODIFIED `CriticNetwork (base_ppo.py:55)`: `state_dim=56 (+8 priv optional)`; add `asymmetric(state_dim, priv_dim)` classmethod.
- MODIFIED `RolloutBuffer (base_ppo.py:72)`: `action_dim` 12->18 (store full, slice per mode).
- MODIFIED `PPOAgent (base_ppo.py:116)`: no structural change; retune `clip 0.2->0.15` jump stage, `entropy 0.01->0.005` post-Stage-0 via `update()` kwargs.

## Dependencies
- No new packages: `gymnasium[mujoco]==1.3.0`, `stable-baselines3[extra]==2.9.0`, `numpy==2.0.2`, `scipy`, `imageio`, `torch 2.13.0` suffice. Remove unused `sympy` import in `go1_env.py`. MuJoCo 3.10 `touch` sensors + `data.contact` lookup for hurdle-hit; CPU-only so keep MLP 256, `n_steps=2048`.

## Testing
- NEW `tests/test_cpg.py`: (1) trot antisymmetry within 1e-6; (2) Hopf 5 s open-loop: `r` bounded, phases lock to `{0,pi}` +-0.1 rad; (3) `params_from_action` clipping to `[freq 1.5-3.0, amps +-50%]`.
- NEW `tests/test_obstacle_env.py`: (1) headless load `scene_obstacles.xml` incl. `hurdle`; (2) 10 resets move hurdle x in range; (3) obs 56 with first-52 prefix match; (4) scripted crouch-extend clears 0.06 m hurdle with no hit; (5) flat `cpg_mode="off"` regression +-5% mean reward over 200 steps.
- MODIFY manual checks: `base_ppo.main()` sanity asserts obs 56 / action 18 in CPG modes; renderers save one GIF per stage and print `jump_success_rate, mean_clearance, hurdle_hit_rate` over 20 eps.
- Gates: Stage 0: 0.5 m/s trot; Stage 1: 5 consecutive bump episodes no fall; Stage 2: >=70% clearance at 0.10 m before 0.15 m.

## Implementation Order
1. Cleanup dead first `go1_env` (lines 1-289) + `sympy` import; verify flat rollout still works.
2. Obstacle world: `scene_obstacles.xml` + `src/terrain/config.py`; headless-load + reposition-on-reset test; keep `scene.xml` for regression.
3. Env obstacle-awareness: `reset_model/_get_obs/step/_get_foot_contacts` + `check_hurdle_collision()`, ext obs 52->56, jump reward/penalty, extended `info_dict`; run flat-regression + randomization checks.
4. Stage-A CPG (fixed trot + residuals): `src/cpg/types.py` + `fixed_trot.py`, wire `cpg_mode="fixed_residual"`; actor dims with 12-dim fallback; train flat trot to parity.
5. Rough curriculum: Stage 1 `mode=rough`, `residual_scale` 0.05->0.15; bump-traversal gate.
6. Jump shaping: Stage 2 `mode=hurdle` + `jump_reward_w/hurdle_hit_penalty` (approach vel + airtime/clearance + landing bonus); scripted-jump smoke then train to >=70% at 0.10 m.
7. Stage-B parametric: `src/cpg/parametric.py`, 18-dim actor `[12+6]`, `RolloutBuffer/main()` + `src/train.py` parity; train rough+hurdle with modulated freq/amp + `jump_boost`.
8. Stage-C Hopf + feedback: `src/cpg/hopf.py` with contact feedback; `cpg_mode="hopf"`; stability + integration tests; tune clip/entropy.
9. Renderers: obstacle scene, CPG overlay, per-stage GIFs + 20-episode hurdle eval (0.12-0.15 m) + flat regression GIF.
