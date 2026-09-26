# deep-rl-and-cpg-ws

Deep-RL + CPG locomotion stack for the **Unitree Go1** quadruped in MuJoCo:
an obstacle-aware Gymnasium environment, a three-stage curriculum
(**flat → rough → hurdle jump**), and hybrid controllers that pair a Central
Pattern Generator (CPG) with a PPO residual policy.

Two runnable training stacks are kept side by side:

| Stack | Entry point | Role | Checkpoints |
| :--- | :--- | :--- | :--- |
| Custom PPO | `base_ppo.py` | **Primary** — CPG stages + obstacle curriculum, hand-written actor/critic/loss | `ppo_checkpoint_*.pth` (repo root) |
| Stable-Baselines3 PPO | `src/train.py` | Secondary baseline — CPG `off`, `VecNormalize` | `src/checkpoints/your_run_name/` |

## Status (verified in this working tree)

Run with the in-repo virtualenv interpreter (`./bin/python`, Python 3.14.4):

| Gate | Command | Result |
| :--- | :--- | :--- |
| CPG unit tests | `./bin/python tests/test_cpg.py` | **6/6 passed** |
| Env / obstacle integration tests | `./bin/python tests/test_obstacle_env.py` | **9/9 passed** |
| SB3 training launch | `./bin/python -m src.train` | starts; detects stale 49-dim `VecNormalize` and refits fresh 56-dim stats |
| Env spec probe | `go1_env` with each `cpg_mode` | obs `(56,)`, action `(12,)` / `(18,)`, `dt=0.01`, `frame_skip=5`, privileged obs `(9,)` |
| Custom PPO launch | `./bin/python base_ppo.py` | sanity check passes (`Obs shape: (56,)`) and resumes from `ppo_checkpoint_latest.pth` (global step 6,000,640) |
| Custom-PPO renderer | `./bin/python customs_eval_render.py` | runs; resolves `ppo_checkpoint_latest.pth` → `actor=(56,12+0)`, `cpg_mode=fixed_residual`, prints hit/success/clearance line |

> `pytest` is **not** installed in the venv, so both test modules ship a
> `__main__` shim and are run as plain scripts (see [Tests](#tests)).

## Repository layout

```
base_ppo.py                    Custom PPO (actor/critic, GAE buffer, curriculum, checkpointing)
customs_eval_render.py         Rollout renderer for the custom PPO checkpoints (GIF + CPG overlay)
src/
  go1_env.py                   go1_env: MuJoCo Gymnasium env (obs/reward/terrain/CPG wiring)
  train.py                     Stable-Baselines3 PPO baseline
  eval_render.py               SB3 evaluation + optional GIF
  terrain/config.py            TerrainConfig: hurdle/bump placement & per-episode randomization
  cpg/
    types.py                   CPGParams, OscillatorState, trot phase offsets & coupling
    fixed_trot.py              FixedTrotCPG — open-loop trot + tall stance
    parametric.py              ParametricCPG — policy-modulated freq/amp/phase/duty/jump_boost
    hopf.py                    HopfCPGNetwork — 4 coupled Hopf oscillators + contact feedback
tests/
  test_cpg.py                  CPG kinematics/stability (6 tests)
  test_obstacle_env.py         Env obs/reward/randomization integration (9 tests)
mujoco_menagerie/              Vendored MuJoCo Menagerie clone (unitree_go1 assets, own licenses)
gifs/                          Renderer output (empty in the committed tree; gitignored)
implementation_plan.md         Design doc: obstacles + staged CPG plan
balance_walk_analysis.md       Root-cause analysis of the balance/walk failures found in the CPG + reward
clean_up.md                    Log of the repo cleanup pass (archive-then-verify)
```

## Install / environment

The repo already contains a virtualenv at its root (`bin/`, `lib/`, `lib64`,
`share/`, `pyvenv.cfg` — all gitignored). Use it, or build an equivalent one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "gymnasium[mujoco]==1.3.0" "stable-baselines3[extra]==2.9.0" \
            numpy scipy imageio torch
```

Versions the current venv resolves to (CPU is used for training; torch is a CUDA build):

| Package | Version |
| :--- | :--- |
| Python | 3.14.4 |
| gymnasium | 1.3.0 |
| mujoco | 3.10.0 |
| stable-baselines3 | 2.9.0 |
| torch | 2.13.0+cu130 |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| imageio | 2.37.3 |
| cv2 (optional) | 5.0.0 — CPG overlay in `customs_eval_render.py` |

**Robot assets.** `mujoco_menagerie/unitree_go1/` supplies `go1.xml` (with the
four `*_Touch` foot sensors the env reads), `scene.xml` (flat world) and
`scene_obstacles.xml` (three low bumps + a hurdle). The vendored scenes are
never edited in place — obstacle geometry is moved at runtime instead.

> ⚠️ **Absolute paths.** `base_ppo.py`, `src/train.py`, `src/eval_render.py` and
> the `xml_file` default of `go1_env.__init__` hardcode
> `/home/plsh/rl_env2/...`. Clone to that path or update those constants.

## The environment — `src/go1_env.py`

`go1_env(xml_file=..., terrain_config=TerrainConfig(...), cpg_mode=..., ...)` extends
`gymnasium.envs.mujoco.MujocoEnv` with `frame_skip=5` on a `0.002 s` MuJoCo step,
so the control step is `dt = 0.01 s` (100 Hz).

### Observation — 56-dim (`include_ext_obs=True`, the default)

| Block | Dim | Content |
| :--- | :--- | :--- |
| Position | 31 | trunk `z` (1; `x,y` dropped via `exclude_current_position_from_observation`), trunk roll/pitch/yaw (3), joint positions (12), gravity vector in body frame (3), previous 12 residual actions (12) |
| Velocity | 18 | trunk linear velocity (3), trunk angular velocity (3), joint velocities (12) |
| Target | 3 | commanded body-frame velocity `[0.6, 0.0, 0.0]` |
| Obstacle extras | 4 | `dist_to_hurdle/5` (clipped to `[-2, 10]`), normalized hurdle height, trunk clearance over hurdle top, vertical velocity `vz` |

* `include_ext_obs=False` restores the **52-dim** legacy observation (its first 52
  entries are byte-identical to the 56-dim prefix — a regression test enforces this).
* `_get_privileged_obs()` returns a **9-dim** extras vector (`ext_obs` + 4 foot
  contacts + trunk `z`) for an asymmetric critic (`CriticNetwork.asymmetric`).

### Action

| `cpg_mode` | Action dim | Meaning |
| :--- | :--- | :--- |
| `off` | 12 | `home_qpos[7:] + action * 0.25` (legacy joint-position control) |
| `fixed_residual` | 12 | open-loop trot targets + `action * residual_scale` |
| `parametric` | 18 | `[12 residual │ 6 mod]` — mod vector drives CPG params |
| `hopf` | 18 | `[12 residual │ 6 mod]` + foot-contact feedback in the oscillator dynamics |

Actions are box-bounded to `[-1, 1]`; the `go1_env` default `residual_scale` is
`0.25` (`base_ppo.py` overrides it with a curriculum schedule, see below).

### Reward

```
reward =  3.0 * velocity_reward                                   # exp(-4*(vx_local - 0.6)^2)
        + 0.001                                                   # survival bonus
        - 0.005 * mean((action12 - last12)^2)                     # action smoothness
        - navigation_penalty                                      # 0.5*vy^2 + 0.3*yaw_rate^2 + 0.5*heading_err^2
        - form_weight * kinematic_penalty
        + 2.0 * jump_bonus                                        # approach + airtime/clearance (hurdle stages)
        + landing_bonus                                           # <= landing_bonus_w (first clean touchdown)
        - 5.0 * hurdle_hit                                        # contact with the hurdle
reward = clip(reward, -10.0, 10.0)
```

where `kinematic_penalty = 2.0*trot_flag + 5.0*max(0, 0.10 - |diag_height_diff|)^2
+ 3.0*(roll^2 + pitch^2) + 0.005*Σaction^2 + 0.0005*Σjoint_vel^2`.

Two shaping switches matter:

* **`form_weight`** is read as `getattr(self, "form_weight", 0.0)` — i.e. **all**
  kinematic penalties are off unless the training loop sets the attribute.
  `base_ppo.py` sets it every step (0.2 → 1.0).
* `hurdle_hit_penalty`, `jump_reward_w` and `landing_bonus_w` are constructor
  arguments (`5.0`, `2.0`, `1.0` by default).

**Episode ends** when the trunk leaves `healthy_z_range = (0.25, 0.45)` (reward
forced to `-2.0`), or on hurdle contact when `hurdle_hit_terminate=True`.
`info` exposes the full breakdown: `velocity_reward`, `local_x/y_velocity`,
`current_height`, `distance_from_origin`, `action_smoothness_penalty`,
`navigation_penalty_total`, `kinematic_penalty_scaled`, `raw_trot_flag`,
`raw_clearance_error`, `raw_posture_penalty`, `active_form_weight`,
`dist_to_hurdle`, `clearance_m`, `hurdle_hit`, `jump_bonus`, `landing_bonus`,
`jump_success`, `cpg_phase`, `cpg_freq`, `cpg_mode`, `jump_boost`, `last_mod`.

## CPG modes — `src/cpg/`

Leg order is always **`[FR, FL, RR, RL]`** (3 joints each: hip, thigh, calf);
the diagonal pairs `(FL, RR)` and `(FR, RL)` are antiphase.

| Mode | Class | Behaviour |
| :--- | :--- | :--- |
| `off` | – | No CPG; legacy direct joint targeting. |
| `fixed_residual` | `FixedTrotCPG` | Analytic trot around a tall stance `[0, 0.8, -1.5]` with a one-cycle ease-in ramp; the policy only adds `residual_scale`-scaled corrections. All four thigh axes are `[0,1,0]`, so swing-forward is a **negative** delta on every leg (`THIGH_DIRS = [-1,-1,-1,-1]`). |
| `parametric` | `ParametricCPG` | Policy emits 6 mod values → `[freq (1.5–3.0 Hz), thigh amp ±50%, calf amp ±50%, phase offset, duty (0.3–0.7), jump_boost (0–1)]`; `jump_boost` blends the trot toward a symmetric crouch-tuck for jumping. |
| `hopf` | `HopfCPGNetwork` | Four Hopf amplitude oscillators (`dr = α(μ − r²)r`) with Kuramoto trot coupling and a per-leg foot-contact phase pull (`k_fb`). Gains can be retuned live with `set_gains()`. |

## Terrain & curriculum — `src/terrain/config.py`

`TerrainConfig` (`mode: flat | rough | hurdle | mixed`) drives per-episode
placement so a single XML covers many episodes:

* **Hurdle** — `hurdle_x` resampled each reset from `hurdle_x_range = (2.5, 5.0)`,
  box geometry (height `0.12 m`, thickness `0.08 m`, width `2.0 m`) written into
  the `hurdle` geom after `reset_model()`.
* **Bumps** — `bump1..bump3` x-centres jittered by `±0.25 m` and heights scaled
  from `(0.5, 1.5)×` the XML base on `rough`/`mixed`/`hurdle`.
* Helpers: `sample_hurdle_x()`, `sample_bump_poses()`, `sample_terrain(rng, stage)`,
  `hurdle_geom_size()`, `hurdle_geom_pos()`.

Curriculum stages used by both trainers:

| Stage | World | Terrain mode |
| :--- | :--- | :--- |
| 0 | `scene.xml` | `flat` |
| 1 | `scene_obstacles.xml` | `rough` (bumps only) |
| 2 | `scene_obstacles.xml` | `hurdle` (bumps + hurdle) |

## Training

### Custom PPO (primary) — `base_ppo.py`

```bash
./bin/python base_ppo.py                 # default CPG mode: fixed_residual
GO1_CPG_MODE=hopf ./bin/python base_ppo.py
```

* Registers the env as `gym id 'Go1Env-v0'` (entry point is the `go1_env` callable).
* Architecture: shared trunk `Linear(state_dim→256) → Tanh → Linear(256→256) → Tanh`,
  then a **residual head** (12) plus, for `parametric`/`hopf`, a **mod head** (6);
  `residual_log_std` / `mod_log_std` are learned diagonal std parameters.
  `ActorNetwork.load_legacy_state_dict()` warm-starts 52/12 checkpoints into a
  56/12+ actor (shared trunk + residual head only).
* Critic: `Linear(state_dim(+priv)→256) → Tanh → 256 → Tanh → 1`;
  `CriticNetwork.asymmetric(...)` supports the 9-dim privileged vector.
* PPO: `RolloutBuffer(2048)`, 5 epochs, minibatch 64, `clip_epsilon=0.2`
  (`0.15` in the jump stage), `value_coef=0.5`, `entropy_coef=0.01`
  (`0.005` from the rough stage), grad-norm clip `0.5`, linearly decayed
  `lr = 1e-4`, `max_training_timesteps = 6_010_000`.
* Curriculum: stage 0 `< 1.5 M` steps (flat), stage 1 `< 3.0 M` (rough),
  stage 2 above (hurdle). On a stage switch the env is rebuilt and reset.
* `form_weight` schedule: `0.2` for the first 1 M steps, ramping
  `0.2 → 1.0` between 1 M and 2 M, `1.0` afterwards.
* Residual authority: `0.05` on flat, `0.15` on rough/hurdle (CPG modes);
  `GO1_RESIDUAL_SCALE` applies only when `CPG_MODE=off`.
  For `hopf`, `k_fb` is `0.1` on flat (near open-loop) and `GO1_CPG_K_FB` otherwise.
* Resume/checkpointing: loads `ppo_checkpoint_latest.pth` if present (with
  legacy shape fallback) and every 10 PPO updates writes both
  `ppo_checkpoint_latest.pth` and `ppo_checkpoint_<global_step>.pth` in the repo
  root, storing actor/critic/optimizers, `global_step`, `episode_count`,
  `cpg_mode`, `state_dim`, `action_dim`.
* A 200-step random-policy sanity check runs before training and asserts the
  observation shape matches the actor input.

> The root `ppo_checkpoint_*.pth` files were trained against the pre-fix CPG
> (inverted thigh signs) — see `balance_walk_analysis.md` before resuming from
> them; the recommended path is a fresh Stage-0 run.

### Stable-Baselines3 PPO (secondary) — `src/train.py`

```bash
./bin/python -m src.train                # run as a module from the repo root
GO1_STAGE=2 ./bin/python -m src.train
```

`src/train.py` executes at import time (no `main()` guard), so it must be run as
a module from the repo root — `./bin/python src/train.py` cannot resolve
`from src.go1_env import go1_env`. It builds a `VecNormalize`-wrapped
`make_vec_env(make_go1_env, n_envs=4)` and trains `PPO("MlpPolicy")` for
2,500,000 steps (`lr=1e-4`, `n_steps=512`, `batch_size=64`, `n_epochs=10`,
`gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.0`, `vf_coef=0.5`,
`max_grad_norm=0.5`). `CheckpointCallback(save_freq=15000)` with `n_envs=4`
produces `rl_model_<env_steps>_steps.zip` every 60,000 env steps; a custom
callback saves `latest_vecnormalize.pkl` alongside, and `final` /
`final_vecnormalize.pkl` are written at the end. Logs go to
`src/logs/<RUN_NAME>/` for TensorBoard (`./bin/tensorboard --logdir src/logs`).

Note `RUN_NAME = "your_run_name"` and the world paths are module constants —
edit them (or export the env vars below) before a real run. The CPG path is
intentionally not used by this trainer (`GO1_CPG_MODE` defaults to `off`).

### Environment-variable reference

| Var | Used by | Default | Effect |
| :--- | :--- | :--- | :--- |
| `GO1_CPG_MODE` | `base_ppo.py` | `fixed_residual` | `off` \| `fixed_residual` \| `parametric` \| `hopf` |
| `GO1_RESIDUAL_SCALE` | `base_ppo.py` | `0.10` | Residual scale, **only** when `CPG_MODE=off` |
| `GO1_CPG_K_FB` | `base_ppo.py` | `0.5` | Hopf foot-contact feedback gain (stages ≥ 1) |
| `GO1_CPG_COUPLING` | `base_ppo.py` | `2.0` | Hopf Kuramoto coupling strength |
| `GO1_LANDING_BONUS_W` | `base_ppo.py` | `1.0` | Weight of the landing-stability bonus |
| `GO1_HURDLE_HIT_TERMINATE` | `base_ppo.py` | `0` | `1` ends the episode on hurdle contact |
| `GO1_STAGE0_END` | `base_ppo.py` | `1500000` | Step where flat → rough |
| `GO1_STAGE1_END` | `base_ppo.py` | `3000000` | Step where rough → hurdle |
| `GO1_STAGE` | `src/train.py` | `0` | Curriculum stage 0/1/2 |
| `GO1_CPG_MODE` | `src/train.py` | `off` | SB3 path stays CPG-off by default |
| `GO1_FRESH_VECNORM` | `src/train.py` | `0` | `1` forces fresh `VecNormalize` stats (required for 56-dim runs) |
| `GO1_EVAL_STAGE` | `src/eval_render.py` | `flat` | `flat` \| `hurdle` |
| `GO1_EVAL_XML` | `src/eval_render.py` | derived | Overrides the evaluation XML |
| `GO1_SAVE_GIF` | `src/eval_render.py` | `"0"` | ⚠️ inverted flag: unset/`"0"` **enables** GIF saving, any other value disables it |
| `GO1_EVAL_STAGE` | `customs_eval_render.py` | `hurdle` | `flat` uses `scene.xml`, else `scene_obstacles.xml` |

## Evaluation & rendering

Both renderers run the env with `render_mode="human"`, so they need a display
(`DISPLAY` is set in this workspace; MuJoCo opens its own window). Note that in
`human` mode Gymnasium's MuJoCo `render()` returns `None`, so *no frames are
captured* and no GIF is written — switch the env to `render_mode="rgb_array"`
if you actually want the GIF.

```bash
# SB3 policy (expects rl_model_120000_steps.zip + latest_vecnormalize.pkl)
./bin/python -m src.eval_render           # GO1_EVAL_STAGE=flat|hurdle
```

`src/eval_render.py` disables normalization at eval time
(`VecNormalize.training = False`, `norm_reward = False`), tracks the `trunk`
body with the camera, runs up to 1000 steps / 20 episodes and prints
`hit_rate`, `success_rate`, `mean_clearance`, `mean_landing_bonus`
(plus `output_<stage>.gif` when GIF saving is on).

> ⚠️ **This command currently aborts on the shipped artifacts**: the checked-in
> SB3 `VecNormalize` stats are 49-dim while the env is 56-dim, so it raises
> `AssertionError: spaces must have the same shape: (49,) != (56,)`. Train a
> fresh 56-dim SB3 run first (`./bin/python -m src.train`) or repoint
> `CHECKPOINT_DIR` at new artifacts.

```bash
# Custom PPO policy (resolves ppo_checkpoint_latest.pth, infers dims)
./bin/python customs_eval_render.py       # GO1_EVAL_STAGE=hurdle (default)
```

`customs_eval_render.py` resolves `ppo_checkpoint_latest.pth` (falling back to
`ppo_checkpoint_6000640.pth` and `checkpoints/`), rebuilds `ActorNetwork` from
the checkpoint's stored `state_dim`/`action_dim` (legacy 52/12 checkpoints
warm-start too), reads the checkpoint's own `cpg_mode`, renders up to 2000 steps
/ 20 episodes, draws a per-leg swing-bar + `phase/boost/dist/clear` overlay when
`cv2` is available (cv2 5.0.0 is installed here), and writes
`gifs/go1_<stage>_<cpg_mode>_rollout.gif` at 30 fps when frames are captured.
A verified run prints e.g. `Episodes: 1 | jump_eps: 1 hit_rate=0.00
success_rate=0.00 mean_clearance=0.184m mean_landing_bonus=0.000` followed by
`No frames were captured; GIF was not saved.` (the `human`-mode caveat above).

## Tests

`pytest` is not installed in the shipped venv, so the modules have `__main__`
shims. Run them from the repo root with the venv interpreter:

```bash
./bin/python tests/test_cpg.py            # -> 6/6 CPG tests passed
./bin/python tests/test_obstacle_env.py   # -> 9/9 obstacle-env tests passed
```

`tests/test_cpg.py` covers trot antisymmetry (diagonals swap after half a
cycle), swing-leg thigh sign regression, `params_from_action` clipping to the
safe CPG ranges, 5 s open-loop Hopf stability (oscillation + bounded amplitudes
+ phase locking), feedback-advances-phase, and live gain retuning.

`tests/test_obstacle_env.py` covers the 56-dim default obs with neutral extras,
the 52-dim compat flag, per-episode hurdle-x and bump randomization, jump
`info` keys, prefix-preservation between the 52/56 obs, `landing_bonus == 0` on
flat, the `hurdle_hit_terminate` branch, and a 200-step flat regression with
finite rewards. If you install pytest, `python3 -m pytest tests -q` works as well.

## Artifacts & file conventions

| Path | Produced by | Contents |
| :--- | :--- | :--- |
| `ppo_checkpoint_latest.pth`, `ppo_checkpoint_<step>.pth` | `base_ppo.py` | Custom PPO actor/critic/optimizer state + metadata |
| `src/checkpoints/your_run_name/` | `src/train.py` | SB3 `rl_model_*_steps.zip`, `latest_vecnormalize.pkl`, `final*` |
| `src/logs/your_run_name/` | `src/train.py` | TensorBoard event files |
| `gifs/` | renderers | `<stage>` rollout GIFs (empty in git) |

Everything above is gitignored (`*.pth`, `*.zip`, `*.pkl`, `src/logs/`, `gifs/`,
plus the venv and caches), so the tracked tree is only code, tests and docs.
`git ls-files` lists `.gitignore`, `README.md`, `base_ppo.py`,
`customs_eval_render.py`, `implementation_plan.md`, `src/**` and `tests/**`;
`balance_walk_analysis.md`, `clean_up.md` and the vendored `mujoco_menagerie/`
are currently untracked.

## Known limitations & gotchas

1. **Hardcoded absolute paths** — `base_ppo.py`, `src/train.py`,
   `src/eval_render.py` and the `go1_env` default `xml_file` assume
   `/home/plsh/rl_env2`.
2. **`form_weight` defaults to 0.0** — outside `base_ppo.py` the kinematic
   penalties (trot rhythm, clearance, posture, effort) are *silently disabled*.
   Set `env.unwrapped.form_weight` if you write your own training loop.
3. **Stale SB3 artifacts** — the shipped `src/checkpoints/your_run_name/` files
   are pre-obstacle (49-dim `VecNormalize` stats, 52-dim policy). `src/train.py`
   catches the mismatch and refits fresh 56-dim stats, dropping the old
   checkpoint (`GO1_FRESH_VECNORM=1` makes that explicit), but
   `src/eval_render.py` loads them directly and therefore **aborts** with
   `spaces must have the same shape: (49,) != (56,)` until a fresh SB3 run exists.
   The custom-PPO renderer is unaffected — verified working (`customs_eval_render.py`
   resolves the root checkpoint and infers 56/12+0 dims automatically).
4. **Pre-fix root checkpoints** — the `ppo_checkpoint_*.pth` files were trained
   with the inverted `THIGH_DIRS` CPG; see `balance_walk_analysis.md`.
5. **`pytest` missing** from the venv → use the script shims above.
6. **`GO1_SAVE_GIF` is inverted** in `src/eval_render.py` (comparison against
   `"0"`), and GIF capture slows rendering substantially.
7. **Renderers need a display** and, because they use `render_mode="human"`,
   `env.render()` returns `None` — so `customs_eval_render.py` prints
   "No frames were captured" instead of writing a GIF. Use `rgb_array` (with a
   display or EGL/OSMesa) to capture frames.
8. **Hopf/parametric modes have unit-test coverage only** — no trained
   policy is checked in for them.
9. `mujoco_menagerie/` is a vendored clone (large, untracked) — never edit the
   vendored `scene.xml` / `go1.xml`; add new scenes or move geoms at runtime.

## Further reading

| Doc | Contents |
| :--- | :--- |
| `implementation_plan.md` | Obstacles + staged CPG design, types, files, test gates and implementation order |
| `balance_walk_analysis.md` | Root-cause analysis of the five compounding reward/CPG bugs and the remediation plan |

## Credits & license

Robot model and scenes come from the vendored
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`unitree_go1` package (BSD-3-Clause, © Unitree Robotics — see
`mujoco_menagerie/unitree_go1/LICENSE`). Training uses
[Gymnasium](https://gymnasium.farama.org/), [MuJoCo](https://mujoco.org/),
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and
[PyTorch](https://pytorch.org/). No license file is present for the project code
itself.
