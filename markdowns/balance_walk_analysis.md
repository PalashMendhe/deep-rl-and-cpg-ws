# Unitree Go1 Locomotion Analysis: Why the Bot Fails to Balance & Walk

## Executive Summary

An in-depth empirical and kinematic investigation of the codebase (`/home/plsh/rl_env2`) reveals **5 compounding bugs** across the Central Pattern Generator (CPG), the Gymnasium environment reward formulation, and the curriculum training schedule. 

Together, these flaws create contradictory physical objectives and destroy the learning signal:
1. **The CPG actively commands half the legs to drag backward during swing**, creating an asymmetric, self-braking gait.
2. **The environment never rewards staying upright** (`timestep_reward` is defined but omitted in `step()`), leaving falling/sliding forward as the only way to gain positive reward.
3. **The curriculum zeros out all kinematic penalties for the first 1,000,000 steps** (`form_weight = 0.0`), allowing the policy to entrench degenerate sliding/falling habits.
4. **The clearance penalty punishes the robot even when standing completely still**, penalizing neutral support.
5. **The forward velocity reward is unbounded upward**, rewarding diving or falling forward at high speed over controlled trotting.

---

## Root Cause Deep-Dive

### 🔴 Bug 1 (Critical): `THIGH_DIRS` Kinematic Sign Inversion in `fixed_trot.py`

* **File:** [`src/cpg/fixed_trot.py`](file:///home/plsh/rl_env2/src/cpg/fixed_trot.py#L22) (also consumed by [`src/cpg/parametric.py`](file:///home/plsh/rl_env2/src/cpg/parametric.py#L10) and [`src/cpg/hopf.py`](file:///home/plsh/rl_env2/src/cpg/hopf.py))

#### The Problem
In `fixed_trot.py`, the thigh direction multipliers are defined as:
```python
# CURRENT CODE (BUGGY)
THIGH_DIRS = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)  # Leg order: [FR, FL, RR, RL]
```

The accompanying code comment states:
```python
# Axis inversions so thigh sweep drives forward motion, not moonwalk.
```

However, empirical inspection of the MuJoCo model (`unitree_go1/scene.xml`) confirms that **all four thigh joints share identical rotation axes and kinematic orientations**:
* `FR_thigh_joint`: axis = `[0, 1, 0]`
* `FL_thigh_joint`: axis = `[0, 1, 0]`
* `RR_thigh_joint`: axis = `[0, 1, 0]`
* `RL_thigh_joint`: axis = `[0, 1, 0]`

#### Empirical Kinematic Verification
Perturbing the thigh joints by $+0.5\text{ rad}$ and measuring foot site positions in world Cartesian coordinates:
* **FR foot displacement:** $\Delta x = -0.127\text{ m}$ (foot moves **backward**)
* **FL foot displacement:** $\Delta x = -0.127\text{ m}$ (foot moves **backward**)
* **RR foot displacement:** $\Delta x = -0.127\text{ m}$ (foot moves **backward**)
* **RL foot displacement:** $\Delta x = -0.127\text{ m}$ (foot moves **backward**)

For **all 4 legs**, increasing the thigh angle ($+$) sweeps the foot **backward**, while decreasing the thigh angle ($-$) swings the foot **forward**.

#### Impact on Gait
During trot swing phase ($\sin(\phi) > 0$):
$$\Delta \theta_{\text{thigh}} = \text{thigh\_amp} \times \sin(\phi) \times \text{THIGH\_DIR}$$

| Leg | Phase | `THIGH_DIR` | $\Delta \theta_{\text{thigh}}$ | Foot Motion | Intended Motion | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FR** | `phase_2` | **-1.0** | Negative ($-$) | **Forward** ($+x$) | Forward Swing | ✅ **Correct** |
| **FL** | `phase_1` | **+1.0** | Positive ($+$) | **Backward** ($-x$) | Forward Swing | ❌ **DRAGGING BACKWARD** |
| **RR** | `phase_1` | **+1.0** | Positive ($+$) | **Backward** ($-x$) | Forward Swing | ❌ **DRAGGING BACKWARD** |
| **RL** | `phase_2` | **-1.0** | Negative ($-$) | **Forward** ($+x$) | Forward Swing | ✅ **Correct** |

When diagonal pair 1 (FL + RR) swings, both legs are driven backward into the ground/stance, opposing forward progress. When diagonal pair 2 (FR + RL) swings, they swing forward. The robot experiences severe yaw torques, internal mechanical fighting, and cannot balance or propel itself.

#### Solution
All legs must have the same negative sign:
```python
# CORRECTED
THIGH_DIRS = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
```

---

### 🔴 Bug 2 (Critical): `timestep_reward` Omitted from Reward Formula

* **File:** [`src/go1_env.py`](file:///home/plsh/rl_env2/src/go1_env.py#L36) & [`src/go1_env.py`](file:///home/plsh/rl_env2/src/go1_env.py#L473-L482)

#### The Problem
In `go1_env.__init__`:
```python
timestep_reward = 0.001,  # Reward the bot for each timestep it is healthy.
```
However, in `go1_env.step()`:
```python
reward = (
    self._forward_reward * velocity_reward
    - self._contact_penalty * action_smoothness_penalty
    - navigation_penalty
    - scaled_kinematic_penalty
    + self._jump_reward_w * jump_bonus
    + landing_bonus
    - hurdle_penalty
)
```
`self._timestep_reward` is completely omitted from the summation.

#### Impact
There is **zero positive reward for simply standing, balancing, and surviving**. The only positive term available during standard locomotion is `forward_reward * velocity_reward`.
Because standing upright yields $\approx 0$ reward, while pitching forward and collapsing delivers positive forward COM displacement for several steps, the agent actively learns that **diving forward onto its chest/knees is more rewarding than attempting to stay upright**.

#### Solution
Include a meaningful survival bonus in `step()`:
```python
reward = (
    self._forward_reward * velocity_reward
    + self._timestep_reward
    - self._contact_penalty * action_smoothness_penalty
    - navigation_penalty
    - scaled_kinematic_penalty
    + self._jump_reward_w * jump_bonus
    + landing_bonus
    - hurdle_penalty
)
```
*(Recommended value: increase `timestep_reward` from `0.001` to `0.01` or `0.02` per step to give a consistent baseline survival gradient).*

---

### 🟠 Bug 3 (High): Curriculum Schedule Multiplies Kinematics by Zero for 1M Steps

* **File:** [`base_ppo.py`](file:///home/plsh/rl_env2/base_ppo.py#L425-L436)

#### The Problem
In `base_ppo.py`:
```python
if global_step < 1000000:
    form_weight = 0.0
elif global_step > 2000000:
    form_weight = 1.0
else:
    form_weight = (global_step - 1000000) / 1000000.0

env.unwrapped.form_weight = form_weight
```

And in `go1_env.py`:
```python
scaled_kinematic_penalty = raw_kinematic_penalty * current_form_weight
```

`raw_kinematic_penalty` contains:
* `2.0 * trot_penalty_flag` (punishing invalid trotting rhythm)
* `clearance_penalty` (foot lift requirements)
* `posture_penalty = 3.0 * (roll**2 + pitch**2)` (upright torso stability)
* `effort_penalty`
* `joint_vel_penalty`

#### Impact
For the first **1,000,000 environment steps** ($\approx 488$ policy updates):
* Roll and pitch are completely unpenalized.
* Trotting rhythm is completely unpenalized.
* Joint velocities and action jitter are unpenalized.

The policy spends 1M steps learning an uncontrolled shuffle or roll. By the time `form_weight` starts ramping up at step 1,000,000, the policy's action distribution has already converged to a high-entropy/degenerate local optimum that cannot easily transition to a rhythmic trot.

#### Solution
Start with a non-zero baseline `form_weight` (e.g. $0.15 - 0.20$) from step 0 so the agent never discovers that falling or tilting is penalty-free:
```python
min_form = 0.2
if global_step < 1000000:
    form_weight = min_form
elif global_step > 2000000:
    form_weight = 1.0
else:
    form_weight = min_form + (1.0 - min_form) * ((global_step - 1000000) / 1000000.0)
```

---

### 🟡 Bug 4 (Medium): `clearance_penalty` Punishes Stationary Upright Stance

* **File:** [`src/go1_env.py`](file:///home/plsh/rl_env2/src/go1_env.py#L405-L411)

#### The Problem
In `go1_env.py`:
```python
diagonal_1_height = fl_z + rr_z
diagonal_2_height = fr_z + rl_z

target_height_diff = 0.10
actual_height_diff = abs(diagonal_1_height - diagonal_2_height)
clearance_penalty = 10.0 * ((actual_height_diff - target_height_diff) ** 2)
```

When the robot is resting or standing in nominal tall stance with all 4 feet in ground contact:
$$fl_z \approx fr_z \approx rl_z \approx rr_z \approx 0.0$$
$$\text{actual\_height\_diff} = |0 - 0| = 0.0$$
$$\text{clearance\_penalty} = 10.0 \times (0.0 - 0.10)^2 = 10.0 \times 0.01 = \mathbf{0.10\text{ per step}}$$

#### Impact
Even when `form_weight > 0`, standing still and stable incurs an ongoing penalty. Furthermore, summing $z$ coordinates across diagonals ($fl_z + rr_z$) allows degenerate geometries (e.g., $FL = 0.10$, $RR = -0.05 \implies \text{sum} = 0.05$) to satisfy or violate the objective unpredictably.

#### Solution
Only penalize clearance during swing phases, or reward actual swing height rather than applying a quadratic penalty around an exact target diff:
```python
# Better: Reward positive clearance of swing feet rather than penalizing all-grounded stance
swing_clearance = max(0.0, actual_height_diff)
clearance_penalty = 5.0 * max(0.0, target_height_diff - actual_height_diff) ** 2
```

---

### 🟡 Bug 5 (Medium): `velocity_reward` is Unbounded Linear & Rewards Collapsing

* **File:** [`src/go1_env.py`](file:///home/plsh/rl_env2/src/go1_env.py#L381)

#### The Problem
```python
velocity_reward = local_x_vel / (self._target_velocity[0] + 1e-8)
```
With target velocity $v_x^* = 0.6\text{ m/s}$ and `_forward_reward = 3.0`:
* If the robot falls forward rapidly at $1.8\text{ m/s}$, `velocity_reward = 3.0`, giving a huge $+9.0$ reward contribution.
* Conversely, any backward motion yields an unbounded negative penalty.

#### Impact
This linear formulation creates strong gradient pressure to maximize instantaneous forward velocity regardless of gait stability, encouraging dynamic lunging, tipping, or falling over controlled, balanced stepping.

#### Solution
Use a target velocity tracking kernel (Gaussian/exponential or capped clip):
```python
# Gaussian tracking kernel centered around target velocity:
vel_err = local_x_vel - self._target_velocity[0]
velocity_reward = np.exp(-4.0 * (vel_err ** 2))
```
Or a clipped linear reward:
```python
velocity_reward = np.clip(local_x_vel / (self._target_velocity[0] + 1e-8), -0.5, 1.2)
```

---

## Code Fix Summary Table

| Issue | File & Location | Current Line | Fixed Line |
| :--- | :--- | :--- | :--- |
| **Bug 1: Thigh sign** | [`src/cpg/fixed_trot.py#L22`](file:///home/plsh/rl_env2/src/cpg/fixed_trot.py#L22) | `[-1.0, 1.0, 1.0, -1.0]` | `[-1.0, -1.0, -1.0, -1.0]` |
| **Bug 2: Survival bonus** | [`src/go1_env.py#L473-L481`](file:///home/plsh/rl_env2/src/go1_env.py#L473-L481) | *Omitted from reward sum* | `+ self._timestep_reward` added |
| **Bug 3: Zero form weight**| [`base_ppo.py#L425-L430`](file:///home/plsh/rl_env2/base_ppo.py#L425-L430) | `form_weight = 0.0` | `form_weight = 0.2` (minimum baseline) |
| **Bug 4: Clearance penalty**| [`src/go1_env.py#L410`](file:///home/plsh/rl_env2/src/go1_env.py#L410) | `10.0 * ((diff - 0.10)**2)` | `5.0 * max(0.0, 0.10 - diff)**2` |
| **Bug 5: Velocity reward** | [`src/go1_env.py#L381`](file:///home/plsh/rl_env2/src/go1_env.py#L381) | `local_x_vel / v_target` | `np.exp(-4.0 * (local_x_vel - v_target)**2)` |

---

## Action Plan for Remediation & Retraining

1. **Apply the Code Fixes**:
   * Update `THIGH_DIRS` in `src/cpg/fixed_trot.py`.
   * Update `reward` calculation and `clearance_penalty` in `src/go1_env.py`.
   * Update `form_weight` schedule in `base_ppo.py`.
2. **Handle Existing Checkpoint Compatibility**:
   * Existing checkpoints (e.g., `ppo_checkpoint_latest.pth` at step 3,051,520) were trained against the inverted CPG. The policy learned residuals to fight the broken CPG rather than complement a healthy trot.
   * **Recommendation:** Restart Stage 0 training from scratch (`global_step = 0`) on flat ground, or warm-start only the trunk while zeroing the residual policy output layer (`actor.residual_mean.weight.data.zero_()`, `actor.residual_mean.bias.data.zero_()`).
3. **Verify with Visual Rendering**:
   * Run `customs_eval_render.py` or `tests/test_obstacle_env.py` to visually confirm that all 4 feet swing in the forward direction during their respective swing phases.
