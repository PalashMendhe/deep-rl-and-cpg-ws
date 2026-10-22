"""Fixed open-loop trot generator — Flat-Body Trot (feature/flat-body-trot).

Generates a biological trot gait where the trunk stays flat and parallel
to the floor at all times. Only the leg joints move; the body does NOT
bob or pitch with each stride.

Key design decisions vs. the old dog-like gait:
  - FLAT_STANCE: slightly crouched (thigh=0.7, calf=-1.4) so the CoM is
    lower and more stable — the body has less vertical travel per step.
  - thigh_amp reduced (0.25 → 0.15): smaller thigh swing → less body pitch.
  - calf_amp reduced (-0.35 → -0.20): lighter calf retraction → smoother
    foot clearance without pulling the body up.
  - Hip joints get a small abduction drive (hip_amp=0.08 rad): lateral hip
    oscillation is the biological hallmark of a real trot; keeps the foot
    path wide enough for clearance without tilting the trunk.

Leg order: [FR, FL, RR, RL], 3 joints each (hip, thigh, calf).
"""
import numpy as np

from .types import CPGParams, TROT_PHASE_OFFSETS


# Flat-body stance: crouched but level. Compared to TALL_STANCE
# (thigh=0.8, calf=-1.5), the knees are slightly more bent and the
# calf slightly less folded so the trunk sits stably at ~0.30 m.
FLAT_STANCE = np.array([
    0.0, 0.7, -1.4,  # FR
    0.0, 0.7, -1.4,  # FL
    0.0, 0.7, -1.4,  # RR
    0.0, 0.7, -1.4,  # RL
], dtype=np.float64)

# Keep TALL_STANCE as an alias for any legacy code that still imports it.
TALL_STANCE = FLAT_STANCE

# Thigh joints: +angle sweeps every foot backward (-x), so
# swing-forward needs a NEGATIVE delta on ALL legs.
THIGH_DIRS = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)

# Hip joints: lateral abduction during swing keeps feet wide and
# prevents body sway. Diagonal pairs are antiphase:
#   FR/RL abduct while FL/RR adduct, then swap.
# Sign convention: +hip = abduction (outward) on all Go1 hip joints.
HIP_DIRS = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)


class FixedTrotCPG:
    """Open-loop trot CPG for flat-body gait.

    Args:
        freq_hz:   Trot frequency in Hz (default 2.0).
        thigh_amp: Thigh sine amplitude in radians (default 0.15).
        calf_amp:  Calf retraction amplitude in radians (default -0.20).
        hip_amp:   Hip lateral abduction amplitude in radians (default 0.08).
        stance:    12-dim home joint angles (defaults to FLAT_STANCE).
    """
    def __init__(self, freq_hz=2.0, thigh_amp=0.15, calf_amp=-0.20,
                 hip_amp=0.08, stance=None):
        self.params = CPGParams(
            frequency_hz=float(freq_hz),
            thigh_amp=float(thigh_amp),
            calf_amp=float(calf_amp),
            hip_amp=float(hip_amp),
        )
        self.stance = (FLAT_STANCE.copy() if stance is None
                       else np.asarray(stance, dtype=np.float64))
        self.phase = 0.0

    def reset(self, phase=0.0):
        self.phase = float(phase)
        return self.targets_at(self.phase)

    def step(self, dt):
        self.phase += 2.0 * np.pi * self.params.frequency_hz * dt
        return self.targets_at(self.phase)

    def targets_at(self, phase):
        """Flat-body trot joint targets at absolute phase (radians).

        Diagonal pair (FL, RR) is on phase_1; (FR, RL) is on phase_2 = phase_1 + π.
        Each leg:
          hip   — cosine-based lateral abduction (small, keeps foot wide in swing)
          thigh — sine-based fore-aft swing      (primary propulsion)
          calf  — half-cosine retraction         (foot clearance during swing only)
        """
        p = self.params
        phase_1 = phase          # FL, RR
        phase_2 = phase + np.pi  # FR, RL

        # --- Thigh: sine drive ---
        thigh_1 = p.thigh_amp * np.sin(phase_1)
        thigh_2 = p.thigh_amp * np.sin(phase_2)
        # [FR, FL, RR, RL]
        thigh = np.array([thigh_2, thigh_1, thigh_1, thigh_2])

        # --- Calf: retract only during swing (positive cosine half) ---
        calf_1 = p.calf_amp * np.clip(np.cos(phase_1), 0.0, 1.0)
        calf_2 = p.calf_amp * np.clip(np.cos(phase_2), 0.0, 1.0)
        # [FR, FL, RR, RL]
        calf = np.array([calf_2, calf_1, calf_1, calf_2])

        # --- Hip: small lateral abduction, cosine-based ---
        # Abduct during swing (positive cosine), adduct during stance.
        hip_1 = p.hip_amp * np.clip(np.cos(phase_1), 0.0, 1.0)
        hip_2 = p.hip_amp * np.clip(np.cos(phase_2), 0.0, 1.0)
        # [FR, FL, RR, RL] — HIP_DIRS flips sign for proper leg geometry
        hip = np.array([hip_2, hip_1, hip_1, hip_2])

        # Build joint offset vector [hip, thigh, calf] per leg × 4
        offsets = np.zeros(12)
        offsets[0::3] = hip * HIP_DIRS    # hip joints
        offsets[1::3] = thigh * THIGH_DIRS  # thigh joints
        offsets[2::3] = calf               # calf joints

        # Ease-in ramp: linearly scale offsets up over the first full cycle
        # to prevent a jerk from the initial zero-velocity state.
        ramp = min(1.0, phase / (2.0 * np.pi)) if phase > 0 else 0.0
        return self.stance + offsets * ramp

