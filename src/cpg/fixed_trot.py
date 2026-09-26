"""Fixed open-loop trot generator (Stage-A CPG).

Ported from the analytic prototype that lived in the removed dead
go1_env class: tall stance [hip=0, thigh=0.8, calf=-1.5] + diagonal
sine/cosine wave with ease-in ramp. The RL policy only adds small
residual joint corrections on top.
Leg order: [FR, FL, RR, RL], 3 joints each (hip, thigh, calf).
"""
import numpy as np

from .types import CPGParams, TROT_PHASE_OFFSETS


TALL_STANCE = np.array([
    0.0, 0.8, -1.5,  # FR
    0.0, 0.8, -1.5,  # FL
    0.0, 0.8, -1.5,  # RR
    0.0, 0.8, -1.5,  # RL
], dtype=np.float64)

# All four thigh joints share axis [0,1,0]; empirically +angle sweeps every
# foot backward (-x), so swing-forward needs a NEGATIVE delta on ALL legs.
# (A previous [-1,+1,+1,-1] mix made FL/RR drag backward during swing.)
THIGH_DIRS = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)


class FixedTrotCPG:
    def __init__(self, freq_hz=2.0, thigh_amp=0.25, calf_amp=-0.35,
                 stance=None):
        self.params = CPGParams(frequency_hz=float(freq_hz),
                                thigh_amp=float(thigh_amp),
                                calf_amp=float(calf_amp))
        self.stance = (TALL_STANCE.copy() if stance is None
                       else np.asarray(stance, dtype=np.float64))
        self.phase = 0.0

    def reset(self, phase=0.0):
        self.phase = float(phase)
        return self.targets_at(self.phase)

    def step(self, dt):
        self.phase += 2.0 * np.pi * self.params.frequency_hz * dt
        return self.targets_at(self.phase)

    def targets_at(self, phase):
        """Pure trot trajectory at absolute phase (radians)."""
        p = self.params
        phase_1 = phase
        phase_2 = phase + np.pi
        thigh_1 = p.thigh_amp * np.sin(phase_1)
        thigh_2 = p.thigh_amp * np.sin(phase_2)
        calf_1 = p.calf_amp * np.clip(np.cos(phase_1), 0.0, 1.0)
        calf_2 = p.calf_amp * np.clip(np.cos(phase_2), 0.0, 1.0)
        # Diagonal pairing: (FL,RR) on phase_1, (FR,RL) on phase_2.
        thigh = np.array([thigh_2, thigh_1, thigh_1, thigh_2])
        calf = np.array([calf_2, calf_1, calf_1, calf_2])
        offsets = np.zeros(12)
        offsets[1::3] = thigh * THIGH_DIRS
        offsets[2::3] = calf
        ramp = min(1.0, phase / (2.0 * np.pi)) if phase > 0 else 0.0
        return self.stance + offsets * ramp
