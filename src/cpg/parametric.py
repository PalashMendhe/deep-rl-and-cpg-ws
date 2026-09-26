"""Parametric CPG (Stage-B): policy modulates trot parameters.

The policy outputs a 6-dim mod vector in [-1,1]:
  [d_freq, d_thigh_amp, d_calf_amp, d_phase_offset, duty, jump_boost]
mapped to safe CPG ranges. jump_boost blends the trot toward a
symmetric crouch-tuck (all thighs flex, calves fold) used for jump.
"""
import numpy as np

from .fixed_trot import FixedTrotCPG, TALL_STANCE, THIGH_DIRS
from .types import CPGParams

FREQ_RANGE = (1.5, 3.0)
DUTY_RANGE = (0.3, 0.7)

# Symmetric tuck offsets added proportional to jump_boost.
TUCK_OFFSETS = np.array([
    0.0, 0.45, -0.55,
    0.0, 0.45, -0.55,
    0.0, 0.45, -0.55,
    0.0, 0.45, -0.55,
], dtype=np.float64)


class ParametricCPG(FixedTrotCPG):
    def params_from_action(self, mod):
        mod = np.clip(np.asarray(mod, dtype=np.float64).flatten(), -1.0, 1.0)
        assert mod.shape == (6,), mod.shape
        base = self.params
        freq = float(np.clip(2.0 + mod[0] * 0.5, *FREQ_RANGE))
        thigh = float(base.thigh_amp * (1.0 + mod[1] * 0.5))
        calf = float(base.calf_amp * (1.0 + mod[2] * 0.5))
        offsets = base.phase_offsets + mod[3] * 0.3
        duty = float(np.clip(0.5 + mod[4] * 0.2, *DUTY_RANGE))
        boost = float(np.clip((mod[5] + 1.0) / 2.0, 0.0, 1.0))
        return CPGParams(phase=self.phase, frequency_hz=freq,
                         thigh_amp=thigh, calf_amp=calf,
                         phase_offsets=offsets, duty_factor=duty,
                         jump_boost=boost)

    def rollout_targets(self, params, phase):
        """Trot at phase under params, blended toward tuck by jump_boost."""
        saved_params, saved_phase = self.params, self.phase
        self.params = params
        # Temporarily reuse targets_at with per-leg phase offsets.
        trot = self._targets_with_offsets(params, phase)
        out = (1.0 - params.jump_boost) * trot + params.jump_boost * (
            TALL_STANCE + TUCK_OFFSETS)
        self.params, self.phase = saved_params, saved_phase
        return out

    def _targets_with_offsets(self, params, phase, ramp_override=None):
        thigh = params.thigh_amp * np.sin(phase + params.phase_offsets)
        calf = params.calf_amp * np.clip(
            np.cos(phase + params.phase_offsets), 0.0, 1.0)
        offsets = np.zeros(12)
        offsets[1::3] = thigh * THIGH_DIRS
        offsets[2::3] = calf
        if ramp_override is not None:
            ramp = float(ramp_override)
        else:
            ramp = min(1.0, phase / (2.0 * np.pi)) if phase > 0 else 0.0
        return self.stance + offsets * ramp
