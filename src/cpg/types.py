"""CPG shared types: parameters, oscillator state, trot coupling."""
from dataclasses import dataclass, field
import numpy as np


# Leg order used everywhere: [FR, FL, RR, RL].
TROT_PHASE_OFFSETS = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float64)

# Kuramoto-style trot coupling: diagonal pairs in phase (+1),
# off-diagonal pairs antiphase (-1). Rows/cols order FR, FL, RR, RL.
TROT_COUPLING = np.array([
    [0.0, -1.0, -1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, -1.0, 0.0],
], dtype=np.float64)


@dataclass
class CPGParams:
    phase: float = 0.0
    frequency_hz: float = 2.0
    thigh_amp: float = 0.25
    calf_amp: float = -0.35
    phase_offsets: np.ndarray = field(
        default_factory=lambda: TROT_PHASE_OFFSETS.copy())
    duty_factor: float = 0.5
    jump_boost: float = 0.0  # 0 normal trot, 1 full crouch-tuck


@dataclass
class OscillatorState:
    r: np.ndarray = field(default_factory=lambda: np.ones(4))
    phi: np.ndarray = field(default_factory=lambda: TROT_PHASE_OFFSETS.copy())
