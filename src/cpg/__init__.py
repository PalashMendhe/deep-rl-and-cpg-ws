"""CPG package: fixed trot generator, parametric and Hopf extensions."""
from .fixed_trot import FixedTrotCPG
from .parametric import ParametricCPG
from .hopf import HopfCPGNetwork
from .types import CPGParams, OscillatorState, TROT_COUPLING, TROT_PHASE_OFFSETS

__all__ = ["FixedTrotCPG", "ParametricCPG", "HopfCPGNetwork",
           "CPGParams", "OscillatorState", "TROT_COUPLING",
           "TROT_PHASE_OFFSETS"]

