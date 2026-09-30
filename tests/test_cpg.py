"""Step-3/4 CPG unit tests: antisymmetry, clipping, Hopf stability.

Run from repo root: timeout 200 python3 tests/test_cpg.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.cpg.fixed_trot import FixedTrotCPG
from src.cpg.parametric import ParametricCPG
from src.cpg.hopf import HopfCPGNetwork


def test_trot_antisymmetry():
    cpg = FixedTrotCPG()
    a = cpg.targets_at(2 * np.pi + 0.7)
    b = cpg.targets_at(2 * np.pi + 0.7 + np.pi)
    # Diagonal pairs swap: FR<->FL thigh, RR<->RL thigh (with axis dirs).
    assert a.shape == (12,)
    # After half cycle the two diagonals exchange roles exactly (uniform
    # THIGH_DIRS: all thigh joints share axis [0,1,0], so swing-forward is a
    # negative delta on every leg; stance legs mirror it).
    assert np.allclose(a[1] - 0.8, b[4] - 0.8, atol=1e-6)
    assert np.allclose(a[4] - 0.8, b[1] - 0.8, atol=1e-6)


def test_swing_legs_drive_forward():
    # Kinematic regression for Bug 1: a leg in swing (its sin > 0) must get
    # a NEGATIVE thigh delta (empirically +angle sweeps every foot backward).
    cpg = FixedTrotCPG()
    phase = 2 * np.pi + np.pi / 2  # sin(phase_1)=+1: FL,RR swing; FR,RL stance
    t = cpg.targets_at(phase)
    deltas = t[1::3] - 0.8  # [FR, FL, RR, RL] thigh deltas
    assert deltas[1] < 0 and deltas[2] < 0, deltas  # swing pair forward
    assert deltas[0] > 0 and deltas[3] > 0, deltas  # stance pair pushes back


def test_params_from_action_clipping():
    cpg = ParametricCPG()
    p = cpg.params_from_action(np.ones(6) * 5.0)
    assert 1.5 <= p.frequency_hz <= 3.0
    assert 0.3 <= p.duty_factor <= 0.7
    assert 0.0 <= p.jump_boost <= 1.0
    p0 = cpg.params_from_action(np.zeros(6))
    assert abs(p0.frequency_hz - 2.0) < 1e-9
    assert abs(p0.jump_boost - 0.0) < 1e-9


def test_hopf_stability_open_loop():
    cpg = HopfCPGNetwork()
    cpg.reset()
    dt = 0.002
    traj = []
    for _ in range(2500):  # 5 s
        out = cpg.step(dt)
        assert out.shape == (12,)
        assert np.all(np.isfinite(out))
        traj.append(out.copy())
    traj = np.array(traj)
    # Oscillators must actually oscillate (frozen stance = mapping bug).
    swing = traj[:, 1].max() - traj[:, 1].min()
    assert swing > 0.3, swing
    assert np.all((cpg.state.r >= 0.3) & (cpg.state.r <= 1.7))
    dphi = (cpg.state.phi - cpg.state.phi[0]) % (2 * np.pi)
    # Diagonal legs (RR index 2... note order FR,FL,RR,RL: RR~pi, RL~0)
    assert abs(dphi[3]) < 0.3 or abs(dphi[3] - 2 * np.pi) < 0.3


def test_hopf_feedback_advances_phase():
    cpg = HopfCPGNetwork()
    cpg.reset()
    dt = 0.002
    ref = HopfCPGNetwork()
    ref.reset()
    for _ in range(500):
        cpg.step(dt, feedback=np.array([1.0, 0, 0, 0]))
        ref.step(dt)
    adv = (cpg.state.phi[0] - ref.state.phi[0]) % (2 * np.pi)
    assert adv > 0.05, adv


def test_hopf_gain_retune():
    cpg = HopfCPGNetwork()
    cpg.set_gains(k_fb=1.5, coupling_strength=3.0, alpha=30.0)
    assert cpg.k_fb == 1.5
    assert cpg.coupling_strength == 3.0
    assert cpg.alpha == 30.0
    out = cpg.step(0.002)
    assert out.shape == (12,) and np.all(np.isfinite(out))


if __name__ == "__main__":
    test_trot_antisymmetry()
    print("PASS test_trot_antisymmetry")
    test_swing_legs_drive_forward()
    print("PASS test_swing_legs_drive_forward")
    test_params_from_action_clipping()
    print("PASS test_params_from_action_clipping")
    test_hopf_stability_open_loop()
    print("PASS test_hopf_stability_open_loop")
    test_hopf_feedback_advances_phase()
    print("PASS test_hopf_feedback_advances_phase")
    test_hopf_gain_retune()
    print("PASS test_hopf_gain_retune")
    print("6/6 CPG tests passed")
