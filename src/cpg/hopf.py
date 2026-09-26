"""Hopf/Kuramoto CPG network (Stage-C): coupled oscillators + feedback.

Four Hopf amplitude oscillators with Kuramoto phase coupling locked to
the trot topology (diagonal in phase). Foot-contact feedback injects a
phase-reset pull toward stance/swing based on touch-sensor error.
Euler-integrated at env dt; joint mapping reuses ParametricCPG.
"""
import numpy as np

from .parametric import ParametricCPG
from .types import CPGParams, OscillatorState, TROT_COUPLING, TROT_PHASE_OFFSETS


class HopfCPGNetwork(ParametricCPG):
    def __init__(self, freq_hz=2.0, thigh_amp=0.25, calf_amp=-0.35,
                 mu=1.0, alpha=20.0, coupling_strength=2.0, k_fb=0.5):
        super().__init__(freq_hz=freq_hz, thigh_amp=thigh_amp,
                         calf_amp=calf_amp)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.coupling_strength = float(coupling_strength)
        self.k_fb = float(k_fb)
        self.state = OscillatorState()
        # Desired trot phase differences relative to leg 0 (FR).
        self.desired_dphi = TROT_PHASE_OFFSETS - TROT_PHASE_OFFSETS[0]

    def reset(self, phase=0.0):
        self.phase = float(phase)
        self.state = OscillatorState(
            r=np.ones(4),
            phi=(TROT_PHASE_OFFSETS + phase).copy(),
        )
        return self.targets_at(phase)

    def step(self, dt, feedback=None, mod=None):
        """Advance oscillators; feedback[4] in [-1,1] per leg (contact error).

        mod[6] optionally updates CPG params first (Stage-B+C combined).
        Returns 12-dim joint targets.
        """
        if mod is not None:
            self.params = self.params_from_action(mod)
            self.phase = float(self.params.phase)
        if feedback is None:
            feedback = np.zeros(4)
        fb = np.asarray(feedback, dtype=np.float64).flatten()
        assert fb.shape == (4,), fb.shape
        omega = 2.0 * np.pi * self.params.frequency_hz
        r, phi = self.state.r.copy(), self.state.phi.copy()
        # Hopf amplitude dynamics: dr = alpha*(mu - r^2)*r.
        r += dt * self.alpha * (self.mu - r ** 2) * r
        r = np.clip(r, 0.3, 1.7)
        # Kuramoto phase dynamics with trot locking + feedback pull.
        for i in range(4):
            coupling = np.sum(
                TROT_COUPLING[i] * np.sin(phi - phi[i] - (
                    self.desired_dphi - self.desired_dphi[i])))
            phi[i] += dt * (omega + self.coupling_strength * coupling
                            + self.k_fb * fb[i])
        phi = np.mod(phi, 2.0 * np.pi)  # keep phases bounded; differences unaffected
        self.state = OscillatorState(r=r, phi=phi)
        self.phase = float(np.mean(phi))
        # Map mean amplitude onto thigh/calf, per-leg phase offsets.
        params = CPGParams(
            phase=self.phase, frequency_hz=self.params.frequency_hz,
            thigh_amp=self.params.thigh_amp * float(np.mean(r)),
            calf_amp=self.params.calf_amp * float(np.mean(r)),
            phase_offsets=(phi - phi[0]).copy(),
            duty_factor=self.params.duty_factor,
            jump_boost=self.params.jump_boost)
        # Per-leg absolute phases are phi[i] = phi[0] + offsets[i]; pass the
        # rotating reference phi[0] as base phase so the waveform advances.
        # ramp=1: oscillators drive full amplitude from step 0 (reset ease-in
        # already handled via targets_at).
        return self._targets_with_offsets(params, float(phi[0]),
                                          ramp_override=1.0)

    def set_gains(self, k_fb=None, coupling_strength=None, alpha=None):
        """Retune feedback/coupling gains live (e.g. curriculum schedule)."""
        if k_fb is not None:
            self.k_fb = float(k_fb)
        if coupling_strength is not None:
            self.coupling_strength = float(coupling_strength)
        if alpha is not None:
            self.alpha = float(alpha)
