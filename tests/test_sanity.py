import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from signal_model import generate_signals
from filter_design import (
    design_fir_bandpass, design_iir_bandpass, design_resonator_tf,
    apply_filter, compensate_fir_delay, poles_zeros,
)
from adaptive import lms_filter
from analysis import detect_fault_frequency


@pytest.fixture(scope="module")
def cfg():
    return Config()


def test_bpfo_formula(cfg):
    """BPFO = (N/2) f_r (1 - (d/D) cos(phi)); check against hand calc."""
    n, fr = cfg.n_balls, cfg.shaft_freq_hz
    ratio = cfg.ball_diameter_mm / cfg.pitch_diameter_mm
    expected = 0.5 * n * fr * (1.0 - ratio * np.cos(np.deg2rad(cfg.contact_angle_deg)))
    assert cfg.bpfo_hz == pytest.approx(expected, rel=1e-9)
    assert cfg.bpfo_hz == pytest.approx(89.6, abs=0.1)
    assert cfg.shaft_freq_hz == pytest.approx(25.0)


def test_detection_at_high_snr(cfg):
    """At a generous input SNR, both FIR and IIR pipelines must detect
    the fault within the documented frequency tolerance."""
    sig = generate_signals(cfg, snr_db=10.0)
    fir = design_fir_bandpass(cfg)
    iir = design_iir_bandpass(cfg)

    y_fir = compensate_fir_delay(fir, apply_filter(fir, sig["faulty"]))
    y_iir = apply_filter(iir, sig["faulty"])

    r_fir = detect_fault_frequency(y_fir, cfg.fs, cfg)
    r_iir = detect_fault_frequency(y_iir, cfg.fs, cfg)

    assert r_fir["detected"]
    assert r_iir["detected"]
    assert abs(r_fir["peak_freq"] - cfg.bpfo_hz) < 1.0
    assert abs(r_iir["peak_freq"] - cfg.bpfo_hz) < 1.0


def test_raw_signal_not_detected_at_moderate_snr(cfg):
    """Sanity check on the project's core premise: the raw (unfiltered)
    signal should NOT pass the detection rule even at a moderate SNR,
    because the fault is masked by shaft harmonics."""
    sig = generate_signals(cfg, snr_db=10.0)
    r_raw = detect_fault_frequency(sig["faulty"], cfg.fs, cfg)
    assert not r_raw["detected"]


def test_lms_error_drops_after_convergence(cfg):
    """LMS squared error in the second half of adaptation must be much
    smaller than in the first few hundred samples (it is learning)."""
    sig = generate_signals(cfg, snr_db=10.0)
    error, _ = lms_filter(sig["faulty"], sig["reference"], cfg.lms_taps, cfg.lms_default_mu)

    early_power = np.mean(error[:200] ** 2)
    late_power = np.mean(error[len(error) // 2:] ** 2)
    assert late_power < early_power


def test_iir_filter_stable(cfg):
    """All poles of the IIR Butterworth band-pass must lie strictly
    inside the unit circle (Z-transform stability test)."""
    iir = design_iir_bandpass(cfg)
    _, poles, stable = poles_zeros(iir)
    assert stable
    assert np.all(np.abs(poles) < 1.0)


def test_resonator_tf_stable_and_matches_impulse_response(cfg):
    """The 2-pole resonator H(z) must be stable and its impulse
    response must match the analytic damped-sinusoid impulse response."""
    from scipy.signal import lfilter
    from signal_model import resonator_impulse_response

    fr = design_resonator_tf(cfg)
    _, poles, stable = poles_zeros(fr)
    assert stable

    impulse = np.zeros(512)
    impulse[0] = 1.0
    h_from_tf = lfilter(fr.b, fr.a, impulse)
    h_from_tf /= np.max(np.abs(h_from_tf))
    h_analytic = resonator_impulse_response(cfg)

    assert np.allclose(h_from_tf, h_analytic, atol=1e-9)


def test_fir_filter_always_stable_no_poles(cfg):
    """An FIR filter is BIBO stable by construction (no poles outside
    the origin)."""
    fir = design_fir_bandpass(cfg)
    _, poles, stable = poles_zeros(fir)
    assert stable
    assert len(poles) == 0
