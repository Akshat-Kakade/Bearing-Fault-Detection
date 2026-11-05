# FIR and IIR band-pass filter design and analysis.

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal

from config import Config


@dataclass
class FilterResult:
    name: str
    b: np.ndarray
    a: np.ndarray
    sos: np.ndarray = None
    order: int = 0
    delay_samples: float = 0.0  # constant group delay, used to align FIR output


def design_resonator_tf(cfg: Config) -> FilterResult:
    """Exact 2-pole discrete transfer function H(z) for the structural
    resonance, via the impulse-invariant transform.

    The continuous resonator H(s) = wd / ((s + zeta*wn)^2 + wd^2) has
    impulse response h(t) = exp(-zeta*wn*t) sin(wd*t). Sampling h(t) at
    T = 1/fs (impulse invariance) gives the standard second-order
    damped-sinusoid z-transform pair:
        H(z) = r*sin(wd*T) * z^-1 / (1 - 2*r*cos(wd*T) z^-1 + r^2 z^-2)
    with r = exp(-zeta*wn*T), a complex-conjugate pole pair at
    z = r * exp(+-j*wd*T). This is the actual 2-pole system behind
    signal_model.resonator_impulse_response (the two give numerically
    identical impulse responses, since impulse invariance of a strictly
    proper rational H(s) is exact up to aliasing).
    """
    wn = 2.0 * np.pi * cfg.f_resonance
    zeta = cfg.damping_ratio
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    t_sample = cfg.dt
    r = np.exp(-zeta * wn * t_sample)

    b = np.array([0.0, r * np.sin(wd * t_sample)])
    a = np.array([1.0, -2.0 * r * np.cos(wd * t_sample), r ** 2])
    return FilterResult(name="Resonator H(z)", b=b, a=a, order=2)


def design_fir_bandpass(cfg: Config) -> FilterResult:
    """Linear-phase FIR band-pass via the Kaiser window method.

    kaiserord() picks the minimum order and the window beta that meet
    the requested stopband attenuation and transition width; firwin()
    then builds the band-pass impulse response (zero-phase by
    construction: a linear phase filter has a constant group delay of
    (order)/2 samples).
    """
    nyq = cfg.fs / 2.0
    width = cfg.fir_transition_width
    numtaps, beta = signal.kaiserord(cfg.fir_stopband_atten_db, width / nyq)
    if numtaps % 2 == 0:
        numtaps += 1  # odd length -> symmetric, Type I linear-phase FIR
    b = signal.firwin(
        numtaps, [cfg.bp_low, cfg.bp_high], window=("kaiser", beta),
        pass_zero=False, fs=cfg.fs,
    )
    order = numtaps - 1
    return FilterResult(name="FIR (Kaiser)", b=b, a=np.array([1.0]),
                         order=order, delay_samples=order / 2.0)


def design_iir_bandpass(cfg: Config) -> FilterResult:
    """Butterworth band-pass IIR filter, order cfg.iir_order, as SOS.

    Second-order sections (sosfilt) are used instead of a single
    transfer-function (b, a) pair because cascaded biquads are far less
    sensitive to coefficient round-off than one high-order (b, a) pair.
    An IIR filter has no constant group delay, so its output is not
    time-aligned with the input the way the FIR's is.
    """
    nyq = cfg.fs / 2.0
    sos = signal.butter(
        cfg.iir_order, [cfg.bp_low / nyq, cfg.bp_high / nyq], btype="bandpass", output="sos"
    )
    b, a = signal.sos2tf(sos)
    return FilterResult(name="IIR (Butterworth)", b=b, a=a, sos=sos,
                         order=2 * cfg.iir_order, delay_samples=0.0)


def frequency_response(fr: FilterResult, cfg: Config, n_points: int = 4096):
    """Magnitude, phase and group delay vs frequency (Hz).

    For an SOS filter (our IIR design) we evaluate the response from
    the second-order sections directly with sosfreqz: converting an
    order-8 cascade to a single (b, a) pair is numerically ill
    conditioned and corrupts both the magnitude and the group delay.
    Group delay is then obtained as -d(phase)/d(omega) by finite
    differences, which is well-behaved away from zeros on the unit
    circle.
    """
    if fr.sos is not None:
        w, h = signal.sosfreqz(fr.sos, worN=n_points, fs=cfg.fs)
    else:
        w, h = signal.freqz(fr.b, fr.a, worN=n_points, fs=cfg.fs)
    phase = np.unwrap(np.angle(h))
    domega = (w[1] - w[0]) * (2.0 * np.pi / cfg.fs)  # angular-frequency step, rad/sample
    gd = -np.gradient(phase, domega)  # group delay in samples
    return w, h, gd


def poles_zeros(fr: FilterResult):
    # Poles and zeros of H(z), and whether the filter is stable.

    with warnings.catch_warnings():
        # A leading zero in b (e.g. the resonator's pure z^-1 delay) is
        # intentional, not ill-conditioning; scipy's generic warning
        # about it is a false positive here.
        warnings.filterwarnings("ignore", category=signal.BadCoefficients)
        z, p, _ = signal.tf2zpk(fr.b, fr.a)
    stable = np.all(np.abs(p) < 1.0)
    return z, p, stable


def apply_filter(fr: FilterResult, x: np.ndarray) -> np.ndarray:
    """Zero-phase-safe filtering: sosfilt for IIR (stable cascade),
    plain convolution-based filtering for FIR."""
    
    if fr.sos is not None:
        return signal.sosfilt(fr.sos, x)
    return signal.lfilter(fr.b, fr.a, x)


def compensate_fir_delay(fr: FilterResult, y: np.ndarray) -> np.ndarray:
    """Shift a FIR output left by its constant group delay so it lines
    up in time with the input (and with the IIR output) for comparison."""
    shift = int(round(fr.delay_samples))
    aligned = np.empty_like(y)
    if shift > 0:
        aligned[:-shift] = y[shift:]
        aligned[-shift:] = 0.0
    else:
        aligned[:] = y
    return aligned
