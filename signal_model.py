"""Healthy and faulty bearing vibration signal generation.

The faulty signal models a bearing outer-race defect as a periodic train
of impacts (with small random timing jitter and amplitude variation)
convolved with the impulse response of a second-order structural
resonance. Shaft harmonics, white Gaussian noise and 50 Hz mains-hum
interference are then added on top, exactly as a real accelerometer
would pick them up.
"""
import numpy as np

from config import Config


def resonator_impulse_response(cfg: Config, n_taps: int = 512) -> np.ndarray:
    """Impulse response h[n] of the 2nd-order structural resonance.

    Uses the analytic damped-sinusoid form of a continuous 2nd-order
    system h(t) = exp(-zeta*wn*t) * sin(wd*t), sampled at fs (this is
    the impulse-invariant mapping from the continuous resonator to a
    discrete-time sequence; see filter_design.py for the matching H(z)).
    """
    wn = 2.0 * np.pi * cfg.f_resonance
    zeta = cfg.damping_ratio
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    n = np.arange(n_taps)
    t = n * cfg.dt
    h = np.exp(-zeta * wn * t) * np.sin(wd * t)
    h /= np.max(np.abs(h))  # normalise peak to 1 so impact amplitudes are meaningful
    return h


def _impact_train(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Periodic impulse train at the BPFO period with jitter and amplitude noise."""
    n_samples = cfg.n_samples
    period_samples = cfg.fs / cfg.bpfo_hz
    n_impacts = int(cfg.duration * cfg.bpfo_hz) + 2

    train = np.zeros(n_samples)
    nominal_times = np.arange(n_impacts) * period_samples
    jitter = rng.uniform(-cfg.impact_jitter_frac, cfg.impact_jitter_frac, n_impacts) * period_samples
    amplitudes = 1.0 + rng.uniform(
        -cfg.impact_amplitude_var_frac, cfg.impact_amplitude_var_frac, n_impacts
    )

    impact_indices = np.round(nominal_times + jitter).astype(int)
    valid = (impact_indices >= 0) & (impact_indices < n_samples)
    train[impact_indices[valid]] = amplitudes[valid]
    return train


def shaft_harmonics(cfg: Config, t: np.ndarray) -> np.ndarray:
    """Sum of shaft-speed harmonics (1x, 2x, 3x f_r) with decreasing amplitude."""
    fr = cfg.shaft_freq_hz
    signal = np.zeros_like(t)
    for k, amp in enumerate(cfg.shaft_harmonic_amplitudes, start=1):
        signal += amp * np.sin(2.0 * np.pi * k * fr * t)
    return signal


def mains_hum(cfg: Config, t: np.ndarray, rng: np.random.Generator,
              phase: float = 0.0, amp_scale: float = 1.0, noise_std: float = 0.0) -> np.ndarray:
    """50 Hz mains hum plus a weak 100 Hz harmonic, used for both the
    contaminated signal and its (differently phased/scaled) reference channel."""
    f0 = cfg.mains_hum_freq
    hum = amp_scale * cfg.mains_hum_amplitude * np.sin(2.0 * np.pi * f0 * t + phase)
    hum += amp_scale * cfg.mains_hum_2nd_harmonic_amplitude * np.sin(2.0 * np.pi * 2 * f0 * t + phase)
    if noise_std > 0:
        hum += rng.normal(0.0, noise_std, t.shape)
    return hum


def _noise_for_snr(cfg: Config, fault_component: np.ndarray, snr_db: float,
                    rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise scaled so that the input SNR (fault power /
    noise power, in dB) matches the requested value."""
    fault_power = np.mean(fault_component ** 2)
    noise_power = fault_power / (10.0 ** (snr_db / 10.0))
    return rng.normal(0.0, np.sqrt(noise_power), fault_component.shape)


def generate_signals(cfg: Config, snr_db: float = None, seed: int = None):
    """Build the healthy signal, faulty signal and mains-hum reference channel.

    Returns a dict with keys: t, healthy, faulty, reference, h_resonance,
    fault_component (the clean convolved impact train, used only to
    calibrate the noise level / for SNR bookkeeping).
    """
    if snr_db is None:
        snr_db = cfg.input_snr_db
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    t = np.arange(cfg.n_samples) * cfg.dt
    h = resonator_impulse_response(cfg)

    impacts = _impact_train(cfg, rng)
    fault_component = np.convolve(impacts, h, mode="full")[: cfg.n_samples]  # Convolution

    harmonics = shaft_harmonics(cfg, t)
    noise = _noise_for_snr(cfg, fault_component, snr_db, rng)
    hum = mains_hum(cfg, t, rng, phase=0.0, amp_scale=1.0, noise_std=0.0)

    healthy = harmonics + _noise_for_snr(cfg, fault_component, snr_db, rng) + hum
    faulty = fault_component + harmonics + noise + hum

    reference = mains_hum(
        cfg, t, rng,
        phase=cfg.reference_phase_offset_rad,
        amp_scale=cfg.reference_amplitude_scale,
        noise_std=cfg.reference_noise_std,
    )

    return {
        "t": t,
        "healthy": healthy,
        "faulty": faulty,
        "reference": reference,
        "h_resonance": h,
        "fault_component": fault_component,
        "snr_db": snr_db,
    }
