import numpy as np
from scipy import signal

from config import Config


def fft_spectrum(x: np.ndarray, fs: float):
    #Single-sided amplitude spectrum via the FFT.

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x)) / n
    spectrum[1:-1] *= 2.0  # fold the negative-frequency energy into the positive side
    return freqs, spectrum


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Envelope of a band-passed signal via the analytic signal.

    The Hilbert transform turns x[n] into an analytic signal x[n] + j
    H{x}[n] whose magnitude is the envelope: it removes the carrier
    (the resonance ringing at ~3 kHz) and keeps the slow amplitude
    modulation caused by repeated fault impacts at BPFO.
    """
    return np.abs(signal.hilbert(x))


def envelope_spectrum(x: np.ndarray, fs: float):
    # FFT of the Hilbert envelope (with its DC component removed).

    env = hilbert_envelope(x)
    env = env - np.mean(env)
    return fft_spectrum(env, fs)


def spectrogram(x: np.ndarray, fs: float, nperseg: int = 512, noverlap: int = 384):
    # Short-time Fourier transform magnitude, for the time-frequency view.
    
    f, t, sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    return f, t, sxx


def detect_fault_frequency(x: np.ndarray, fs: float, cfg: Config):
    """Apply the project's detection rule to a (band-passed) signal.

    Detection rule: the highest peak of the envelope spectrum within
    [detect_band_low, detect_band_high] Hz must lie within
    detect_freq_tolerance_hz of BPFO, and exceed detect_peak_to_median_ratio
    times the median magnitude of that band.

    Returns dict(detected: bool, peak_freq: float, peak_mag: float,
                  median_mag: float, ratio: float).
    """
    freqs, spec = envelope_spectrum(x, fs)
    band = (freqs >= cfg.detect_band_low) & (freqs <= cfg.detect_band_high)
    band_freqs = freqs[band]
    band_spec = spec[band]

    if len(band_spec) == 0:
        return {"detected": False, "peak_freq": np.nan, "peak_mag": 0.0,
                "median_mag": 0.0, "ratio": 0.0}

    peak_idx = np.argmax(band_spec)
    peak_freq = band_freqs[peak_idx]
    peak_mag = band_spec[peak_idx]
    median_mag = np.median(band_spec)
    ratio = peak_mag / median_mag if median_mag > 0 else np.inf

    within_tol = abs(peak_freq - cfg.bpfo_hz) <= cfg.detect_freq_tolerance_hz
    strong_enough = ratio >= cfg.detect_peak_to_median_ratio
    detected = bool(within_tol and strong_enough)

    return {"detected": detected, "peak_freq": float(peak_freq), "peak_mag": float(peak_mag),
            "median_mag": float(median_mag), "ratio": float(ratio)}


def snr_improvement_db(x_in: np.ndarray, x_out: np.ndarray, fault_component: np.ndarray) -> float:
    """Output SNR minus input SNR, both defined as fault-component power
    over (signal - fault-component) power, in dB."""
    def _snr(signal_x):
        noise_est = signal_x - fault_component[: len(signal_x)]
        fault_power = np.mean(fault_component[: len(signal_x)] ** 2)
        noise_power = np.mean(noise_est ** 2)
        return 10.0 * np.log10(fault_power / noise_power) if noise_power > 0 else np.inf

    return _snr(x_out) - _snr(x_in)
