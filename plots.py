"""All figure-generation code for the bearing-fault DSP pipeline.

Every function renders one named figure (fig01 .. fig10) to figures/ as
a 300 dpi PNG. Style is kept uniform here: white background, serif
font, labelled axes with units, colour-blind-safe colours, no seaborn.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

from config import Config
from analysis import fft_spectrum, hilbert_envelope, envelope_spectrum, spectrogram

FIGURES_DIR = "figures"

# Colour-blind-safe qualitative palette (Wong, 2011)
COLOR_HEALTHY = "#0072B2"   # blue
COLOR_FAULTY = "#D55E00"    # vermillion
COLOR_FIR = "#009E73"       # green
COLOR_IIR = "#CC79A7"       # pink
COLOR_RAW = "#56B4E9"       # sky blue
COLOR_ACCENT = "#E69F00"    # orange

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
})


def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig01_time_domain(cfg: Config, sig: dict):
    """fig01: healthy vs faulty signal in the time domain, zoomed to a
    few shaft revolutions, so individual fault impacts are visible."""
    n_rev = 4
    zoom_samples = int(n_rev * cfg.fs / cfg.shaft_freq_hz)
    t = sig["t"][:zoom_samples] * 1000.0  # ms

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
    axes[0].plot(t, sig["healthy"][:zoom_samples], color=COLOR_HEALTHY)
    axes[0].set_title("Healthy bearing")
    axes[0].set_ylabel("Amplitude (g)")

    axes[1].plot(t, sig["faulty"][:zoom_samples], color=COLOR_FAULTY)
    axes[1].set_title("Faulty bearing (outer-race defect)")
    axes[1].set_ylabel("Amplitude (g)")
    axes[1].set_xlabel("Time (ms)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Time-domain vibration signal ({n_rev} shaft revolutions)")
    fig.tight_layout()
    return _save(fig, "fig01_time_domain.png")


def fig02_raw_fft(cfg: Config, sig: dict):
    """fig02: raw FFT of healthy vs faulty signal, BPFO marked, showing
    the fault is NOT visible in the raw spectrum."""
    f_h, s_h = fft_spectrum(sig["healthy"], cfg.fs)
    f_f, s_f = fft_spectrum(sig["faulty"], cfg.fs)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(f_h, s_h, color=COLOR_HEALTHY, label="Healthy", alpha=0.8)
    ax.plot(f_f, s_f, color=COLOR_FAULTY, label="Faulty", alpha=0.8)
    ax.axvline(cfg.bpfo_hz, color=COLOR_ACCENT, linestyle="--", linewidth=1.0,
               label=f"BPFO = {cfg.bpfo_hz:.1f} Hz")
    ax.set_xlim(0, 500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Raw FFT: fault is not visible despite being present")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig02_raw_fft.png")


def fig03_resonator(cfg: Config, h: np.ndarray):
    """fig03: resonator impulse response, frequency response, and
    pole-zero plot with a stability statement.

    The frequency response and poles/zeros come from the exact 2-pole
    discrete transfer function H(z) (impulse-invariant transform), not
    from treating the truncated impulse response as an FIR filter."""
    from filter_design import design_resonator_tf, frequency_response, poles_zeros

    n = len(h)
    t_ms = np.arange(n) / cfg.fs * 1000.0

    fr = design_resonator_tf(cfg)
    w, hf, _ = frequency_response(fr, cfg, n_points=4096)
    z, p, stable = poles_zeros(fr)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    axes[0].plot(t_ms, h, color=COLOR_HEALTHY)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("h[n]")
    axes[0].set_title("Impulse response")
    axes[0].grid(alpha=0.3)

    axes[1].plot(w, 20 * np.log10(np.abs(hf) + 1e-12), color=COLOR_FAULTY)
    axes[1].axvline(cfg.f_resonance, color=COLOR_ACCENT, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title(f"Frequency response ($f_n$={cfg.f_resonance:.0f} Hz)")
    axes[1].grid(alpha=0.3)

    theta = np.linspace(0, 2 * np.pi, 256)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8)
    axes[2].scatter(p.real, p.imag, marker="x", color=COLOR_FAULTY, label="poles", zorder=3)
    if len(z) > 0:
        axes[2].scatter(z.real, z.imag, marker="o", facecolors="none",
                         edgecolors=COLOR_HEALTHY, label="zeros", zorder=3)
    axes[2].set_xlabel("Re(z)")
    axes[2].set_ylabel("Im(z)")
    stability = "stable" if stable else "UNSTABLE"
    axes[2].set_title(f"Pole-zero plot ({stability})")
    axes[2].set_aspect("equal")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "fig03_resonator.png")


def fig04_fir_vs_iir(cfg: Config, fir, iir):
    """fig04: FIR vs IIR band-pass comparison: magnitude, phase/group
    delay, and IIR pole-zero plot."""
    from filter_design import frequency_response, poles_zeros

    w_fir, h_fir, gd_fir = frequency_response(fir, cfg)
    w_iir, h_iir, gd_iir = frequency_response(iir, cfg)
    z_iir, p_iir, stable = poles_zeros(iir)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))

    ax = axes[0, 0]
    ax.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), color=COLOR_FIR, label="FIR")
    ax.plot(w_iir, 20 * np.log10(np.abs(h_iir) + 1e-12), color=COLOR_IIR, label="IIR")
    ax.axvspan(cfg.bp_low, cfg.bp_high, color="grey", alpha=0.15, label="pass-band")
    ax.set_xlim(0, cfg.fs / 2)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Magnitude response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(w_fir, np.unwrap(np.angle(h_fir)), color=COLOR_FIR, label="FIR")
    ax.plot(w_iir, np.unwrap(np.angle(h_iir)), color=COLOR_IIR, label="IIR")
    ax.set_xlim(cfg.bp_low - 200, cfg.bp_high + 200)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Phase response (near pass-band)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(w_fir, gd_fir / cfg.fs * 1000.0, color=COLOR_FIR, label="FIR")
    ax.plot(w_iir, gd_iir / cfg.fs * 1000.0, color=COLOR_IIR, label="IIR")
    ax.set_xlim(cfg.bp_low - 200, cfg.bp_high + 200)
    ax.set_ylim(0, max(np.max(gd_fir), np.max(gd_iir[np.isfinite(gd_iir)])) / cfg.fs * 1000.0 * 1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ms)")
    ax.set_title("Group delay: FIR is constant, IIR is not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8)
    ax.scatter(p_iir.real, p_iir.imag, marker="x", color=COLOR_IIR, label="poles", zorder=3)
    ax.scatter(z_iir.real, z_iir.imag, marker="o", facecolors="none",
               edgecolors=COLOR_IIR, label="zeros", zorder=3)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    stability = "stable" if stable else "UNSTABLE"
    ax.set_title(f"IIR pole-zero plot ({stability})")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(f"FIR (order {fir.order}) vs IIR Butterworth (order {iir.order}) band-pass")
    fig.tight_layout()
    return _save(fig, "fig04_fir_vs_iir.png")


def fig05_filtered_signals(cfg: Config, sig: dict, y_fir: np.ndarray, y_iir: np.ndarray):
    """fig05: filtered faulty signal, FIR vs IIR, FIR delay compensated."""
    n_rev = 6
    zoom_samples = int(n_rev * cfg.fs / cfg.shaft_freq_hz)
    t = sig["t"][:zoom_samples] * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
    axes[0].plot(t, y_fir[:zoom_samples], color=COLOR_FIR)
    axes[0].set_title("FIR band-pass output (delay-compensated)")
    axes[0].set_ylabel("Amplitude (g)")

    axes[1].plot(t, y_iir[:zoom_samples], color=COLOR_IIR)
    axes[1].set_title("IIR band-pass output")
    axes[1].set_ylabel("Amplitude (g)")
    axes[1].set_xlabel("Time (ms)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Band-pass filtered faulty signal around the resonance")
    fig.tight_layout()
    return _save(fig, "fig05_filtered_signals.png")


def fig06_envelope_spectrum(cfg: Config, y_fir: np.ndarray, y_iir: np.ndarray):
    """fig06: Hilbert envelope spectrum for FIR and IIR outputs, BPFO
    and harmonics marked."""
    f_fir, s_fir = envelope_spectrum(y_fir, cfg.fs)
    f_iir, s_iir = envelope_spectrum(y_iir, cfg.fs)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(f_fir, s_fir, color=COLOR_FIR, label="FIR pipeline", alpha=0.85)
    ax.plot(f_iir, s_iir, color=COLOR_IIR, label="IIR pipeline", alpha=0.85)
    for k in range(1, 5):
        ax.axvline(k * cfg.bpfo_hz, color=COLOR_ACCENT, linestyle="--", linewidth=0.8,
                   label=f"BPFO harmonics" if k == 1 else None)
    ax.set_xlim(0, cfg.detect_band_high)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Envelope spectrum amplitude")
    ax.set_title(f"Envelope spectrum (BPFO = {cfg.bpfo_hz:.1f} Hz)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig06_envelope_spectrum.png")


def fig07_spectrogram(cfg: Config, sig: dict, y_filtered: np.ndarray):
    """fig07: spectrogram before and after band-pass filtering."""
    f1, t1, sxx1 = spectrogram(sig["faulty"], cfg.fs)
    f2, t2, sxx2 = spectrogram(y_filtered, cfg.fs)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    im0 = axes[0].pcolormesh(t1, f1, 10 * np.log10(sxx1 + 1e-12), shading="gouraud", cmap="viridis")
    axes[0].set_title("Before band-pass")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], label="Power (dB)")

    im1 = axes[1].pcolormesh(t2, f2, 10 * np.log10(sxx2 + 1e-12), shading="gouraud", cmap="viridis")
    axes[1].set_title("After band-pass (IIR)")
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(im1, ax=axes[1], label="Power (dB)")

    fig.suptitle("Spectrogram of the faulty signal")
    fig.tight_layout()
    return _save(fig, "fig07_spectrogram.png")


def fig08_lms(cfg: Config, sig: dict, step_size_results: dict, default_mu: float,
              before: np.ndarray, after: np.ndarray):
    """fig08: LMS before/after hum removal, and convergence curves for
    multiple step sizes mu."""
    n_rev = 10
    zoom_samples = int(n_rev * cfg.fs / cfg.mains_hum_freq)
    t = sig["t"][:zoom_samples] * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    axes[0].plot(t, before[:zoom_samples], color=COLOR_RAW, label="Before (raw + hum)", alpha=0.8)
    axes[0].plot(t, after[:zoom_samples], color=COLOR_FAULTY, label="After LMS", alpha=0.8)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude (g)")
    axes[0].set_title(f"LMS hum removal (μ={default_mu})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    colors = [COLOR_HEALTHY, COLOR_FIR, COLOR_ACCENT, COLOR_FAULTY, COLOR_IIR]
    for i, (mu, curve) in enumerate(step_size_results.items()):
        axes[1].plot(curve, color=colors[i % len(colors)], label=f"μ={mu}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Smoothed squared error (dB)")
    axes[1].set_title("LMS convergence vs step size")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "fig08_lms.png")


def fig09_detection_rate(cfg: Config, snr_levels, rates_raw, rates_fir, rates_iir):
    """fig09: Monte Carlo detection rate vs input SNR for raw/FIR/IIR."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(snr_levels, rates_raw, "o-", color=COLOR_RAW, label="Raw envelope (no band-pass)")
    ax.plot(snr_levels, rates_fir, "s-", color=COLOR_FIR, label="FIR pipeline")
    ax.plot(snr_levels, rates_iir, "^-", color=COLOR_IIR, label="IIR pipeline")
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Fault detection rate vs input SNR ({cfg.sweep_trials_per_level} trials/level)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig09_detection_rate.png")


def fig10_freq_error_snr_improvement(cfg: Config, snr_levels, freq_err_fir, freq_err_iir,
                                      snr_imp_fir, snr_imp_iir):
    """fig10: frequency estimation error and output SNR improvement vs
    input SNR, for the FIR and IIR pipelines."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    axes[0].plot(snr_levels, freq_err_fir, "s-", color=COLOR_FIR, label="FIR")
    axes[0].plot(snr_levels, freq_err_iir, "^-", color=COLOR_IIR, label="IIR")
    axes[0].set_xlabel("Input SNR (dB)")
    axes[0].set_ylabel("|Detected - BPFO| (Hz)")
    axes[0].set_title("Frequency estimation error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(snr_levels, snr_imp_fir, "s-", color=COLOR_FIR, label="FIR")
    axes[1].plot(snr_levels, snr_imp_iir, "^-", color=COLOR_IIR, label="IIR")
    axes[1].set_xlabel("Input SNR (dB)")
    axes[1].set_ylabel("Output SNR improvement (dB)")
    axes[1].set_title("SNR improvement vs input SNR")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "fig10_freq_error_snr_improvement.png")
