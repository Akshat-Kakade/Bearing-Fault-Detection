"""End-to-end pipeline runner for the bearing-fault DSP simulation.

Running `python main.py` regenerates every figure, the metrics CSV and
the summary report, and prints a short results table to the terminal.
"""
import csv
import os
import time

import numpy as np

from config import CFG
from signal_model import generate_signals
from filter_design import (
    design_fir_bandpass, design_iir_bandpass, apply_filter, compensate_fir_delay,
)
from adaptive import lms_filter, convergence_curve
from analysis import detect_fault_frequency, snr_improvement_db
from experiments import run_snr_sweep
import plots

RESULTS_DIR = "results"


def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(plots.FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("Bearing Fault Detection via Vibration Signal Analysis")
    print("=" * 70)
    print(f"fs = {CFG.fs} Hz, duration = {CFG.duration} s, "
          f"shaft speed = {CFG.shaft_speed_rpm} rpm (f_r = {CFG.shaft_freq_hz:.2f} Hz)")
    print(f"BPFO (theoretical) = {CFG.bpfo_hz:.3f} Hz")

    # ---- Step 1: signal generation --------------------------------
    sig = generate_signals(CFG)
    plots.fig01_time_domain(CFG, sig)
    plots.fig02_raw_fft(CFG, sig)
    raw_detect = detect_fault_frequency(sig["faulty"], CFG.fs, CFG)
    print(f"\n[fig01, fig02] Raw-signal detection (no filtering): "
          f"detected={raw_detect['detected']}, peak={raw_detect['peak_freq']:.2f} Hz "
          "(expected to fail -- fault masked by shaft harmonics/noise)")

    # ---- Step 2: resonator + filter design -------------------------
    plots.fig03_resonator(CFG, sig["h_resonance"])

    fir = design_fir_bandpass(CFG)
    iir = design_iir_bandpass(CFG)
    plots.fig04_fir_vs_iir(CFG, fir, iir)

    y_fir = compensate_fir_delay(fir, apply_filter(fir, sig["faulty"]))
    y_iir = apply_filter(iir, sig["faulty"])
    plots.fig05_filtered_signals(CFG, sig, y_fir, y_iir)

    fir_detect = detect_fault_frequency(y_fir, CFG.fs, CFG)
    iir_detect = detect_fault_frequency(y_iir, CFG.fs, CFG)
    print(f"[fig04, fig05] FIR order={fir.order}, group delay={fir.delay_samples:.1f} samples; "
          f"IIR order={iir.order}, zero constant delay")
    print(f"  FIR detection: detected={fir_detect['detected']}, peak={fir_detect['peak_freq']:.2f} Hz")
    print(f"  IIR detection: detected={iir_detect['detected']}, peak={iir_detect['peak_freq']:.2f} Hz")

    plots.fig06_envelope_spectrum(CFG, y_fir, y_iir)
    plots.fig07_spectrogram(CFG, sig, y_iir)

    # ---- Step 3: LMS adaptive filtering (applied before band-pass) --
    step_curves = {}
    for mu in CFG.lms_step_sizes:
        err, _ = lms_filter(sig["faulty"], sig["reference"], CFG.lms_taps, mu)
        step_curves[mu] = convergence_curve(err)

    err_default, _ = lms_filter(sig["faulty"], sig["reference"], CFG.lms_taps, CFG.lms_default_mu)
    plots.fig08_lms(CFG, sig, step_curves, CFG.lms_default_mu, sig["faulty"], err_default)
    hum_reduction_db = 10.0 * np.log10(np.var(sig["faulty"]) / np.var(err_default))
    print(f"\n[fig08] LMS (taps={CFG.lms_taps}, default mu={CFG.lms_default_mu}): "
          f"variance reduced by {hum_reduction_db:.1f} dB after convergence")

    # ---- Step 4: Monte Carlo SNR sweep -----------------------------
    print("\nRunning Monte Carlo SNR sweep "
          f"({CFG.sweep_snr_db_min} to {CFG.sweep_snr_db_max} dB, "
          f"{CFG.sweep_trials_per_level} trials/level)...")
    sweep = run_snr_sweep(CFG, fir, iir)
    plots.fig09_detection_rate(CFG, sweep["snr_levels"], sweep["rate_raw"],
                                sweep["rate_fir"], sweep["rate_iir"])
    plots.fig10_freq_error_snr_improvement(
        CFG, sweep["snr_levels"], sweep["freq_err_fir"], sweep["freq_err_iir"],
        sweep["snr_imp_fir"], sweep["snr_imp_iir"],
    )

    # ---- Step 5: metrics.csv, summary.md, terminal table -----------
    _write_metrics_csv(sweep)
    _write_summary_md(fir, iir, raw_detect, fir_detect, iir_detect, hum_reduction_db, sweep)
    _print_summary_table(raw_detect, fir_detect, iir_detect, fir, iir, sweep)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f} s. Figures in figures/, metrics in results/.")


def _write_metrics_csv(sweep):
    path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "detection_rate_raw", "detection_rate_fir",
                          "detection_rate_iir", "freq_err_fir_hz", "freq_err_iir_hz",
                          "snr_improvement_fir_db", "snr_improvement_iir_db"])
        for i, snr in enumerate(sweep["snr_levels"]):
            writer.writerow([
                snr, sweep["rate_raw"][i], sweep["rate_fir"][i], sweep["rate_iir"][i],
                sweep["freq_err_fir"][i], sweep["freq_err_iir"][i],
                sweep["snr_imp_fir"][i], sweep["snr_imp_iir"][i],
            ])


def _write_summary_md(fir, iir, raw_detect, fir_detect, iir_detect, hum_reduction_db, sweep):
    path = os.path.join(RESULTS_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("# Simulation summary\n\n")
        f.write(f"BPFO (theoretical): {CFG.bpfo_hz:.3f} Hz\n\n")
        f.write(f"At input SNR = {CFG.input_snr_db} dB:\n\n")
        f.write(f"- Raw signal: detected={raw_detect['detected']}, "
                f"peak={raw_detect['peak_freq']:.2f} Hz\n")
        f.write(f"- FIR pipeline: detected={fir_detect['detected']}, "
                f"peak={fir_detect['peak_freq']:.2f} Hz, order={fir.order}\n")
        f.write(f"- IIR pipeline: detected={iir_detect['detected']}, "
                f"peak={iir_detect['peak_freq']:.2f} Hz, order={iir.order}\n")
        f.write(f"- LMS hum suppression: {hum_reduction_db:.1f} dB variance reduction\n\n")
        f.write("## SNR sweep (see results/metrics.csv for full data)\n\n")
        f.write("| SNR (dB) | Rate (raw) | Rate (FIR) | Rate (IIR) |\n")
        f.write("|---|---|---|---|\n")
        for i, snr in enumerate(sweep["snr_levels"]):
            f.write(f"| {snr:.0f} | {sweep['rate_raw'][i]:.2f} | "
                    f"{sweep['rate_fir'][i]:.2f} | {sweep['rate_iir'][i]:.2f} |\n")


def _print_summary_table(raw_detect, fir_detect, iir_detect, fir, iir, sweep):
    print("\n" + "-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    print(f"{'Metric':<35}{'Value':>35}")
    print(f"{'BPFO theoretical (Hz)':<35}{CFG.bpfo_hz:>35.3f}")
    print(f"{'BPFO detected, FIR (Hz)':<35}{fir_detect['peak_freq']:>35.3f}")
    print(f"{'BPFO detected, IIR (Hz)':<35}{iir_detect['peak_freq']:>35.3f}")
    print(f"{'FIR filter order':<35}{fir.order:>35d}")
    print(f"{'IIR filter order':<35}{iir.order:>35d}")

    mid_idx = len(sweep["snr_levels"]) // 2
    for label, idx in [("high SNR (0 dB)", -1), ("mid SNR", mid_idx), ("low SNR (-20 dB)", 0)]:
        print(f"{'Detection rate FIR @ ' + label:<35}{sweep['rate_fir'][idx]:>35.2f}")
        print(f"{'Detection rate IIR @ ' + label:<35}{sweep['rate_iir'][idx]:>35.2f}")
    print("-" * 70)


if __name__ == "__main__":
    main()
