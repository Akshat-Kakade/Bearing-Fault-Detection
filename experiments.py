# Monte Carlo SNR sweep experiment.

import numpy as np

from config import Config
from signal_model import generate_signals
from filter_design import apply_filter, compensate_fir_delay
from analysis import detect_fault_frequency, snr_improvement_db


def run_snr_sweep(cfg: Config, fir, iir):
    """Sweep input SNR from sweep_snr_db_min to sweep_snr_db_max and run
    sweep_trials_per_level independent trials at each level.

    For every trial we build a fresh faulty signal (new random impact
    jitter/amplitude and new noise draw), then test fault detection on
    (a) the raw signal's envelope spectrum with no band-pass at all,
    (b) the FIR pipeline, (c) the IIR pipeline. We also record the
    frequency estimation error and the output SNR improvement for the
    two filtered pipelines.

    Returns a dict of numpy arrays, one entry per metric, indexed by
    SNR level.
    """
    snr_levels = np.arange(
        cfg.sweep_snr_db_min, cfg.sweep_snr_db_max + 1e-9, cfg.sweep_snr_db_step
    )
    n_levels = len(snr_levels)
    n_trials = cfg.sweep_trials_per_level

    rate_raw = np.zeros(n_levels)
    rate_fir = np.zeros(n_levels)
    rate_iir = np.zeros(n_levels)
    freq_err_fir = np.zeros(n_levels)
    freq_err_iir = np.zeros(n_levels)
    snr_imp_fir = np.zeros(n_levels)
    snr_imp_iir = np.zeros(n_levels)

    base_seed = cfg.seed
    for li, snr_db in enumerate(snr_levels):
        det_raw = det_fir = det_iir = 0
        errs_fir, errs_iir, imps_fir, imps_iir = [], [], [], []

        for trial in range(n_trials):
            seed = base_seed + 1000 * (li + 1) + trial
            sig = generate_signals(cfg, snr_db=snr_db, seed=seed)
            faulty = sig["faulty"]

            y_fir = compensate_fir_delay(fir, apply_filter(fir, faulty))
            y_iir = apply_filter(iir, faulty)

            r_raw = detect_fault_frequency(faulty, cfg.fs, cfg)
            r_fir = detect_fault_frequency(y_fir, cfg.fs, cfg)
            r_iir = detect_fault_frequency(y_iir, cfg.fs, cfg)

            det_raw += int(r_raw["detected"])
            det_fir += int(r_fir["detected"])
            det_iir += int(r_iir["detected"])

            errs_fir.append(abs(r_fir["peak_freq"] - cfg.bpfo_hz))
            errs_iir.append(abs(r_iir["peak_freq"] - cfg.bpfo_hz))
            imps_fir.append(snr_improvement_db(faulty, y_fir, sig["fault_component"]))
            imps_iir.append(snr_improvement_db(faulty, y_iir, sig["fault_component"]))

        rate_raw[li] = det_raw / n_trials
        rate_fir[li] = det_fir / n_trials
        rate_iir[li] = det_iir / n_trials
        freq_err_fir[li] = np.mean(errs_fir)
        freq_err_iir[li] = np.mean(errs_iir)
        snr_imp_fir[li] = np.mean(imps_fir)
        snr_imp_iir[li] = np.mean(imps_iir)

    return {
        "snr_levels": snr_levels,
        "rate_raw": rate_raw,
        "rate_fir": rate_fir,
        "rate_iir": rate_iir,
        "freq_err_fir": freq_err_fir,
        "freq_err_iir": freq_err_iir,
        "snr_imp_fir": snr_imp_fir,
        "snr_imp_iir": snr_imp_iir,
    }
