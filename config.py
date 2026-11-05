# Central configuration for the bearing-fault DSP simulation.

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Config:
    # ---- Acquisition -----------------------------------------------
    fs: float = 12_000.0          # sampling rate, Hz
    duration: float = 2.0         # signal length, s
    seed: int = 20260107          # fixed RNG seed for reproducibility

    # ---- Shaft / bearing geometry (6205-type ball bearing) ---------
    shaft_speed_rpm: float = 1500.0
    n_balls: int = 9
    ball_diameter_mm: float = 7.94
    pitch_diameter_mm: float = 39.04
    contact_angle_deg: float = 0.0

    # ---- Structural resonance (2nd-order system excited by impacts) -
    f_resonance: float = 3000.0   # natural frequency, Hz
    damping_ratio: float = 0.05

    # ---- Fault impact train randomness ------------------------------
    impact_jitter_frac: float = 0.01      # +/-1% random slip on impact times
    impact_amplitude_var_frac: float = 0.10  # +/-10% amplitude variation per impact

    # ---- Shaft harmonics (present in both healthy and faulty signal) -
    shaft_harmonic_amplitudes: tuple = (1.0, 0.5, 0.25)  # 1x, 2x, 3x f_r

    # ---- Noise / interference ---------------------------------------
    input_snr_db: float = 10.0    # default SNR for single-run figures
    mains_hum_freq: float = 50.0
    mains_hum_amplitude: float = 0.6
    mains_hum_2nd_harmonic_amplitude: float = 0.15
    reference_phase_offset_rad: float = 0.9
    reference_amplitude_scale: float = 0.8
    reference_noise_std: float = 0.05

    # ---- FIR / IIR band-pass design around the resonance -------------
    bp_low: float = 2250.0
    bp_high: float = 3750.0
    fir_stopband_atten_db: float = 60.0
    fir_transition_width: float = 150.0   # Hz, used by kaiserord
    iir_order: int = 4                    # Butterworth band-pass order

    # ---- LMS adaptive filter ------------------------------------------
    lms_taps: int = 32
    lms_step_sizes: tuple = (0.0005, 0.002, 0.008, 0.02)
    lms_default_mu: float = 0.002

    # ---- Detection rule (envelope spectrum peak test) -----------------
    detect_band_low: float = 20.0
    detect_band_high: float = 500.0
    detect_freq_tolerance_hz: float = 2.0
    detect_peak_to_median_ratio: float = 5.0

    # ---- Monte Carlo SNR sweep -----------------------------------------
    sweep_snr_db_min: float = -20.0
    sweep_snr_db_max: float = 0.0
    sweep_snr_db_step: float = 2.0
    sweep_trials_per_level: int = 50

    @property
    def shaft_freq_hz(self) -> float:
        """Shaft rotation frequency f_r in Hz (1500 rpm = 25 Hz)."""
        return self.shaft_speed_rpm / 60.0

    @property
    def bpfo_hz(self) -> float:
        """Ball Pass Frequency Outer-race (BPFO), the fault ground truth.

        Syllabus topic: Signals and Systems (deriving the characteristic
        frequency of a physical fault model from first principles).

        BPFO = (N/2) * f_r * (1 - (d/D) * cos(phi))
        """
        n, fr = self.n_balls, self.shaft_freq_hz
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm
        phi = np.deg2rad(self.contact_angle_deg)
        return 0.5 * n * fr * (1.0 - ratio * np.cos(phi))

    @property
    def n_samples(self) -> int:
        return int(round(self.fs * self.duration))

    @property
    def dt(self) -> float:
        return 1.0 / self.fs


CFG = Config()
