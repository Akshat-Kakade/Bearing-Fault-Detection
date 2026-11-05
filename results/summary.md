# Simulation summary

BPFO (theoretical): 89.620 Hz

At input SNR = 10.0 dB:

- Raw signal: detected=False, peak=25.00 Hz
- FIR pipeline: detected=True, peak=89.50 Hz, order=292
- IIR pipeline: detected=True, peak=89.50 Hz, order=8
- LMS hum suppression: 3.4 dB variance reduction

## SNR sweep (see results/metrics.csv for full data)

| SNR (dB) | Rate (raw) | Rate (FIR) | Rate (IIR) |
|---|---|---|---|
| -20 | 0.00 | 0.00 | 0.00 |
| -18 | 0.00 | 0.04 | 0.04 |
| -16 | 0.00 | 0.48 | 0.52 |
| -14 | 0.00 | 0.94 | 0.88 |
| -12 | 0.00 | 1.00 | 1.00 |
| -10 | 0.00 | 1.00 | 1.00 |
| -8 | 0.00 | 1.00 | 1.00 |
| -6 | 0.00 | 1.00 | 1.00 |
| -4 | 0.00 | 1.00 | 1.00 |
| -2 | 0.00 | 1.00 | 1.00 |
| 0 | 0.00 | 1.00 | 1.00 |
