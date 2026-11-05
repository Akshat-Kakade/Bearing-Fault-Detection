import numpy as np

def lms_filter(primary: np.ndarray, reference: np.ndarray, n_taps: int, mu: float):
    """Normalised-free LMS adaptive noise canceller.

    Update rule (Haykin, Adaptive Filter Theory):
        y[n]  = w[n]^T u[n]            (filter output: interference estimate)
        e[n]  = d[n] - y[n]            (error: the cleaned signal)
        w[n+1] = w[n] + mu * e[n] * u[n]
    where d[n] is the primary (contaminated) signal, u[n] is the tap
    vector of the most recent n_taps reference samples, and w[n] is the
    adaptive filter's weight vector.

    Returns (error, weight_history_norm):
      error             the cleaned output e[n], same length as primary
      weight_norm       ||w[n]||^2 at each step, useful for plotting
                        convergence (we actually return the smoothed
                        squared error in main plotting code; this
                        function just returns the raw adaptation).
    """
    n = len(primary)
    w = np.zeros(n_taps)
    error = np.zeros(n)
    u = np.zeros(n_taps)  # sliding tap-delay line of reference samples

    for i in range(n):
        u[1:] = u[:-1]
        u[0] = reference[i]
        y = np.dot(w, u)
        error[i] = primary[i] - y
        w += mu * error[i] * u

    return error, w

def convergence_curve(error: np.ndarray, smoothing: int = 200) -> np.ndarray:
    # Smoothed squared error in dB vs iteration, for convergence plots.
    sq = error ** 2
    kernel = np.ones(smoothing) / smoothing
    smoothed = np.convolve(sq, kernel, mode="valid")
    smoothed = np.maximum(smoothed, 1e-12)  # avoid log(0)
    return 10.0 * np.log10(smoothed)
