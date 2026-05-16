# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Time-domain features
# ================================================================
def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value."""
    return np.mean(np.abs(x), axis=2)

def std(x: np.ndarray) -> np.ndarray:
    """Standard deviation."""
    return np.std(x, axis=2)

def var(x: np.ndarray) -> np.ndarray:
    """Variance."""
    return np.var(x, axis=2)

def maxav(x: np.ndarray) -> np.ndarray:
    """Maximum absolute value."""
    return np.max(np.abs(x), axis=2)

def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square."""
    return np.sqrt(np.mean(x**2, axis=2))

def wl(x: np.ndarray) -> np.ndarray:
    """Waveform length."""
    return np.sum(np.abs(np.diff(x, axis=2)), axis=2)

def ssc(x: np.ndarray) -> np.ndarray:
    """Slope sign changes."""
    dx = np.diff(x, axis=2)
    return np.sum((dx[:, :, :-1] * dx[:, :, 1:]) < 0, axis=2)

def zc(x: np.ndarray) -> np.ndarray:
    """Zero crossings."""
    return np.sum(np.diff(np.signbit(x), axis=2), axis=2)

def log_det(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log detector.

    Uses abs(x) because EMG can be negative.
    """
    return np.exp(np.mean(np.log(np.abs(x) + eps), axis=2))

def wamp(x: np.ndarray, threshold: float = 20e-6) -> np.ndarray:
    """
    Willison amplitude.

    Counts how often abs(diff) exceeds a threshold.
    Threshold assumes signal is in volts.
    """
    dx = np.abs(np.diff(x, axis=2))
    return np.sum(dx > threshold, axis=2)

def iav(x: np.ndarray) -> np.ndarray:
    """Integrated absolute value."""
    return np.sum(np.abs(x), axis=2)

def ssi(x: np.ndarray) -> np.ndarray:
    """Simple square integral."""
    return np.sum(x**2, axis=2)

def aac(x: np.ndarray) -> np.ndarray:
    """
    Average amplitude change.

    Similar to waveform length, but normalized by window length.
    """
    return np.mean(np.abs(np.diff(x, axis=2)), axis=2)

def dasdv(x: np.ndarray) -> np.ndarray:
    """
    Difference absolute standard deviation value.

    RMS of the first difference.
    """
    dx = np.diff(x, axis=2)
    return np.sqrt(np.mean(dx**2, axis=2))

def ptp_amp(x: np.ndarray) -> np.ndarray:
    """Peak-to-peak amplitude."""
    return np.max(x, axis=2) - np.min(x, axis=2)

def median_abs(x: np.ndarray) -> np.ndarray:
    """Median absolute value."""
    return np.median(np.abs(x), axis=2)

def iqr(x: np.ndarray) -> np.ndarray:
    """Interquartile range."""
    q75 = np.percentile(x, 75, axis=2)
    q25 = np.percentile(x, 25, axis=2)
    return q75 - q25

def myop(x: np.ndarray, threshold: float = 20e-6) -> np.ndarray:
    """
    Myopulse percentage rate.

    Fraction of samples whose absolute value exceeds a threshold.
    Threshold assumes signal is in volts.
    """
    return np.mean(np.abs(x) > threshold, axis=2)

def hjorth_mobility(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Hjorth mobility.

    Measures relative amount of signal variation.
    """
    dx = np.diff(x, axis=2)

    var_x = np.var(x, axis=2)
    var_dx = np.var(dx, axis=2)

    return np.sqrt(var_dx / np.maximum(var_x, eps))

def hjorth_complexity(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Hjorth complexity.

    Measures how much the signal shape changes compared with a pure sine-like signal.
    """
    dx = np.diff(x, axis=2)
    ddx = np.diff(dx, axis=2)

    var_x = np.var(x, axis=2)
    var_dx = np.var(dx, axis=2)
    var_ddx = np.var(ddx, axis=2)

    mobility_x = np.sqrt(var_dx / np.maximum(var_x, eps))
    mobility_dx = np.sqrt(var_ddx / np.maximum(var_dx, eps))

    return mobility_dx / np.maximum(mobility_x, eps)


# ================================================================
# 2. Section: Frequency-domain scalar features
# ================================================================
def fft_power(x: np.ndarray) -> np.ndarray:
    """
    One-sided FFT power spectrum.

    Returns:
        shape (n_epochs, n_channels, n_freqs)
    """
    fft = np.fft.rfft(x, axis=2)
    return np.abs(fft) ** 2

def fft_freqs(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    One-sided FFT frequency vector.

    Returns:
        shape (n_freqs,)
    """
    return np.fft.rfftfreq(x.shape[2], d=1.0 / sfreq)

def total_power(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Total spectral power."""
    power = fft_power(x)
    return np.sum(power, axis=2)

def mean_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Mean frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    denom = np.sum(power, axis=2)
    denom = np.maximum(denom, 1e-12)

    return np.sum(power * f[None, None, :], axis=2) / denom

def median_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Median frequency based on cumulative spectral power."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    cumulative_power = np.cumsum(power, axis=2)
    half_power = cumulative_power[:, :, -1:] / 2.0

    idx = np.argmax(cumulative_power >= half_power, axis=2)

    return f[idx]

def peak_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Peak frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    idx = np.argmax(power, axis=2)

    return f[idx]


# ================================================================
# 3. Section: Mapped
# ================================================================
TIME_FEATURE_FUNCTIONS = [
    mav,
    std,
    var,
    maxav,
    rms,
    wl,
    ssc,
    zc,
    log_det,
    wamp,
    iav,
    ssi,
    aac,
    dasdv,
    ptp_amp,
    median_abs,
    iqr,
    myop,
    hjorth_mobility,
    hjorth_complexity,
]

FREQ_FEATURE_FUNCTIONS = [
    total_power,
    mean_freq,
    median_freq,
    peak_freq,
]
