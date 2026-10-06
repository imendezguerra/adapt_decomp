"""A synthetic HD-EMG recording with known motor unit firings, for the end-to-end tests.

The fixtures built on it (synthetic_recording, synthetic_calibration) are in conftest.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticRecording:
    """HD-EMG simulated from known motor unit firings, see make_synthetic_recording.

    Attributes:
        emg (np.ndarray): EMG with shape (samples, channels).
        spikes (np.ndarray): Ground-truth binary spike trains with shape (samples, units).
        fs (int): Sampling frequency, in Hz.
        n_cal (int): Samples of the stationary calibration window emg[:n_cal].
    """

    emg: np.ndarray
    spikes: np.ndarray
    fs: int
    n_cal: int


def make_synthetic_recording(
    seed: int = 0,
    fs: int = 2048,
    cal_s: float = 5.0,
    dyn_s: float = 15.0,
    n_ch: int = 16,
    n_mu: int = 4,
    muap_ms: float = 10.0,
    snr_db: float = 20.0,
) -> SyntheticRecording:
    """Simulate a convolutive HD-EMG mixture whose MUAPs drift after the calibration window.

    Each unit fires at 9-16 Hz with 10% inter-spike jitter; its MUAP on each
    channel is a random Gaussian derivative. The MUAPs are fixed over the first
    cal_s seconds, then each morphs linearly into an independent random MUAP by
    the end of the recording: a fixed decomposition loses the units, an
    adaptive one can track them.

    Returns:
        SyntheticRecording: The EMG, its ground truth and the calibration window.
    """
    rng = np.random.default_rng(seed)
    n_samples = int((cal_s + dyn_s) * fs)
    n_cal = int(cal_s * fs)
    length = int(muap_ms / 1000 * fs)
    t = np.linspace(-3, 3, length)

    spikes = np.zeros((n_samples, n_mu), dtype=np.int32)
    for unit in range(n_mu):
        rate = rng.uniform(9, 16)
        k = int(rng.integers(0, fs // 10))
        while k < n_samples:
            spikes[k, unit] = 1
            k += max(int(fs / rate * (1 + 0.1 * rng.standard_normal())), int(0.03 * fs))

    def random_muaps() -> np.ndarray:
        width = rng.uniform(0.3, 0.8, (n_mu, 1, n_ch))
        shift = rng.uniform(-1, 1, (n_mu, 1, n_ch))
        gain = rng.standard_normal((n_mu, 1, n_ch))
        tt = t[None, :, None] - shift
        return -gain * tt * np.exp(-(tt**2) / (2 * width**2))

    start, end = random_muaps(), random_muaps()
    emg = np.zeros((n_samples + length, n_ch))
    for unit in range(n_mu):
        for i in np.flatnonzero(spikes[:, unit]):
            a = max(0.0, (i - n_cal) / (n_samples - n_cal))
            emg[i : i + length] += (1 - a) * start[unit] + a * end[unit]
    emg = emg[:n_samples]
    emg += rng.standard_normal(emg.shape) * np.sqrt(emg.var() / 10 ** (snr_db / 10))
    return SyntheticRecording(emg.astype(np.float32), spikes, fs, n_cal)
