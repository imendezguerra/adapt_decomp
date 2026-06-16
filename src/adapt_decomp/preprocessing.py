"""Functions to preprocess EMG signals"""

from typing import Literal, Optional
import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import butter, filtfilt, firwin2, iirnotch


def bandpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[list] = None,
    order: Optional[int] = 4,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a bandpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): List with the cutoff frequencies of the filter.
            Default is [20, 500].
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define cutoff frequencies
    if cutoff is None:
        cutoff = [20, 500]

    # Define filter
    sos = signal.butter(order, cutoff, btype='band', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out


def highpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 20,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a highpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): High cutoff frequency of the filter. Default is 20.
        order (Optional[int]): The order of the filter. Default is 2.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out


def lowpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 500,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a lowpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): Low cutoff frequency of the filter. Default is 20.
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out

 
def remove_powerline(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 50,
    width: Optional[float] = 1,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Remove powerline noise from the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): Cutoff frequency of the filter. Default is 50.
        width (Optional[float]): Width of the filter. Default is 1. 
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.
    """

    # Build cutoff
    cutoff = [cutoff - width/2, cutoff + width/2]

    # Define filter
    sos = signal.butter(order, cutoff, btype='bandstop', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out


def bandpass(
    data: np.ndarray,
    fs: float,
    highpass: float = 20.0,
    lowpass: float = 500.0,
    order: int = 4,
    ftype: Literal["butter", "firwin2"] = "butter",
) -> np.ndarray:
    """Bandpass filter with explicit high/low cutoff frequencies.

    Args:
        data:     [T, C] EMG array.
        fs:       Sampling frequency in Hz.
        highpass: High-pass cutoff frequency in Hz.
        lowpass:  Low-pass cutoff frequency in Hz.
        order:    Filter order (butter) or number of taps (firwin2).
        ftype:    Filter design method.

    Returns:
        Filtered array, same shape as input.
    """
    if ftype == "butter":
        b, a = butter(order, [highpass, lowpass], fs=fs, btype="band")
        return filtfilt(b, a, data, axis=0)
    if ftype == "firwin2":
        nyq = fs / 2
        f = [0, highpass * 0.9, highpass, lowpass, lowpass * 1.1, nyq]
        m = [0, 0, 1, 1, 0, 0]
        fir_coeff = firwin2(order, f, m, fs=fs)
        return filtfilt(fir_coeff, [1.0], data, axis=0)
    raise ValueError(f"Unknown filter type: {ftype!r}")


def notch_harmonics(
    data: np.ndarray,
    fs: float,
    f0: float = 50.0,
    n_harmonics: int = 3,
    width: float = 1.0,
    order: int = 2,
    ftype: Literal["butter", "fft", "iirnotch"] = "butter",
) -> np.ndarray:
    """Remove powerline noise and its harmonics.

    Args:
        data:        [T, C] EMG array.
        fs:          Sampling frequency in Hz.
        f0:          Fundamental powerline frequency in Hz (50 or 60).
        n_harmonics: Number of harmonics to notch (including fundamental).
        width:       Half-bandwidth of each notch in Hz (±width).
        order:       Filter order (butter) or quality factor (iirnotch).
        ftype:       Filter design method.

    Returns:
        Filtered array, same shape as input.
    """
    harmonics = f0 * np.arange(1, n_harmonics + 1)
    out = data.copy()

    if ftype == "butter":
        for freq in harmonics:
            b, a = butter(order, [freq - width, freq + width], fs=fs, btype="bandstop")
            out = filtfilt(b, a, out, axis=0)

    elif ftype == "fft":
        n_samples = data.shape[0]
        spectrum = rfft(out, axis=0)
        freqs = rfftfreq(n_samples, d=1.0 / fs)
        for freq in harmonics:
            mask = np.abs(freqs - freq) <= width
            spectrum[mask, :] = 0
        out = irfft(spectrum, n=n_samples, axis=0)

    elif ftype == "iirnotch":
        for freq in harmonics:
            b, a = iirnotch(freq, order, fs)
            out = filtfilt(b, a, out, axis=0)

    else:
        raise ValueError(f"Unknown filter type: {ftype!r}")

    return out


def replace_bad_channels(
    data: np.ndarray,
    bad_ch: list[int] | np.ndarray,
    ch_map: np.ndarray,
    layout: Literal["samples_first", "channels_first", "grid"] = "samples_first",
) -> np.ndarray:
    """Replace bad channels with the mean of their spatial neighbours.

    Args:
        data:    EMG array. Layouts:
                   ``samples_first`` = ``[T, C]``,
                   ``channels_first`` = ``[C, T]``,
                   ``grid`` = ``[rows, cols, T]``.
        bad_ch:  0-based indices of bad channels, processed in order.
        ch_map:  ``[rows, cols]`` 0-based channel map; empty cells are ``-1``.
        layout:  Data layout.

    Returns:
        Copy of data with bad channels replaced.
    """
    remaining = list(np.asarray(bad_ch, dtype=int).flatten())
    out = data.copy()
    n_rows, n_cols = ch_map.shape
    offsets = np.array(
        [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]],
        dtype=int,
    )

    while remaining:
        curr = int(remaining.pop(0))
        row_idx, col_idx = np.nonzero(ch_map == curr)
        if row_idx.size == 0:
            continue
        r0, c0 = int(row_idx[0]), int(col_idx[0])
        coords = np.array([[r0, c0]], dtype=int) + offsets  # [8, 2]
        in_bounds = (
            (coords[:, 0] >= 0) & (coords[:, 0] < n_rows)
            & (coords[:, 1] >= 0) & (coords[:, 1] < n_cols)
        )
        coords = coords[in_bounds]
        neigh_ch = ch_map[coords[:, 0], coords[:, 1]]
        if remaining:
            good = (neigh_ch >= 0) & ~np.isin(neigh_ch, remaining)
        else:
            good = neigh_ch >= 0
        coords = coords[good]
        neigh_ch = neigh_ch[good]
        if neigh_ch.size == 0:
            continue

        if layout == "samples_first":
            out[:, curr] = out[:, neigh_ch].mean(axis=1)
        elif layout == "channels_first":
            out[curr, :] = out[neigh_ch, :].mean(axis=0)
        elif layout == "grid":
            out[r0, c0, :] = out[coords[:, 0], coords[:, 1], :].mean(axis=0)
        else:
            raise ValueError(f"Unknown layout: {layout!r}")

    return out