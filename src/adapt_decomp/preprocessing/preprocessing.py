"""Functions to preprocess EMG signals"""

from typing import Literal
import numpy as np
from scipy.signal import butter, sosfilt


def preprocess_emg(
    data: np.ndarray,
    fs: float,
    highpass: float = 20.0,
    lowpass: float = 500.0,
    filter_order: int = 4,
    notch_filter: bool = True,
    notch_freq: float = 50.0,
    notch_width_hz: float = 1.0,
    notch_n_harmonics: int = 3,
    notch_order: int = 2,
) -> np.ndarray:
    """Causal bandpass + optional harmonic notch. The single preprocessing path shared
    by CBSS calibration (CBSS._preprocess_emg) and online adaptation (Data.preprocess_emg).

    Args:
        data:              [T, C] EMG array.
        fs:                Sampling frequency in Hz.
        highpass, lowpass: Bandpass cutoffs in Hz.
        filter_order:      Butterworth order for the bandpass stage.
        notch_filter:      Whether to notch out powerline harmonics.
        notch_freq:        Fundamental powerline frequency in Hz (50 or 60).
        notch_width_hz:    Half-bandwidth of each notch in Hz (±notch_width_hz).
        notch_n_harmonics: Number of harmonics to notch (including the fundamental).
        notch_order:       Butterworth order for each notch stage.

    Returns:
        Filtered array, same shape as input, float32.
    """
    out = np.asarray(data, dtype=np.float64)
    sos_bp = butter(filter_order, [highpass, lowpass], fs=fs, btype="band", output="sos")
    out = sosfilt(sos_bp, out, axis=0)
    if notch_filter:
        for harmonic in notch_freq * np.arange(1, notch_n_harmonics + 1):
            sos_notch = butter(
                notch_order, [harmonic - notch_width_hz, harmonic + notch_width_hz],
                fs=fs, btype="bandstop", output="sos",
            )
            out = sosfilt(sos_notch, out, axis=0)
    return out.astype(np.float32)


def filter_kwargs(cfg) -> dict:
    """Build preprocess_emg() kwargs from an AdaptConfig or CBSSConfig.

    Both config dataclasses declare identically-named lowcut/highcut/powerline/
    powerline_freq/notch_* fields for this shared filter, so this one mapping
    keeps CBSS calibration (cbss/core.py::_preprocess_emg) and online adaptation
    (data_structures.py::Data.preprocess_emg) from drifting out of sync.
    """
    return dict(
        highpass=cfg.lowcut,
        lowpass=cfg.highcut,
        filter_order=cfg.filter_order,
        notch_filter=cfg.powerline,
        notch_freq=cfg.powerline_freq,
        notch_width_hz=cfg.notch_width_hz,
        notch_n_harmonics=cfg.notch_n_harmonics,
        notch_order=cfg.notch_order,
    )


def replace_bad_channels(
    data: np.ndarray,
    bad_ch: list[int] | np.ndarray,
    ch_map: np.ndarray,
    layout: Literal["samples_first", "channels_first", "grid"] = "samples_first",
) -> np.ndarray:
    """Replace bad channels with the mean of their spatial neighbours.

    Args:
        data:    EMG array. Layouts:
                   samples_first = [T, C],
                   channels_first = [C, T],
                   grid = [rows, cols, T].
        bad_ch:  0-based indices of bad channels, processed in order.
        ch_map:  [rows, cols] 0-based channel map; empty cells are -1.
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