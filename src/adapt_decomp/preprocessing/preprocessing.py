"""Functions to preprocess EMG signals"""

from typing import List, Literal, Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfilt


def _build_sos_stages(
    fs: float,
    highpass: float,
    lowpass: float,
    filter_order: int,
    notch_filter: bool,
    notch_freq: float,
    notch_width_hz: float,
    notch_n_harmonics: int,
    notch_order: int,
) -> List[np.ndarray]:
    """Build the bandpass plus optional harmonic notch sos filter stages.

    Shared by preprocess_emg and preprocess_emg_stateful so the two can't
    drift on which filters they apply.

    Args:
        fs (float): Sampling frequency in Hz.
        highpass (float): Bandpass low cutoff in Hz.
        lowpass (float): Bandpass high cutoff in Hz.
        filter_order (int): Butterworth order for the bandpass stage.
        notch_filter (bool): Whether to notch out powerline harmonics.
        notch_freq (float): Fundamental powerline frequency in Hz.
        notch_width_hz (float): Half-bandwidth of each notch in Hz.
        notch_n_harmonics (int): Number of harmonics to notch, including
            the fundamental.
        notch_order (int): Butterworth order for each notch stage.

    Returns:
        List[np.ndarray]: sos arrays, bandpass first, then each notch
        harmonic in order.
    """
    stages = [butter(filter_order, [highpass, lowpass], fs=fs, btype="band", output="sos")]
    if notch_filter:
        for harmonic in notch_freq * np.arange(1, notch_n_harmonics + 1):
            stages.append(butter(
                notch_order, [harmonic - notch_width_hz, harmonic + notch_width_hz],
                fs=fs, btype="bandstop", output="sos",
            ))
    return stages


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
    sos_stages = _build_sos_stages(
        fs, highpass, lowpass, filter_order,
        notch_filter, notch_freq, notch_width_hz, notch_n_harmonics, notch_order,
    )
    for sos in sos_stages:
        out = sosfilt(sos, out, axis=0)
    return out.astype(np.float32)


def preprocess_emg_stateful(
    data: np.ndarray,
    fs: float,
    zi: Optional[List[np.ndarray]] = None,
    highpass: float = 20.0,
    lowpass: float = 500.0,
    filter_order: int = 4,
    notch_filter: bool = True,
    notch_freq: float = 50.0,
    notch_width_hz: float = 1.0,
    notch_n_harmonics: int = 3,
    notch_order: int = 2,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Filter one EMG chunk, carrying sosfilt state across calls.

    Args:
        data (np.ndarray): EMG chunk with shape (T, C).
        fs (float): Sampling frequency in Hz.
        zi (Optional[List[np.ndarray]]): Per-stage filter state from a
            previous call, or None on the first call.
        highpass (float): Bandpass low cutoff in Hz.
        lowpass (float): Bandpass high cutoff in Hz.
        filter_order (int): Butterworth order for the bandpass stage.
        notch_filter (bool): Whether to notch out powerline harmonics.
        notch_freq (float): Fundamental powerline frequency in Hz.
        notch_width_hz (float): Half-bandwidth of each notch in Hz.
        notch_n_harmonics (int): Number of harmonics to notch, including
            the fundamental.
        notch_order (int): Butterworth order for each notch stage.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: Filtered chunk (float32, same
        shape as data) and the state list to pass into the next call.
    """
    out = np.asarray(data, dtype=np.float64)
    sos_stages = _build_sos_stages(
        fs, highpass, lowpass, filter_order,
        notch_filter, notch_freq, notch_width_hz, notch_n_harmonics, notch_order,
    )
    zi_new = []
    for i, sos in enumerate(sos_stages):
        stage_zi = zi[i] if zi is not None else np.zeros((sos.shape[0], 2, out.shape[1]))
        out, stage_zi_out = sosfilt(sos, out, axis=0, zi=stage_zi)
        zi_new.append(stage_zi_out)
    return out.astype(np.float32), zi_new


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


def select_channels(
    emg: np.ndarray,
    ch_mask: Optional[np.ndarray],
    ch_map: Optional[np.ndarray],
    interpolate: bool,
) -> np.ndarray:
    """Apply bad-channel handling: interpolate, drop, or no-op.

    Shared by CBSS._preprocess_emg (calibration) and Data.__init__ (online
    adaptation).

    Args:
        emg: EMG array with shape (samples, raw_channels).
        ch_mask: Boolean mask with shape (raw_channels,), True = keep.
            None = no channel selection at all.
        ch_map: 0-based channel map with shape (rows, cols), only used for
            interpolation.
        interpolate: Interpolate (True) vs drop (False) bad channels, when
            ch_mask is set. Ignored when ch_mask is None.

    Returns:
        np.ndarray: EMG with shape (samples, raw_channels) -- unchanged, or
        with bad channels interpolated in place -- or (samples,
        good_channels) if bad channels were dropped instead.
    """
    if interpolate and ch_mask is not None and ch_map is not None:
        bad_ch = np.nonzero(~ch_mask)[0]
        return replace_bad_channels(emg, bad_ch, ch_map, layout="samples_first")
    if ch_mask is not None:
        return emg[:, ch_mask]
    return emg


def validate_channel_selection(
    ch_mask: Optional[np.ndarray],
    ch_map: Optional[np.ndarray],
    replace_bad_channels: bool,
    n_raw_channels: int,
) -> None:
    """Validate ch_mask/ch_map/replace_bad_channels against the raw channel count.

    Shared by Data._select_channels (online adaptation, eager mode) and
    AdaptDecomp._preprocess_batch_raw (online adaptation, streaming mode)

    Args:
        ch_mask (Optional[np.ndarray]): Boolean mask with shape (raw_channels,).
        ch_map (Optional[np.ndarray]): 0-based channel map with shape (rows, cols).
        replace_bad_channels (bool): Whether interpolation is requested.
        n_raw_channels (int): Raw channel count to validate ch_mask against.

    Raises:
        ValueError: If replace_bad_channels is True with ch_map unset, or
            ch_mask's length disagrees with n_raw_channels.
    """
    if replace_bad_channels and ch_map is None:
        raise ValueError(
            "AdaptConfig.replace_bad_channels=True requires ch_map to be "
            "set: interpolation needs the electrode grid to find each bad "
            "channel's spatial neighbours."
        )
    if ch_mask is not None and ch_mask.shape[0] != n_raw_channels:
        raise ValueError(
            f"AdaptConfig.ch_mask has length {ch_mask.shape[0]} but the "
            f"raw emg has {n_raw_channels} channels: ch_mask.shape[0] must "
            "equal emg's raw channel count."
        )