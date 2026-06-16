"""Calibration step: run CBSS on a specified segment of an EMG recording."""

from __future__ import annotations

from copy import copy
from typing import Union

import numpy as np
import torch

from adapt_decomp.cbss import CBSS, CBSSConfig, CBSSResult


def calibrate_from_indices(
    emg: Union[np.ndarray, torch.Tensor],
    timestamps: Union[np.ndarray, torch.Tensor],
    calib_indices: Union[slice, np.ndarray],
    cbss_config: CBSSConfig | None = None,
) -> CBSSResult:
    """Run CBSS on ``emg[calib_indices]`` and return a calibration-ready ``CBSSResult``.

    The returned result always has ``.emg`` populated (required by
    ``AdaptDecomp.from_calibration``). No unit filtering is applied — call
    ``select_units_unsupervised()`` or ``select_units_supervised()`` afterward.

    Args:
        emg:           [T, C] full EMG recording (numpy or torch).
        timestamps:    [T] sample times in seconds.
        calib_indices: Which samples to use for calibration — a ``slice``,
                       an integer index array, or a boolean mask.
        cbss_config:   CBSS configuration. Defaults to ``CBSSConfig()``.
                       ``save_emg`` is forced to ``True`` regardless.

    Returns:
        ``CBSSResult`` with ``.emg`` set to the calibration-window EMG and
        ``.sources`` / ``.spikes`` aligned to the same window.

    Raises:
        ValueError: If no units are found or inputs are inconsistent.
    """
    emg_np = _to_numpy(emg)
    ts_np = _to_numpy(timestamps)

    if emg_np.ndim != 2:
        raise ValueError(f"emg must be 2-D [T, C], got shape {emg_np.shape}")
    if ts_np.ndim != 1 or ts_np.shape[0] != emg_np.shape[0]:
        raise ValueError(
            f"timestamps must be 1-D with length {emg_np.shape[0]}, got {ts_np.shape}"
        )

    emg_calib = emg_np[calib_indices]
    ts_calib = ts_np[calib_indices]

    if emg_calib.shape[0] < 2:
        raise ValueError(
            f"Calibration window has only {emg_calib.shape[0]} samples — too short for CBSS."
        )

    config = copy(cbss_config) if cbss_config is not None else CBSSConfig()
    config.save_emg = True  # required for AdaptDecomp.from_calibration()

    result = CBSS(config).decompose(emg_calib, ts_calib)

    if result.sources.shape[1] == 0:
        raise ValueError(
            "CBSS found no motor units in the calibration window. "
            "Try relaxing sil_th or increasing search_iter in CBSSConfig."
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
