"""PyTorch motor-unit property functions (canonical implementations)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from adapt_decomp.ops import find_peaks_multisource


def get_muaps(
    spike_trains: torch.Tensor,
    emg_ch_array: torch.Tensor,
    half_win: int,
) -> torch.Tensor:
    """MUAPs via vectorised window averaging.

    Args:
        spike_trains: [T, M] bool/int tensor.
        emg_ch_array: [rows, cols, T] tensor.
        half_win:     Half-window in samples.

    Returns:
        [M, rows, cols, 2*half_win].
    """
    rows, cols, n_samples = emg_ch_array.shape
    n_mu = spike_trains.shape[1]
    win = 2 * half_win
    muaps = torch.zeros(n_mu, rows, cols, win, dtype=emg_ch_array.dtype, device=emg_ch_array.device)
    offsets = torch.arange(-half_win, half_win, device=emg_ch_array.device)
    for unit in range(n_mu):
        firings = spike_trains[:, unit].nonzero(as_tuple=True)[0]
        valid = firings[(firings >= half_win) & (firings + half_win <= n_samples - 1)]
        if valid.numel() == 0:
            continue
        idx = valid.unsqueeze(1) + offsets
        muaps[unit] = emg_ch_array[:, :, idx].mean(dim=2)
    return muaps


def get_base_and_spike_vals(
    spike_train: torch.Tensor,
    ipt2: torch.Tensor,
    ext_fact: int,
    min_dist: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (base_vals, spike_vals) for one unit from squared source."""
    # Use the canonical multi-source peak finder on a single [T, 1] view
    peak_mask, _ = find_peaks_multisource(ipt2.unsqueeze(1), min_dist)
    peak_idx = peak_mask[:, 0].nonzero(as_tuple=True)[0]
    peak_idx = peak_idx[peak_idx >= ext_fact + 1]

    spike_mask = spike_train.bool()
    spike_idx = spike_mask.nonzero(as_tuple=True)[0]
    spike_idx = spike_idx[spike_idx >= ext_fact + 1]

    base_idx = peak_idx[~torch.isin(peak_idx, spike_idx)]
    if not base_idx.any():
        base_idx = (~spike_mask).nonzero(as_tuple=True)[0]
        base_idx = base_idx[base_idx >= ext_fact + 1]

    return ipt2[base_idx], ipt2[spike_idx]


def get_silhouette_measure(
    spike_trains: torch.Tensor,
    ipts: torch.Tensor,
    ext_fact: int,
    min_peak_dist: int = 0,
    spike_centr: Optional[torch.Tensor] = None,
    base_centr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Silhouette measure for each unit using resolved spike-train labels."""
    ipts2 = ipts ** 2
    n_mu = spike_trains.shape[1]
    sil = torch.zeros(n_mu, dtype=ipts.dtype, device=ipts.device)
    min_dist = max(1, int(min_peak_dist))
    for unit in range(n_mu):
        base_vals, spike_vals = get_base_and_spike_vals(
            spike_trains[:, unit], ipts2[:, unit], ext_fact, min_dist
        )
        if spike_vals.numel() == 0 or base_vals.numel() == 0:
            continue
        sc = spike_vals.median() if spike_centr is None else spike_centr[unit]
        bc = base_vals.median() if base_centr is None else base_centr[unit]
        within = ((spike_vals - sc) ** 2).sum()
        between = ((spike_vals - bc) ** 2).sum()
        denom = torch.maximum(within, between)
        if denom > 0:
            sil[unit] = (between - within) / denom
    return sil


def get_pulse_to_noise_ratio(
    spike_trains: torch.Tensor,
    ipts: torch.Tensor,
    ext_fact: int,
    min_peak_dist: int = 0,
) -> torch.Tensor:
    """Pulse-to-noise ratio in dB using resolved spike-train labels."""
    ipts2 = ipts ** 2
    n_mu = spike_trains.shape[1]
    pnr = torch.full((n_mu,), float("nan"), dtype=ipts.dtype, device=ipts.device)
    min_dist = max(1, int(min_peak_dist))
    for unit in range(n_mu):
        base_vals, spike_vals = get_base_and_spike_vals(
            spike_trains[:, unit], ipts2[:, unit], ext_fact, min_dist
        )
        if spike_vals.numel() == 0 or base_vals.numel() == 0:
            continue
        baseline_mean = base_vals.mean()
        if baseline_mean > 0:
            pnr[unit] = 20.0 * torch.log10(spike_vals.mean() / baseline_mean)
    return pnr


def get_discharge_rate(
    spike_trains: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    """Mean discharge rate in Hz for each unit."""
    n_mu = spike_trains.shape[1]
    dr = torch.zeros(n_mu, dtype=timestamps.dtype, device=timestamps.device)
    for unit in range(n_mu):
        times = timestamps[spike_trains[:, unit].bool()]
        n_spikes = times.numel()
        if n_spikes == 0:
            continue
        total = times[-1] - times[0]
        if total == 0:
            continue
        isi = times.diff()
        active = total - isi[isi > 0.25].sum()
        if active > 0:
            dr[unit] = n_spikes / active
    return dr


def get_coefficient_of_variation(
    spike_trains: torch.Tensor,
    timestamps: torch.Tensor,
    discard_peri_isi: Optional[float] = 0.25,
) -> torch.Tensor:
    """Coefficient of variation of ISI as a ratio (e.g. 0.35 = 35%)."""
    n_mu = spike_trains.shape[1]
    cov = torch.full((n_mu,), float("nan"), dtype=timestamps.dtype, device=timestamps.device)
    for unit in range(n_mu):
        times = timestamps[spike_trains[:, unit].bool()]
        if times.numel() < 2:
            continue
        isi = times.diff()
        if discard_peri_isi is not None:
            isi = isi[isi < discard_peri_isi]
        if isi.numel() < 2 or isi.mean() == 0:
            continue
        cov[unit] = isi.std() / isi.mean()
    return cov


def emg_to_ch_array(emg: torch.Tensor, ch_map: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Reshape [T, C] EMG to [rows, cols, T] using a 0-based channel map."""
    ch_map_t = torch.as_tensor(ch_map, dtype=torch.long, device=emg.device)
    n_samples = emg.shape[0]
    rows, cols = ch_map_t.shape
    ch_array = torch.zeros(rows, cols, n_samples, dtype=emg.dtype, device=emg.device)
    valid = ch_map_t >= 0
    ch_array[valid, :] = emg[:, ch_map_t[valid]].T
    return ch_array


# Aliases matching the parent package naming convention
get_muaps_torch = get_muaps
get_base_and_spike_vals_torch = get_base_and_spike_vals
get_silhouette_measure_torch = get_silhouette_measure
get_pulse_to_noise_ratio_torch = get_pulse_to_noise_ratio
get_discharge_rate_torch = get_discharge_rate
get_coefficient_of_variation_torch = get_coefficient_of_variation
emg_to_ch_array_torch = emg_to_ch_array
