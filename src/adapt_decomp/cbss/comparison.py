"""Spike-train comparison helpers: RoA and duplicate removal."""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger
import numpy as np
import torch

from adapt_decomp.cbss.signal_props import get_coefficient_of_variation


def spikes_dict_to_binary(
    spikes_dict: Dict[int, np.ndarray],
    n_samples: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert ``{unit: spike_indices}`` to a ``[T, n_units]`` int32 binary matrix."""
    spike_trains = torch.zeros(n_samples, len(spikes_dict), dtype=torch.int32, device=device)
    for unit, idx in spikes_dict.items():
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        idx_t = idx_t[(idx_t >= 0) & (idx_t < n_samples)]
        if idx_t.numel() > 0:
            spike_trains[idx_t, int(unit)] = 1
    return spike_trains


def shift_train(train: torch.Tensor, lag: int) -> torch.Tensor:
    """Shift a binary train by ``lag`` samples, zero-filling edges."""
    shifted = torch.zeros_like(train)
    if lag > 0:
        shifted[lag:] = train[:-lag]
    elif lag < 0:
        shifted[:lag] = train[-lag:]
    else:
        shifted = train.clone()
    return shifted


def dilate_train(train: torch.Tensor, tol: int) -> torch.Tensor:
    """Mark samples within ``tol`` samples of any spike."""
    train = train.bool()
    if tol <= 0:
        return train
    spike_idx = train.nonzero(as_tuple=True)[0]
    if spike_idx.numel() == 0:
        return train
    offsets = torch.arange(-tol, tol + 1, device=train.device)
    idx = spike_idx[:, None] + offsets[None, :]
    idx = idx[(idx >= 0) & (idx < train.numel())]
    out = torch.zeros_like(train, dtype=torch.bool)
    out[idx] = True
    return out


def rate_of_agreement_pair(
    train_a: torch.Tensor,
    train_b: torch.Tensor,
    tol_spike: int,
    tol_train: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, int]:
    """Rate of agreement for one pair with a small lag search."""
    train_a = train_a.bool()
    train_b = train_b.bool()
    dtype = dtype or torch.float32
    n_a = train_a.sum()
    n_b = train_b.sum()
    if n_a == 0 or n_b == 0:
        return torch.tensor(0.0, dtype=dtype, device=train_a.device), 0

    dil_a = dilate_train(train_a, tol_spike)
    dil_b = dilate_train(train_b, tol_spike)
    best_lag = 0
    best_corr = torch.tensor(-1, dtype=torch.int64, device=train_a.device)
    for lag in range(-tol_train, tol_train + 1):
        corr = (dil_a & shift_train(dil_b, lag)).sum()
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    aligned_b = shift_train(train_b, best_lag)
    matched_a = (train_a & dilate_train(aligned_b, tol_spike)).sum()
    matched_b = (aligned_b & dil_a).sum()
    common = torch.minimum(matched_a, matched_b).to(dtype)
    denom = n_a.to(dtype) + n_b.to(dtype) - common
    if denom <= 0:
        return torch.tensor(0.0, dtype=dtype, device=train_a.device), best_lag
    return common / denom, best_lag


def rate_of_agreement_all(
    spike_trains: torch.Tensor,
    fs: float,
    *,
    tol_spike_ms: float = 1.0,
    tol_train_ms: float = 40.0,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the symmetric pairwise RoA matrix."""
    n_units = spike_trains.shape[1]
    dtype = dtype or torch.float32
    roa = torch.eye(n_units, dtype=dtype, device=spike_trains.device)
    lags = torch.zeros(n_units, n_units, dtype=torch.long, device=spike_trains.device)
    tol_spike = max(1, round(tol_spike_ms / 1000.0 * fs))
    tol_train = round(tol_train_ms / 1000.0 * fs)
    for i in range(n_units):
        for j in range(i + 1, n_units):
            score, lag = rate_of_agreement_pair(
                spike_trains[:, i], spike_trains[:, j], tol_spike, tol_train, dtype=dtype
            )
            roa[i, j] = score
            roa[j, i] = score
            lags[i, j] = lag
            lags[j, i] = -lag
    return roa, lags


def remove_duplicates(
    result: Dict,
    fs: float,
    *,
    roa_th: float = 0.3,
    tol_train_ms: float = 40.0,
    tol_spike_ms: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    """Remove duplicate motor units (by RoA), keeping the unit with lower CoV-ISI.

    Operates on ``result['sources']`` [T, n_mu] and ``result['spikes_dict']``.
    Per-unit arrays preserved: ``sil``, ``cov``, ``spikes_centr``, ``base_centr``,
    ``sep_vectors`` [dim, n_mu].
    """
    sources = result.get("sources")
    spikes_dict = result.get("spikes_dict", {})
    if sources is None or len(spikes_dict) < 2:
        return result

    _device = torch.device(device)
    sources_t = torch.as_tensor(sources, dtype=dtype, device=_device)
    n_samples = sources_t.shape[0]

    spike_trains = spikes_dict_to_binary(spikes_dict, n_samples, device=_device)
    n_units = spike_trains.shape[1]
    if n_units < 2:
        return result

    timestamps = torch.arange(n_samples, dtype=dtype, device=_device) / fs
    roa, _ = rate_of_agreement_all(spike_trains, fs, tol_spike_ms=tol_spike_ms, tol_train_ms=tol_train_ms, dtype=dtype)
    cov = get_coefficient_of_variation(spike_trains, timestamps, None)
    cov = torch.where(torch.isnan(cov), torch.full_like(cov, torch.inf), cov)

    keep = torch.ones(n_units, dtype=torch.bool, device=_device)
    pair_idx = torch.triu_indices(n_units, n_units, offset=1, device=_device)
    pair_roa = roa[pair_idx[0], pair_idx[1]]
    for order_idx in torch.argsort(pair_roa, descending=True):
        score = pair_roa[order_idx]
        if score <= roa_th:
            break
        i = int(pair_idx[0, order_idx].item())
        j = int(pair_idx[1, order_idx].item())
        if not (bool(keep[i]) and bool(keep[j])):
            continue
        if cov[i] <= cov[j]:
            keep[j] = False
            removed, kept = j, i
        else:
            keep[i] = False
            removed, kept = i, j
        if verbose:
            logger.debug(f"Removed duplicate unit {removed} (RoA={float(score):.3f}, kept unit {kept})")

    keep_idx = keep.nonzero(as_tuple=True)[0]
    if keep_idx.numel() == n_units:
        return result

    keep_np = keep_idx.cpu().numpy()
    result["sources"] = sources_t[:, keep_idx].cpu().numpy()
    result["spikes_dict"] = {
        new_id: spikes_dict[int(old_id)] for new_id, old_id in enumerate(keep_np)
    }
    for key in ("sil", "cov", "spikes_centr", "base_centr"):
        value = result.get(key)
        if value is not None and len(np.asarray(value)) == n_units:
            result[key] = np.asarray(value)[keep_np]
    sep_vectors = result.get("sep_vectors")
    if sep_vectors is not None:
        sep_vectors = np.asarray(sep_vectors)
        if sep_vectors.ndim == 2 and sep_vectors.shape[1] == n_units:
            result["sep_vectors"] = sep_vectors[:, keep_np]

    return result
