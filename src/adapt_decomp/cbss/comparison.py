"""Spike-train comparison helpers: RoA and duplicate removal."""

from __future__ import annotations

from typing import Dict

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
    cov = get_coefficient_of_variation(spike_trains, timestamps, None)
    cov_np = torch.where(torch.isnan(cov), torch.full_like(cov, torch.inf), cov).cpu().numpy()

    from adapt_decomp.utils import rate_of_agreement
    roa_vals, pairs, _ = rate_of_agreement(
        None, spike_trains.cpu().numpy(), fs=int(fs),
        tol_spike_ms=tol_spike_ms, tol_train_ms=tol_train_ms,
    )
    sort_order = np.argsort(roa_vals)[::-1]
    keep = np.ones(n_units, dtype=bool)
    for sort_idx in sort_order:
        score = float(roa_vals[sort_idx])
        if score <= roa_th:
            break
        i, j = pairs[sort_idx]
        if not (keep[i] and keep[j]):
            continue
        if cov_np[i] <= cov_np[j]:
            keep[j] = False
            removed, kept = j, i
        else:
            keep[i] = False
            removed, kept = i, j
        if verbose:
            logger.debug(f"Removed duplicate unit {removed} (RoA={score:.3f}, kept unit {kept})")

    keep_idx = np.where(keep)[0]
    if len(keep_idx) == n_units:
        return result

    result["sources"] = sources_t[:, keep_idx].cpu().numpy()
    result["spikes_dict"] = {
        new_id: spikes_dict[int(old_id)] for new_id, old_id in enumerate(keep_idx)
    }
    for key in ("sil", "cov", "spikes_centr", "base_centr"):
        value = result.get(key)
        if value is not None and len(np.asarray(value)) == n_units:
            result[key] = np.asarray(value)[keep_idx]
    sep_vectors = result.get("sep_vectors")
    if sep_vectors is not None:
        sep_vectors = np.asarray(sep_vectors)
        if sep_vectors.ndim == 2 and sep_vectors.shape[1] == n_units:
            result["sep_vectors"] = sep_vectors[:, keep_idx]

    return result
