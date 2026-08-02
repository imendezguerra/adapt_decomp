"""Unit selection after CBSS calibration — unsupervised and supervised paths."""

from __future__ import annotations

from typing import Optional

import numpy as np

from adapt_decomp.cbss import CBSSResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_units_unsupervised(
    result: CBSSResult,
    *,
    sil_th: Optional[float] = None,
    pnr_th: Optional[float] = None,
    dr_min: Optional[float] = None,
    dr_max: Optional[float] = None,
    cov_th: Optional[float] = None,
) -> CBSSResult:
    """Keep units that pass ALL provided quality thresholds (``None`` = skip criterion).

    Args:
        result:  ``CBSSResult`` from ``calibrate_from_indices()``.
        sil_th:  Minimum silhouette score (units with ``sil >= sil_th`` are kept).
        pnr_th:  Minimum pulse-to-noise ratio.
        dr_min:  Minimum discharge rate (pps).
        dr_max:  Maximum discharge rate (pps).
        cov_th:  Maximum coefficient of variation of inter-spike intervals.

    Returns:
        New ``CBSSResult`` with only the selected units.

    Raises:
        ValueError: If no units survive the filters.
    """
    n_mu = result.sources.shape[1]
    mask = np.ones(n_mu, dtype=bool)

    if sil_th is not None:
        if result.sil is None:
            raise ValueError("result.sil is None — cannot apply sil_th filter.")
        mask &= result.sil >= sil_th

    if pnr_th is not None:
        if result.pnr is None:
            raise ValueError("result.pnr is None — cannot apply pnr_th filter.")
        mask &= result.pnr >= pnr_th

    if dr_min is not None:
        if result.dr is None:
            raise ValueError("result.dr is None — cannot apply dr_min filter.")
        mask &= result.dr >= dr_min

    if dr_max is not None:
        if result.dr is None:
            raise ValueError("result.dr is None — cannot apply dr_max filter.")
        mask &= result.dr <= dr_max

    if cov_th is not None:
        if result.cov is None:
            raise ValueError("result.cov is None — cannot apply cov_th filter.")
        mask &= result.cov <= cov_th

    n_kept = int(mask.sum())
    if n_kept == 0:
        raise ValueError(
            "No units survived unsupervised quality filtering. "
            "Loosen one or more thresholds or check CBSSConfig."
        )
    return _subset_cbss_result(result, mask)


def select_units_supervised(
    result: CBSSResult,
    gt_spikes: np.ndarray,
    *,
    roa_th: float = 0.5,
    tol_ms: float = 0.5,
    fs: Optional[float] = None,
) -> CBSSResult:
    """Match each decomposed unit to the best GT unit by RoA; keep matches above ``roa_th``.

    Args:
        result:    ``CBSSResult`` from ``calibrate_from_indices()``.
        gt_spikes: ``[T_calib, M_gt]`` binary (int or bool) spike matrix aligned to the
                   calibration window.  Must have the same number of samples as
                   ``result.sources``.
        roa_th:    Minimum rate-of-agreement to keep a unit (default 0.5).
        tol_ms:    Tolerance window for coincident spikes in milliseconds (default 0.5).
        fs:        Sampling frequency in Hz.  If ``None``, inferred from
                   ``result.timestamps`` (required if timestamps is absent).

    Returns:
        New ``CBSSResult`` with:
        - only units whose best GT match has ``RoA >= roa_th``
        - ``gt_matched_indices[i]`` set to the index of the matched GT unit

    Raises:
        ValueError: If no units match or ``fs`` cannot be determined.
    """
    T = result.sources.shape[0]
    n_mu = result.sources.shape[1]

    if gt_spikes.shape[0] != T:
        raise ValueError(
            f"gt_spikes has {gt_spikes.shape[0]} samples but result.sources has {T}. "
            "gt_spikes must be aligned to the calibration window."
        )
    fs = _resolve_fs(fs, result)

    from adapt_decomp.utils import rate_of_agreement
    roa_vals, pairs, _ = rate_of_agreement(
        gt_spikes.astype(np.float32),
        result.spikes.astype(np.float32),
        fs=int(fs),
        tol_spike_ms=tol_ms,
    )
    # pairs[i] = (gt_idx, dec_idx), sorted by dec_idx ascending — 1:1 global greedy assignment
    mask = np.zeros(n_mu, dtype=bool)
    gt_matched = np.zeros(n_mu, dtype=np.int64)
    roa_by_dec_idx = np.zeros(n_mu, dtype=np.float64)
    for (gt_idx, dec_idx), roa_val in zip(pairs, roa_vals):
        if roa_val >= roa_th:
            mask[dec_idx] = True
            gt_matched[dec_idx] = gt_idx
            roa_by_dec_idx[dec_idx] = roa_val

    n_kept = int(mask.sum())
    if n_kept == 0:
        raise ValueError(
            f"No units had RoA >= {roa_th} with any ground-truth unit. "
            "Lower roa_th or check the calibration."
        )

    subset = _subset_cbss_result(result, mask)
    subset.gt_matched_indices = gt_matched[mask].astype(np.int64)
    subset.roa = roa_by_dec_idx[mask]
    return subset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _subset_cbss_result(result: CBSSResult, mask: np.ndarray) -> CBSSResult:
    """Return a new ``CBSSResult`` keeping only units where ``mask`` is True.

    ``mask`` may be a boolean array or an integer index array.
    """
    idx = np.where(mask)[0] if mask.dtype == bool else mask

    def _sel1d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return np.asarray(arr)[idx]

    def _sel2d_axis1(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return np.asarray(arr)[:, idx]

    def _sel4d_axis0(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return np.asarray(arr)[idx]

    sources = _sel2d_axis1(result.sources)
    spikes = _sel2d_axis1(result.spikes)
    new_spikes_dict = {
        new_i: result.spikes_dict[int(old_i)]
        for new_i, old_i in enumerate(idx)
        if int(old_i) in result.spikes_dict
    }

    return CBSSResult(
        sources=sources,
        spikes=spikes,
        spikes_dict=new_spikes_dict,
        sil=_sel1d(result.sil),
        cov=_sel1d(result.cov),
        sep_vectors=_sel2d_axis1(result.sep_vectors),
        whitening=result.whitening,
        extension_mean=result.extension_mean,
        spikes_centr=_sel1d(result.spikes_centr),
        base_centr=_sel1d(result.base_centr),
        pca_components=result.pca_components,
        pca_mean=result.pca_mean,
        pnr=_sel1d(result.pnr),
        dr=_sel1d(result.dr),
        muaps=_sel4d_axis0(result.muaps),
        emg=result.emg,
        timestamps=result.timestamps,
        wh_loss=result.wh_loss,
        sv_loss=result.sv_loss,
        total_loss=result.total_loss,
        centroid_loss=result.centroid_loss,
        wh_trace=result.wh_trace,
        gt_matched_indices=_sel1d(result.gt_matched_indices),
        roa=_sel1d(result.roa),
    )


def _resolve_fs(fs: Optional[float], result: CBSSResult) -> float:
    if fs is not None:
        return float(fs)
    if result.timestamps is not None and len(result.timestamps) > 1:
        diffs = np.diff(result.timestamps)
        return float(1.0 / np.median(diffs))
    raise ValueError(
        "Cannot determine sampling frequency: pass fs= or ensure result.timestamps is set."
    )
