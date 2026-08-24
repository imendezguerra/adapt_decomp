"""Pure tensor utility functions for online adaptive EMG decomposition.

All functions operate without autograd. Heavy functions are wrapped in
@torch.no_grad() to prevent accidental graph construction in the online path.

log_cosh/contrast_fn (used by update_sv_spike_gated below) and
find_peaks_multisource (used by find_peaks_multisource-calling code below)
are NOT defined here even though they're pure tensor math too: log_cosh/
contrast_fn live in adapt_decomp.cbss.ica (their primary consumer, the
fixed-point ICA solve) and find_peaks_multisource lives in
adapt_decomp.spikes.detection (spike-detection code, consumed by both cbss
and adaptation) -- both are imported below rather than duplicated, so this
module stays adaptation-only and cbss/spikes never need to depend on it.

stable_cov, conversely, moved IN from adapt_decomp.spikes.metrics: despite
living in a "shared" module, its only real consumer was always
adaptation/data_structures.py::Decomposition's covariance-with-shrinkage
computations -- cbss/whitening.py::whiten computes its own covariance with a
different (unshrunk, eigenvalue-regularised) scheme and never called it. Since
nothing outside adaptation actually used it, it belongs with the rest of this
module's adaptation-only tensor primitives instead of spikes/, which is
reserved for genuinely cbss+adaptation-shared spike-detection/metrics code.
"""

from typing import Literal, Optional, Tuple

import torch

from adapt_decomp.cbss.ica import log_cosh
from adapt_decomp.spikes.detection import find_peaks_multisource

__all__ = [
    "clip_global_delta",
    "clip_rowwise_delta",
    "stable_cov",
    "orthonormalize_rows_qr",
    "orthonormalize_rows_gram_schmidt",
    "orthonormalize_rows",
    "find_peaks_multisource",
    "classify_peaks_from_adaptive_centroids",
    "update_centroids_from_peaks",
    "compute_contrast_error",
    "update_sv_spike_gated",
    "gate_spikes_by_iqr",
]


@torch.no_grad()
def stable_cov(
    X: torch.Tensor,
    rowvar: bool = False,
    rho: Optional[float] = None,
    I: Optional[torch.Tensor] = None,
    ddof: int = 1,
) -> torch.Tensor:
    """Compute a symmetric, optionally shrunk, sample covariance matrix.

    Used wherever a covariance-with-shrinkage-toward-identity is needed --
    e.g. Decomposition's Rz_cal/Rz computations in data_structures.py.
    Accepts either a single 2-D matrix or a batch of them (leading batch
    dimension), so the same call covers both a one-off covariance and a
    batched sliding-window sweep (previously bmm'd inline).

    Args:
        X (torch.Tensor): Data with shape (..., samples, feats) if rowvar is
            False, or (..., feats, samples) if rowvar is True. A leading
            batch dimension (or several) is supported and broadcasts through.
        rowvar (bool, optional): Whether each row of X is a variable (True)
            or each column is (False). Defaults to False.
        rho (Optional[float], optional): Shrinkage weight toward the
            identity matrix, in [0, 1]. None disables shrinkage. Defaults
            to None.
        I (Optional[torch.Tensor], optional): Identity matrix to shrink
            toward, with shape (feats, feats). Defaults to None, which
            builds one on X's device/dtype when rho is set.
        ddof (int, optional): Delta degrees of freedom -- the covariance
            divisor is samples - ddof (floored at 1). 1 (default)
            gives the usual unbiased sample covariance; pass 0 to divide
            by the raw sample count instead (e.g. to match a caller's
            existing, already-validated normalisation exactly).

    Returns:
        torch.Tensor: Symmetric covariance matrix with shape (..., feats, feats).
    """
    if rowvar:
        samples = X.shape[-1]
        Xc = X - X.mean(dim=-1, keepdim=True)
        C = (Xc @ Xc.transpose(-1, -2)) / max(1, samples - ddof)
    else:
        samples = X.shape[-2]
        Xc = X - X.mean(dim=-2, keepdim=True)
        C = (Xc.transpose(-1, -2) @ Xc) / max(1, samples - ddof)

    # Make symmetric
    C = 0.5 * (C + C.transpose(-1, -2))

    # Shrinkage towards identity
    if rho is not None:
        if I is None:
            I = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
        C = (1 - rho) * C + rho * I
    return C


@torch.no_grad()
def clip_global_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    max_rel_delta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale delta so its global norm does not exceed max_rel_delta * ||reference||."""
    delta_norm = torch.linalg.norm(delta)
    ref_norm = torch.linalg.norm(reference)
    max_norm = max_rel_delta * (ref_norm + eps)
    scale = torch.clamp(max_norm / (delta_norm + eps), max=1.0)
    return delta * scale


@torch.no_grad()
def clip_rowwise_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    max_rel_delta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale each row of delta so its norm does not exceed max_rel_delta * ||reference_row||."""
    delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    ref_norm = torch.linalg.norm(reference, dim=1, keepdim=True)
    max_norm = max_rel_delta * (ref_norm + eps)
    scale = torch.clamp(max_norm / (delta_norm + eps), max=1.0)
    return delta * scale


@torch.no_grad()
def orthonormalize_rows_qr(sv: torch.Tensor) -> torch.Tensor:
    """QR-based row orthonormalization.

    sv: [M, D] — returns matrix with approximately orthonormal rows.
    Uses QR on sv.T so rows of sv map to columns of Q, which are orthonormal.
    Avoids eigendecomposition, keeping online runtime predictable.
    """
    Q, _ = torch.linalg.qr(sv.T, mode="reduced")
    return Q.T


@torch.no_grad()
def orthonormalize_rows_gram_schmidt(sv: torch.Tensor) -> torch.Tensor:
    """Classical Gram-Schmidt row orthonormalization.

    sv: [M, D] — returns matrix with orthonormal rows.
    Sequentially projects each row onto the complement of all prior rows.
    Less numerically stable than QR for large M; prefer QR in production.
    """
    M = sv.shape[0]
    Q = torch.empty_like(sv)
    for i in range(M):
        v = sv[i].clone()
        for j in range(i):
            v = v - (v @ Q[j]) * Q[j]
        Q[i] = v / torch.linalg.norm(v).clamp_min(1e-8)
    return Q


@torch.no_grad()
def orthonormalize_rows(
    sv: torch.Tensor,
    method: Literal["qr", "gram_schmidt", "none"] = "qr",
) -> torch.Tensor:
    """Dispatcher for row orthonormalization. Use 'none' for ablations only."""
    if method == "qr":
        return orthonormalize_rows_qr(sv)
    if method == "gram_schmidt":
        return orthonormalize_rows_gram_schmidt(sv)
    if method == "none":
        return sv
    raise ValueError(f"Unknown orthonormalization method: {method!r}")


@torch.no_grad()
def classify_peaks_from_adaptive_centroids(
    sources_det: torch.Tensor,
    peak_mask: torch.Tensor,
    spike_centroids: torch.Tensor,
    base_centroids: torch.Tensor,
) -> torch.Tensor:
    """Classify peak candidates as spikes using adaptive online centroids.

    Threshold is the midpoint between base and spike centroids.
    Uses spike_centroids and base_centroids (adaptive), NOT frozen calibration values.

    Returns spike_mask: [N, M] bool.
    """
    threshold = base_centroids + 0.5 * (spike_centroids - base_centroids)
    return peak_mask & (sources_det > threshold[None, :])


@torch.no_grad()
def update_centroids_from_peaks(
    sources: torch.Tensor,
    peak_mask: torch.Tensor,
    spike_mask: torch.Tensor,
    spike_centroids: torch.Tensor,
    base_centroids: torch.Tensor,
    peak_power: float = 2.0,
    centroid_momentum: float = 0.95,
    min_spikes_for_centroid: int = 1,
    min_base_peaks_for_centroid: int = 1,
    use_abs_for_detection: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update adaptive spike and baseline centroids via EMA from detected peaks.

    Source loop is intentional: centroid update is inherently per-source stateful.
    Only updates a centroid when enough peaks exist; reverts if spike <= base.

    Returns (new_spike_centroids, new_base_centroids), both shape [M].
    """
    sources_det = sources.abs().pow(peak_power) if use_abs_for_detection else sources.pow(peak_power)
    base_mask = peak_mask & ~spike_mask
    M = sources.shape[1]

    new_spike = spike_centroids.clone()
    new_base = base_centroids.clone()

    for j in range(M):
        spike_vals = sources_det[spike_mask[:, j], j]
        base_vals = sources_det[base_mask[:, j], j]

        if spike_vals.numel() >= min_spikes_for_centroid:
            candidate = (
                centroid_momentum * spike_centroids[j]
                + (1.0 - centroid_momentum) * spike_vals.mean()
            )
            new_spike[j] = candidate

        if base_vals.numel() >= min_base_peaks_for_centroid:
            candidate = (
                centroid_momentum * base_centroids[j]
                + (1.0 - centroid_momentum) * base_vals.mean()
            )
            new_base[j] = candidate

    # Revert any source where the ordering invariant would be violated
    valid = new_spike > (new_base + eps)
    new_spike = torch.where(valid, new_spike, spike_centroids)
    new_base = torch.where(valid, new_base, base_centroids)

    return new_spike, new_base


@torch.no_grad()
def compute_contrast_error(
    sources: torch.Tensor,
    spike_mask: torch.Tensor,
    kappa_cal: torch.Tensor,
    contrast_scope: str = "batch_based",
    sigma_kappa_cal: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the z-scored contrast error against calibration.

    Shared by update_sv_spike_gated (where e_sv drives the natural-gradient
    step) and _compute_sv_diag in adaptation/core.py (where it is only
    logged, on the adapt_sv=False path) -- one formula instead of two
    independently-maintained copies.

    contrast_scope controls the estimator:
        "batch_based" — kappa = log_cosh(sources).mean(dim=0) over all N
                        samples; every source is reported active.
        "spike_based" — kappa = log_cosh(sources[spike_mask]).mean per
                        source; a source with zero trusted spikes is
                        reported inactive (kappa/e_sv are still computed,
                        just not meaningful for it -- callers decide what
                        to do with active).

    Args:
        sources (torch.Tensor): Estimated source signals, with shape (N, M).
        spike_mask (torch.Tensor): Trusted spike mask, with shape (N, M),
            bool. Only consumed when contrast_scope is "spike_based".
        kappa_cal (torch.Tensor): Calibration contrast mean, with shape (M,).
        contrast_scope (str, optional): "batch_based" or "spike_based".
            Defaults to "batch_based".
        sigma_kappa_cal (Optional[torch.Tensor], optional): Calibration
            contrast std, with shape (M,). Defaults to None, which disables
            z-scoring (equivalent to a std of 1).
        eps (float, optional): Floor applied to sigma_kappa_cal before
            dividing, so a degenerate (near-zero) calibration std can't blow
            up e_sv. Defaults to 1e-8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (kappa, e_sv, active, spike_counts), each with shape (M,) --
        kappa/e_sv float, active bool, spike_counts float.
    """
    sigma_kappa_cal = sigma_kappa_cal if sigma_kappa_cal is not None else torch.ones_like(kappa_cal)
    spike_counts = spike_mask.to(sources.dtype).sum(dim=0)
    if contrast_scope == "batch_based":
        kappa = log_cosh(sources).mean(dim=0)
        active = torch.ones(sources.shape[1], dtype=torch.bool, device=sources.device)
    else:
        mask_f = spike_mask.to(sources.dtype)
        kappa = (log_cosh(sources) * mask_f).sum(dim=0) / spike_counts.clamp_min(1.0)
        active = spike_counts >= 1
    e_sv = (kappa - kappa_cal) / sigma_kappa_cal.clamp_min(eps)
    return kappa, e_sv, active, spike_counts


@torch.no_grad()
def update_sv_spike_gated(
    sv: torch.Tensor,
    Z: torch.Tensor,
    sources: torch.Tensor,
    kappa_cal: torch.Tensor,
    spike_mask: torch.Tensor,
    max_rel_delta_sv: float,
    contrast_scope: str = "batch_based",
    eps: float = 1e-8,
    sigma_kappa_cal: Optional[torch.Tensor] = None,
    lr_sv: float = 1e-3,
    ema_gradnorm_sv: Optional[torch.Tensor] = None,
    ema_alpha: float = 0.95,
    lr_mode: Literal["fixed", "rel_error"] = "rel_error",
) -> tuple[torch.Tensor, dict]:
    """Separation matrix update with retained contrast error.

    Gradient uses tanh(sources) — the exact derivative of log_cosh.

    contrast_scope controls both kappa estimation and gradient direction:
        "batch_based" — kappa = log_cosh(sources).mean(dim=0) over all N samples;
                        gradient = tanh(sources).T @ Z / N  (full-batch ICA step,
                        decoupled from spike detection)
        "spike_based" — kappa = log_cosh(sources[spike_mask]).mean per source;
                        gradient is spike-gated; sources with fewer than 1
                        trusted spike get zero delta

    The reported contrast_error for inactive sources (spike_based mode, no
    trusted spikes) is always a fixed -3.0 penalty rather than exclusion
    (NaN) from the loss. Does not affect grad_sv, which stays masked to zero
    for inactive sources regardless: only the loss/diagnostic value changes,
    never the sv update.

    The natural-gradient direction (grad_sv, per row) is normalized to unit scale by
    an EMA of its own norm (ema_gradnorm_sv, updated here and returned in diag for the
    caller to persist) before being scaled by lr_sv and the *signed* e_sv -- so the
    applied step tracks how wrong the model actually is, rather than always being
    clipped to a fixed size (max_rel_delta_sv is now a rare safety net only; pass
    ema_gradnorm_sv=None to seed cold, e.g. the first batch of a fresh trial).

    lr_mode="fixed" drops the signed e_sv factor entirely, flips the sign, and skips the
    EMA direction-normalization below -- giving the raw, un-normalized
    natural-gradient ASCENT sv += lr_sv * grad_sv, reproducing main (v1)'s
    fixed-learning-rate sv update (main had no error term and no normalization
    either). The step shrinks on its own as sv approaches its fixed point
    (grad_sv -> 0), the same self-damping behaviour main relied on for stability --
    the EMA-normalized branch below would otherwise force a constant relative step
    every batch regardless of convergence. The sign flip (vs. the default branch's
    -e_sv*grad_sv) is required: main's update always increases contrast, whereas the
    default branch is a signed control step that can push contrast either way
    depending on e_sv's sign. Because this step is unnormalized, lr_sv means something
    different here than in the default branch and must be tuned separately.

    Returns (sv_new, diagnostics_dict).
    """
    N, M = sources.shape

    kappa, e_sv, active, spike_counts = compute_contrast_error(
        sources, spike_mask, kappa_cal, contrast_scope, sigma_kappa_cal, eps,
    )
    if contrast_scope == "batch_based":
        # Full-batch ICA natural-gradient direction — no spike gating
        G = torch.tanh(sources)
        grad_sv = (G.T @ Z) / N
    else:
        # Spike-gated natural-gradient direction
        mask_f = spike_mask.to(sources.dtype)
        G = mask_f * torch.tanh(sources)
        grad_sv = (G.T @ Z) / spike_counts.clamp_min(1.0)[:, None]
        grad_sv = grad_sv * active[:, None]

    # Compute the gradient norm via EMA for relative error normalisation
    grad_sv_norm = torch.linalg.norm(grad_sv, dim=1)   # [M], instantaneous
    new_ema_gradnorm_sv = (
        grad_sv_norm.detach() if ema_gradnorm_sv is None
        else (ema_alpha * ema_gradnorm_sv + (1 - ema_alpha) * grad_sv_norm).detach()
    )
    sv_row_norm = torch.linalg.norm(sv, dim=1, keepdim=True)
    if lr_mode == "fixed": # Fixed learning rate
        delta_sv_target = lr_sv * grad_sv
    else: # learning rate with relative error and norm normalisation via EMA
        delta_sv_target = (
            -lr_sv * sv_row_norm * e_sv[:, None]
            * grad_sv / (new_ema_gradnorm_sv[:, None] + eps)
        )

    # Clip update 
    delta_sv_raw_norm = torch.linalg.norm(delta_sv_target, dim=1)   # pre-safety-clip target norm
    delta_sv = clip_rowwise_delta(delta_sv_target, sv, max_rel_delta_sv, eps)  # rare safety net

    # Compute new separation vectors and orthonormalise
    sv_new = sv + delta_sv
    sv_new = orthonormalize_rows_qr(sv_new)

    # Diagnostics
    _nan = torch.tensor(float("nan"), device=sources.device, dtype=sources.dtype)
    _fallback = torch.full_like(e_sv, -3.0)
    diag = {
        "kappa":          torch.where(active, kappa,  _nan),        
        "contrast_error": torch.where(active, e_sv,   _fallback),  
        "spike_counts":   spike_counts,
        "active":         active,
        "delta_sv_norm":     torch.linalg.norm(delta_sv, dim=1),
        "delta_sv_raw_norm": delta_sv_raw_norm,
        "ema_gradnorm_sv":    new_ema_gradnorm_sv,
        "orthogonality_error": torch.linalg.norm(
            sv_new @ sv_new.T - torch.eye(M, device=sv.device, dtype=sv.dtype)
        ),
    }
    return sv_new, diag


@torch.no_grad()
def gate_spikes_by_iqr(
    sources: torch.Tensor,
    spike_mask: torch.Tensor,
    Q75_cal: torch.Tensor,
    IQR_cal: torch.Tensor,
    gate_factor: float,
    peak_power: float = 2.0,
    use_abs_for_detection: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return trusted_spike_mask: spike_mask with amplitude outliers excluded.

    Outliers are spikes whose detection-domain amplitude exceeds the Tukey upper
    fence Q75_cal + gate_factor * IQR_cal. Excluded spikes still appear in the
    output spike train but do not update centroids or separation vectors.
    """
    sources_det = sources.abs().pow(peak_power) if use_abs_for_detection else sources.pow(peak_power)
    upper_gate = Q75_cal + gate_factor * IQR_cal.clamp_min(eps)  # [M]
    return spike_mask & (sources_det <= upper_gate[None, :])            # [N, M] bool
