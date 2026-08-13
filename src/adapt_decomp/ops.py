"""Pure tensor utility functions for adaptive EMG decomposition.

All functions operate without autograd. Heavy functions are wrapped in
@torch.no_grad() to prevent accidental graph construction in the online path.
"""

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Stable log(cosh(x)) = x + softplus(-2x) - log(2).

    Avoids overflow at large |x| that naive log(cosh(x)) would produce.
    tanh(x) is the exact derivative, used in the gradient of sv.
    """
    return x + F.softplus(-2.0 * x) - math.log(2.0)


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
def find_peaks_multisource(
    sources: torch.Tensor,
    min_dist: int,
    peak_power: float = 2.0,
    use_abs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized multi-source peak detector using 1-D max-pool NMS.

    sources: [N, M] source matrix.

    Suppresses flat plateaus (requires a strict local max on both neighbours)
    so tied runs cannot emit multiple peaks.
    Returns:
        peak_mask:   [N, M] bool — candidate peak locations
        sources_det: [N, M] — detection-domain values (|sources|^peak_power or sources^peak_power)
    """
    N, M = sources.shape
    sources_det = sources.abs().pow(peak_power) if use_abs else sources.pow(peak_power)

    win = 2 * min_dist + 1
    # Apply max-pool across time for all sources simultaneously: input [1, M, N]
    pooled = (
        F.max_pool1d(
            sources_det.T.unsqueeze(0).float(),
            kernel_size=win,
            stride=1,
            padding=min_dist,
        )
        .squeeze(0)
        .T.to(sources_det.dtype)
    )

    strict_mask = torch.zeros_like(sources_det, dtype=torch.bool)
    if N >= 3:
        strict_mask[1:-1] = (sources_det[1:-1] > sources_det[:-2]) & (sources_det[1:-1] > sources_det[2:])
    peak_mask = strict_mask & (sources_det == pooled) & (sources_det > 0)

    return peak_mask, sources_det


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
    lr_alone: bool = False,
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

    lr_alone=True drops the signed e_sv factor entirely, flips the sign, and skips the
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

    sigma_kappa_cal = sigma_kappa_cal if sigma_kappa_cal is not None else torch.ones_like(kappa_cal)
    if contrast_scope == "batch_based":
        kappa = log_cosh(sources).mean(dim=0)
        e_sv = (kappa - kappa_cal) / sigma_kappa_cal

        # Full-batch ICA natural-gradient direction — no spike gating
        G = torch.tanh(sources)
        grad_sv = (G.T @ Z) / N
        active = torch.ones(M, dtype=torch.bool, device=sources.device)
        spike_counts = spike_mask.to(sources.dtype).sum(dim=0)   # kept for diagnostics only
    else:
        mask_f = spike_mask.to(sources.dtype)
        spike_counts = mask_f.sum(dim=0)
        kappa = (log_cosh(sources) * mask_f).sum(dim=0) / spike_counts.clamp_min(1.0)
        e_sv = (kappa - kappa_cal) / sigma_kappa_cal

        # Spike-gated natural-gradient direction
        G = mask_f * torch.tanh(sources)
        active = spike_counts >= 1
        grad_sv = (G.T @ Z) / spike_counts.clamp_min(1.0)[:, None]
        grad_sv = grad_sv * active[:, None]

    # Normalize the natural-gradient direction to unit scale via an EMA of its own
    # norm (per unit), then scale by lr_sv and the full signed e_sv -- restores step
    # size proportional to how wrong the model is, instead of a fixed-size step.
    # (lr_alone bypasses this normalization entirely -- see docstring above; the EMA
    # is still tracked/returned below so state stays consistent if the flag changes
    # between runs.)
    grad_sv_norm = torch.linalg.norm(grad_sv, dim=1)   # [M], instantaneous
    new_ema_gradnorm_sv = (
        grad_sv_norm.detach() if ema_gradnorm_sv is None
        else (ema_alpha * ema_gradnorm_sv + (1 - ema_alpha) * grad_sv_norm).detach()
    )
    sv_row_norm = torch.linalg.norm(sv, dim=1, keepdim=True)
    if lr_alone:
        # Main (v1)'s raw, un-normalized natural-gradient ASCENT: no EMA
        # direction-normalization, no error term (see docstring above).
        delta_sv_target = lr_sv * grad_sv
    else:
        delta_sv_target = (
            -lr_sv * sv_row_norm * e_sv[:, None]
            * grad_sv / (new_ema_gradnorm_sv[:, None] + eps)
        )
    delta_sv_raw_norm = torch.linalg.norm(delta_sv_target, dim=1)   # pre-safety-clip target norm
    delta_sv = clip_rowwise_delta(delta_sv_target, sv, max_rel_delta_sv, eps)  # rare safety net

    sv_new = sv + delta_sv
    sv_new = orthonormalize_rows_qr(sv_new)

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
def kmeans2_1d(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Globally optimal 1-D k=2 clustering via sorted threshold sweep.

    For 1-D data the optimal 2-means partition is always a single threshold.
    Sorting + prefix sums evaluate all N-1 split points in O(N) after the
    O(N log N) sort — guaranteed global optimum, no random restarts needed.

    Returns:
        labels:    [N] LongTensor — 0 for lower cluster, 1 for upper cluster.
        centroids: [2] tensor    — [c_lower, c_upper].
    """
    sorted_vals, sort_idx = vals.sort()
    N = sorted_vals.shape[0]
    cs  = sorted_vals.cumsum(0)
    cs2 = (sorted_vals ** 2).cumsum(0)
    ks  = torch.arange(1, N, device=vals.device, dtype=vals.dtype)
    n0, n1  = ks, N - ks
    s0, s2_0 = cs[:-1], cs2[:-1]
    s1, s2_1 = cs[-1] - s0, cs2[-1] - s2_0
    cost = (s2_0 - s0 ** 2 / n0) + (s2_1 - s1 ** 2 / n1)
    k = int(cost.argmin().item()) + 1
    labels_sorted = torch.zeros(N, dtype=torch.long, device=vals.device)
    labels_sorted[k:] = 1
    labels = torch.empty_like(labels_sorted)
    labels[sort_idx] = labels_sorted
    c0 = sorted_vals[:k].median()
    c1 = sorted_vals[k:].median()
    return labels, torch.stack([c0, c1])


@torch.no_grad()
def contrast_fn(
    u: torch.Tensor,
    fn: Literal["logcosh", "square", "cube", "smooth_abs"] = "logcosh",
    *,
    contrast_exp: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a contrast function and its derivative.

    The ``logcosh`` branch uses :func:`log_cosh` for numerical stability.

    Args:
        u:            [T] estimated source signal.
        fn:           Which contrast function to use.
        contrast_exp: Exponent for ``smooth_abs`` (default 3.0).

    Returns:
        ``(g_u, dg_u)`` — contrast value and derivative, both ``[T]``.
    """
    if fn == "logcosh":
        tanh_u = torch.tanh(u)
        return tanh_u, 1.0 - tanh_u ** 2
    if fn == "square":
        return u ** 2, 2.0 * u
    if fn == "smooth_abs":
        eps = 1e-3
        a = contrast_exp
        g_u  = (eps + u ** 2) ** ((a - 3) / 2) * (a * u ** 2 + eps)
        dg_u = (a - 1) * u * (eps + u ** 2) ** ((a - 5) / 2) * (a * u ** 2 + 3 * eps)
        return g_u, dg_u
    # cube
    return u ** 3, 3.0 * u ** 2


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
