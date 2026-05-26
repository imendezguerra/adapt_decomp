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
    tanh(x) is the exact derivative, used in the gradient of B.
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
def orthonormalize_rows_qr(B: torch.Tensor) -> torch.Tensor:
    """QR-based row orthonormalization.

    B: [M, D] — returns matrix with approximately orthonormal rows.
    Uses QR on B.T so rows of B map to columns of Q, which are orthonormal.
    Avoids eigendecomposition, keeping online runtime predictable.
    """
    Q, _ = torch.linalg.qr(B.T, mode="reduced")
    return Q.T


@torch.no_grad()
def orthonormalize_rows_gram_schmidt(B: torch.Tensor) -> torch.Tensor:
    """Classical Gram-Schmidt row orthonormalization.

    B: [M, D] — returns matrix with orthonormal rows.
    Sequentially projects each row onto the complement of all prior rows.
    Less numerically stable than QR for large M; prefer QR in production.
    """
    M = B.shape[0]
    Q = torch.empty_like(B)
    for i in range(M):
        v = B[i].clone()
        for j in range(i):
            v = v - (v @ Q[j]) * Q[j]
        Q[i] = v / torch.linalg.norm(v).clamp_min(1e-8)
    return Q


@torch.no_grad()
def orthonormalize_rows(
    B: torch.Tensor,
    method: Literal["qr", "gram_schmidt", "none"] = "qr",
) -> torch.Tensor:
    """Dispatcher for row orthonormalization. Use 'none' for ablations only."""
    if method == "qr":
        return orthonormalize_rows_qr(B)
    if method == "gram_schmidt":
        return orthonormalize_rows_gram_schmidt(B)
    if method == "none":
        return B
    raise ValueError(f"Unknown orthonormalization method: {method!r}")


@torch.no_grad()
def find_peaks_multisource(
    Y: torch.Tensor,
    min_dist: int,
    peak_power: float = 2.0,
    strict: bool = True,
    use_abs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized multi-source peak detector using 1-D max-pool NMS.

    Y: [N, M] source matrix.

    Strict mode suppresses flat plateaus so tied runs cannot emit multiple peaks.
    Returns:
        peak_mask: [N, M] bool — candidate peak locations
        Y_det:     [N, M] — detection-domain values (|Y|^peak_power or Y^peak_power)
    """
    N, M = Y.shape
    Y_det = Y.abs().pow(peak_power) if use_abs else Y.pow(peak_power)

    win = 2 * min_dist + 1
    # Apply max-pool across time for all sources simultaneously: input [1, M, N]
    pooled = (
        F.max_pool1d(
            Y_det.T.unsqueeze(0).float(),
            kernel_size=win,
            stride=1,
            padding=min_dist,
        )
        .squeeze(0)
        .T.to(Y_det.dtype)
    )

    if strict:
        strict_mask = torch.zeros_like(Y_det, dtype=torch.bool)
        if N >= 3:
            strict_mask[1:-1] = (Y_det[1:-1] > Y_det[:-2]) & (Y_det[1:-1] > Y_det[2:])
        peak_mask = strict_mask & (Y_det == pooled) & (Y_det > 0)
    else:
        peak_mask = (Y_det == pooled) & (Y_det > 0)

    return peak_mask, Y_det


@torch.no_grad()
def classify_peaks_from_adaptive_centroids(
    Y_det: torch.Tensor,
    peak_mask: torch.Tensor,
    spike_centroid: torch.Tensor,
    base_centroid: torch.Tensor,
) -> torch.Tensor:
    """Classify peak candidates as spikes using adaptive online centroids.

    Threshold is the midpoint between base and spike centroids.
    Uses spike_centroid and base_centroid (adaptive), NOT frozen calibration values.

    Returns spike_mask: [N, M] bool.
    """
    threshold = base_centroid + 0.5 * (spike_centroid - base_centroid)
    return peak_mask & (Y_det > threshold[None, :])


@torch.no_grad()
def update_centroids_from_peaks(
    Y: torch.Tensor,
    peak_mask: torch.Tensor,
    spike_mask: torch.Tensor,
    spike_centroid: torch.Tensor,
    base_centroid: torch.Tensor,
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

    Returns (new_spike_centroid, new_base_centroid), both shape [M].
    """
    Y_det = Y.abs().pow(peak_power) if use_abs_for_detection else Y.pow(peak_power)
    base_mask = peak_mask & ~spike_mask
    M = Y.shape[1]

    new_spike = spike_centroid.clone()
    new_base = base_centroid.clone()

    for j in range(M):
        spike_vals = Y_det[spike_mask[:, j], j]
        base_vals = Y_det[base_mask[:, j], j]

        if spike_vals.numel() >= min_spikes_for_centroid:
            candidate = (
                centroid_momentum * spike_centroid[j]
                + (1.0 - centroid_momentum) * spike_vals.mean()
            )
            new_spike[j] = candidate

        if base_vals.numel() >= min_base_peaks_for_centroid:
            candidate = (
                centroid_momentum * base_centroid[j]
                + (1.0 - centroid_momentum) * base_vals.mean()
            )
            new_base[j] = candidate

    # Revert any source where the ordering invariant would be violated
    valid = new_spike > (new_base + eps)
    new_spike = torch.where(valid, new_spike, spike_centroid)
    new_base = torch.where(valid, new_base, base_centroid)

    return new_spike, new_base


@torch.no_grad()
def update_B_spike_gated(
    B: torch.Tensor,
    Z: torch.Tensor,
    Y: torch.Tensor,
    kappa_cal: torch.Tensor,
    spike_mask: torch.Tensor,
    max_rel_delta_b: float,
    min_spikes_for_update: int,
    orthonormalization: str = "qr",
    contrast_scope: str = "batch_based",
    eps: float = 1e-8,
    sigma_kappa_cal: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Separation matrix update with retained contrast error.

    Gradient uses tanh(Y) — the exact derivative of log_cosh.
    clip_rowwise_delta enforces a hard ceiling on each row of ΔB for pathological batches.

    contrast_scope controls both kappa estimation and gradient direction:
        "batch_based" — kappa = log_cosh(Y).mean(dim=0) over all N samples;
                        gradient = tanh(Y).T @ Z / N  (full-batch ICA step,
                        decoupled from spike detection)
        "spike_based" — kappa = log_cosh(Y[spike_mask]).mean per source;
                        gradient is spike-gated; sources with fewer than
                        min_spikes_for_update spikes get zero delta

    Returns (B_new, diagnostics_dict).
    """
    N, M = Y.shape

    sigma_kappa_cal = sigma_kappa_cal if sigma_kappa_cal is not None else torch.ones_like(kappa_cal)
    if contrast_scope == "batch_based":
        kappa = log_cosh(Y).mean(dim=0)
        e_b = (kappa - kappa_cal) / sigma_kappa_cal

        # Full-batch ICA natural-gradient direction — no spike gating
        G = torch.tanh(Y)
        grad_B = (G.T @ Z) / N
        active = torch.ones(M, dtype=torch.bool, device=Y.device)
        spike_counts = spike_mask.to(Y.dtype).sum(dim=0)   # kept for diagnostics only
    else:
        mask_f = spike_mask.to(Y.dtype)
        spike_counts = mask_f.sum(dim=0)
        kappa = (log_cosh(Y) * mask_f).sum(dim=0) / spike_counts.clamp_min(1.0)
        e_b = (kappa - kappa_cal) / sigma_kappa_cal

        # Spike-gated natural-gradient direction
        G = mask_f * torch.tanh(Y)
        active = spike_counts >= min_spikes_for_update
        grad_B = (G.T @ Z) / spike_counts.clamp_min(1.0)[:, None]
        grad_B = grad_B * active[:, None]

    delta_B = -e_b[:, None] * grad_B
    delta_B = clip_rowwise_delta(delta_B, B, max_rel_delta_b, eps)

    B_new = B + delta_B
    B_new = orthonormalize_rows(B_new, method=orthonormalization)

    _nan = torch.tensor(float("nan"), device=Y.device, dtype=Y.dtype)
    diag = {
        "kappa":          torch.where(active, kappa,  _nan),
        "contrast_error": torch.where(active, e_b,    _nan),
        "spike_counts":   spike_counts,
        "active":         active,
        "delta_B_norm":   torch.linalg.norm(delta_B, dim=1),
        "orthogonality_error": torch.linalg.norm(
            B_new @ B_new.T - torch.eye(M, device=B.device, dtype=B.dtype)
        ),
    }
    return B_new, diag


@torch.no_grad()
def gate_spikes_by_iqr(
    Y: torch.Tensor,
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
    Y_det = Y.abs().pow(peak_power) if use_abs_for_detection else Y.pow(peak_power)
    upper_gate = Q75_cal + gate_factor * IQR_cal.clamp_min(eps)  # [M]
    return spike_mask & (Y_det <= upper_gate[None, :])            # [N, M] bool


