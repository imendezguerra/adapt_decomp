"""Pure tensor utility functions for adaptive EMG decomposition.

All functions operate without autograd. Heavy functions are wrapped in
@torch.no_grad() to prevent accidental graph construction in the online path.
"""

import math
import torch
import torch.nn.functional as F
from typing import Literal


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
def orthonormalize_rows(
    B: torch.Tensor,
    method: Literal["qr", "none"] = "qr",
) -> torch.Tensor:
    """Dispatcher for row orthonormalization. Use 'none' for ablations only."""
    if method == "qr":
        return orthonormalize_rows_qr(B)
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
    eta_b: float,
    max_rel_delta_b: float,
    min_spikes_for_update: int,
    orthonormalization: str = "qr",
    contrast_scope: str = "batch_based",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Spike-gated separation matrix update with retained contrast error.

    Gradient uses tanh(Y) — the exact derivative of log_cosh.
    eta_b scales the step in the normal operating regime; clip_global_delta
    enforces a hard ceiling on ||ΔB||_F for pathological batches.
    contrast_scope:
        "batch_based" — kappa = log_cosh(Y).mean(dim=0) over all N samples
        "spike_based" — kappa = log_cosh(Y[spike_mask]).mean per source (spike times only)
    B update is always gated: sources with fewer than min_spikes_for_update spikes
    get zero delta regardless of contrast_scope.

    Returns (B_new, diagnostics_dict).
    """
    N, M = Y.shape

    if contrast_scope == "batch_based":
        kappa = log_cosh(Y).mean(dim=0)
    else:
        mask_f = spike_mask.to(Y.dtype)
        counts = mask_f.sum(dim=0)
        kappa = (log_cosh(Y) * mask_f).sum(dim=0) / counts.clamp_min(1.0)

    e_b_raw = kappa - kappa_cal

    mask = spike_mask.to(Y.dtype)
    # tanh is the derivative of log_cosh — used here for the natural-gradient step
    G = mask * torch.tanh(Y)
    spike_counts = mask.sum(dim=0)
    active = spike_counts >= min_spikes_for_update

    grad_B = (G.T @ Z) / spike_counts.clamp_min(1.0)[:, None]
    grad_B = grad_B * active[:, None]

    delta_B = -eta_b * e_b_raw[:, None] * grad_B
    delta_B = clip_global_delta(delta_B, B, max_rel_delta_b, eps)

    B_new = B + delta_B
    B_new = orthonormalize_rows(B_new, method=orthonormalization)

    _nan = torch.tensor(float("nan"), device=Y.device, dtype=Y.dtype)
    diag = {
        "kappa":          torch.where(active, kappa,    _nan),
        "contrast_error": torch.where(active, e_b_raw,  _nan),
        "spike_counts":   spike_counts,
        "active":         active,
        "delta_B_norm":   torch.linalg.norm(delta_B, dim=1),
        "orthogonality_error": torch.linalg.norm(
            B_new @ B_new.T - torch.eye(M, device=B.device, dtype=B.dtype)
        ),
    }
    return B_new, diag
