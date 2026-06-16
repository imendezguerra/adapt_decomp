"""Fast fixed-point ICA for CBSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from adapt_decomp.ops import contrast_fn as _contrast_fn


@dataclass
class FastICAResult:
    """Result of one fixed-point ICA solve."""
    w: torch.Tensor
    converged: bool
    n_iter: int
    delta: float
    collapsed: bool


def _normalize(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = w.norm()
    return w / norm if norm > eps else w


def _gram_schmidt_deflate(
    w: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Orthonormalize single vector w against columns of basis."""
    if basis.numel() > 0:
        w = w - basis @ (basis.T @ w)
    return _normalize(w, eps)


def _fast_fixed_point_ica(
    w: torch.Tensor,
    z: torch.Tensor,
    contrast_fun_type: Literal["logcosh", "square", "cube", "smooth_abs"],
    max_iter: int,
    tol: float = 1e-6,
    eps: float = 1e-12,
    deflation_basis: Optional[torch.Tensor] = None,
    contrast_exp: Optional[float] = None,
) -> FastICAResult:
    """Fixed-point ICA iteration for one component.

    Args:
        w:                [D] initial unit vector.
        z:                [D, T] whitened data (features × samples).
        contrast_fun_type: Contrast function name.
        max_iter:         Maximum iterations.
        tol:              Convergence tolerance on |w_new · w| ≈ 1.
        eps:              Numerical stability constant.
        deflation_basis:  Vectors to orthogonalise against at every step.
        contrast_exp:     Exponent for ``smooth_abs``.
    """
    T = z.shape[1]
    converged = False
    collapsed = False
    delta = float("inf")
    n_iter = 0
    ce = contrast_exp if contrast_exp is not None else 3.0

    for iter_idx in range(max_iter):
        n_iter = iter_idx + 1
        u = w @ z  # [T]
        g_u, dg_u = _contrast_fn(u, fn=contrast_fun_type, contrast_exp=ce)
        w_new = (z @ g_u) / T - dg_u.mean() * w
        if not torch.isfinite(w_new).all() or w_new.norm() <= eps:
            collapsed = True
            w = w_new
            break
        w_new = _normalize(w_new, eps)
        if deflation_basis is not None and deflation_basis.numel() > 0:
            w_new = _gram_schmidt_deflate(w_new, deflation_basis, eps)
            if not torch.isfinite(w_new).all() or w_new.norm() <= eps:
                collapsed = True
                w = w_new
                break
        delta = float((torch.abs(w_new @ w) - 1.0).abs().item())
        if delta < tol:
            w = w_new
            converged = True
            break
        w = w_new

    return FastICAResult(w=w, converged=converged, n_iter=n_iter, delta=delta, collapsed=collapsed)
