"""Fast fixed-point ICA for CBSS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FastICAResult:
    """Result of one fixed-point ICA solve."""
    w: torch.Tensor
    converged: bool
    n_iter: int
    delta: float
    collapsed: bool


# ---------------------------------------------------------------------------
# Contrast function -- shared math primitive.
# ---------------------------------------------------------------------------

def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Stable log(cosh(x)) = x + softplus(-2x) - log(2).

    Avoids overflow at large |x| that naive log(cosh(x)) would produce.
    tanh(x) is the exact derivative, used in the gradient of sv.

    Args:
        x (torch.Tensor): Input tensor, any shape.

    Returns:
        torch.Tensor: log(cosh(x)), same shape as x.
    """
    return x + F.softplus(-2.0 * x) - math.log(2.0)


@torch.no_grad()
def contrast_fn(
    u: torch.Tensor,
    fn: Literal["logcosh", "square", "cube", "smooth_abs"] = "logcosh",
    *,
    contrast_exp: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a contrast function and its derivative.

    The logcosh branch uses log_cosh for numerical stability.

    Args:
        u (torch.Tensor): Estimated source signal with shape (samples,).
        fn (Literal["logcosh", "square", "cube", "smooth_abs"], optional):
            Which contrast function to use. Defaults to "logcosh".
        contrast_exp (float, optional): Exponent for "smooth_abs". Defaults
            to 3.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (g_u, dg_u) -- contrast value
        and derivative, both with shape (samples,).
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
        contrast_exp:     Exponent for smooth_abs.
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
        g_u, dg_u = contrast_fn(u, fn=contrast_fun_type, contrast_exp=ce)
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
