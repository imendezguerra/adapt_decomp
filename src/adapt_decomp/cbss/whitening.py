"""CBSS whitening, extension, and PCA reduction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


def extend_emg(emg: torch.Tensor, ext_fact: int) -> torch.Tensor:
    """Extend EMG with time-delayed copies for convolutive mixing.

    Args:
        emg:      [T, C] EMG tensor.
        ext_fact: Extension factor.

    Returns:
        [T, C * ext_fact] — column block i holds emg shifted forward by i samples.
    """
    T, C = emg.shape
    out = torch.zeros(T, C * ext_fact, dtype=emg.dtype, device=emg.device)
    for i in range(ext_fact):
        out[i:, C * i : C * (i + 1)] = emg[: T - i, :]
    return out


def pca_reduction(
    emg_ext: torch.Tensor,
    n_components: Optional[int],
    pca_model: Optional[PCA] = None,
) -> Tuple[torch.Tensor, Optional[PCA]]:
    """Optional PCA dimensionality reduction before whitening.

    Returns:
        (reduced_tensor, fitted_pca_model). pca_model is None when n_components is None.
    """
    if n_components is None:
        return emg_ext, None
    d = min(n_components, emg_ext.shape[0], emg_ext.shape[1])
    if d < 1:
        raise ValueError("PCA requires at least one sample and one feature.")
    if pca_model is None:
        pca_model = PCA(n_components=d)
    emg_pca_np = pca_model.fit_transform(emg_ext.cpu().numpy())
    emg_pca = torch.from_numpy(emg_pca_np).to(device=emg_ext.device, dtype=emg_ext.dtype)
    return emg_pca, pca_model


def whiten(
    emg_ext: torch.Tensor,
    method: str = "ZCA",
    regularization: str | float | None = "auto",
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whiten extended (and optionally PCA-reduced) EMG.

    Args:
        emg_ext:       [T, D] input.
        method:        ``"ZCA"`` (preserves spatial structure) or ``"PCA"``.
        regularization: ``"auto"`` (mean of lower eigenvalues), float, or None.
        eps:           Numerical stability constant.

    Returns:
        (emg_whitened [T, D], whitening_matrix W [D, D]).
    """
    cov = (emg_ext.T @ emg_ext) / max(1, emg_ext.shape[0] - 1)
    evals_cpu, evecs_cpu = torch.linalg.eigh(cov.cpu())
    evals_cpu = evals_cpu.clamp(min=0.0)

    if regularization == "auto":
        low_evals = evals_cpu[: len(evals_cpu) // 2]
        reg = float(torch.mean(low_evals).item()) if low_evals.numel() > 0 else 0.0
    elif isinstance(regularization, float):
        reg = float(regularization)
    elif regularization is None:
        reg = 0.0
    else:
        raise ValueError(f"Unknown regularization: {regularization!r}")

    inv_sqrt = (evals_cpu + reg + eps).rsqrt()
    evecs = evecs_cpu.to(emg_ext.device)
    inv_sqrt_t = inv_sqrt.to(emg_ext.device)

    if method == "ZCA":
        W = evecs @ torch.diag(inv_sqrt_t) @ evecs.T
    elif method == "PCA":
        W = torch.diag(inv_sqrt_t) @ evecs.T
    else:
        raise ValueError(f"Unknown whitening method: {method!r}")

    return emg_ext @ W.T, W
