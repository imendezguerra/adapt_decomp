"""CBSS whitening, extension, and PCA reduction."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from sklearn.decomposition import PCA


def extend_emg(
    emg: torch.Tensor,
    ext_fact: int,
    ext_mode: Literal["block", "toeplitz"] = "block",
) -> torch.Tensor:
    """Extend EMG with time-delayed copies for convolutive mixing.

    Args:
        emg:            [T, C] EMG tensor.
        ext_fact:       Extension factor.
        ext_mode: "block" (default) — column block i holds ALL channels
            shifted by i samples: cols = [ch0..chC @ delay0, ch0..chC @ delay1, ...].
            "toeplitz" — each channel's own ext_fact delayed copies are kept
            together, so each channel's block of columns is itself a Toeplitz
            (constant-diagonal) matrix: cols = [ch0 @ delay0..delay(L-1), ch1 @
            delay0..delay(L-1), ...]. Standard convolutive-EMG-mixing convention
            (Negro et al. 2016). Must match Config.ext_mode used downstream
            by any online adaptation applied to this calibration's output.

    Returns:
        [T, C * ext_fact], column order set by ext_mode.
    """
    T, C = emg.shape
    out = torch.zeros(T, C * ext_fact, dtype=emg.dtype, device=emg.device)
    for i in range(ext_fact):
        out[i:, C * i : C * (i + 1)] = emg[: T - i, :]
    if ext_mode == "toeplitz":
        out = out.view(T, ext_fact, C).permute(0, 2, 1).reshape(T, C * ext_fact)
    elif ext_mode != "block":
        raise ValueError(
            f"Unknown ext_mode: {ext_mode!r}. Expected 'block' or 'toeplitz'."
        )
    return out


def pca_reduction(
    emg_ext: torch.Tensor,
    n_components: Optional[int],
    pca_model: Optional[PCA] = None,
) -> Tuple[torch.Tensor, Optional[PCA]]:
    """Optional PCA dimensionality reduction before whitening.

    If pca_model is provided (already fitted), it is reused via .transform()
    rather than refit, so the same projection learned at calibration time can
    be applied to new data. If pca_model is None, a fresh PCA is fit here.

    Returns:
        (reduced_tensor, fitted_pca_model). pca_model is None when n_components is None.
    """
    if n_components is None:
        return emg_ext, None
    d = min(n_components, emg_ext.shape[0], emg_ext.shape[1])
    if d < 1:
        raise ValueError("PCA requires at least one sample and one feature.")
    emg_ext_np = emg_ext.cpu().numpy()
    if pca_model is None:
        pca_model = PCA(n_components=d)
        emg_pca_np = pca_model.fit_transform(emg_ext_np)
    else:
        emg_pca_np = pca_model.transform(emg_ext_np)
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
    elif isinstance(regularization, (int, float)) and not isinstance(regularization, bool):
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
