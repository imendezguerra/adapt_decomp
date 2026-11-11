"""Whitening functions for EMG data."""

from typing import Literal, Tuple

import torch


def whiten(
    emg_ext: torch.Tensor,
    method: Literal["ZCA", "PCA"] = "ZCA",
    regularization: str | float | None = "auto",
    eps: float = 1e-10,
    compute_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whiten EMG data using ZCA or PCA whitening.

    The covariance and its eigendecomposition run on the CPU in compute_dtype: the
    extended EMG's covariance is ill-conditioned, and float32 rounds its small eigenvalues.

    Args:
        emg_ext (torch.Tensor): Extended EMG data of shape (samples, channels).
        method (Literal["ZCA", "PCA"]): Whitening method. Defaults to "ZCA".
        regularization (str | float | None, optional): Regularization strategy. Defaults to "auto".
        eps (float, optional): Numerical stability constant. Defaults to 1e-10.
        compute_dtype (torch.dtype, optional): Precision of the covariance and
            eigendecomposition. Defaults to torch.float64.

    Raises:
        ValueError: If an unknown regularization strategy or whitening method is provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Whitened EMG data and the whitening matrix,
        both in emg_ext's dtype and device.
    """
    x = emg_ext.to(device="cpu", dtype=compute_dtype)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp(min=0.0)

    if regularization == "auto":
        low_evals = evals[: len(evals) // 2]
        reg = float(torch.mean(low_evals).item()) if low_evals.numel() > 0 else 0.0
    elif isinstance(regularization, (int, float)) and not isinstance(regularization, bool):
        reg = float(regularization)
    elif regularization is None:
        reg = 0.0
    else:
        raise ValueError(f"Unknown regularization: {regularization!r}")

    inv_sqrt = (evals + reg + eps).rsqrt()

    if method == "ZCA":
        W = evecs @ torch.diag(inv_sqrt) @ evecs.T
    elif method == "PCA":
        W = torch.diag(inv_sqrt) @ evecs.T
    else:
        raise ValueError(f"Unknown whitening method: {method!r}. Expected 'ZCA' or 'PCA'.")

    W = W.to(device=emg_ext.device, dtype=emg_ext.dtype)
    return emg_ext @ W.T, W
