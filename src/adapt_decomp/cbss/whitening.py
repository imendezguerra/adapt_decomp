"""Whitening functions for EMG data."""

from typing import Literal, Tuple
import torch


def whiten(
    emg_ext: torch.Tensor,
    method: Literal["ZCA", "PCA"] = "ZCA",
    regularization: str | float | None = "auto",
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whiten EMG data using ZCA or PCA whitening.

    Args:
        emg_ext (torch.Tensor): Extended EMG data of shape (samples, channels).
        method (Literal["ZCA", "PCA"]): Whitening method. Defaults to "ZCA".
        regularization (str | float | None, optional): Regularization strategy. Defaults to "auto".
        eps (float, optional): Numerical stability constant. Defaults to 1e-10.

    Raises:
        ValueError: If an unknown regularization strategy or whitening method is provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Whitened EMG data and the whitening matrix.
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
        raise ValueError(f"Unknown whitening method: {method!r}. Expected 'ZCA' or 'PCA'.")

    return emg_ext @ W.T, W
