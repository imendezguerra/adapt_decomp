"""EMG extension functions."""

import torch
from typing import Literal, Optional

def extend_data(
    data: torch.Tensor,
    ext_fact: int,
    device: Optional[torch.device] = None,
    ext_mode: Literal["block", "toeplitz"] = "block",
) -> torch.Tensor:
    """Extend data by a given factor (time-delay embedding).

    Each of the ext_fact column blocks is the raw channel matrix shifted by
    one additional sample, giving shape (samples, channels * ext_fact). The
    resulting extended-channel space lets a downstream whitening matrix
    capture temporal correlations across ext_fact consecutive samples in a
    single linear projection. Shared by both cbss (calibration) and
    data_structures.Data/Decomposition (online adaptation).

    Args:
        data (torch.Tensor): EMG data of shape (samples, channels).
        ext_fact (int): Extension factor.
        device (Optional[torch.device]): Device for the extended data. Defaults
            to None, which reuses data's own device.
        ext_mode (Literal["block", "toeplitz"], optional): "block" (default) —
            column block i holds ALL channels shifted by i samples.
            "toeplitz" — each channel's own ext_fact delayed copies are kept
            together, so each channel's block of columns is itself a
            Toeplitz (constant-diagonal) matrix. Same extended width
            (channels * ext_fact) either way. Defaults to "block".

    Raises:
        ValueError: If ext_mode is not "block" or "toeplitz".

    Returns:
        torch.Tensor: Extended data of shape (samples, channels * ext_fact),
        with the same dtype as data.
    """
    if device is None:
        device = data.device
    samples, chs = data.shape
    data_ext = torch.zeros((samples, int(chs * ext_fact)), device=device, dtype=data.dtype)
    for i in range(ext_fact):
        data_ext[i:samples, chs * i: chs * (i + 1)] = data[0:(samples - i), :]
    if ext_mode == "toeplitz":
        data_ext = (
            data_ext.view(samples, ext_fact, chs)
            .permute(0, 2, 1)
            .reshape(samples, chs * ext_fact)
        )
    elif ext_mode != "block":
        raise ValueError(f"Unknown ext_mode: {ext_mode!r}. Expected 'block' or 'toeplitz'.")
    return data_ext