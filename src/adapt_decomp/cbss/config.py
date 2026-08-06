"""CBSS configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import yaml

from adapt_decomp.config import validate_literals


@dataclass
class CBSSConfig:
    """Configuration for the CBSS decomposition algorithm."""

    # Preprocessing
    # Shares preprocessing.preprocess_emg with the online AdaptDecomp Config

    fs: float = 2048.0
    preprocess_emg: bool = True
    lowcut: Optional[float] = 20.0
    highcut: Optional[float] = 500.0
    filter_order: int = 4
    powerline: bool = True
    powerline_freq: float = 50.0
    notch_width_hz: float = 1.0      # half-bandwidth per notch, in Hz
    notch_n_harmonics: int = 3
    notch_order: int = 2
    replace_bad_channels: bool = False
    bad_chs: Optional[list] = None
    ch_map: Optional[np.ndarray] = None

    # Extension
    # Must match the online adaptation Config.ext_fact used downstream on this
    # calibration's output (Decomposition.__init__ validates this at construction).
    ext_fact: int = 10

    # "block" (default) — column block i holds ALL channels shifted by i samples.
    # "toeplitz" — each channel's own ext_fact delays are grouped together, so
    # each channel's block of columns is itself a Toeplitz matrix (standard
    # convolutive-EMG-mixing convention). Same extended width (C*ext_fact)
    # either way, so a mismatch against Config.ext_mode used downstream
    # is NOT caught by shape validation -- keep the two in sync manually.
    ext_mode: Literal["block", "toeplitz"] = "block"

    # PCA dimensionality reduction before whitening (None = skip PCA)
    n_components: Optional[int] = None

    # Whitening
    whitening_method: Literal["ZCA", "PCA"] = "ZCA"
    regularization: Literal["auto"] | float | None = "auto"
    eps: float = 1e-10

    # ICA
    contrast_fun: Literal["logcosh", "square", "cube", "smooth_abs"] = "square"
    contrast_exp: float = 3.0
    search_iter: int = 100
    ica_iter: int = 100
    ica_tol: float = 1e-4

    # Spike detection
    spike_det_exp: float = 2.0
    spike_min_dist_ms: float = 10.0

    # Refinement loop
    refinement_loop: bool = True
    refinement_mode: Literal["cov", "sil"] = "sil"
    refine_max_iter: int = 20

    # Quality control
    sil_th: float = 0.9
    min_spikes: int = 10

    # Duplicate removal
    roa_th: float = 0.3
    run_duplicate_removal: bool = True

    # Compute properties
    compute_properties: bool = True

    # Result storage
    save_emg: bool = True

    # Compute device (None = auto: CUDA > MPS > CPU)
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32

    # Reproducibility
    random_seed: Optional[int] = 1909

    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        if isinstance(self.dtype, str):
            self.dtype = _dtype_from_string(self.dtype)
        if self.device is not None:
            self.device = str(self.device)
        if self.ch_map is not None and not isinstance(self.ch_map, np.ndarray):
            self.ch_map = np.asarray(self.ch_map)
        validate_literals(self)

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            out[key] = _to_yaml_safe(value)
        return out

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CBSSConfig":
        with Path(path).open("r") as f:
            data = yaml.safe_load(f) or {}
        if data.get("ch_map") is not None:
            data["ch_map"] = np.asarray(data["ch_map"])
        if data.get("dtype") is not None:
            data["dtype"] = _dtype_from_string(data["dtype"])
        return cls(**data)


def _to_yaml_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, (torch.device, Path)):
        return str(value)
    return value


def _dtype_from_string(value: str) -> torch.dtype:
    name = value.replace("torch.", "")
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {value!r}") from exc
