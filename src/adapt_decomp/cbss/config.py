"""CBSS configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import yaml


@dataclass
class CBSSConfig:
    """Configuration for the CBSS decomposition algorithm."""

    # Preprocessing
    fs: float = 2048.0
    preprocess_emg: bool = True
    highpass_cutoff_hz: Optional[float] = 20.0
    lowpass_cutoff_hz: Optional[float] = 500.0
    filter_order: int = 4
    filter_type: Literal["butter", "firwin2"] = "butter"
    notch_filter: bool = True
    notch_freq_hz: float = 50.0
    notch_quality_factor: float = 30.0
    notch_n_harmonics: int = 3
    notch_filter_type: Literal["butter", "firwin2"] = "butter"
    replace_bad_channels: bool = False
    bad_chs: Optional[list] = None
    ch_map: Optional[np.ndarray] = None

    # Extension
    ext_fact: int = 12

    # PCA dimensionality reduction before whitening (None = skip PCA)
    n_components: Optional[int] = None

    # Whitening
    whitening_method: Literal["ZCA", "PCA"] = "ZCA"
    regularization: Literal["auto"] | float | None = "auto"
    eps: float = 1e-10

    # ICA
    solver: Literal["fast_ica"] = "fast_ica"
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
    cov_th: float = 0.35

    # Quality control
    sil_th: float = 0.9
    min_spikes: int = 10

    # Duplicate removal
    roa_th: float = 0.3
    run_duplicate_removal: bool = True

    # Compute properties
    compute_properties: bool = True

    # Result storage
    save_emg: bool = False

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
