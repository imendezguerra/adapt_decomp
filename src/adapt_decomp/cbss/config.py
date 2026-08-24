"""CBSS configuration dataclass."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import yaml

from adapt_decomp.utils import dtype_from_string, to_yaml_safe, validate_literals


@dataclass
class CBSSConfig:
    """Configuration for the CBSS decomposition algorithm."""

    # Preprocessing
    # Shares preprocessing.preprocess_emg with the online AdaptDecomp AdaptConfig

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
    # Must match the online adaptation AdaptConfig.ext_fact used downstream on this
    # calibration's output (Decomposition.__init__ validates this at construction).
    ext_fact: int = 10

    # "block" (default) — column block i holds ALL channels shifted by i samples.
    # "toeplitz" — each channel's own ext_fact delays are grouped together, so
    # each channel's block of columns is itself a Toeplitz matrix. 
    ext_mode: Literal["block", "toeplitz"] = "block"

    # PCA dimensionality reduction before whitening (None = skip PCA)
    n_components: Optional[int] = None

    # Whitening
    whitening_method: Literal["ZCA", "PCA"] = "ZCA"
    regularization: Literal["auto"] | float | None = "auto"
    eps: float = 1e-10

    # ICA
    contrast_fun: Literal["logcosh", "square", "cube", "smooth_abs"] = "square"
    contrast_exp: float = 3.0       # Only used for smooth_abs
    search_iter: int = 100
    ica_iter: int = 100
    ica_tol: float = 1e-4

    # Spike detection
    spike_det_exp: float = 2.0
    spike_min_dist_ms: float = 10.0  # Minimum inter-spike distance in ms
    spike_min_dist: int = field(init=False)  # Derived: spike_min_dist_ms in samples

    # Refinement loop
    refinement_loop: bool = True
    refinement_mode: Literal["cov_isi", "sil"] = "sil"
    refine_max_iter: int = 20

    # Quality control
    sil_th: float = 0.9
    min_spikes: int = 10

    # Duplicate removal
    roa_th: float = 0.3
    run_duplicate_removal: bool = True

    # Unit selection (post-hoc filter, applied inside decompose() after CBSS
    # finds units). None = keep every discovered unit, matching decompose()'s
    # behaviour with no selection configured.
    # - unsupervised: selection based on motor unit properties
    # - supervised: selection based on ground truth
    selection: Literal["unsupervised", "supervised", None] = None 
    selection_kwargs: Optional[Dict[str, Any]] = None  # forwarded to CBSSResult.select_unsupervised()/select_supervised()

    # Compute properties
    compute_properties: bool = True

    # Result storage
    save_emg: bool = True

    # Compute device (None = auto: CUDA > MPS > CPU)
    device: Optional[Literal['cpu', 'mps', 'cuda']] = 'cpu'
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
            self.dtype = dtype_from_string(self.dtype)
        if self.device is not None:
            self.device = str(self.device)
        if self.ch_map is not None and not isinstance(self.ch_map, np.ndarray):
            self.ch_map = np.asarray(self.ch_map)
        self.spike_min_dist = max(1, round(self.spike_min_dist_ms / 1000 * self.fs))
        validate_literals(self)

    def to_dict(self) -> dict:
        """Serialise config fields to a YAML-safe dict.

        Excludes derived (init=False) fields such as spike_min_dist, so that
        from_yaml(to_dict()) round-trips cleanly through the constructor
        instead of raising on an unexpected keyword argument.

        Returns:
            dict: Mapping of constructor field name to YAML-safe value.
        """
        out = {}
        for f in fields(self):
            if not f.init:
                continue
            out[f.name] = to_yaml_safe(getattr(self, f.name))
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
            data["dtype"] = dtype_from_string(data["dtype"])
        return cls(**data)
