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
    """Configuration for the CBSS decomposition algorithm.

    The preprocessing, channel and extension fields are shared with AdaptConfig: an
    adaptation built with AdaptDecomp.from_calibration() takes them from this config.

    Attributes:
        fs (float): Sampling frequency, in Hz.
        preprocess_emg (bool): Whether to band-pass and notch filter the EMG before
            extension.
        lowcut (Optional[float]): High-pass cutoff, in Hz. None skips the high-pass.
        highcut (Optional[float]): Low-pass cutoff, in Hz. None skips the low-pass.
        filter_order (int): Butterworth band-pass filter order.
        powerline (bool): Whether to notch out the powerline frequency and its
            harmonics.
        powerline_freq (float): Powerline frequency, in Hz (50 or 60).
        notch_width_hz (float): Half-bandwidth of each notch, in Hz.
        notch_n_harmonics (int): Number of powerline harmonics notched out, the
            fundamental included.
        notch_order (int): Notch filter order.
        replace_bad_channels (bool): False drops the channels that ch_mask marks as
            bad; True interpolates them from their neighbours on ch_map.
        ch_mask (Optional[np.ndarray]): Boolean channel mask with shape (channels,),
            True to keep. None keeps every channel.
        ch_map (Optional[np.ndarray]): Electrode grid layout with shape (rows, cols),
            holding raw channel indices. Required by replace_bad_channels and MUAP
            computation.
        ext_fact (int): Extension factor: the number of delayed copies of each
            channel.
        ext_mode (Literal["block", "toeplitz"]): Column order of the extended EMG.
        n_components (Optional[int]): Number of PCA components kept before whitening.
            None skips PCA. The fitted PCA is reused unchanged by AdaptDecomp.
        whitening_method (Literal["ZCA", "PCA"]): Whitening transform.
        regularization (Union[Literal["auto"], float, None]): Value added to the
            covariance eigenvalues before whitening. "auto" uses the mean of the
            smaller half of the eigenvalues; None adds nothing.
        eps (float): Numerical stability constant for whitening and normalisation.
        contrast_fun (Literal["logcosh", "square", "cube", "smooth_abs"]): Contrast
            function of the fixed-point ICA.
        contrast_exp (float): Exponent of the "smooth_abs" contrast. Ignored by the
            other contrasts.
        search_iter (int): Number of ICA initialisations tried, each of which can
            yield one unit.
        ica_iter (int): Maximum fixed-point iterations per initialisation.
        ica_tol (float): Fixed-point convergence tolerance.
        spike_det_exp (float): Power the source is raised to before peak detection.
        spike_min_dist_ms (float): Minimum inter-spike interval, in ms.
        spike_min_dist (int): spike_min_dist_ms in samples. Derived, not set by the
            caller.
        refinement_loop (bool): Whether to refine each converged unit by re-estimating
            its separation vector from its own spikes.
        refinement_mode (Literal["cov_isi", "sil"]): Metric the refinement loop
            improves: the silhouette (higher is better) or the coefficient of
            variation of the inter-spike intervals (lower is better).
        refine_max_iter (int): Maximum refinement iterations per unit.
        sil_th (float): Minimum silhouette for a unit to be kept.
        min_spikes (int): Minimum number of spikes for a unit to be kept.
        roa_th (float): Score on duplicate_metric at or above which two units count
            as duplicates.
        duplicate_metric (Literal["roa", "f1", "match"]): Agreement score compared with
            roa_th: rate of agreement, F1, or common spikes over the larger spike count
            (muniverse's duplicate score).
        duplicate_keep_by (Literal["cov_isi", "sil"]): Unit kept from each group of
            duplicates: lowest CoV-ISI or highest silhouette.
        duplicate_tol_spike_ms (float): Spike coincidence tolerance for duplicate
            detection, in ms.
        duplicate_tol_train_ms (float): Largest lag searched between two units for
            duplicate detection, in ms.
        run_duplicate_removal (bool): Whether to remove duplicate units.
        selection (Literal["unsupervised", "supervised", None]): Unit selection
            applied at the end of decompose(): on unit properties ("unsupervised") or
            against ground truth ("supervised"). None keeps every unit.
        selection_kwargs (Optional[Dict[str, Any]]): Keyword arguments for
            CBSSResult.select_unsupervised() or select_supervised(). Its gt_spikes,
            being data, is left out of to_dict()/to_yaml() and embedded configs.
        compute_properties (bool): Whether to compute each unit's pulse-to-noise ratio,
            discharge rate and MUAPs. Required by unsupervised selection.
        save_emg (bool): Whether to store the calibration EMG and timestamps in the
            result. Required to build an AdaptDecomp from it.
        device (Optional[Literal["cpu", "mps", "cuda"]]): Compute device. None picks
            CUDA, then MPS, then the CPU.
        dtype (torch.dtype): Floating-point precision; CBSS runs in float32 only.
        random_seed (Optional[int]): Seed of the ICA initialisation order. None gives
            a different decomposition on every run.
        verbose (bool): Whether to print progress.
    """

    # Preprocessing
    fs: float = 2048.0
    preprocess_emg: bool = True
    lowcut: Optional[float] = 20.0
    highcut: Optional[float] = 500.0
    filter_order: int = 4
    powerline: bool = True
    powerline_freq: float = 50.0
    notch_width_hz: float = 1.0
    notch_n_harmonics: int = 3
    notch_order: int = 2
    replace_bad_channels: bool = False
    ch_mask: Optional[np.ndarray] = None
    ch_map: Optional[np.ndarray] = None

    # Extension
    ext_fact: int = 10
    ext_mode: Literal["block", "toeplitz"] = "block"

    # PCA
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
    spike_min_dist: int = field(init=False)

    # Refinement loop
    refinement_loop: bool = True
    refinement_mode: Literal["cov_isi", "sil"] = "sil"
    refine_max_iter: int = 20

    # Quality control
    sil_th: float = 0.9
    min_spikes: int = 10

    # Duplicate removal
    roa_th: float = 0.3
    duplicate_metric: Literal["roa", "f1", "match"] = "roa"
    duplicate_keep_by: Literal["cov_isi", "sil"] = "cov_isi"
    duplicate_tol_spike_ms: float = 0.5
    duplicate_tol_train_ms: float = 40.0
    run_duplicate_removal: bool = True

    # Unit selection
    selection: Literal["unsupervised", "supervised", None] = None
    selection_kwargs: Optional[Dict[str, Any]] = None

    # Compute properties
    compute_properties: bool = True

    # Result storage
    save_emg: bool = True

    # Compute device
    device: Optional[Literal["cpu", "mps", "cuda"]] = "cpu"
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
        if self.dtype != torch.float32:
            raise ValueError(f"CBSS supports float32 only, got dtype={self.dtype}.")
        if self.device is not None:
            self.device = str(self.device)
        if self.ch_map is not None and not isinstance(self.ch_map, np.ndarray):
            self.ch_map = np.asarray(self.ch_map)
        if self.ch_mask is not None and not isinstance(self.ch_mask, np.ndarray):
            self.ch_mask = np.asarray(self.ch_mask, dtype=bool)
        self.spike_min_dist = max(1, round(self.spike_min_dist_ms / 1000 * self.fs))
        validate_literals(self)

    def to_dict(self) -> dict:
        """Serialise config fields to a YAML-safe dict.

        Excludes derived (init=False) fields such as spike_min_dist, so that
        from_yaml(to_dict()) round-trips cleanly through the constructor
        instead of raising on an unexpected keyword argument. Also excludes
        selection_kwargs["gt_spikes"]: ground truth is data, not config.

        Returns:
            dict: Mapping of constructor field name to YAML-safe value.
        """
        out = {}
        for f in fields(self):
            if not f.init:
                continue
            out[f.name] = to_yaml_safe(getattr(self, f.name))
        if out["selection_kwargs"] is not None:
            out["selection_kwargs"] = {
                k: v for k, v in out["selection_kwargs"].items() if k != "gt_spikes"
            }
        return out

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CBSSConfig":
        with Path(path).open("r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_dict(cls, data: dict) -> "CBSSConfig":
        """Build a CBSSConfig from a dict written by to_dict().

        Args:
            data (dict): Mapping of constructor field name to YAML-safe value.

        Returns:
            CBSSConfig: A new instance, with ch_map/ch_mask/dtype restored to
            their array/dtype types.
        """
        data = dict(data)
        if data.get("ch_map") is not None:
            data["ch_map"] = np.asarray(data["ch_map"])
        if data.get("ch_mask") is not None:
            data["ch_mask"] = np.asarray(data["ch_mask"], dtype=bool)
        if data.get("dtype") is not None:
            data["dtype"] = dtype_from_string(data["dtype"])
        return cls(**data)
