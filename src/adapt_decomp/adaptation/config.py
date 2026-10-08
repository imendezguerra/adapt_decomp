"""Configuration dataclass for adaptive EMG decomposition."""

from dataclasses import dataclass, field, fields
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import yaml

from adapt_decomp.utils import to_yaml_safe, validate_literals


@dataclass
class _LegacyConfig:
    """Fields retained only so old YAML files load without error. Not used by any logic.

    Exception: contrast_fun. Online adaptation's sv update always uses log_cosh
    (see adaptation/ops.py::update_sv_spike_gated and
    adaptation/data_structures.py::Decomposition.init_sv_update, both import
    log_cosh directly from cbss.ica with no dispatch on this field) -- unlike
    CBSSConfig.contrast_fun, which is real and drives calibration's fixed-point
    ICA. Narrowed to a single-value Literal so a config can't silently claim a
    contrast that isn't actually applied; kept here (not promoted to a live
    AdaptConfig field) since there is still nothing to configure.
    """

    contrast_fun: Literal["logcosh"] = "logcosh"
    spike_height_mult: int = 3
    spike_prev_weight: int = 5
    cov_alpha: float = 0.1


@dataclass
class AdaptConfig(_LegacyConfig):
    """Configuration of the online adaptation.

    The preprocessing, channel and extension fields, and spike_det_exp, must match the
    calibration: AdaptDecomp.from_calibration() overwrites them with the CBSSConfig's
    values. Tuned values come from the presets (AdaptConfig.from_preset()).

    Attributes:
        contrast_fun (Literal["logcosh"]): Legacy, ignored: kept so old YAML files load.
        spike_height_mult (int): Legacy, ignored.
        spike_prev_weight (int): Legacy, ignored.
        cov_alpha (float): Legacy, ignored.
        fs (int): Sampling frequency, in Hz.
        device (Literal["cpu", "cuda", "mps", None]): Compute device. None picks CUDA,
            then MPS, then the CPU.
        lowcut (float): High-pass cutoff, in Hz.
        highcut (float): Low-pass cutoff, in Hz.
        filter_order (int): Butterworth band-pass filter order.
        powerline (bool): Whether to notch out the powerline frequency and its
            harmonics.
        powerline_freq (float): Powerline frequency, in Hz (50 or 60).
        notch_width_hz (float): Half-bandwidth of each notch, in Hz.
        notch_n_harmonics (int): Number of powerline harmonics notched out, the
            fundamental included.
        notch_order (int): Notch filter order.
        ch_mask (Optional[np.ndarray]): Boolean channel mask with shape (channels,),
            True to keep. None keeps every channel.
        ch_map (Optional[np.ndarray]): Electrode grid layout with shape (rows, cols),
            holding raw channel indices. Only needed to interpolate bad channels online.
        replace_bad_channels (bool): False drops the channels that ch_mask marks as
            bad; True interpolates them from their neighbours on ch_map.
        ext_fact (int): Extension factor: the number of delayed copies of each
            channel.
        ext_mode (Literal["block", "toeplitz"]): Column order of the extended EMG.
        batch_ms (int): Batch duration, in ms. Each batch is one adaptation step.
        batch_size (int): batch_ms in samples. Derived, not set by the caller.
        adapt_wh (bool): Whether to adapt the whitening matrix.
        adapt_sv (bool): Whether to adapt the separation vectors.
        adapt_sd (bool): Whether to adapt the spike detection centroids.
        compute_loss (bool): Whether to compute the whitening and separation-vector
            losses. Needed for hyperparameter searches, not for the adaptation itself.
        sv_loss_reduction (Literal["sum", "mean"]): How sv_loss_total reduces across
            units per batch: "mean" weighs every recording the same in a pooled search
            whatever its unit count; "sum" is the 1.0.0 behaviour.
        save_params (bool): Whether to write the adapted parameters of every batch to
            the HDF5 file given as save_path.
        wh_learning_rate (float): Step size of the whitening update. NeuroMotion: 7e-3 |
            Wrist: 1e-3 | Forearm: 2e-3 | MUniverse: 3.3e-2.
        sv_learning_rate (float): Step size of the separation-vector update.
            NeuroMotion: 3e-3 | Wrist: 5e-4 | Forearm: 5e-4 | MUniverse: 4.9e-3.
        lr_mode (Literal["fixed", "rel_error"]): "fixed" takes a plain step of the
            learning rate along the gradient (the 1.0 behaviour); "rel_error" scales a
            unit-norm step by the normalised error, so it shrinks as the error does.
        wh_mode (Literal["kl_to_identity", "kl_to_cal"]): Whitening error: the KL
            divergence of the whitened covariance to the identity, or to the
            calibration's whitened covariance.
        wh_sv_coupling (bool): Whether each whitening update also applies its
            first-order frame correction to the separation vectors.
        contrast_scope (Literal["batch_based", "spike_based"]): Samples the
            separation-vector contrast is computed on: the detected spikes only, or the
            whole batch.
        sv_epochs (int): Maximum separation-vector updates per batch.
        sv_tol (float): Convergence tolerance that stops the separation-vector updates
            early when sv_epochs > 1.
        spike_min_dist_ms (int): Minimum inter-spike interval, in ms.
        spike_min_dist (int): spike_min_dist_ms in samples. Derived, not set by the
            caller.
        spike_det_exp (float): Power the source is raised to before peak detection.
        centroid_momentum (float): Momentum of the spike and baseline centroid updates,
            from 0 (follow each batch) to 1 (never move). NeuroMotion, Wrist, Forearm:
            0.8 | MUniverse: 0.6.
        shrinkage (float): Tikhonov shrinkage added to the whitening FIFO covariance.
        eps (float): Numerical stability floor.
        safety_clip_multiplier_wh (float): Caps the relative size of each whitening
            update at this multiple of wh_learning_rate.
        safety_clip_multiplier_sv (float): Caps the relative size of each
            separation-vector update at this multiple of sv_learning_rate.
        ema_alpha (float): Exponential moving average weight of the running centring
            mean in online mode and of the update norms in lr_mode="rel_error".
        fifo_length (Optional[int]): Number of extended samples in the whitening
            covariance FIFO. None uses twice the extended dimension.
        source_fifo_batches (int): Past batches of sources kept to detect spikes at the
            start of each batch.
        source_fifo_from_calib (bool): Whether to seed the source FIFO with the
            calibration's last sources, for EMG that starts where calibration ends,
            e.g. process_data(emg[b:]) after a calibration window [a, b).
        max_sigma_batches (int): Maximum calibration batches used to estimate the
            calibration statistics the losses are normalised by.
        debug (bool): Whether to store per-batch diagnostics in
            AdaptationResult.diagnostics.
    """

    # General
    fs: int = 2048
    device: Literal["cpu", "cuda", "mps", None] = None

    # Preprocessing
    lowcut: float = 20
    highcut: float = 500
    filter_order: int = 4
    powerline: bool = True
    powerline_freq: float = 50
    notch_width_hz: float = 1.0
    notch_n_harmonics: int = 3
    notch_order: int = 2

    # Bad channels
    ch_mask: Optional[np.ndarray] = None
    ch_map: Optional[np.ndarray] = None
    replace_bad_channels: bool = False

    # Extension
    ext_fact: int = 10
    ext_mode: Literal["block", "toeplitz"] = "block"

    # Adaptation flags
    batch_ms: int = 100
    adapt_wh: bool = True
    adapt_sv: bool = True
    adapt_sd: bool = True
    compute_loss: bool = True
    sv_loss_reduction: Literal["sum", "mean"] = "mean"
    save_params: bool = False

    # Hyperparameters
    wh_learning_rate: float = 5e-3
    sv_learning_rate: float = 1e-3
    lr_mode: Literal["fixed", "rel_error"] = "fixed"

    # Whitening
    wh_mode: Literal["kl_to_identity", "kl_to_cal"] = "kl_to_identity"
    wh_sv_coupling: bool = False

    # Separation vectors
    contrast_scope: Literal["batch_based", "spike_based"] = "spike_based"
    sv_epochs: int = 1
    sv_tol: float = 1e-4

    # Spike detection
    spike_min_dist_ms: int = 10
    spike_min_dist: int = field(init=False)
    spike_det_exp: float = 2.0
    centroid_momentum: float = 0.95

    # Numerical stability
    shrinkage: float = 1e-3
    eps: float = 1e-7
    safety_clip_multiplier_wh: float = 20.0
    safety_clip_multiplier_sv: float = 20.0
    ema_alpha: float = 0.95

    # FIFOs and calibration statistics
    fifo_length: Optional[int] = None
    source_fifo_batches: int = 2
    source_fifo_from_calib: bool = False
    max_sigma_batches: int = 300

    # Debugging
    debug: bool = False

    def __post_init__(self) -> None:
        if self.ch_map is not None and not isinstance(self.ch_map, np.ndarray):
            self.ch_map = np.asarray(self.ch_map)
        if self.ch_mask is not None and not isinstance(self.ch_mask, np.ndarray):
            self.ch_mask = np.asarray(self.ch_mask, dtype=bool)
        self.spike_min_dist = int(self.spike_min_dist_ms * self.fs / 1000)
        self.batch_size = int(self.batch_ms * self.fs / 1000)
        validate_literals(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config fields to a YAML-safe dict.

        Excludes derived (init=False) fields such as spike_min_dist, so that
        from_yaml(to_dict()) round-trips cleanly through the constructor
        instead of raising on an unexpected keyword argument. batch_size is
        also derived (set in __post_init__) but, unlike spike_min_dist, isn't
        a declared dataclass field at all, so it's excluded automatically.

        Returns:
            Dict[str, Any]: Mapping of constructor field name to YAML-safe value.
        """
        return {f.name: to_yaml_safe(getattr(self, f.name)) for f in fields(self) if f.init}

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write this config to a YAML file, creating parent directories as needed.

        Args:
            path (Union[str, Path]): Destination file path.

        Returns:
            None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AdaptConfig":
        """Load an AdaptConfig from a YAML file written by to_yaml().

        Args:
            path (Union[str, Path]): Path to a YAML file mapping constructor
                field names to values.

        Returns:
            AdaptConfig: A new instance built from the file's fields, with
            derived fields (spike_min_dist, batch_size) recomputed by
            __post_init__ and every Literal-typed field validated.
        """
        with Path(path).open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_preset(cls, name: str) -> "AdaptConfig":
        """Load one of the configs shipped with the package.

        Args:
            name (str): Preset name, one of PRESETS: "muniverse" (tuned on the FDSI
                benchmark pool), "neuromotion" (the NeuroMotion simulation of the
                tutorial), "wrist" and "forearm" (the experimental recordings of the
                JNE 2024 paper, with the electrodes on the wrist or the forearm), or
                "fixed" (every adaptation switched off, the baseline).

        Returns:
            AdaptConfig: A new instance built from the preset's YAML file.

        Raises:
            ValueError: If name is not one of PRESETS.
        """
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name!r}. Expected one of {list(PRESETS)}.")
        return cls.from_yaml(preset_path(name))


# Configs shipped with the package, in adaptation/presets/<name>.yaml
PRESETS = ("muniverse", "neuromotion", "wrist", "forearm", "fixed")


def preset_path(name: str) -> Path:
    """Return the path of a preset's YAML file.

    Args:
        name (str): Preset name, one of PRESETS.

    Returns:
        Path: The preset's YAML file inside the installed package.

    Raises:
        ValueError: If name is not one of PRESETS.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Expected one of {list(PRESETS)}.")
    return Path(str(files("adapt_decomp.adaptation") / "presets" / f"{name}.yaml"))
