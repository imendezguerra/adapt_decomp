"""Configuration dataclass for adaptive EMG decomposition."""

import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

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
    """Configuration parameters."""

    # General parameters
    fs: int = 2048
    device: Literal["cpu", "cuda", "mps", None] = None

    # Preprocessing parameters
    # Shares preprocessing.preprocess_emg with CBSS calibration (CBSSConfig) -- keep
    # these in sync with CBSSConfig's equivalents so the whitening reference computed
    # at calibration and the online covariance see EMG at the same scale.
    lowcut: float = 20
    highcut: float = 500
    filter_order: int = 4
    powerline: bool = True
    powerline_freq: float = 50
    notch_width_hz: float = 1.0
    notch_n_harmonics: int = 3
    notch_order: int = 2

    # Extension parameters (to be inherited from calibration)
    ext_fact: int = 10
    ext_mode: Literal["block", "toeplitz"] = "block" 

    # Decomposition adaptation flags
    batch_ms: int = 100
    adapt_wh: bool = True        # Adapt whitening
    adapt_sv: bool = True        # Adapt separation vectors
    adapt_sd: bool = True        # Adapt spike detection
    compute_loss: bool = True    # Log wh_loss and sv_loss
    save_params: bool = False    # Save newly adapted parameters per batch

    # Main adaptation hyperparameters to tune
    wh_learning_rate: float = 5e-3
    sv_learning_rate: float = 1e-3

    # ---- Adaptation behaviour -----

    # Learning rate mode: "fixed" (previously lr_alone=True) or "rel_error" (previously lr_alone=False)
    lr_mode: Literal["fixed", "rel_error"] = "fixed"

    # Whitening
    wh_mode: Literal["kl_to_identity", "kl_to_cal"] = "kl_to_identity"  # Reference point for calibration
    wh_sv_coupling: bool = False    # Propagate the first-order frame correction from each wh step to sv.

    # Separation vectors 
    contrast_scope: Literal["batch_based", "spike_based"] = "batch_based" # Samples to use for separation vector update
    sv_epochs: int = 1       # Max number of separation vector updates per batch
    sv_tol: float = 1e-4     # Convergence tolerance in case multiple updates per batch for early stopping

    # Spike detection
    spike_min_dist_ms: int = 10              # Minimum inter-spike distance in ms
    spike_min_dist: int = field(init=False)  # Derived: samples
    spike_det_exp: float = 2.0               # Exponent for spike detection
    centroid_momentum: float = 0.95          # Momentum for centroid EMA update

    # ---- Constants ----
    
    # Numerical stability constants
    shrinkage: float = 1e-3         # Tikhonov shrinkage on per-FIFO covariance
    eps: float = 1e-7               # Numerical stability floor

    # Safety clip multipliers
    safety_clip_multiplier_wh: float = 20.0
    safety_clip_multiplier_sv: float = 20.0
    ema_alpha: float = 0.95  # Used for update scaling based on EMA norm

    # Fifo constants for calibration parameter estimation
    fifo_length: Optional[int] = None  # If None, defaults to 2x number of varaibles
    source_fifo_batches: int = 2    # Past batches of sources prepended for edge spike support
    max_sigma_batches: int = 300    # Max number of calibration batches used to compute mean and std of signal properties

    # Debugging
    debug: bool = False

    def __post_init__(self) -> None:
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
        return {
            f.name: to_yaml_safe(getattr(self, f.name))
            for f in fields(self)
            if f.init
        }

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


def load_yaml(file_path: str) -> Dict:
    """Load a YAML file into a dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    defaults_path: str = "configs/model_configs/default_neuromotion.yml",
    wandb_config=None,
) -> AdaptConfig:
    """Load YAML config and apply optional wandb sweep overrides."""
    defaults = load_yaml(defaults_path)
    if wandb_config:
        for key, value in wandb_config.items():
            if key in defaults:
                defaults[key] = value
    return AdaptConfig(**defaults)
