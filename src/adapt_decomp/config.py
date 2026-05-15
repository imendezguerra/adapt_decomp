"""Configuration dataclass for adaptive EMG decomposition."""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


@dataclass
class Config:
    """Configuration parameters."""

    # General parameters
    fs: int = 2048
    device: Literal["cpu", "cuda", "mps", None] = None

    # Preprocessing parameters
    lowcut: float = 20
    highcut: float = 500
    powerline: bool = True
    powerline_freq: float = 50

    # Decomposition parameters
    ext_fact: int = 10

    # Decomposition adaptation flags
    batch_ms: int = 100
    adapt_wh: bool = True
    adapt_sv: bool = True
    adapt_sd: bool = True
    compute_loss: bool = True
    save_params: bool = False

    # --- Whitening mode ---
    # "kl_to_identity" — drives KL(Rz ‖ I) toward the calibration value K_cal.
    #   Update direction: (Rz − I) @ V.  Error: K − K_cal.
    # "kl_to_cal"      — drives KL(Rz ‖ Rz_cal) to zero.
    #   Update direction: (Rz_cal⁻¹ Rz − I) @ V.  Error: KL(Rz ‖ Rz_cal).
    #   Zero update iff Rz = Rz_cal (unique fixed point at calibration statistics).
    wh_mode: Literal["kl_to_identity", "kl_to_cal"] = "kl_to_identity"

    # --- Adaptation hyperparameters ---
    shrinkage: float = 1e-3         # Tikhonov shrinkage on per-FIFO covariance
    eps: float = 1e-7               # Numerical stability floor

    # Trust-region step bounds — the sole rate controls for V and B.
    # V moves at most max_rel_delta_v * ||V|| per batch (Frobenius norm).
    # Each B row moves at most max_rel_delta_b * ||b_j|| per batch.
    max_rel_delta_v: float = 5e-3
    max_rel_delta_b: float = 1e-3

    min_spikes_for_update: int = 1      # Minimum spike count to allow B row update

    orthonormalization: str = "qr"      # "qr" or "none"

    # Contrast scope: how kappa and kappa_cal are computed.
    #   "batch_based"  — log_cosh(Y).mean(dim=0) over all N samples in the batch
    #   "spike_based"  — log_cosh(Y) averaged only at detected spike times
    # Calibration kappa_cal is computed the same way for consistency.
    # B update is always gated on detected spikes regardless of scope.
    contrast_scope: Literal["batch_based", "spike_based"] = "batch_based"

    # --- Spike detection ---
    peak_power: float = 2.0
    strict_peaks: bool = True
    use_abs_for_detection: bool = True

    spike_dist_ms: int = 10         # Minimum inter-spike distance in ms
    spike_dist: int = field(init=False)  # Derived: samples

    # --- FIFO buffers ---
    # fifo_length: number of samples to keep for whitening covariance estimation.
    # 0 means auto = 2 × D (extended channels). Hard floor = D.
    fifo_length: int = 0
    source_fifo_batches: int = 2    # Past batches of Y prepended for edge spike support

    # --- Centroid adaptation ---
    centroid_momentum: float = 0.95
    min_spikes_for_centroid: int = 1
    min_base_peaks_for_centroid: int = 3

    # --- Debug ---
    debug: bool = False

    # --- Legacy parameters kept for backward compatibility ---
    wh_learning_rate: float = 0.0   # unused — kept so existing YAML files load without error
    sv_learning_rate: float = 0.0   # unused — kept so existing YAML files load without error
    sv_epochs: int = 1
    sv_tol: float = 1e-4
    contrast_fun: Literal["logcosh", "cube"] = "logcosh"
    cov_reg_eps: float = 1e-6
    wh_error_clamp: float = 1e3
    sv_error_clamp: float = 10.0
    spike_height_mult: int = 3
    spike_prev_weight: int = 5

    def __post_init__(self) -> None:
        self.spike_dist = int(self.spike_dist_ms * self.fs / 1000)
        self.batch_size = int(self.batch_ms * self.fs / 1000)


def load_yaml(file_path: str) -> Dict:
    """Load a YAML file into a dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    defaults_path: str = "configs/model_configs/default_neuromotion.yml",
    wandb_config=None,
) -> Config:
    """Load YAML config and apply optional wandb sweep overrides."""
    defaults = load_yaml(defaults_path)
    if wandb_config:
        for key, value in wandb_config.items():
            if key in defaults:
                defaults[key] = value
    return Config(**defaults)
