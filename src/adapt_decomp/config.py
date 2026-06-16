"""Configuration dataclass for adaptive EMG decomposition."""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


@dataclass
class _LegacyConfig:
    """Fields retained only so old YAML files load without error. Not used by any logic."""
    compute_loss: bool = True
    log_wh_trace: bool = False
    log_centroid_loss: bool = False
    wh_learning_rate: float = 0.0
    sv_learning_rate: float = 0.0
    sv_epochs: int = 1
    sv_tol: float = 1e-4
    contrast_fun: Literal["logcosh", "cube"] = "logcosh"
    cov_reg_eps: float = 1e-6
    wh_error_clamp: float = 1e3
    sv_error_clamp: float = 10.0
    spike_height_mult: int = 3
    spike_prev_weight: int = 5


@dataclass
class Config(_LegacyConfig):
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
    extension_method: Literal["block", "toeplitz"] = "block"

    # Decomposition adaptation flags
    batch_ms: int = 100
    adapt_wh: bool = True
    adapt_sv: bool = True
    adapt_sd: bool = True
    log_loss: bool = True    # Log wh_loss, sv_loss, centroid_loss and wh_trace each batch
    save_params: bool = False

    # --- Whitening mode ---
    # "kl_to_identity" — drives KL(Rz ‖ I) toward the calibration value K_cal.
    #   Update direction: (Rz − I) @ V.  Error: K − K_cal.
    # "kl_to_cal"      — drives KL(Rz ‖ Rz_cal) to zero.
    #   Update direction: (Rz_cal⁻¹ Rz − I) @ V.  Error: KL(Rz ‖ Rz_cal).
    #   Zero update iff Rz = Rz_cal (unique fixed point at calibration statistics).
    wh_mode: Literal["kl_to_identity", "kl_to_cal"] = "kl_to_identity"

    # Propagate the first-order frame correction from each V step to B.
    # Keeps B aligned with V's coordinate frame without waiting for the contrast
    # gradient to discover the mismatch through kappa drift.
    wh_b_coupling: bool = False

    # --- Adaptation hyperparameters ---
    shrinkage: float = 1e-3         # Tikhonov shrinkage on per-FIFO covariance
    eps: float = 1e-7               # Numerical stability floor

    # Trust-region safety clips — hard ceiling on any single update.
    # V moves at most max_rel_delta_v * ||V|| per batch (Frobenius norm).
    # B moves at most max_rel_delta_b * ||B|| per batch (global Frobenius norm).
    max_rel_delta_v: float = 1e-1
    max_rel_delta_b: float = 1e-1

    min_spikes_for_update: int = 1      # Minimum spike count to allow B row update

    orthonormalization: str = "qr"      # "qr" (default), "gram_schmidt", or "none"

    # Contrast scope: how kappa and kappa_cal are computed for the B update.
    #   "batch_based"  — log_cosh(Y).mean(dim=0) over all N samples; gradient is also
    #                    batch-averaged (tanh(Y).T @ Z / N), decoupled from spike detection
    #   "spike_based"  — log_cosh(Y) averaged only at detected spike times; gradient
    #                    is spike-gated (tanh(Y[spike_mask]).T @ Z / spike_counts)
    # Calibration kappa_cal is computed the same way for consistency.
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

    # --- Calibration sigma estimation ---
    # Maximum number of calibration windows used to estimate sigma_K_cal and sigma_kappa_cal.
    # Windows are sampled uniformly across the calibration recording.
    # 0 = use all windows. On CPU, capping at 200-300 gives stable estimates with ~8× speedup.
    max_sigma_batches: int = 300

    # --- Centroid adaptation ---
    centroid_momentum: float = 0.95
    min_spikes_for_centroid: int = 1
    min_base_peaks_for_centroid: int = 1

    # --- B fixed-point iterations ---
    max_iter_b: int = 1       # Max ICA fixed-point iterations per batch for B (1 = single step)
    tol_b: float = 1e-4       # Early-exit threshold: ‖B_new − B_old‖_F / ‖B_old‖_F < tol_b

    # --- Optimisation ---
    trace_check: bool = True                                    # Reject diverged trials in run_optimisation via trace ratio guard
    trace_check_mode: Literal["last", "median"] = "median"     # "last": endpoint batch only; "median": robust to tail extremes
    # "single_obj": wh_loss + sv_loss combined scalar (single-objective, unchanged behaviour)
    # "multi_obj":  (wh_loss, sv_loss, centroid_loss) 3-objective tuple (multi-objective)
    optim_loss: Literal["single_obj", "multi_obj"] = "single_obj"

    # --- Debug ---
    debug: bool = False

    # --- IQR spike gate ---
    adapt_iqr_gate: bool = True
    iqr_gate_factor: float = 3.0

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
