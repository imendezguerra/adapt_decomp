"""Configuration dataclass for adaptive EMG decomposition."""

import yaml
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, get_args, get_origin


def validate_literals(obj) -> None:
    """Raise ValueError if any Literal[...]-typed dataclass field holds a value
    outside its declared choices.

    Config/CBSSConfig fields are set via plain setattr() in several places
    (config_overrides, base_config, categorical Optuna search spaces), which
    bypasses dataclass construction entirely -- a typo'd value would otherwise
    silently fall through whatever `==` checks read that field downstream
    instead of raising. Call this after every such override.
    """
    for f in dataclasses.fields(obj):
        if get_origin(f.type) is Literal:
            value = getattr(obj, f.name)
            choices = get_args(f.type)
            if value not in choices:
                raise ValueError(
                    f"{type(obj).__name__}.{f.name} = {value!r} is not one of {choices!r}"
                )


@dataclass
class _LegacyConfig:
    """Fields retained only so old YAML files load without error. Not used by any logic."""
    log_wh_trace: bool = False
    log_centroid_loss: bool = False
    contrast_fun: Literal["logcosh", "cube"] = "logcosh"
    cov_reg_eps: float = 1e-6
    wh_error_clamp: float = 1e3
    sv_error_clamp: float = 10.0
    spike_height_mult: int = 3
    spike_prev_weight: int = 5
    # cov_alpha (EMA-style covariance regularisation) has no true replacement here --
    # shrinkage (Config, below) is a different quantity (one-shot Tikhonov shrinkage on
    # the FIFO covariance, not an EMA decay rate). Kept inert purely so main's tracked
    # configs (which set cov_alpha) load without raising TypeError.
    cov_alpha: float = 0.1
    # Superseded by wh_learning_rate/sv_learning_rate + safety_clip_multiplier_wh/sv (see
    # Config below): these used to be both the step size AND the clip ceiling, which meant
    # the clip was engaged on ~100% of batches and silently discarded the error term's
    # magnitude. Kept here inert so old YAML/JSON configs still load without error.
    max_rel_delta_v: float = 1e-1
    max_rel_delta_b: float = 1e-1


@dataclass
class Config(_LegacyConfig):
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

    # Decomposition parameters
    ext_fact: int = 10

    # Decomposition adaptation flags
    batch_ms: int = 100
    adapt_wh: bool = True
    adapt_sv: bool = True
    adapt_sd: bool = True
    compute_loss: bool = True    # Log wh_loss, sv_loss, centroid_loss and wh_trace each batch
    save_params: bool = False

    # --- Whitening mode ---
    # "kl_to_identity" — drives KL(Rz ‖ I) toward the calibration value K_cal.
    #   Update direction: (Rz − I) @ wh.  Error: K − K_cal.
    # "kl_to_cal"      — drives KL(Rz ‖ Rz_cal) to zero.
    #   Update direction: (Rz_cal⁻¹ Rz − I) @ wh.  Error: KL(Rz ‖ Rz_cal).
    #   Zero update iff Rz = Rz_cal (unique fixed point at calibration statistics).
    wh_mode: Literal["kl_to_identity", "kl_to_cal"] = "kl_to_identity"

    # Propagate the first-order frame correction from each wh step to sv.
    # Keeps sv aligned with wh's coordinate frame without waiting for the contrast
    # gradient to discover the mismatch through kappa drift.
    wh_b_coupling: bool = False

    # --- Adaptation hyperparameters ---
    shrinkage: float = 1e-3         # Tikhonov shrinkage on per-FIFO covariance
    eps: float = 1e-7               # Numerical stability floor

    # --- Learning rate + safety clip ---
    # NOTE ON BACKWARD COMPATIBILITY: wh_learning_rate/sv_learning_rate are REUSED names
    # from main (v1), but the formula behind them is NOT the same quantity. In v1 these
    # were direct multipliers on a raw gradient. Here, the natural-gradient direction
    # (direction@wh for wh, grad_sv row for sv) is first normalized to unit scale (via an
    # EMA of its own norm, so a single noisy/near-zero batch can't dominate the
    # normalization), then scaled by wh_learning_rate/sv_learning_rate and the actual
    # signed error e_wh/e_sv.
    # Actual step ~ lr * ||ref|| * e, so it shrinks toward zero as e -> 0 and grows for
    # genuine drift -- unlike the old max_rel_delta_{v,b} scheme, whose clip engaged on
    # ~100% of batches (verified empirically) and reduced every step to a fixed size,
    # discarding e's magnitude entirely and keeping only its sign.
    # Practical implication: a wh_learning_rate/sv_learning_rate value tuned against main
    # (v1) does NOT carry over to this version -- it will silently run with a very
    # different (and probably wrong) effective step size. Old configs need re-tuning,
    # not blind reuse of the numeric value.
    wh_learning_rate: float = 5e-3
    sv_learning_rate: float = 1e-3

    # Rare safety net only -- NOT tuned by Optuna. Effective ceiling is always
    # safety_clip_multiplier_{wh,sv} * wh_learning_rate/sv_learning_rate * ||ref||, so it
    # scales automatically with whatever lr gets tuned to and can never independently
    # collapse back to a near-always-binding threshold (which fixing it at an absolute
    # constant would risk).
    safety_clip_multiplier_wh: float = 20.0
    safety_clip_multiplier_sv: float = 20.0

    # Smoothing rate for the EMA of ||direction@wh|| / ||grad_sv_row|| used to normalize
    # the natural-gradient direction. Keeps the normalization denominator stable across
    # batches (a single low-spike-count batch can't make the direction estimate noisy)
    # while the direction itself stays fully responsive to the current batch.
    ema_alpha: float = 0.95

    # --- Learning-rate-alone ablation ---
    # False (default): current behaviour -- wh_learning_rate/sv_learning_rate scales the
    # direction-normalized natural gradient AND the signed calibration error e_wh/e_sv (step
    # magnitude tracks how wrong the model currently is; sign target-tracks calibration
    # statistics).
    # True: drop the e_wh/e_sv factor -- wh_learning_rate/sv_learning_rate alone sets a
    # constant relative step applied every batch, direction-normalized but otherwise
    # unconditional. This is the closest available reproduction of main (v1)'s
    # fixed-learning-rate update, now built on the EMA-normalized direction rather than
    # main's raw gradient. For sv this also flips the update from error-correcting descent
    # to natural-gradient ASCENT -- main's sv update maximized contrast unconditionally (no
    # error term at all); see ops.py::update_sv_spike_gated and adaptation.py::_update_wh
    # for why wh's sign is unaffected but sv's must flip.
    lr_alone: bool = False

    min_spikes_for_update: int = 1      # Minimum spike count to allow sv row update

    # --- Silence penalty (opt-in ablation of NaN-exclusion) ---
    # When a unit has fewer than min_spikes_for_update trusted spikes in a batch
    # (spike_based contrast_scope only -- batch_based never gates), its
    # contrast_error is normally excluded (NaN) from sv_loss so it doesn't bias
    # the loss, but this also hides real failures (whitening/sv-update collapse)
    # behind indistinguishable "no spike" windows. When silence_penalty is True,
    # a fixed z-score (silence_penalty_zscore) is used instead of NaN. The sv
    # update itself is unaffected either way -- grad_sv stays masked to zero for
    # inactive units, only the reported/optimised loss value changes.
    silence_penalty: bool = False
    silence_penalty_zscore: float = -3.0

    orthonormalization: str = "qr"      # "qr" (default), "gram_schmidt", or "none"

    # Contrast scope: how kappa and kappa_cal are computed for the sv update.
    #   "batch_based"  — log_cosh(sources).mean(dim=0) over all N samples; gradient is also
    #                    batch-averaged (tanh(sources).T @ Z / N), decoupled from spike detection
    #   "spike_based"  — log_cosh(sources) averaged only at detected spike times; gradient
    #                    is spike-gated (tanh(sources[spike_mask]).T @ Z / spike_counts)
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
    source_fifo_batches: int = 2    # Past batches of sources prepended for edge spike support

    # --- Calibration sigma estimation ---
    # Maximum number of calibration windows used to estimate sigma_K_cal and sigma_kappa_cal.
    # Windows are sampled uniformly across the calibration recording.
    # 0 = use all windows. On CPU, capping at 200-300 gives stable estimates with ~8× speedup.
    max_sigma_batches: int = 300

    # --- Centroid adaptation ---
    centroid_momentum: float = 0.95
    min_spikes_for_centroid: int = 1
    min_base_peaks_for_centroid: int = 1

    # --- sv fixed-point iterations ---
    sv_epochs: int = 1       # Max ICA fixed-point iterations per batch for sv (1 = single step)
    sv_tol: float = 1e-4     # Early-exit threshold: ‖sv_new − sv_old‖_F / ‖sv_old‖_F < sv_tol

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
        validate_literals(self)


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
