"""Optuna-based hyperparameter optimisation for AdaptDecomp.

Self-contained: no dependency on decomposition.cbss. All inputs are raw tensors
using the same conventions as AdaptDecomp.__init__ (sep_vectors is [M, D]).

Default search space
--------------------
DEFAULT_PARAM_SPACE covers the three parameters whose empirical optimal values
cluster in well-defined ranges across simulated and experimental datasets:

    lr_v              — whitening learning rate (log-uniform)
    lr_b              — separation-vector learning rate (log-uniform)
    centroid_momentum — EMA momentum for spike/base centroid tracking (uniform)

batch_ms is intentionally excluded from the default space because changing it
alters the covariance-estimation window and the kappa_cal reference, requiring
dedicated experiments. To include it, add it to the param_space dict explicitly:
    param_space = {**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}
"""

from __future__ import annotations

import copy
from typing import Literal, Optional

import torch

from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.config import Config, validate_literals

DEFAULT_PARAM_SPACE: dict = {
    "lr_v":   ("log_float", 1e-4, 5e-2),
    "lr_b":   ("log_float", 1e-4, 1e-1),
    # "centroid_momentum": ("float",     0.70, 0.98),
}


def _suggest_overrides(trial, param_space: dict) -> dict:
    """Suggest one value per param_space entry for this trial.

    Shared by optimize_adapt_decomp and optimize_adapt_decomp_pooled so both
    interpret a param_space spec identically.
    """
    overrides = {}
    for name, spec in param_space.items():
        kind = spec[0]
        if kind == "log_float":
            overrides[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            overrides[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            overrides[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            overrides[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown param_space kind: {kind!r}")
    return overrides


def optimize_adapt_decomp(
    emg: torch.Tensor,
    whitening: torch.Tensor,
    sep_vectors: torch.Tensor,
    base_centroids: torch.Tensor,
    spike_centroids: torch.Tensor,
    emg_calib: torch.Tensor,
    ipts_calib: torch.Tensor,
    spikes_calib: torch.Tensor,
    param_space: dict,
    *,
    base_config: Optional[dict] = None,
    n_trials: int = 100,
    preprocess: bool = False,
    sampler=None,
    config: Optional[Config] = None,
    optim_mode: Literal["single", "multiobjective"] = "single",
) -> tuple:
    """Search for optimal AdaptDecomp hyperparameters using Bayesian optimisation.

    Reuses a single AdaptDecomp instance across all Optuna trials, resetting
    calibration state between runs via _reset_params(). The single-objective
    loss is wh_loss + sv_loss (centroid_loss excluded — it is 0-2% of total
    and has a mild anti-signal correlation with RoA). For multiobjective,
    returns (wh_loss, sv_loss, centroid_loss) as a 3-objective Pareto problem.

    Default sampler for single-objective: CmaEsSampler (n_startup_trials=15).
    CMA-ES is preferred over TPE because the optimal (delta_v, delta_b,
    centroid_momentum) configurations are jointly constrained along a ridge
    in parameter space; CMA-ES learns this covariance, TPE cannot.

    param_space format::

        {
            "lr_v":   ("log_float", 1e-4, 5e-2),
            "lr_b":   ("log_float", 1e-4, 1e-1),
            # "centroid_momentum": ("float",     0.70, 0.98),
        }

    Use DEFAULT_PARAM_SPACE for the recommended defaults. To also search
    batch_ms, extend it: {**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}.

    Returns (best_config_dict, optuna.Study).
    For multiobjective, study.best_trials returns the full Pareto front;
    best_config_dict is selected by minimum sum of objective values.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optimize_adapt_decomp requires optuna: pip install optuna")

    run_config = config if config is not None else Config()
    if base_config:
        for k, v in base_config.items():
            setattr(run_config, k, v)
    # optim_mode is the single source of truth for how many values
    # run_optimisation() returns -- keep run_config.optim_loss in sync so a
    # caller never has to separately/correctly set both.
    run_config.optim_loss = "multi_obj" if optim_mode == "multiobjective" else "single_obj"
    validate_literals(run_config)

    adapter = AdaptDecomp(
        emg=emg,
        whitening=whitening,
        sep_vectors=sep_vectors,
        base_centroids=base_centroids,
        spike_centroids=spike_centroids,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        preprocess=preprocess,
        config=run_config,
    )

    def objective(trial):
        overrides = _suggest_overrides(trial, param_space)
        return adapter.run_optimisation(config_overrides=overrides)

    if optim_mode == "multiobjective":
        study = optuna.create_study(
            directions=["minimize", "minimize", "minimize"],
            sampler=sampler if sampler is not None else optuna.samplers.NSGAIISampler(),
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler if sampler is not None else optuna.samplers.CmaEsSampler(
                n_startup_trials=15,
            ),
        )

    study.optimize(objective, n_trials=n_trials)

    if optim_mode == "multiobjective":
        # Select the Pareto-front trial with minimum sum of objective values
        best_trial = min(study.best_trials, key=lambda t: sum(t.values))
        best_params = best_trial.params
    else:
        best_params = study.best_params

    best_config = {**(base_config or {}), **best_params, "optim_loss": run_config.optim_loss}
    return best_config, study


def optimize_adapt_decomp_pooled(
    recordings: list,
    param_space: dict,
    *,
    base_config: Optional[dict] = None,
    n_trials: int = 100,
    preprocess: bool = False,
    sampler=None,
    config: Optional[Config] = None,
) -> tuple:
    """Search for hyperparameters minimising the SUMMED single-objective loss
    across multiple recordings at once, instead of one reference recording.

    Use this instead of optimize_adapt_decomp to test whether a single
    (lr_v, lr_b) generalises across conditions rather than overfitting to
    whichever single recording was used as reference. Recordings to hold out
    for a generalisation check simply aren't included in `recordings`.

    Each element of `recordings` is a dict with the same keys as
    optimize_adapt_decomp's tensor arguments: emg, whitening, sep_vectors,
    base_centroids, spike_centroids, emg_calib, ipts_calib, spikes_calib.

    Single-objective only -- summing a 3-tuple (multi_obj) across recordings
    would no longer be a meaningful Pareto problem, so multiobjective pooling
    isn't supported here.

    The divergence guard (trace_check) already returns 1e10 per recording when
    it diverges, so a setting that blows up on even one pooled recording
    dominates the sum rather than being averaged away.

    Returns (best_config_dict, optuna.Study). Per-recording losses for each
    trial are stored in trial.user_attrs['loss_<i>'] (index into `recordings`).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optimize_adapt_decomp_pooled requires optuna: pip install optuna")

    run_config_template = config if config is not None else Config()
    if base_config:
        for k, v in base_config.items():
            setattr(run_config_template, k, v)
    run_config_template.optim_loss = "single_obj"
    validate_literals(run_config_template)

    adapters = [
        AdaptDecomp(
            emg=rec["emg"],
            whitening=rec["whitening"],
            sep_vectors=rec["sep_vectors"],
            base_centroids=rec["base_centroids"],
            spike_centroids=rec["spike_centroids"],
            emg_calib=rec["emg_calib"],
            ipts_calib=rec["ipts_calib"],
            spikes_calib=rec["spikes_calib"],
            preprocess=preprocess,
            config=copy.deepcopy(run_config_template),
        )
        for rec in recordings
    ]

    def objective(trial):
        overrides = _suggest_overrides(trial, param_space)
        losses = [a.run_optimisation(config_overrides=overrides) for a in adapters]
        for i, loss_i in enumerate(losses):
            trial.set_user_attr(f"loss_{i}", loss_i)
        return sum(losses)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.CmaEsSampler(n_startup_trials=15),
    )
    study.optimize(objective, n_trials=n_trials)

    best_config = {**(base_config or {}), **study.best_params, "optim_loss": "single_obj"}
    return best_config, study


def run_with_optimization(
    emg: torch.Tensor,
    whitening: torch.Tensor,
    sep_vectors: torch.Tensor,
    base_centroids: torch.Tensor,
    spike_centroids: torch.Tensor,
    emg_calib: torch.Tensor,
    ipts_calib: torch.Tensor,
    spikes_calib: torch.Tensor,
    param_space: dict,
    *,
    base_config: Optional[dict] = None,
    n_trials: int = 100,
    preprocess: bool = False,
    sampler=None,
    config: Optional[Config] = None,
    optim_mode: Literal["single", "multiobjective"] = "single",
) -> tuple:
    """Optimise hyperparameters then run the full decomposition with the best config.

    Returns (outputs_dict, best_config_dict, optuna.Study).
    outputs_dict is the raw AdaptDecomp.run() output (keys: spikes, ipts, wh_loss, …).
    For multiobjective, study.best_trials exposes the full Pareto front.
    """
    best_config, study = optimize_adapt_decomp(
        emg=emg,
        whitening=whitening,
        sep_vectors=sep_vectors,
        base_centroids=base_centroids,
        spike_centroids=spike_centroids,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        param_space=param_space,
        base_config=base_config,
        n_trials=n_trials,
        preprocess=preprocess,
        sampler=sampler,
        config=config,
        optim_mode=optim_mode,
    )

    run_config = copy.deepcopy(config) if config is not None else Config()
    for k, v in best_config.items():
        setattr(run_config, k, v)
    if "batch_ms" in best_config:
        run_config.batch_size = int(run_config.batch_ms * run_config.fs / 1000)
    validate_literals(run_config)

    adapter = AdaptDecomp(
        emg=emg,
        whitening=whitening,
        sep_vectors=sep_vectors,
        base_centroids=base_centroids,
        spike_centroids=spike_centroids,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        preprocess=preprocess,
        config=run_config,
    )
    outputs = adapter.run()
    return outputs, best_config, study
