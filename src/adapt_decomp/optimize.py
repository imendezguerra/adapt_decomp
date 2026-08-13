"""Optuna-based hyperparameter optimisation for AdaptDecomp.

Self-contained: no dependency on decomposition.cbss. All inputs are raw tensors
using the same conventions as AdaptDecomp.__init__ (sep_vectors is [M, D]).

Default search space
--------------------
DEFAULT_PARAM_SPACE covers two of the three parameters whose empirical optimal
values cluster in well-defined ranges across simulated and experimental
datasets (the third, centroid_momentum, is included below but commented out
pending further validation):

    wh_learning_rate  — whitening learning rate (log-uniform)
    sv_learning_rate  — separation-vector learning rate (log-uniform)
    centroid_momentum — EMA momentum for spike/base centroid tracking (uniform)

batch_ms is intentionally excluded from the default space because changing it
alters the covariance-estimation window and the kappa_cal reference, requiring
dedicated experiments. To include it, add it to the param_space dict explicitly:
    param_space = {**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.config import Config, validate_literals

DEFAULT_PARAM_SPACE: dict = {
    "wh_learning_rate":   ("log_float", 1e-4, 5e-2),
    "sv_learning_rate":   ("log_float", 1e-4, 1e-1),
    # "centroid_momentum": ("float",     0.70, 0.98),
}


def _suggest_overrides(trial, param_space: dict) -> dict:
    """Suggest one value per param_space entry for this trial."""
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
    base_centr: torch.Tensor,
    spikes_centr: torch.Tensor,
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
) -> tuple:
    """Search for optimal AdaptDecomp hyperparameters using Bayesian optimisation.

    Reuses a single AdaptDecomp instance across all Optuna trials, resetting
    calibration state between runs via _reset_params(). The loss is
    wh_loss + sv_loss (centroid_loss excluded — it is 0-2% of total and has a
    mild anti-signal correlation with RoA).

    Default sampler: CmaEsSampler (n_startup_trials=15). CMA-ES is preferred
    over TPE because the optimal (delta_v, delta_b, centroid_momentum)
    configurations are jointly constrained along a ridge in parameter space;
    CMA-ES learns this covariance, TPE cannot.

    param_space format::

        {
            "wh_learning_rate":   ("log_float", 1e-4, 5e-2),
            "sv_learning_rate":   ("log_float", 1e-4, 1e-1),
            # "centroid_momentum": ("float",     0.70, 0.98),
        }

    Use DEFAULT_PARAM_SPACE for the recommended defaults. To also search
    batch_ms, extend it: {**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}.

    Returns (best_config_dict, optuna.Study).
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
    validate_literals(run_config)

    adapter = AdaptDecomp(
        emg=emg,
        whitening=whitening,
        sep_vectors=sep_vectors,
        base_centr=base_centr,
        spikes_centr=spikes_centr,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        preprocess=preprocess,
        config=run_config,
    )

    def objective(trial):
        overrides = _suggest_overrides(trial, param_space)
        return adapter.run_optimisation(config_overrides=overrides)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.CmaEsSampler(
            n_startup_trials=15,
        ),
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_config = {**(base_config or {}), **best_params}
    return best_config, study


def run_with_optimization(
    emg: torch.Tensor,
    whitening: torch.Tensor,
    sep_vectors: torch.Tensor,
    base_centr: torch.Tensor,
    spikes_centr: torch.Tensor,
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
) -> tuple:
    """Optimise hyperparameters then run the full decomposition with the best config.

    Returns (outputs_dict, best_config_dict, optuna.Study).
    outputs_dict is the raw AdaptDecomp.run() output (keys: spikes, ipts, wh_loss, …).
    """
    best_config, study = optimize_adapt_decomp(
        emg=emg,
        whitening=whitening,
        sep_vectors=sep_vectors,
        base_centr=base_centr,
        spikes_centr=spikes_centr,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        param_space=param_space,
        base_config=base_config,
        n_trials=n_trials,
        preprocess=preprocess,
        sampler=sampler,
        config=config,
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
        base_centr=base_centr,
        spikes_centr=spikes_centr,
        emg_calib=emg_calib,
        ipts_calib=ipts_calib,
        spikes_calib=spikes_calib,
        preprocess=preprocess,
        config=run_config,
    )
    outputs = adapter.run()
    return outputs, best_config, study
