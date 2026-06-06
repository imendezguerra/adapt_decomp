"""Optuna-based hyperparameter optimisation for AdaptDecomp.

Self-contained: no dependency on decomposition.cbss. All inputs are raw tensors
using the same conventions as AdaptDecomp.__init__ (sep_vectors is [M, D]).

Default search space
--------------------
DEFAULT_PARAM_SPACE covers the three parameters whose empirical optimal values
cluster in well-defined ranges across simulated and experimental datasets:

    max_rel_delta_v   — whitening learning rate (log-uniform)
    max_rel_delta_b   — separation-vector learning rate (log-uniform)
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
from adapt_decomp.config import Config

DEFAULT_PARAM_SPACE: dict = {
    "max_rel_delta_v":   ("log_float", 1e-4, 5e-2),
    "max_rel_delta_b":   ("log_float", 1e-3, 1e-1),
    "centroid_momentum": ("float",     0.70, 0.98),
}


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
            "max_rel_delta_v":   ("log_float", 1e-4, 5e-2),
            "max_rel_delta_b":   ("log_float", 1e-3, 1e-1),
            "centroid_momentum": ("float",     0.70, 0.98),
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

    best_config = {**(base_config or {}), **best_params}
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
