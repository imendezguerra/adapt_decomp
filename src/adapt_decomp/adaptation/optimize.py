"""Optuna-based hyperparameter optimisation for AdaptDecomp."""

from __future__ import annotations

import copy
import pickle
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from loguru import logger

from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.spikes.comparison import rate_of_agreement_paired
from adapt_decomp.utils import validate_literals
from adapt_decomp.utils.loaders import PooledDatasetMemory, PooledDatasetDisk

# ------------------------------------------------------------------
# Param space
# ------------------------------------------------------------------

DEFAULT_PARAM_SPACE: dict = {
    "wh_learning_rate":   ("log_float", 1e-4, 5e-2),
    "sv_learning_rate":   ("log_float", 1e-4, 1e-1),
}


# ------------------------------------------------------------------
# Objective scoring
# ------------------------------------------------------------------

ObjectiveName = Literal["sv_loss", "wh_loss", "total_loss", "roa"]

# Maps each loss-valued ObjectiveName to the AdaptationResult field it reads.
_OBJECTIVE_FIELD: Dict[str, str] = {
    "sv_loss": "sv_loss_median",
    "wh_loss": "wh_loss_median",
    "total_loss": "total_loss",
}
_VALID_OBJECTIVES: Tuple[str, ...] = (*_OBJECTIVE_FIELD, "roa")


def _base_losses(outputs: AdaptationResult) -> Dict[str, float]:
    """Read one trial's guarded per-run losses off outputs.

    Args:
        outputs (AdaptationResult): A single run's result, with compute_loss=True
            (so wh_loss_median/sv_loss_median/total_loss are all set).

    Returns:
        Dict[str, float]: {"sv_loss": ..., "wh_loss": ..., "total_loss": ...}.
    """
    return {name: getattr(outputs, field).item() for name, field in _OBJECTIVE_FIELD.items()}


def _roa_loss(roa_mean: float, diverged: bool) -> float:
    """Invert a mean RoA (%) into a guarded, lower-is-better loss for objective="roa".

    Args:
        roa_mean (float): Mean rate of agreement against ground truth, on a 0-100
            scale.
        diverged (bool): Whether this run's base losses already hit the 1e10
            divergence sentinel (see AdaptDecomp._compute_losses()).

    Returns:
        float: 100.0 - roa_mean, or 1e10 if diverged or roa_mean is NaN.
    """
    if diverged or np.isnan(roa_mean):
        return 1e10
    return 100.0 - roa_mean


# ------------------------------------------------------------------
# Trial building blocks
# ------------------------------------------------------------------

def _suggest_overrides(trial: optuna.trial.Trial, param_space: dict) -> dict:
    """Suggest one value per param_space entry for this trial.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple, where kind is "log_float", "float", or "int" (or
            (kind, choices) for "categorical"). See
            optimize_adapt_decomp_pooled_memory's docstring for the full
            format and DEFAULT_PARAM_SPACE.

    Returns:
        dict: Parameter name -> suggested value, one entry per param_space
        key.
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


def _build_trial_config(run_config: AdaptConfig, overrides: dict) -> AdaptConfig:
    """Deep-copy run_config and apply a trial's suggested parameter overrides.

    Args:
        run_config (AdaptConfig): Base configuration to copy from, never
            mutated.
        overrides (dict): Parameter name -> value, typically from
            suggest_overrides(). Any AdaptConfig field name is accepted.

    Returns:
        AdaptConfig: A new instance with overrides applied, batch_size
        recomputed from batch_ms if batch_ms was overridden, compute_loss
        forced to True, and validate_literals() already run.
    """
    # Deep-copy run_config to avoid mutating the caller's instance.
    trial_config = copy.deepcopy(run_config)

    # Apply the trial's suggested overrides on top of the copy.
    for k, v in overrides.items():
        setattr(trial_config, k, v)

    # Compute batch_size from batch_ms if the trial suggested a new batch_ms.
    if "batch_ms" in overrides:
        trial_config.batch_size = int(trial_config.batch_ms * trial_config.fs / 1000)

    # Force loss computation for the optimisation
    trial_config.compute_loss = True

    # Validate the trial_config to ensure all fields are valid before running the trial.
    validate_literals(trial_config)
    return trial_config


# ------------------------------------------------------------------
# Best-result persistence
# ------------------------------------------------------------------

def _save_best_trial(
    best_dir: Path,
    outputs: Union[AdaptationResult, Dict[str, AdaptationResult]],
    trial_config: AdaptConfig,
) -> None:
    """Overwrite best_dir's saved AdaptationResult(s) + config with an improving trial.

    Args:
        best_dir (Path): Directory to write into (already created by the
            caller). "result.pkl" for a single AdaptationResult, or one
            "<dataset>.pkl" per entry for a Dict[str, AdaptationResult]
            (pooled search).
        outputs (Union[AdaptationResult, Dict[str, AdaptationResult]]): This
            trial's result(s), see AdaptationResult.save().
        trial_config (AdaptConfig): This trial's resolved configuration,
            written alongside outputs as "config.yaml".

    Returns:
        None
    """
    if isinstance(outputs, dict):
        for dataset, result in outputs.items():
            result.save(best_dir / f"{dataset}.pkl")
    else:
        outputs.save(best_dir / "result.pkl")

    trial_config.to_yaml(best_dir / "config.yaml")


def _finalize_best_result(best_dir: Path, study: optuna.Study) -> None:
    """Pickle the completed study into best_dir and log where everything landed.

    Args:
        best_dir (Path): Directory already holding the best trial's
            AdaptationResult(s)/config.yaml, written by _save_best_trial()
            over the course of the search.
        study (optuna.Study): The completed Optuna study.

    Returns:
        None
    """
    with open(best_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)
    saved = ", ".join(sorted(p.name for p in best_dir.iterdir()))
    logger.info(f"Saved best trial to {best_dir} ({saved})")


def _promote_temp_to_best(
    temp_dir: Path, best_dir: Path, dataset_names: Iterable[str], trial_config: AdaptConfig,
    trial_number: int,
) -> None:
    """Copy this trial's already-saved per-dataset results from temp_dir to best_dir.

    Args:
        temp_dir (Path): Directory holding this trial's
            "<trial_number>_<dataset>.pkl" files, written by the trial loop
            before this is called.
        best_dir (Path): Directory to overwrite, already created by the
            caller.
        dataset_names (Iterable[str]): Every dataset name to copy over
            (typically pool.keys()).
        trial_config (AdaptConfig): This trial's resolved configuration,
            written alongside the copied results as "config.yaml".
        trial_number (int): This trial's Optuna trial.number, matching the
            scratch filenames written by the trial loop -- unique per trial
            even under n_jobs > 1, so concurrent trials never read each
            other's files.

    Returns:
        None
    """
    for dataset in dataset_names:
        shutil.copy2(temp_dir / f"{trial_number}_{dataset}.pkl", best_dir / f"{dataset}.pkl")
    trial_config.to_yaml(best_dir / "config.yaml")


# ------------------------------------------------------------------
# Shared pooled-trial body
# ------------------------------------------------------------------

def _run_one_dataset(
    trial: optuna.trial.Trial,
    name: str,
    emg: Union[torch.Tensor, np.ndarray],
    calibration: CBSSResult,
    cbss_config: CBSSConfig,
    preprocess: bool,
    gt_paired_bin: Optional[np.ndarray],
    run_config: AdaptConfig,
    overrides: dict,
    objective: ObjectiveName,
    compute_roa: bool,
    roa_kwargs: Optional[dict],
) -> Tuple[AdaptationResult, Dict[str, Any], AdaptConfig]:
    """Run one trial's AdaptDecomp for a single dataset, logging its user_attrs.

    Shared by optimize_adapt_decomp_pooled_memory and optimize_adapt_decomp_pooled_disk's
    trial loops -- the sole per-dataset runner for both, whether their pool has one dataset
    or many.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        name (str): This dataset's pool key, used to namespace its
            trial.set_user_attr() entries (e.g. "sv_loss_<name>").
        emg (Union[torch.Tensor, np.ndarray]): Online EMG to decompose, with
            shape (samples, channels).
        calibration (CBSSResult): This dataset's calibration result.
        cbss_config (CBSSConfig): The CBSSConfig that produced calibration.
        preprocess (bool): Whether to preprocess emg before extension.
        gt_paired_bin (Optional[np.ndarray]): Ground-truth binary spike
            train matched to calibration's units, with shape (samples, M).
            Required when compute_roa is True.
        run_config (AdaptConfig): Base configuration this trial's config is
            deep-copied from, before overrides are applied.
        overrides (dict): This trial's suggested parameter overrides, from
            suggest_overrides(), shared across every dataset in the pool.
        objective (ObjectiveName): Which scalar this dataset's "loss" entry
            reads -- "sv_loss", "wh_loss", "total_loss", or "roa".
        compute_roa (bool): If True, score RoA against gt_paired_bin and
            include it in the returned losses.
        roa_kwargs (Optional[dict]): Extra keyword arguments forwarded to
            rate_of_agreement_paired() when compute_roa is True.

    Returns:
        Tuple[AdaptationResult, Dict[str, Any], AdaptConfig]: outputs, this
        dataset's result; losses, {"sv_loss", "wh_loss", "total_loss",
        "loss"}, plus {"roa", "roa_mean", "roa_per_unit"} when compute_roa is
        True; trial_config, this dataset's resolved AdaptConfig.
    """
    trial_config = _build_trial_config(run_config, overrides)

    adapter = AdaptDecomp.from_calibration(
        emg, calibration=calibration, cbss_config=cbss_config,
        adapt_config=trial_config, preprocess=preprocess,
    )
    outputs = adapter.run()

    losses = _base_losses(outputs)
    diverged = losses["total_loss"] >= 1e10
    for loss_name, value in losses.items():
        trial.set_user_attr(f"{loss_name}_{name}", value)

    if compute_roa:
        pred_spikes = adapter.spikes.detach().cpu().numpy().astype(np.float32)
        roa_vals, _, _ = rate_of_agreement_paired(gt_paired_bin, pred_spikes, **roa_kwargs)
        outputs.roa = np.asarray(roa_vals, dtype=np.float32)  # travels with outputs.save()
        roa_mean = float(np.nanmean(roa_vals)) * 100
        roa_per_unit = [float(x) for x in roa_vals]
        roa_loss = _roa_loss(roa_mean, diverged)
        trial.set_user_attr(f"roa_mean_{name}", roa_mean)
        trial.set_user_attr(f"roa_per_unit_{name}", roa_per_unit)
        trial.set_user_attr(f"roa_{name}", roa_loss)
        losses["roa_mean"] = roa_mean
        losses["roa_per_unit"] = roa_per_unit
        losses["roa"] = roa_loss

    losses["loss"] = losses["roa" if objective == "roa" else objective]
    trial.set_user_attr(f"loss_{name}", losses["loss"])

    return outputs, losses, trial_config


# ------------------------------------------------------------------
# Search entry points
# ------------------------------------------------------------------

def optimize_adapt_decomp_pooled_memory(
    *,
    pool: Dict[str, PooledDatasetMemory],
    objective: ObjectiveName = "total_loss",
    base_config: Optional[AdaptConfig] = None,
    compute_roa: bool = False,
    roa_kwargs: Optional[dict] = None,
    param_space: dict,
    n_trials: int = 100,
    n_jobs: int = 1,
    sampler=None,
    random_seed: Optional[int] = 1909,
    best_result_path: Optional[str] = None,
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[AdaptConfig, optuna.Study]:
    """Search hyperparameters shared across every dataset in pool, summing per-dataset loss.

    Applies one suggested parameter set per trial to a fresh AdaptDecomp per
    dataset (via PooledDatasetMemory.resolve() + _run_one_dataset());
    the objective is the SUM of per-dataset losses.

    Args:
        pool (Dict[str, PooledDatasetMemory]): Dataset name -> its
            calibration and optional ground truth, read via
            PooledDatasetMemory.resolve() before each dataset's run. Every
            dataset is evaluated on every trial. Each dataset's cbss_config
            wins over base_config's shared fields on disagreement, applied
            fresh every trial inside from_calibration(), see
            adaptation.core.reconcile_with_calib_config.
        objective (ObjectiveName, optional): Which scalar to score a trial
            on, "sv_loss", "wh_loss", "total_loss", or "roa" (implies
            compute_roa=True and requires every dataset's gt_paired_bin).
            Defaults to "total_loss".
        base_config (Optional[AdaptConfig], optional): Resolved base
            AdaptConfig instance each trial/dataset is deep-copied from,
            before param_space overrides are applied. Defaults to None,
            which uses AdaptConfig().
        compute_roa (bool, optional): If True, log RoA against every
            dataset's gt_paired_bin for every trial ("roa_mean",
            "roa_per_unit", per dataset and pooled) and write it onto the
            winning trial's AdaptationResult.roa. Requires every dataset in
            pool to set gt_paired_bin. Forced to True when objective="roa".
            Defaults to False.
        roa_kwargs (Optional[dict], optional): Extra keyword arguments
            forwarded to rate_of_agreement_paired() (e.g. tol_spike_ms).
            "fs" defaults to the resolved config's fs unless overridden
            here. Ignored when compute_roa is False. Defaults to None.
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple, where kind is "log_float", "float", or "int" (or
            (kind, choices) for "categorical"). Use DEFAULT_PARAM_SPACE for
            the recommended defaults::
                {
                    "wh_learning_rate":   ("log_float", 1e-4, 5e-2),
                    "sv_learning_rate":   ("log_float", 1e-4, 1e-1),
                }
            To also search batch_ms or other parameters, extend it:
            {**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}.
        n_trials (int, optional): Number of Optuna trials to run. Defaults
            to 100.
        n_jobs (int, optional): Passed straight to study.optimize()'s
            n_jobs -- number of trials to run concurrently (thread-based).
            Defaults to 1 (sequential).
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses
            TPESampler(n_startup_trials=15).
        random_seed (Optional[int], optional): Seed for the default
            TPESampler. Note that exact reproducibility only holds for
            n_jobs=1. Defaults to 1909.
        best_result_path (Optional[str], optional): If set, the best trial
            so far is written here as one AdaptationResult per dataset
            ("<dataset>.pkl") plus "config.yaml", and "study.pkl" once
            the search finishes. Defaults to None.
        on_trial (Optional[Callable[[Dict[str, Any]], None]], optional):
            Called once per completed trial with a log dict:
                {"trial_number": int, "loss": float, "objective": str,
                 "sv_loss": float, "wh_loss": float, "total_loss": float,
                 "roa": float, "params": dict,
                 "per_dataset": {name: {"loss": float, "sv_loss": float,
                                         "wh_loss": float, "total_loss": float,
                                         "roa": float, "roa_mean": float,
                                         "roa_per_unit": List[float]}},
                 "roa_mean": float}
            Top-level sv_loss/wh_loss/total_loss/roa are pooled sums;
            per_dataset holds each dataset's own values; roa_mean (top
            level) is the pooled mean of per-dataset RoA means, a
            different aggregation from the pooled "roa" sum. roa/roa_mean
            keys only present when compute_roa is True. Defaults to None.

    Raises:
        ValueError: If objective is not one of "sv_loss", "wh_loss",
            "total_loss", "roa", or if any PooledDatasetMemory in pool has
            gt_paired_bin=None while compute_roa is True or objective="roa"
            (the latter implies the former).

    Returns:
        Tuple[AdaptConfig, optuna.Study]: best_config, the resolved
        base_config with the best trial's parameters applied; study, the
        completed Optuna study. If best_result_path is set, returns
        (outputs, best_config, study) instead, with outputs mapping
        dataset name to its AdaptationResult.
    """
    run_config = base_config if base_config is not None else AdaptConfig()
    validate_literals(run_config)

    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective: {objective!r}; expected one of {_VALID_OBJECTIVES}"
        )

    # objective="roa" needs RoA computed every trial to have anything to score on.
    if objective == "roa":
        compute_roa = True

    if compute_roa:
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}
        missing = [name for name, dataset in pool.items() if dataset.gt_paired_bin is None]
        if missing:
            raise ValueError(
                f"gt_paired_bin is required (compute_roa=True or objective='roa') for every "
                f"dataset in pool; missing for: {missing}"
            )

    # Best-so-far tracking, only active when best_result_path is set.
    best_dir = Path(best_result_path) if best_result_path is not None else None
    if best_dir is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_outputs: Optional[Dict[str, AdaptationResult]] = None
    best_lock = threading.Lock()  # guards best_loss/best_outputs/best_dir writes under n_jobs>1

    # Define the objective to optimise
    def _trial_objective(trial):
        nonlocal best_loss, best_outputs

        # ONE suggestion, shared across every dataset in the pool.
        overrides = _suggest_overrides(trial, param_space)

        # Pool-based value tracking
        pooled_losses = {"sv_loss": 0.0, "wh_loss": 0.0, "total_loss": 0.0}
        pooled_roa_loss = 0.0   # sum of per-dataset _roa_loss(), matching pooled_losses' sum
        roa_means = []          # per-dataset roa_mean, for the separate mean diagnostic below
        trial_outputs: Dict[str, AdaptationResult] = {}
        per_dataset: Dict[str, Dict[str, Any]] = {}
        reference_config: Optional[AdaptConfig] = None

        for name, dataset in pool.items():
            emg, calibration, cbss_config, preprocess, gt_paired_bin = dataset.resolve()
            outputs, losses, trial_config = _run_one_dataset(
                trial, name, emg, calibration, cbss_config, preprocess, gt_paired_bin,
                run_config, overrides, objective, compute_roa, roa_kwargs,
            )
            if reference_config is None:
                reference_config = trial_config  # first dataset's, may differ from others' if their cbss_configs disagreed

            # Update pool trackers
            for loss_name in ("sv_loss", "wh_loss", "total_loss"):
                pooled_losses[loss_name] += losses[loss_name]
            trial_outputs[name] = outputs
            per_dataset[name] = losses
            if compute_roa:
                pooled_roa_loss += losses["roa"]
                roa_means.append(losses["roa_mean"])

        pooled_loss = pooled_roa_loss if objective == "roa" else pooled_losses[objective]

        # Canonical per-trial log record, see on_trial's Args entry above.
        log_vars: Dict[str, Any] = {
            "trial_number": trial.number,
            "loss": pooled_loss,
            "objective": objective,
            **pooled_losses,
            "params": overrides,
            "per_dataset": per_dataset,
        }
        for loss_name, value in pooled_losses.items():
            trial.set_user_attr(loss_name, value)

        if compute_roa:
            roa_mean_pooled = float(np.mean(roa_means))
            trial.set_user_attr("roa_mean_pooled", roa_mean_pooled)
            trial.set_user_attr("roa", pooled_roa_loss)
            log_vars["roa_mean"] = roa_mean_pooled
            log_vars["roa"] = pooled_roa_loss

        if on_trial is not None:
            on_trial(log_vars)

        # Overwrite the saved best-so-far result whenever this trial improves on it.
        # Locked: under n_jobs>1, concurrent trials would otherwise race on the
        # read-then-write of best_loss/best_outputs.
        with best_lock:
            if best_dir is not None and pooled_loss < best_loss:
                _save_best_trial(best_dir, trial_outputs, reference_config)
                best_outputs = trial_outputs
                best_loss = pooled_loss

        # Pooled objective: SUM, not mean, across every dataset (see docstring).
        return pooled_loss

    # Build the Optuna study and run the optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            n_startup_trials=15, seed=random_seed,
        ),
    )
    study.optimize(_trial_objective, n_trials=n_trials, n_jobs=n_jobs)

    # Build the winning AdaptConfig via the same helper every trial used.
    best_config = _build_trial_config(run_config, study.best_params)

    if best_dir is not None:
        _finalize_best_result(best_dir, study)
        return best_outputs, best_config, study

    return best_config, study


def optimize_adapt_decomp_pooled_disk(
    *,
    pool: Dict[str, PooledDatasetDisk],
    objective: ObjectiveName = "total_loss",
    base_config: Optional[AdaptConfig] = None,
    compute_roa: bool = False,
    roa_kwargs: Optional[dict] = None,
    param_space: dict,
    n_trials: int = 100,
    n_jobs: int = 1,
    sampler=None,
    random_seed: Optional[int] = 1909,
    best_result_path: Optional[str] = None,
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[AdaptConfig, optuna.Study]:
    """Memory-lean counterpart to optimize_adapt_decomp_pooled_memory.

    Builds each trial's AdaptDecomp via from_calibration() from on-disk
    paths, loaded and discarded per dataset per trial (PooledDatasetDisk.resolve())
    instead of preloaded once. Otherwise identical (both share
    _run_one_dataset() for the per-dataset run+score). best_result_path
    and the return value differ, since results are never held in memory here.

    Args:
        pool (Dict[str, PooledDatasetDisk]): Dataset name -> its
            on-disk calibration paths, loaded fresh via
            PooledDatasetDisk.resolve() at the start of each dataset's run
            every trial. Every dataset is evaluated on every trial.
        objective (ObjectiveName, optional): See optimize_adapt_decomp_pooled_memory.
            Defaults to "total_loss".
        base_config (Optional[AdaptConfig], optional): Resolved base
            AdaptConfig each trial/dataset is deep-copied from. Defaults
            to None, which uses AdaptConfig().
        compute_roa (bool, optional): See optimize_adapt_decomp_pooled_memory --
            every dataset in pool must set path_gt. Forced to True
            when objective="roa". Defaults to False.
        roa_kwargs (Optional[dict], optional): Extra keyword arguments
            forwarded to rate_of_agreement_paired(). "fs" defaults to
            base_config's fs unless overridden here. Ignored when
            compute_roa is False. Defaults to None.
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple. See optimize_adapt_decomp_pooled_memory's docstring and
            DEFAULT_PARAM_SPACE.
        n_trials (int, optional): Number of Optuna trials to run. Defaults
            to 100.
        n_jobs (int, optional): Passed straight to study.optimize()'s
            n_jobs -- number of trials to run concurrently (thread-based).
            Defaults to 1 (sequential).
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses
            TPESampler(n_startup_trials=15).
        random_seed (Optional[int], optional): Seed for the default
            TPESampler. Note that exact reproducibility only holds for
            n_jobs=1. Defaults to 1909.
        best_result_path (Optional[str], optional): Fully optional, as in
            optimize_adapt_decomp_pooled_memory. When set, this is the only
            way to retrieve per-dataset AdaptationResults (see Returns):
            each trial's per-dataset outputs are written to
            "<best_result_path>_temp" under trial-scoped filenames as
            they're computed (so concurrent trials under n_jobs>1 never
            collide), promoted into best_result_path on an improving trial,
            and the temp directory is deleted once the search finishes.
            Defaults to None.
        on_trial (Optional[Callable[[Dict[str, Any]], None]], optional): See
            optimize_adapt_decomp_pooled_memory, identical log dict shape.

    Raises:
        ValueError: If objective is not one of "sv_loss", "wh_loss",
            "total_loss", "roa", or if any PooledDatasetDisk in pool has
            path_gt=None while compute_roa is True or objective="roa".

    Returns:
        Tuple[AdaptConfig, optuna.Study]: best_config, the resolved
        base_config with the best trial's parameters applied; study, the
        completed Optuna study. Unlike optimize_adapt_decomp_pooled_memory,
        never returns per-dataset AdaptationResults in memory, reload
        from best_result_path instead, e.g.
        AdaptationResult.load(Path(best_result_path) / f"{name}.pkl").
    """
    run_config = base_config if base_config is not None else AdaptConfig()
    validate_literals(run_config)

    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective: {objective!r}; expected one of {_VALID_OBJECTIVES}"
        )

    # objective="roa" needs RoA computed every trial to have anything to score on.
    if objective == "roa":
        compute_roa = True

    if compute_roa:
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}
        missing = [name for name, spec in pool.items() if spec.path_gt is None]
        if missing:
            raise ValueError(
                f"path_gt is required (compute_roa=True or objective='roa') for every "
                f"dataset in pool; missing for: {missing}"
            )

    # Best-so-far tracking + scratch space, both only active when best_result_path is set.
    best_dir = Path(best_result_path) if best_result_path is not None else None
    temp_dir = best_dir.with_name(best_dir.name + "_temp") if best_dir is not None else None
    if best_dir is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_lock = threading.Lock()  # guards best_loss/best_dir promotion under n_jobs>1

    # Define the objective to optimise
    def _trial_objective(trial):
        nonlocal best_loss

        # ONE suggestion, shared across every dataset in the pool.
        overrides = _suggest_overrides(trial, param_space)

        # Pool-based value tracking
        pooled_losses = {"sv_loss": 0.0, "wh_loss": 0.0, "total_loss": 0.0}
        pooled_roa_loss = 0.0   # sum of per-dataset _roa_loss(), matching pooled_losses' sum
        roa_means = []          # per-dataset roa_mean, for the separate mean diagnostic below
        per_dataset: Dict[str, Dict[str, Any]] = {}
        reference_config: Optional[AdaptConfig] = None

        for name, spec in pool.items():
            emg, calibration, cbss_config, preprocess, gt_paired_bin = spec.resolve()
            outputs, losses, trial_config = _run_one_dataset(
                trial, name, emg, calibration, cbss_config, preprocess, gt_paired_bin,
                run_config, overrides, objective, compute_roa, roa_kwargs,
            )
            if reference_config is None:
                reference_config = trial_config  # value-identical across datasets

            # Update pool trackers
            for loss_name in ("sv_loss", "wh_loss", "total_loss"):
                pooled_losses[loss_name] += losses[loss_name]
            per_dataset[name] = losses
            if compute_roa:
                pooled_roa_loss += losses["roa"]
                roa_means.append(losses["roa_mean"])

            # Flush to the scratch directory immediately, nothing here is
            # kept resident past this iteration. Filenames are trial-scoped
            # (trial.number is unique even under n_jobs>1) so concurrent
            # trials never collide on the same path.
            if temp_dir is not None:
                outputs.save(temp_dir / f"{trial.number}_{name}.pkl")

        pooled_loss = pooled_roa_loss if objective == "roa" else pooled_losses[objective]

        # Canonical per-trial log record, see on_trial's Args entry above.
        log_vars: Dict[str, Any] = {
            "trial_number": trial.number,
            "loss": pooled_loss,
            "objective": objective,
            **pooled_losses,
            "params": overrides,
            "per_dataset": per_dataset,
        }
        for loss_name, value in pooled_losses.items():
            trial.set_user_attr(loss_name, value)

        if compute_roa:
            roa_mean_pooled = float(np.mean(roa_means))
            trial.set_user_attr("roa_mean_pooled", roa_mean_pooled)
            trial.set_user_attr("roa", pooled_roa_loss)
            log_vars["roa_mean"] = roa_mean_pooled
            log_vars["roa"] = pooled_roa_loss

        if on_trial is not None:
            on_trial(log_vars)

        # Promote this trial's scratch files whenever it improves on the
        # best pooled loss seen so far. Locked: under n_jobs>1, concurrent
        # trials would otherwise race on the read-then-write of best_loss.
        with best_lock:
            if temp_dir is not None and pooled_loss < best_loss:
                _promote_temp_to_best(temp_dir, best_dir, pool.keys(), reference_config, trial.number)
                best_loss = pooled_loss

        # This trial's own scratch files are no longer needed either way --
        # promotion above already copied them if they won. Deleting now (not
        # just at the final rmtree) keeps temp_dir bounded to roughly n_jobs
        # trials' worth of files instead of growing across the whole search.
        if temp_dir is not None:
            for name in pool:
                (temp_dir / f"{trial.number}_{name}.pkl").unlink(missing_ok=True)

        # Pooled objective: sum of the loss across datasets.
        return pooled_loss

    # Build the Optuna study and run the optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            n_startup_trials=15, seed=random_seed,
        ),
    )
    study.optimize(_trial_objective, n_trials=n_trials, n_jobs=n_jobs)

    # Build the winning AdaptConfig via the same helper every trial used.
    best_config = _build_trial_config(run_config, study.best_params)

    if best_dir is not None:
        _finalize_best_result(best_dir, study)
        shutil.rmtree(temp_dir)  # scratch space only, everything worth keeping is in best_dir

    return best_config, study
