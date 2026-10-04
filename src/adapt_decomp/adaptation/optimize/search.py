"""The search entry point, optimize_adapt_decomp, with its inputs, study and result."""

from __future__ import annotations

import shutil
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna
import torch
from loguru import logger

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.optimize.pareto import (
    SELECTION_RULES,
    FrontSelector,
    SelectionName,
    update_front,
    validate_selection,
)
from adapt_decomp.adaptation.optimize.persistence import (
    evict_front_member,
    promote_trial,
    save_study_snapshot,
)
from adapt_decomp.adaptation.optimize.resources import plan_search_resources
from adapt_decomp.adaptation.optimize.scoring import (
    DEFAULT_PARAM_SPACE,
    ObjectiveName,
    build_trial_config,
    pool_trial,
    suggest_overrides,
    validate_objectives,
)
from adapt_decomp.adaptation.optimize.units import (
    DEFAULT_UNIT_SELECTION_KWARGS,
    UnitSelection,
    has_gt,
    select_pool_units,
    validate_unit_selection,
)
from adapt_decomp.adaptation.optimize.workers import run_trials, score_pool, start_workers
from adapt_decomp.utils import validate_literals
from adapt_decomp.utils.loaders import PooledDataset, PooledDatasetDisk, PooledDatasetMemory


@dataclass
class OptimisationResult:
    """Outcome of optimize_adapt_decomp().

    Attributes:
        best_config (AdaptConfig): The resolved base config with the chosen
            trial's parameters applied.
        study (optuna.Study): The completed Optuna study.
        pareto_front (Optional[List[optuna.trial.FrozenTrial]]):
            study.best_trials for a Pareto search; None for a
            single-objective one.
        outputs (Optional[Dict[str, AdaptationResult]]): The chosen trial's
            per-dataset results, reloaded from best_result_path for an
            in-memory pool. None without best_result_path, or for an on-disk
            pool (reload them from best_result_path instead).
    """

    best_config: AdaptConfig
    study: optuna.Study
    pareto_front: Optional[List[optuna.trial.FrozenTrial]] = None
    outputs: Optional[Dict[str, AdaptationResult]] = None


def _prepare_search(
    pool: Dict[str, PooledDataset],
    objectives: Tuple[ObjectiveName, ...],
    base_config: Optional[AdaptConfig],
    compute_roa: bool,
    roa_kwargs: Optional[dict],
    unit_selection: UnitSelection,
    selection: Union[SelectionName, FrontSelector],
) -> Tuple[AdaptConfig, bool, Optional[dict]]:
    """Validate a search's inputs up front, before any trial runs.

    Args:
        pool, objectives, base_config, compute_roa, roa_kwargs,
        unit_selection, selection: As optimize_adapt_decomp.

    Raises:
        TypeError: If pool mixes in-memory and on-disk entries.
        ValueError: On an empty pool, invalid objectives, unit_selection or
            selection, or missing ground truth when compute_roa,
            "roa" in objectives, or unit_selection="supervised" needs it.

    Returns:
        Tuple[AdaptConfig, bool, Optional[dict]]: run_config, the validated
        base config; compute_roa, forced True when "roa" is an objective;
        roa_kwargs, with "fs" defaulted from run_config when compute_roa.
    """
    run_config = base_config if base_config is not None else AdaptConfig()
    validate_literals(run_config)
    validate_objectives(objectives)

    if not pool:
        raise ValueError("pool is empty.")
    if not all(isinstance(d, PooledDatasetMemory) for d in pool.values()) and not all(
        isinstance(d, PooledDatasetDisk) for d in pool.values()
    ):
        raise TypeError("pool must be all PooledDatasetMemory or all PooledDatasetDisk entries.")
    validate_unit_selection(unit_selection)
    validate_selection(selection, objectives)

    compute_roa = compute_roa or "roa" in objectives
    missing = [name for name, dataset in pool.items() if not has_gt(dataset)]
    if missing and (compute_roa or unit_selection == "supervised"):
        raise ValueError(
            "Ground truth (gt_paired_bin/path_gt) is required for every dataset in pool when "
            "compute_roa=True, 'roa' is an objective, or unit_selection='supervised'; "
            f"missing for: {missing}"
        )
    if compute_roa:
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}
    return run_config, compute_roa, roa_kwargs


# Random start-up trials of the default sampler, run as one batch (see run_trials)
DEFAULT_N_STARTUP_TRIALS: int = 15


def _make_study(
    objectives: Tuple[ObjectiveName, ...],
    sampler: Optional[optuna.samplers.BaseSampler],
    random_seed: Optional[int],
    n_jobs: int,
) -> optuna.Study:
    """Create the minimising study, with multivariate TPE unless sampler is given.

    Args:
        objectives (Tuple[ObjectiveName, ...]): Scored objectives, one study
            direction (and metric name) each.
        sampler (Optional[optuna.samplers.BaseSampler]): Sampler to use as-is.
        random_seed (Optional[int]): Seed for the default sampler.
        n_jobs (int): Trials suggested together; above 1 the default sampler
            uses constant_liar so pending trials aren't suggested twice.

    Returns:
        optuna.Study: The new study.
    """
    with warnings.catch_warnings():  # multivariate/constant_liar/metric names are experimental
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=DEFAULT_N_STARTUP_TRIALS,
                multivariate=True,
                constant_liar=n_jobs > 1,
                seed=random_seed,
            )
        study = optuna.create_study(directions=["minimize"] * len(objectives), sampler=sampler)
        study.set_metric_names(list(objectives))
    return study


def optimize_adapt_decomp(
    *,
    pool: Dict[str, PooledDataset],
    objectives: Union[ObjectiveName, Tuple[ObjectiveName, ...]] = "sv_loss",
    param_space: Optional[dict] = None,
    base_config: Optional[AdaptConfig] = None,
    compute_roa: bool = False,
    roa_kwargs: Optional[dict] = None,
    unit_selection: UnitSelection = None,
    unit_selection_kwargs: Optional[dict] = None,
    selection: Union[SelectionName, FrontSelector] = "min_sv_loss",
    n_trials: int = 100,
    n_jobs: int = 1,
    n_cores: Optional[int] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    random_seed: Optional[int] = 1909,
    best_result_path: Optional[str] = None,
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> OptimisationResult:
    """Search AdaptConfig parameters shared across every dataset in pool.

    Every trial applies one suggested parameter set to a fresh AdaptDecomp
    per dataset; each objective is the SUM of its per-dataset values over
    the pool. One objective runs a single-objective search; two or more run
    a Pareto search (optuna directions=[...]) and pick one front member with
    selection.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> its pool entry, all
            PooledDatasetMemory (preloaded) or all PooledDatasetDisk (loaded
            fresh each trial via resolve()). Each dataset's cbss_config wins
            over base_config's shared fields on disagreement, see
            adaptation.core.reconcile_with_calib_config.
        objectives (Union[ObjectiveName, Tuple[ObjectiveName, ...]], optional):
            "sv_loss", "wh_loss", "total_loss" or "roa" (implies
            compute_roa=True), or a tuple of distinct ones for a Pareto
            search, e.g. DEFAULT_OBJECTIVES = ("wh_loss", "sv_loss").
            Defaults to "sv_loss".
        param_space (Optional[dict], optional): Maps parameter name to a
            (kind, low, high) tuple, where kind is "log_float", "float", or
            "int" (or (kind, choices) for "categorical"). To also search
            batch_ms, extend it: {**DEFAULT_PARAM_SPACE, "batch_ms": ("int",
            50, 200)}. Defaults to None, which uses DEFAULT_PARAM_SPACE.
        base_config (Optional[AdaptConfig], optional): Resolved base config
            each trial/dataset is deep-copied from, before overrides. Its
            sv_loss_reduction sets how each dataset's sv_loss reduces across
            units ("mean" weighs datasets alike whatever their unit count).
            Defaults to None, which uses AdaptConfig().
        compute_roa (bool, optional): If True, log RoA against every
            dataset's ground truth for every trial and write it onto the
            saved AdaptationResult.roa. Defaults to False.
        roa_kwargs (Optional[dict], optional): Extra keyword arguments
            forwarded to rate_of_agreement_paired() (e.g. tol_spike_ms).
            "fs" defaults to base_config's fs. Defaults to None.
        unit_selection (UnitSelection, optional): Which calibration units
            the search adapts and scores: None keeps every unit;
            "unsupervised" keeps units passing
            CBSSResult.unsupervised_mask(**unit_selection_kwargs), leaving
            out datasets with none, and is recommended for recordings
            without ground truth; "supervised" requires every dataset's
            ground truth (its loader already narrowed the calibration to
            GT-matched units). Defaults to None.
        unit_selection_kwargs (Optional[dict], optional): Thresholds for
            "unsupervised". Defaults to None, which uses
            DEFAULT_UNIT_SELECTION_KWARGS = {"cov_th": 0.3}.
        selection (Union[SelectionName, FrontSelector], optional): Pareto
            search only: which front member builds best_config --
            "min_sv_loss" (the front's sv_loss extreme), "knee" (two
            objectives only; see _select_knee), "max_roa_mean" (needs
            compute_roa), or a callable taking study.best_trials. Defaults to
            "min_sv_loss".
        n_trials (int, optional): Number of Optuna trials. Defaults to 100.
        n_jobs (int, optional): Trials suggested together after the
            sampler's random start-up trials (which are always suggested
            together). With random_seed it sets the search: 1 is the
            one-at-a-time search, above 1 a batched one. Defaults to 1.
        n_cores (Optional[int], optional): Physical cores the search may
            use; sets only its speed. Datasets are spread over worker
            processes first (plan_resources), leftover cores become torch
            threads per run; a single worker runs in-process. Defaults to
            None, which uses available_cores().
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses multivariate
            TPESampler(n_startup_trials=DEFAULT_N_STARTUP_TRIALS,
            seed=random_seed), with constant_liar when n_jobs > 1. A given
            sampler's start-up trials are not batched.
        random_seed (Optional[int], optional): Seed for the default sampler.
            Defaults to 1909.
        best_result_path (Optional[str], optional): If set, each trial's
            per-dataset AdaptationResults are staged in
            "<best_result_path>_temp" (deleted at the end) and promoted when
            the trial improves on the best so far ("<dataset>.pkl" plus
            "config.yaml" in best_result_path itself) or joins the Pareto
            front ("trial_<n>/" subdirectories, removed when dominated);
            "study.pkl" is rewritten after every trial. Defaults to None.
        on_trial (Optional[Callable[[Dict[str, Any]], None]], optional):
            Called once per completed trial with a log dict:
                {"trial_number": int, "params": dict,
                 "loss": float, "objective": str,           # single-objective
                 "objectives": tuple, "values": tuple,      # Pareto
                 "sv_loss": float, "wh_loss": float, "total_loss": float,
                 "per_dataset": {name: {"sv_loss", "wh_loss", "total_loss",
                                        "loss" (single-objective), "roa",
                                        "roa_mean", "roa_per_unit"}},
                 "roa": float, "roa_mean": float,           # if compute_roa
                 "on_front": bool}                          # if best_result_path
            Top-level losses and "roa" are pooled sums; "roa_mean" is the
            mean of per-dataset RoA means. Defaults to None.

    Raises:
        TypeError, ValueError: See _prepare_search and plan_search_resources.

    Returns:
        OptimisationResult: best_config, study, pareto_front and outputs.
    """
    objectives = (objectives,) if isinstance(objectives, str) else tuple(objectives)
    param_space = param_space if param_space is not None else DEFAULT_PARAM_SPACE
    unit_selection_kwargs = (
        unit_selection_kwargs
        if unit_selection_kwargs is not None
        else DEFAULT_UNIT_SELECTION_KWARGS
    )
    run_config, compute_roa, roa_kwargs = _prepare_search(
        pool, objectives, base_config, compute_roa, roa_kwargs, unit_selection, selection
    )
    pool = select_pool_units(pool, unit_selection, unit_selection_kwargs)
    single = len(objectives) == 1
    settings = dict(
        run_config=run_config,
        compute_roa=compute_roa,
        roa_kwargs=roa_kwargs,
        unit_selection=unit_selection,
        unit_selection_kwargs=unit_selection_kwargs,
    )

    # Best-so-far / front tracking, scratch space and study snapshots, only
    # active when best_result_path is set. One lock per resource.
    best_dir = Path(best_result_path) if best_result_path is not None else None
    temp_dir = best_dir.with_name(best_dir.name + "_temp") if best_dir is not None else None
    if best_dir is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
    front: Dict[int, Tuple[float, ...]] = {}
    front_lock = threading.Lock()
    save_lock = threading.Lock()

    def member_dir(trial_number: int) -> Path:
        return best_dir if single else best_dir / f"trial_{trial_number}"

    def _trial_objective(trial, group, n_threads):
        # ONE suggestion, shared across every dataset in the pool.
        overrides = suggest_overrides(trial, param_space)
        per_dataset = score_pool(
            pool,
            groups[group] if groups else None,
            overrides,
            temp_dir,
            trial.number,
            settings,
            n_threads,
        )
        if single:
            for losses in per_dataset.values():
                losses["loss"] = losses[objectives[0]]
        values, log_vars = pool_trial(trial, overrides, per_dataset, objectives, compute_roa)

        if best_dir is not None:
            # Locked: concurrent trials would race on the front.
            with front_lock:
                joined, evicted = update_front(front, trial.number, values, keep_ties=not single)
                if joined:
                    trial_config = build_trial_config(run_config, overrides)
                    promote_trial(
                        temp_dir, member_dir(trial.number), trial.number, pool, trial_config
                    )
                if not single:  # the single best is overwritten in place instead
                    for n in evicted:
                        evict_front_member(best_dir, n)
            log_vars["on_front"] = joined
            # This trial's scratch files are no longer needed either way.
            for name in pool:
                (temp_dir / f"{trial.number}_{name}.pkl").unlink(missing_ok=True)

        if on_trial is not None:
            on_trial(log_vars)
        return values[0] if single else values

    # Plan workers from the cores and check memory, before any worker starts
    n_startup = DEFAULT_N_STARTUP_TRIALS if sampler is None else 0
    n_trials_at_once = min(n_trials, max(n_startup, n_jobs))
    plan, shards, n_cores = plan_search_resources(
        pool, n_trials_at_once, n_jobs, n_cores, run_config.fifo_length
    )

    study = _make_study(objectives, sampler, random_seed, n_jobs)
    callbacks = (
        [lambda study, trial: save_study_snapshot(best_dir, study, save_lock)]
        if best_dir is not None
        else []
    )

    # One sharded copy of the pool per worker group; a single worker runs in-process
    in_process = plan.n_groups * plan.workers_per_group == 1
    groups = [] if in_process else [start_workers(pool, shards) for _ in range(plan.n_groups)]
    torch_threads = torch.get_num_threads()
    try:
        run_trials(study, _trial_objective, n_trials, n_jobs, n_startup, n_cores, plan, callbacks)
    finally:
        for group in groups:
            for executor in set(group.values()):
                executor.shutdown()
        torch.set_num_threads(torch_threads)

    pareto_front = None if single else study.best_trials
    if single:
        chosen = study.best_trial
    else:
        chosen = (SELECTION_RULES[selection] if isinstance(selection, str) else selection)(
            pareto_front
        )
    best_config = build_trial_config(run_config, chosen.params)

    outputs = None
    if best_dir is not None:
        shutil.rmtree(temp_dir)  # scratch space only, everything worth keeping is in best_dir
        if isinstance(next(iter(pool.values())), PooledDatasetMemory):
            outputs = {
                name: AdaptationResult.load(member_dir(chosen.number) / f"{name}.pkl")
                for name in pool
            }
        saved = ", ".join(sorted(p.name for p in best_dir.iterdir()))
        logger.info(f"Saved search results to {best_dir} ({saved})")

    return OptimisationResult(best_config, study, pareto_front, outputs)
