"""Optuna-based hyperparameter optimisation for AdaptDecomp.

optimize_adapt_decomp() is the single entry point: one shared parameter
suggestion per trial is scored on every dataset of a pool, with one objective
(single-objective search) or several (Pareto search), from an in-memory or an
on-disk pool, optionally spread over worker processes.
"""

from __future__ import annotations

import copy
import multiprocessing
import pickle
import shutil
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from loguru import logger

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.spikes.comparison import rate_of_agreement_paired
from adapt_decomp.utils import validate_literals
from adapt_decomp.utils.loaders import PooledDatasetDisk, PooledDatasetMemory

PooledDataset = Union[PooledDatasetMemory, PooledDatasetDisk]
FrontSelector = Callable[[List[optuna.trial.FrozenTrial]], optuna.trial.FrozenTrial]

# ------------------------------------------------------------------
# Param space
# ------------------------------------------------------------------

DEFAULT_PARAM_SPACE: dict = {
    "wh_learning_rate": ("log_float", 1e-4, 5e-2),
    "sv_learning_rate": ("log_float", 1e-4, 1e-1),
    "centroid_momentum": ("float", 0.0, 0.95),
}


# ------------------------------------------------------------------
# Objective scoring
# ------------------------------------------------------------------

ObjectiveName = Literal["sv_loss", "wh_loss", "total_loss", "roa"]

# Maps each loss-valued ObjectiveName to the AdaptationResult field it reads.
_OBJECTIVE_FIELD: Dict[str, str] = {
    "sv_loss": "sv_loss_total",
    "wh_loss": "wh_loss_total",
    "total_loss": "total_loss",
}
_VALID_OBJECTIVES: Tuple[str, ...] = (*_OBJECTIVE_FIELD, "roa")

DEFAULT_OBJECTIVES: Tuple[ObjectiveName, ...] = ("wh_loss", "sv_loss")


def _base_losses(outputs: AdaptationResult) -> Dict[str, float]:
    """Read one trial's guarded per-run losses off outputs.

    Args:
        outputs (AdaptationResult): A single run's result, with compute_loss=True
            (so wh_loss_total/sv_loss_total/total_loss are all set).

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
# Unit selection
# ------------------------------------------------------------------

UnitSelection = Literal["unsupervised", "supervised", None]
_VALID_UNIT_SELECTIONS: Tuple[Optional[str], ...] = ("unsupervised", "supervised", None)
DEFAULT_UNIT_SELECTION_KWARGS: dict = {"cov_th": 0.3}


def _has_gt(dataset: PooledDataset) -> bool:
    """Whether a pooled dataset carries ground truth (gt_paired_bin or path_gt).

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        bool: True if its ground truth is set.
    """
    if isinstance(dataset, PooledDatasetMemory):
        return dataset.gt_paired_bin is not None
    return dataset.path_gt is not None


def _select_units(
    calibration: CBSSResult,
    gt_paired_bin: Optional[np.ndarray],
    unit_selection: UnitSelection,
    unit_selection_kwargs: dict,
) -> Optional[Tuple[CBSSResult, Optional[np.ndarray]]]:
    """Drop the calibration units the search should not adapt or score.

    Only "unsupervised" subsets here (CBSSResult.unsupervised_mask, with
    gt_paired_bin's columns subset alike so RoA stays paired); "supervised"
    pools are already narrowed to GT-matched units by their loaders.

    Args:
        calibration (CBSSResult): A dataset's calibration.
        gt_paired_bin (Optional[np.ndarray]): Its paired ground truth, with
            shape (samples, M), or None.
        unit_selection (UnitSelection): "unsupervised", "supervised" or None.
        unit_selection_kwargs (dict): Thresholds for unsupervised_mask.

    Returns:
        Optional[Tuple[CBSSResult, Optional[np.ndarray]]]: The (possibly
        subset) calibration and gt_paired_bin, or None if no unit is kept.
    """
    if unit_selection != "unsupervised":
        return calibration, gt_paired_bin
    mask = calibration.unsupervised_mask(**unit_selection_kwargs)
    if not mask.any():
        return None
    return calibration.subset(mask), gt_paired_bin[:, mask] if gt_paired_bin is not None else None


def _select_pool_units(
    pool: Dict[str, PooledDataset], unit_selection: UnitSelection, unit_selection_kwargs: dict
) -> Dict[str, PooledDataset]:
    """Apply unit selection once up front, leaving out datasets with no unit kept.

    In-memory entries are replaced by their selected copy; on-disk entries
    are kept as paths and selected again (identically) after every resolve().

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        unit_selection (UnitSelection): See _select_units.
        unit_selection_kwargs (dict): See _select_units.

    Raises:
        ValueError: If no dataset keeps any unit.

    Returns:
        Dict[str, PooledDataset]: The pool to search on.
    """
    if unit_selection != "unsupervised":
        return pool
    selected = {}
    for name, dataset in pool.items():
        _, calibration, _, _, gt_paired_bin = dataset.resolve()
        kept = _select_units(calibration, gt_paired_bin, unit_selection, unit_selection_kwargs)
        n_units = calibration.sources.shape[1]
        if kept is None:
            logger.warning(
                f"{name}: no unit of {n_units} passes {unit_selection_kwargs}, left out of the pool"
            )
            continue
        logger.info(f"{name}: {kept[0].sources.shape[1]}/{n_units} units adapted and scored")
        if isinstance(dataset, PooledDatasetMemory):
            dataset = replace(dataset, calibration=kept[0], gt_paired_bin=kept[1])
        selected[name] = dataset
    if not selected:
        raise ValueError(f"No dataset in pool keeps any unit under {unit_selection_kwargs}.")
    return selected


# ------------------------------------------------------------------
# Trial building blocks
# ------------------------------------------------------------------


def _suggest_overrides(trial: optuna.trial.Trial, param_space: dict) -> dict:
    """Suggest one value per param_space entry for this trial.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple, where kind is "log_float", "float", or "int" (or
            (kind, choices) for "categorical"). See optimize_adapt_decomp's
            docstring for the full format and DEFAULT_PARAM_SPACE.

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


def _run_one_dataset(
    emg: Union[torch.Tensor, np.ndarray],
    calibration: CBSSResult,
    cbss_config: CBSSConfig,
    preprocess: bool,
    gt_paired_bin: Optional[np.ndarray],
    trial_config: AdaptConfig,
    compute_roa: bool,
    roa_kwargs: Optional[dict],
) -> Tuple[AdaptationResult, Dict[str, Any]]:
    """Run one trial's AdaptDecomp for a single dataset and score it.

    Args:
        emg (Union[torch.Tensor, np.ndarray]): Online EMG to decompose, with
            shape (samples, channels).
        calibration (CBSSResult): This dataset's calibration result.
        cbss_config (CBSSConfig): The CBSSConfig that produced calibration.
        preprocess (bool): Whether to preprocess emg before extension.
        gt_paired_bin (Optional[np.ndarray]): Ground-truth binary spike
            train matched to calibration's units, with shape (samples, M).
            Required when compute_roa is True.
        trial_config (AdaptConfig): This trial's resolved configuration.
        compute_roa (bool): If True, score RoA against gt_paired_bin and
            include it in the returned losses.
        roa_kwargs (Optional[dict]): Extra keyword arguments forwarded to
            rate_of_agreement_paired() when compute_roa is True.

    Returns:
        Tuple[AdaptationResult, Dict[str, Any]]: outputs, this dataset's
        result (with .roa set when compute_roa); losses, {"sv_loss",
        "wh_loss", "total_loss"}, plus {"roa", "roa_mean", "roa_per_unit"}
        when compute_roa is True.
    """
    adapter = AdaptDecomp.from_calibration(
        calibration=calibration,
        cbss_config=cbss_config,
        adapt_config=trial_config,
    )
    outputs = adapter.process_data(emg, preprocess=preprocess)
    losses: Dict[str, Any] = _base_losses(outputs)

    if compute_roa:
        pred_spikes = outputs.spikes.numpy().astype(np.float32)
        roa_vals, _, _ = rate_of_agreement_paired(gt_paired_bin, pred_spikes, **roa_kwargs)
        outputs.roa = np.asarray(roa_vals, dtype=np.float32)  # travels with outputs.save()
        roa_mean = float(np.nanmean(roa_vals)) * 100
        losses["roa_mean"] = roa_mean
        losses["roa_per_unit"] = [float(x) for x in roa_vals]
        losses["roa"] = _roa_loss(roa_mean, losses["total_loss"] >= 1e10)

    return outputs, losses


def _score_dataset(
    dataset: PooledDataset,
    overrides: dict,
    stage_path: Optional[Path],
    *,
    run_config: AdaptConfig,
    compute_roa: bool,
    roa_kwargs: Optional[dict],
    unit_selection: UnitSelection,
    unit_selection_kwargs: dict,
) -> Dict[str, Any]:
    """Resolve, select and run one pool entry for a trial, staging its outputs.

    The per-dataset unit of work, run in-process or in a worker process
    (_score_in_worker) -- only the small losses dict travels back.

    Args:
        dataset (PooledDataset): Pool entry, loaded via its resolve().
        overrides (dict): This trial's suggested parameter overrides.
        stage_path (Optional[Path]): Where to save this dataset's
            AdaptationResult for best-result promotion, or None.
        run_config, compute_roa, roa_kwargs: As optimize_adapt_decomp.
        unit_selection, unit_selection_kwargs: See _select_units.

    Returns:
        Dict[str, Any]: This dataset's losses; see _run_one_dataset.
    """
    emg, calibration, cbss_config, preprocess, gt_paired_bin = dataset.resolve()
    calibration, gt_paired_bin = _select_units(
        calibration, gt_paired_bin, unit_selection, unit_selection_kwargs
    )
    outputs, losses = _run_one_dataset(
        emg,
        calibration,
        cbss_config,
        preprocess,
        gt_paired_bin,
        _build_trial_config(run_config, overrides),
        compute_roa,
        roa_kwargs,
    )
    if stage_path is not None:
        outputs.save(stage_path)
    return losses


# ------------------------------------------------------------------
# Worker processes: each holds a shard of the pool for the whole search
# ------------------------------------------------------------------

_WORKER_POOL: Dict[str, PooledDataset] = {}


def _init_worker(shard: Dict[str, PooledDataset]) -> None:
    """Worker initializer: keep this worker's shard resident, one torch thread.

    Args:
        shard (Dict[str, PooledDataset]): The pool entries this worker runs.

    Returns:
        None
    """
    torch.set_num_threads(1)
    _WORKER_POOL.update(shard)


def _score_in_worker(
    name: str, overrides: dict, stage_path: Optional[Path], settings: dict
) -> Dict[str, Any]:
    """_score_dataset on a pool entry resident in this worker.

    Args:
        name (str): Pool key of the entry, which must be in this worker's shard.
        overrides (dict): This trial's suggested parameter overrides.
        stage_path (Optional[Path]): See _score_dataset.
        settings (dict): _score_dataset's keyword-only arguments.

    Returns:
        Dict[str, Any]: See _score_dataset.
    """
    return _score_dataset(_WORKER_POOL[name], overrides, stage_path, **settings)


def _dataset_size(dataset: PooledDataset) -> int:
    """Rough cost of running one pool entry, to balance shards.

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        int: Its EMG sample count (in memory) or file size in bytes (on disk).
    """
    if isinstance(dataset, PooledDatasetMemory):
        return len(dataset.emg)
    return Path(dataset.path_emg).stat().st_size


def _start_workers(
    pool: Dict[str, PooledDataset], n_workers: int
) -> Dict[str, ProcessPoolExecutor]:
    """Shard pool over n_workers single-process executors, longest datasets first.

    Each shard is sent to its worker once, at start-up; trials then only
    send overrides. Every dataset always runs in the same worker.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        n_workers (int): Number of worker processes (capped at len(pool)).

    Returns:
        Dict[str, ProcessPoolExecutor]: Dataset name -> the executor owning it.
    """
    shards: List[Dict[str, PooledDataset]] = [{} for _ in range(min(n_workers, len(pool)))]
    loads = [0] * len(shards)
    for name in sorted(pool, key=lambda n: _dataset_size(pool[n]), reverse=True):
        i = loads.index(min(loads))
        shards[i][name] = pool[name]
        loads[i] += _dataset_size(pool[name])

    context = multiprocessing.get_context("spawn")
    owners = {}
    for shard in shards:
        executor = ProcessPoolExecutor(
            max_workers=1, mp_context=context, initializer=_init_worker, initargs=(shard,)
        )
        owners.update(dict.fromkeys(shard, executor))
    return owners


def _score_pool(
    pool: Dict[str, PooledDataset],
    workers: Optional[Dict[str, ProcessPoolExecutor]],
    overrides: dict,
    stage_dir: Optional[Path],
    trial_number: int,
    settings: dict,
) -> Dict[str, Dict[str, Any]]:
    """Score one trial's overrides on every dataset, in pool order.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        workers (Optional[Dict[str, ProcessPoolExecutor]]): Owner executor
            per dataset (from _start_workers), or None to run in-process.
        overrides (dict): This trial's suggested parameter overrides.
        stage_dir (Optional[Path]): Scratch directory for this trial's
            "<trial_number>_<dataset>.pkl" outputs, or None.
        trial_number (int): This trial's Optuna trial.number, unique even
            under n_jobs > 1 so concurrent trials never collide.
        settings (dict): _score_dataset's keyword-only arguments.

    Returns:
        Dict[str, Dict[str, Any]]: Dataset name -> its losses.
    """

    def stage_path(name: str) -> Optional[Path]:
        return stage_dir / f"{trial_number}_{name}.pkl" if stage_dir is not None else None

    if workers is None:
        return {
            name: _score_dataset(dataset, overrides, stage_path(name), **settings)
            for name, dataset in pool.items()
        }
    futures = {
        name: workers[name].submit(_score_in_worker, name, overrides, stage_path(name), settings)
        for name in pool
    }
    return {name: future.result() for name, future in futures.items()}


def _pool_trial(
    trial: optuna.trial.Trial,
    overrides: dict,
    per_dataset: Dict[str, Dict[str, Any]],
    objectives: Tuple[ObjectiveName, ...],
    compute_roa: bool,
) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """Sum per-dataset losses over the pool, record them as user_attrs, build the log dict.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        overrides (dict): This trial's suggested parameter overrides.
        per_dataset (Dict[str, Dict[str, Any]]): Dataset name -> its losses,
            with "loss" added for a single-objective search.
        objectives (Tuple[ObjectiveName, ...]): Scored objectives.
        compute_roa (bool): Whether per_dataset carries RoA.

    Returns:
        Tuple[Tuple[float, ...], Dict[str, Any]]: values, the pooled sum of
        each objective; log_vars, see optimize_adapt_decomp's on_trial.
    """
    for name, losses in per_dataset.items():
        for key, value in losses.items():
            trial.set_user_attr(f"{key}_{name}", value)

    pooled = {key: sum(d[key] for d in per_dataset.values()) for key in _OBJECTIVE_FIELD}
    if compute_roa:
        pooled["roa"] = sum(d["roa"] for d in per_dataset.values())
    values = tuple(pooled[o] for o in objectives)

    if len(objectives) == 1:
        head = {"loss": values[0], "objective": objectives[0]}
    else:
        head = {"objectives": objectives, "values": values}
    log_vars: Dict[str, Any] = {
        "trial_number": trial.number,
        **head,
        **{key: pooled[key] for key in _OBJECTIVE_FIELD},
        "params": overrides,
        "per_dataset": per_dataset,
    }
    for key in _OBJECTIVE_FIELD:
        trial.set_user_attr(key, pooled[key])
    if compute_roa:
        roa_mean_pooled = float(np.mean([d["roa_mean"] for d in per_dataset.values()]))
        trial.set_user_attr("roa_mean_pooled", roa_mean_pooled)
        trial.set_user_attr("roa", pooled["roa"])
        log_vars["roa_mean"] = roa_mean_pooled
        log_vars["roa"] = pooled["roa"]
    return values, log_vars


# ------------------------------------------------------------------
# Best-result tracking and persistence
# ------------------------------------------------------------------


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True iff objective vector a Pareto-dominates b (both minimised, equal length).

    Args:
        a (Tuple[float, ...]): Candidate dominator's objective values.
        b (Tuple[float, ...]): Candidate dominated point's objective values.

    Returns:
        bool: True iff a is no worse than b in every dimension and strictly
        better in at least one -- matching Optuna's own study.best_trials
        convention, so a tie dominates neither point (both stay on the front).
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _update_front(
    front: Dict[int, Tuple[float, ...]],
    trial_number: int,
    values: Tuple[float, ...],
    keep_ties: bool = True,
) -> Tuple[bool, List[int]]:
    """Join trial_number onto the resident front, evicting anything it now dominates.

    With one objective and keep_ties=False the front is the single best
    trial so far, replaced only on strict improvement (as study.best_trial).

    Args:
        front (Dict[int, Tuple[float, ...]]): Currently resident front
            members, trial_number -> objective values. Mutated in place.
        trial_number (int): This trial's Optuna trial.number.
        values (Tuple[float, ...]): This trial's objective values, ordered
            to match objectives.
        keep_ties (bool, optional): Whether a trial tying a resident member
            joins the front. Defaults to True (Pareto convention).

    Returns:
        Tuple[bool, List[int]]: joined, False if some resident member
        already dominates (or, without keep_ties, ties) this trial (front
        left unchanged), True otherwise, in which case trial_number is added
        to front; evicted, the trial numbers removed from front because this
        trial dominates them (always empty when joined is False).
    """
    if any(
        _dominates(existing, values) or (not keep_ties and existing == values)
        for existing in front.values()
    ):
        return False, []
    evicted = [n for n, existing in front.items() if _dominates(values, existing)]
    for n in evicted:
        del front[n]
    front[trial_number] = values
    return True, evicted


def _promote_trial(
    temp_dir: Path,
    member_dir: Path,
    trial_number: int,
    dataset_names: Iterable[str],
    trial_config: AdaptConfig,
) -> None:
    """Copy one trial's staged per-dataset results + config into member_dir.

    Args:
        temp_dir (Path): Directory holding this trial's
            "<trial_number>_<dataset>.pkl" scratch files.
        member_dir (Path): Destination: best_result_path itself for a
            single-objective search, or its "trial_<trial_number>"
            subdirectory for a Pareto front member.
        trial_number (int): This trial's Optuna trial.number.
        dataset_names (Iterable[str]): Every dataset name to copy over.
        trial_config (AdaptConfig): This trial's resolved configuration,
            written alongside the results as "config.yaml".

    Returns:
        None
    """
    member_dir.mkdir(parents=True, exist_ok=True)
    for dataset in dataset_names:
        shutil.copy2(temp_dir / f"{trial_number}_{dataset}.pkl", member_dir / f"{dataset}.pkl")
    trial_config.to_yaml(member_dir / "config.yaml")


def _evict_front_member(best_dir: Path, trial_number: int) -> None:
    """Delete a previously-saved front member's subdirectory.

    Called when a later trial dominates a resident member -- ignore_errors
    so a member that was never actually saved (shouldn't happen, but not
    worth crashing the search over) is a no-op rather than an exception.

    Args:
        best_dir (Path): Front's root directory.
        trial_number (int): The dominated trial's Optuna trial.number.

    Returns:
        None
    """
    shutil.rmtree(best_dir / f"trial_{trial_number}", ignore_errors=True)


def _save_study_snapshot(best_dir: Path, study: optuna.Study, lock: threading.Lock) -> None:
    """Pickle study to best_dir/"study.pkl", overwriting any previous snapshot.

    Passed as a study.optimize(callbacks=[...]) entry -- Optuna invokes
    callbacks once per completed trial, so a crashed run's study.pkl still
    reflects every trial that finished before the crash, not just a final
    one written after study.optimize() returns. Study/InMemoryStorage are
    picklable mid-run by design (strip thread-local/lock state in
    __getstate__, rebuild it in __setstate__).

    Args:
        best_dir (Path): Directory to write "study.pkl" into, already
            created by the caller.
        study (optuna.Study): The study so far.
        lock (threading.Lock): Guards the write against concurrent callback
            invocations under n_jobs>1.

    Returns:
        None
    """
    with lock:
        with open(best_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)


# ------------------------------------------------------------------
# Pareto-front selection
# ------------------------------------------------------------------

SelectionName = Literal["min_sv_loss", "knee", "max_roa_mean"]


def _select_min_sv_loss(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Default Pareto-front selection: the front's own minimum pooled sv_loss member.

    Reads trial.user_attrs["sv_loss"] (always logged pooled, regardless of
    which dimensions objectives actually optimised) rather than
    trial.values, so this works even when "sv_loss" isn't itself one of
    objectives. Always Pareto-optimal by construction and needs no ground
    truth, though it sits at the front's sv_loss extreme.

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed Pareto search.

    Returns:
        optuna.trial.FrozenTrial: The front member with the lowest
        "sv_loss" user_attr.
    """
    return min(pareto_front, key=lambda t: t.user_attrs["sv_loss"])


def _select_max_roa_mean(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Oracle Pareto-front selection: the front's own highest mean RoA member.

    Only meaningful when compute_roa was True (or "roa" was in objectives)
    for the search that produced pareto_front -- otherwise every member's
    "roa_mean_pooled" user_attr is absent and this falls back to picking
    arbitrarily among ties at float("-inf").

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed Pareto search.

    Returns:
        optuna.trial.FrozenTrial: The front member with the highest
        "roa_mean_pooled" user_attr.
    """
    return max(pareto_front, key=lambda t: t.user_attrs.get("roa_mean_pooled", float("-inf")))


def _select_knee(pareto_front: List[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    """Knee-point Pareto-front selection for two objectives.

    Min-max normalises both objectives over the front and returns the member
    farthest from the line through its two extremes: the point where
    improving one objective starts costing the most of the other, rather
    than an extreme of the front. Needs no ground truth. Falls back to
    _select_min_sv_loss for fronts of fewer than three members (no interior
    point).

    Args:
        pareto_front (List[optuna.trial.FrozenTrial]): study.best_trials
            from a completed two-objective Pareto search.

    Raises:
        ValueError: If the front's trials don't have exactly two values.

    Returns:
        optuna.trial.FrozenTrial: The front member farthest from the chord.
    """
    values = np.array([t.values for t in pareto_front], dtype=float)
    if values.shape[1] != 2:
        raise ValueError(f"Knee selection needs exactly 2 objectives, got {values.shape[1]}.")
    if len(pareto_front) < 3:
        return _select_min_sv_loss(pareto_front)
    span = values.max(0) - values.min(0)
    norm = (values - values.min(0)) / np.where(span > 0, span, 1.0)
    start, end = norm[norm[:, 0].argmin()], norm[norm[:, 1].argmin()]
    chord = end - start
    if not np.linalg.norm(chord) > 0:
        return _select_min_sv_loss(pareto_front)
    offsets = norm - start
    distance = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    return pareto_front[int(distance.argmax())]


_SELECTION_RULES: Dict[str, FrontSelector] = {
    "min_sv_loss": _select_min_sv_loss,
    "knee": _select_knee,
    "max_roa_mean": _select_max_roa_mean,
}


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


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


def _validate_objectives(objectives: Tuple[ObjectiveName, ...]) -> None:
    """Check objectives is a non-empty tuple of distinct, known objective names.

    Args:
        objectives (Tuple[ObjectiveName, ...]): Scalars to optimise jointly.

    Raises:
        ValueError: If objectives is empty, contains an unknown name, or
            contains a duplicate.

    Returns:
        None
    """
    unknown = [o for o in objectives if o not in _VALID_OBJECTIVES]
    if unknown or not objectives:
        raise ValueError(f"Unknown objective(s): {unknown}; expected each in {_VALID_OBJECTIVES}")
    if len(set(objectives)) != len(objectives):
        raise ValueError(f"objectives must not contain duplicates, got {objectives!r}")


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
    _validate_objectives(objectives)

    if not pool:
        raise ValueError("pool is empty.")
    if not all(isinstance(d, PooledDatasetMemory) for d in pool.values()) and not all(
        isinstance(d, PooledDatasetDisk) for d in pool.values()
    ):
        raise TypeError("pool must be all PooledDatasetMemory or all PooledDatasetDisk entries.")
    if unit_selection not in _VALID_UNIT_SELECTIONS:
        raise ValueError(
            f"Unknown unit_selection: {unit_selection!r}; expected one of {_VALID_UNIT_SELECTIONS}"
        )
    if not callable(selection) and selection not in _SELECTION_RULES:
        raise ValueError(
            f"Unknown selection: {selection!r}; expected a callable or one of "
            f"{tuple(_SELECTION_RULES)}"
        )
    if selection == "knee" and len(objectives) != 2:
        raise ValueError(f"selection='knee' needs exactly 2 objectives, got {objectives!r}")

    compute_roa = compute_roa or "roa" in objectives
    missing = [name for name, dataset in pool.items() if not _has_gt(dataset)]
    if missing and (compute_roa or unit_selection == "supervised"):
        raise ValueError(
            "Ground truth (gt_paired_bin/path_gt) is required for every dataset in pool when "
            "compute_roa=True, 'roa' is an objective, or unit_selection='supervised'; "
            f"missing for: {missing}"
        )
    if compute_roa:
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}
    return run_config, compute_roa, roa_kwargs


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
        n_jobs (int): Concurrent trials; above 1 the default sampler uses
            constant_liar so pending trials aren't suggested twice.

    Returns:
        optuna.Study: The new study.
    """
    with warnings.catch_warnings():  # multivariate/constant_liar/metric names are experimental
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=15, multivariate=True, constant_liar=n_jobs > 1, seed=random_seed
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
    unit_selection: UnitSelection = "unsupervised",
    unit_selection_kwargs: Optional[dict] = None,
    selection: Union[SelectionName, FrontSelector] = "min_sv_loss",
    n_trials: int = 100,
    n_jobs: int = 1,
    n_workers: int = 1,
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
            the search adapts and scores: "unsupervised" keeps units passing
            CBSSResult.unsupervised_mask(**unit_selection_kwargs), leaving
            out datasets with none; "supervised" requires every dataset's
            ground truth (its loader already narrowed the calibration to
            GT-matched units); None keeps every unit (ablation). Defaults to
            "unsupervised".
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
        n_jobs (int, optional): Concurrent trials (threads), passed to
            study.optimize(). Reproducible from random_seed only at 1.
            Defaults to 1.
        n_workers (int, optional): Worker processes each trial's datasets
            are spread over (longest first; each holds its shard for the
            whole search, one torch thread each). Trials stay sequential, so
            the sampler loses nothing and results stay reproducible.
            Defaults to 1 (in-process).
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses multivariate
            TPESampler(n_startup_trials=15, seed=random_seed), with
            constant_liar when n_jobs > 1.
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
        TypeError, ValueError: See _prepare_search.

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
    pool = _select_pool_units(pool, unit_selection, unit_selection_kwargs)
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

    def _trial_objective(trial):
        # ONE suggestion, shared across every dataset in the pool.
        overrides = _suggest_overrides(trial, param_space)
        per_dataset = _score_pool(pool, workers, overrides, temp_dir, trial.number, settings)
        if single:
            for losses in per_dataset.values():
                losses["loss"] = losses[objectives[0]]
        values, log_vars = _pool_trial(trial, overrides, per_dataset, objectives, compute_roa)

        if best_dir is not None:
            # Locked: concurrent trials (n_jobs > 1) would race on the front.
            with front_lock:
                joined, evicted = _update_front(front, trial.number, values, keep_ties=not single)
                if joined:
                    trial_config = _build_trial_config(run_config, overrides)
                    _promote_trial(
                        temp_dir, member_dir(trial.number), trial.number, pool, trial_config
                    )
                if not single:  # the single best is overwritten in place instead
                    for n in evicted:
                        _evict_front_member(best_dir, n)
            log_vars["on_front"] = joined
            # This trial's scratch files are no longer needed either way.
            for name in pool:
                (temp_dir / f"{trial.number}_{name}.pkl").unlink(missing_ok=True)

        if on_trial is not None:
            on_trial(log_vars)
        return values[0] if single else values

    study = _make_study(objectives, sampler, random_seed, n_jobs)
    callbacks = (
        [lambda study, trial: _save_study_snapshot(best_dir, study, save_lock)]
        if best_dir is not None
        else None
    )
    workers = _start_workers(pool, n_workers) if n_workers > 1 else None
    try:
        study.optimize(_trial_objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=callbacks)
    finally:
        for executor in set((workers or {}).values()):
            executor.shutdown()

    pareto_front = None if single else study.best_trials
    if single:
        chosen = study.best_trial
    else:
        chosen = (_SELECTION_RULES[selection] if isinstance(selection, str) else selection)(
            pareto_front
        )
    best_config = _build_trial_config(run_config, chosen.params)

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


# ------------------------------------------------------------------
# Deprecated entry points -- thin wrappers over optimize_adapt_decomp
# ------------------------------------------------------------------


def _deprecated_search(old_name: str, **kwargs) -> OptimisationResult:
    """Warn that old_name is deprecated, then run optimize_adapt_decomp as it did.

    The old entry points never selected units, so unit_selection is None.

    Args:
        old_name (str): The deprecated function's name, for the warning.
        **kwargs: optimize_adapt_decomp's arguments.

    Returns:
        OptimisationResult: See optimize_adapt_decomp.
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version; use "
        "optimize_adapt_decomp instead (it picks memory/disk and single/Pareto itself).",
        FutureWarning,
        stacklevel=3,
    )
    return optimize_adapt_decomp(unit_selection=None, **kwargs)


def _check_pareto_objectives(objectives: Tuple[ObjectiveName, ...]) -> None:
    """The deprecated Pareto entry points' own check: at least two objectives.

    Args:
        objectives (Tuple[ObjectiveName, ...]): Requested objectives.

    Raises:
        ValueError: If objectives has fewer than two entries.

    Returns:
        None
    """
    if len(objectives) < 2:
        raise ValueError(
            f"objectives must have at least 2 entries for a Pareto search (got {objectives!r}); "
            "use a single objective for a single-objective search instead."
        )


def optimize_adapt_decomp_pooled_memory(
    *,
    pool: Dict[str, PooledDatasetMemory],
    objective: ObjectiveName = "sv_loss",
    param_space: dict,
    best_result_path: Optional[str] = None,
    **kwargs,
):
    """Deprecated: optimize_adapt_decomp with one objective over an in-memory pool.

    Args:
        pool (Dict[str, PooledDatasetMemory]): See optimize_adapt_decomp.
        objective (ObjectiveName, optional): The single objective. Defaults
            to "sv_loss".
        param_space (dict): See optimize_adapt_decomp.
        best_result_path (Optional[str], optional): See optimize_adapt_decomp.
        **kwargs: base_config, compute_roa, roa_kwargs, n_trials, n_jobs,
            sampler, random_seed, on_trial; see optimize_adapt_decomp.

    Returns:
        (best_config, study), or (outputs, best_config, study) when
        best_result_path is set.
    """
    result = _deprecated_search(
        "optimize_adapt_decomp_pooled_memory",
        pool=pool,
        objectives=objective,
        param_space=param_space,
        best_result_path=best_result_path,
        **kwargs,
    )
    if best_result_path is not None:
        return result.outputs, result.best_config, result.study
    return result.best_config, result.study


def optimize_adapt_decomp_pooled_disk(
    *,
    pool: Dict[str, PooledDatasetDisk],
    objective: ObjectiveName = "sv_loss",
    param_space: dict,
    **kwargs,
):
    """Deprecated: optimize_adapt_decomp with one objective over an on-disk pool.

    Args:
        pool (Dict[str, PooledDatasetDisk]): See optimize_adapt_decomp.
        objective (ObjectiveName, optional): The single objective. Defaults
            to "sv_loss".
        param_space (dict): See optimize_adapt_decomp.
        **kwargs: base_config, compute_roa, roa_kwargs, n_trials, n_jobs,
            sampler, random_seed, best_result_path, on_trial; see
            optimize_adapt_decomp.

    Returns:
        (best_config, study); per-dataset results are reloaded from
        best_result_path, e.g. AdaptationResult.load(Path(best_result_path)
        / f"{name}.pkl").
    """
    result = _deprecated_search(
        "optimize_adapt_decomp_pooled_disk",
        pool=pool,
        objectives=objective,
        param_space=param_space,
        **kwargs,
    )
    return result.best_config, result.study


def optimize_adapt_decomp_pooled_memory_pareto(
    *,
    pool: Dict[str, PooledDatasetMemory],
    objectives: Tuple[ObjectiveName, ...] = DEFAULT_OBJECTIVES,
    param_space: dict,
    best_result_path: Optional[str] = None,
    selection_rule: Optional[FrontSelector] = None,
    **kwargs,
):
    """Deprecated: optimize_adapt_decomp with two or more objectives over an in-memory pool.

    Args:
        pool (Dict[str, PooledDatasetMemory]): See optimize_adapt_decomp.
        objectives (Tuple[ObjectiveName, ...], optional): At least two
            distinct objectives. Defaults to DEFAULT_OBJECTIVES.
        param_space (dict): See optimize_adapt_decomp.
        best_result_path (Optional[str], optional): See optimize_adapt_decomp.
        selection_rule (Optional[FrontSelector], optional): Front selection
            callable. Defaults to None ("min_sv_loss").
        **kwargs: base_config, compute_roa, roa_kwargs, n_trials, n_jobs,
            sampler, random_seed, on_trial; see optimize_adapt_decomp.

    Raises:
        ValueError: If objectives has fewer than two entries, or see
            optimize_adapt_decomp.

    Returns:
        (best_config, pareto_front, study), or (outputs, best_config,
        pareto_front, study) when best_result_path is set.
    """
    objectives = tuple(objectives)
    _check_pareto_objectives(objectives)
    result = _deprecated_search(
        "optimize_adapt_decomp_pooled_memory_pareto",
        pool=pool,
        objectives=objectives,
        param_space=param_space,
        best_result_path=best_result_path,
        selection=selection_rule or "min_sv_loss",
        **kwargs,
    )
    if best_result_path is not None:
        return result.outputs, result.best_config, result.pareto_front, result.study
    return result.best_config, result.pareto_front, result.study


def optimize_adapt_decomp_pooled_disk_pareto(
    *,
    pool: Dict[str, PooledDatasetDisk],
    objectives: Tuple[ObjectiveName, ...] = DEFAULT_OBJECTIVES,
    param_space: dict,
    selection_rule: Optional[FrontSelector] = None,
    **kwargs,
):
    """Deprecated: optimize_adapt_decomp with two or more objectives over an on-disk pool.

    Args:
        pool (Dict[str, PooledDatasetDisk]): See optimize_adapt_decomp.
        objectives (Tuple[ObjectiveName, ...], optional): At least two
            distinct objectives. Defaults to DEFAULT_OBJECTIVES.
        param_space (dict): See optimize_adapt_decomp.
        selection_rule (Optional[FrontSelector], optional): Front selection
            callable. Defaults to None ("min_sv_loss").
        **kwargs: base_config, compute_roa, roa_kwargs, n_trials, n_jobs,
            sampler, random_seed, best_result_path, on_trial; see
            optimize_adapt_decomp.

    Raises:
        ValueError: If objectives has fewer than two entries, or see
            optimize_adapt_decomp.

    Returns:
        (best_config, pareto_front, study); front members' results are
        reloaded from best_result_path, e.g. AdaptationResult.load(
        Path(best_result_path) / f"trial_{n}" / f"{name}.pkl").
    """
    objectives = tuple(objectives)
    _check_pareto_objectives(objectives)
    result = _deprecated_search(
        "optimize_adapt_decomp_pooled_disk_pareto",
        pool=pool,
        objectives=objectives,
        param_space=param_space,
        selection=selection_rule or "min_sv_loss",
        **kwargs,
    )
    return result.best_config, result.pareto_front, result.study
