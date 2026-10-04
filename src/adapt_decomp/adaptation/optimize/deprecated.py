"""Deprecated 1.0.0 entry points: thin wrappers over optimize_adapt_decomp."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

from adapt_decomp.adaptation.optimize.pareto import FrontSelector
from adapt_decomp.adaptation.optimize.scoring import DEFAULT_OBJECTIVES, ObjectiveName
from adapt_decomp.adaptation.optimize.search import OptimisationResult, optimize_adapt_decomp
from adapt_decomp.utils.loaders import PooledDatasetDisk, PooledDatasetMemory


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
