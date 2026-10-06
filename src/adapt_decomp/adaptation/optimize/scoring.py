"""What one trial computes on one dataset: the parameter space and objectives,
the suggested overrides, the trial's config, the run and its losses, and their
sum over the pool.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.optimize.units import UnitSelection, select_units
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.spikes.comparison import rate_of_agreement_paired
from adapt_decomp.utils import validate_literals
from adapt_decomp.utils.loaders import PooledDataset

DEFAULT_PARAM_SPACE: dict = {
    "wh_learning_rate": ("log_float", 1e-4, 5e-2),
    "sv_learning_rate": ("log_float", 1e-4, 1e-1),
    "centroid_momentum": ("float", 0.1, 0.9, 0.1),  # 0.1, 0.2, ..., 0.9
}


ObjectiveName = Literal["sv_loss", "wh_loss", "total_loss", "roa"]

# Maps each loss-valued ObjectiveName to the AdaptationResult field it reads.
_OBJECTIVE_FIELD: Dict[str, str] = {
    "sv_loss": "sv_loss_total",
    "wh_loss": "wh_loss_total",
    "total_loss": "total_loss",
}
_VALID_OBJECTIVES: Tuple[str, ...] = (*_OBJECTIVE_FIELD, "roa")

DEFAULT_OBJECTIVES: Tuple[ObjectiveName, ...] = ("wh_loss", "sv_loss")


def validate_objectives(objectives: Tuple[ObjectiveName, ...]) -> None:
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


def param_distributions(param_space: dict) -> Dict[str, BaseDistribution]:
    """The Optuna distribution of each param_space entry.

    Args:
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple, where kind is "log_float", "float", or "int", with an
            optional step for "float" and "int": (kind, low, high, step)
            draws only low, low + step, ..., high. ("categorical", choices)
            picks from a list. See optimize_adapt_decomp's docstring for
            the full format and DEFAULT_PARAM_SPACE.

    Raises:
        ValueError: If an entry's kind is unknown.

    Returns:
        Dict[str, BaseDistribution]: Parameter name -> its distribution.
    """
    distributions = {}
    for name, spec in param_space.items():
        kind = spec[0]
        step = spec[3] if len(spec) > 3 else None
        if kind == "log_float":
            distributions[name] = FloatDistribution(spec[1], spec[2], log=True)
        elif kind == "float":
            distributions[name] = FloatDistribution(spec[1], spec[2], step=step)
        elif kind == "int":
            distributions[name] = IntDistribution(spec[1], spec[2], step=step or 1)
        elif kind == "categorical":
            distributions[name] = CategoricalDistribution(spec[1])
        else:
            raise ValueError(f"Unknown param_space kind: {kind!r}")
    return distributions


def suggest_overrides(trial: optuna.trial.Trial, param_space: dict) -> dict:
    """Suggest one value per param_space entry for this trial.

    A trial asked with study.ask(param_distributions(param_space)) already
    holds every value, so this only reads them back.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        param_space (dict): See param_distributions.

    Returns:
        dict: Parameter name -> suggested value, one entry per param_space
        key.
    """
    overrides = {}
    for name, dist in param_distributions(param_space).items():
        if isinstance(dist, FloatDistribution):
            overrides[name] = trial.suggest_float(
                name, dist.low, dist.high, step=dist.step, log=dist.log
            )
        elif isinstance(dist, IntDistribution):
            overrides[name] = trial.suggest_int(
                name, dist.low, dist.high, step=dist.step, log=dist.log
            )
        else:
            overrides[name] = trial.suggest_categorical(name, dist.choices)
    return overrides


def build_trial_config(run_config: AdaptConfig, overrides: dict) -> AdaptConfig:
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


def score_dataset(
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
        unit_selection, unit_selection_kwargs: See select_units.

    Returns:
        Dict[str, Any]: This dataset's losses; see _run_one_dataset.
    """
    emg, calibration, cbss_config, preprocess, gt_paired_bin = dataset.resolve()
    calibration, gt_paired_bin = select_units(
        calibration, gt_paired_bin, unit_selection, unit_selection_kwargs
    )
    outputs, losses = _run_one_dataset(
        emg,
        calibration,
        cbss_config,
        preprocess,
        gt_paired_bin,
        build_trial_config(run_config, overrides),
        compute_roa,
        roa_kwargs,
    )
    if stage_path is not None:
        outputs.save(stage_path)
    return losses


def pool_trial(
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
