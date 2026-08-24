"""Optuna-based hyperparameter optimisation for AdaptDecomp."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from loguru import logger

from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.spikes.comparison import rate_of_agreement_paired
from adapt_decomp.utils import validate_literals

DEFAULT_PARAM_SPACE: dict = {
    "wh_learning_rate":   ("log_float", 1e-4, 5e-2),
    "sv_learning_rate":   ("log_float", 1e-4, 1e-1),
}


def suggest_overrides(trial: optuna.trial.Trial, param_space: dict) -> dict:
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


def build_trial_config(run_config: AdaptConfig, overrides: dict) -> AdaptConfig:
    """Deep-copy run_config and apply a trial's suggested parameter overrides.

    Args:
        run_config (AdaptConfig): Base configuration to copy from -- never
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


def _compute_total_wh_loss(wh_loss: torch.Tensor) -> float:
    """Compute the total whitening loss for a trial, with NaN guard.

    Args:
        wh_loss (torch.Tensor): Per-batch whitening loss, with shape
            (batches,).

    Returns:
        float: Median wh_loss, or 1e10 if any batch is NaN.
    """
    if torch.any(torch.isnan(wh_loss)):
        return 1e10
    return wh_loss.median().item()


def _compute_total_sv_loss(sv_loss: torch.Tensor) -> float:
    """Compute the total separation-vector loss for a trial, with NaN guard.

    Args:
        sv_loss (torch.Tensor): Per-batch, per-unit separation-vector loss,
            with shape (batches, M).

    Returns:
        float: Median sv_loss over non-NaN entries (torch.nanmedian).
    """
    return sv_loss.nanmedian().item()


def run_trial_adapt_decomp(
    trial_config: AdaptConfig,
    emg: torch.Tensor,
    whitening: torch.Tensor,
    sep_vectors: torch.Tensor,
    base_centr: torch.Tensor,
    spikes_centr: torch.Tensor,
    emg_calib: torch.Tensor,
    ipts_calib: torch.Tensor,
    spikes_calib: torch.Tensor,
    preprocess: bool = False,
    pca_components: Optional[torch.Tensor] = None,
    pca_mean: Optional[torch.Tensor] = None,
) -> Tuple[AdaptDecomp, AdaptationResult]:
    """Build one fresh AdaptDecomp from a resolved config and run it.

    Single source of truth for "AdaptDecomp(...).run()", shared by both
    objectives in this module -- optimize_adapt_decomp's per-trial
    objective and optimize_adapt_decomp_pooled's per-condition-per-trial
    objective. 

    No save_path parameter: AdaptDecomp's own save_path/config.save_params
    write an HDF5 per-batch parameter trace, which would be prohibitively
    expensive to produce for every trial in a search, and the name would be
    easy to confuse with optimize_adapt_decomp's best_result_path (a
    cheap, once-per-improving-trial AdaptationResult snapshot -- see its
    docstring). If a per-batch trace is ever needed for a specific
    config, call AdaptDecomp(..., save_path=...) directly instead of going
    through this module.

    Args:
        trial_config (AdaptConfig): Fully resolved configuration for this
            run -- typically the output of build_trial_config().
        emg (torch.Tensor): Online EMG to decompose, with shape
            (samples, channels).
        whitening (torch.Tensor): Calibration whitening matrix, with shape
            (D, D) (or (n, D) when the calibration used PCA reduction).
        sep_vectors (torch.Tensor): Calibration separation matrix, with
            shape (M, D) (or (M, n) when the calibration used PCA reduction).
        base_centr (torch.Tensor): Baseline centroids, with shape (M,).
        spikes_centr (torch.Tensor): Spike centroids, with shape (M,).
        emg_calib (torch.Tensor): Calibration EMG, with shape
            (samples_calib, channels).
        ipts_calib (torch.Tensor): Calibration source signals, with shape
            (samples_calib, M).
        spikes_calib (torch.Tensor): Calibration spike trains, with shape
            (samples_calib, M).
        preprocess (bool, optional): Whether to preprocess emg before
            extension. Defaults to False.
        pca_components (Optional[torch.Tensor], optional): Fitted PCA
            components from the calibration (sklearn PCA.components_
            convention), with shape (n, D). Required whenever whitening/
            sep_vectors are dimensioned for a PCA-reduced space (i.e. the
            calibration used CBSSConfig.n_components). Defaults to None (no
            PCA reduction).
        pca_mean (Optional[torch.Tensor], optional): Fitted PCA mean from
            the calibration, with shape (D,). Required together with
            pca_components. Defaults to None.

    Returns:
        Tuple[AdaptDecomp, AdaptationResult]: adapter, the built instance
        (kept around for e.g. adapter.decomp.trace_cal or adapter.spikes);
        and outputs, its run() result.
    """
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
        adapt_config=trial_config,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )
    return adapter, adapter.run()


def compute_trial_loss(adapter: AdaptDecomp, outputs: AdaptationResult) -> float:
    """Score one run_trial_adapt_decomp() result for Optuna, guarding against divergence.

    Args:
        adapter (AdaptDecomp): Instance returned by run_trial_adapt_decomp(),
            used here only for adapter.decomp.trace_cal.
        outputs (AdaptationResult): Its run() result, used here for
            outputs.wh_trace/wh_loss/sv_loss. Requires
            trial_config.compute_loss=True (build_trial_config() sets this).

    Returns:
        float: wh_loss + sv_loss (each NaN-guarded, see
        _compute_total_wh_loss/_compute_total_sv_loss), or 1e10 if the
        trace ratio (wh_trace / trace_cal) falls outside (0.1, 50.0) --
        i.e. the whitening diverged for this trial.
    """
    # Guard against extreme trace ratios (wh_trace / trace_cal) that indicate divergence.
    trace_ratios = outputs.wh_trace / adapter.decomp.trace_cal
    agg = trace_ratios.median()
    if not (0.1 < agg.item() < 50.0):
        return 1e10

    # Otherwise, return the total loss (wh_loss + sv_loss) for this trial.
    return _compute_total_wh_loss(outputs.wh_loss) + _compute_total_sv_loss(outputs.sv_loss)


def _save_best_trial(
    best_dir: Path,
    outputs: Union[AdaptationResult, Dict[str, AdaptationResult]],
    trial_config: AdaptConfig,
) -> None:
    """Overwrite best_dir's saved AdaptationResult(s) + config with an improving trial.

    Args:
        best_dir (Path): Directory to write into (already created by the
            caller). "result.pkl" for a single AdaptationResult, or one
            "<condition>.pkl" per entry for a Dict[str, AdaptationResult]
            (pooled search).
        outputs (Union[AdaptationResult, Dict[str, AdaptationResult]]): This
            trial's result(s) -- see AdaptationResult.save().
        trial_config (AdaptConfig): This trial's resolved configuration,
            written alongside outputs as "config.yaml" (AdaptConfig.to_yaml())
            for future reference (e.g. reloading without re-running the
            search) -- YAML rather than pickle so it's human-readable and
            diffable, matching how configs are already stored elsewhere in
            the project (configs/model_configs/*.yml).

    Returns:
        None
    """
    if isinstance(outputs, dict):
        for cond, result in outputs.items():
            result.save(best_dir / f"{cond}.pkl")
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


def optimize_adapt_decomp(
    *,
    emg: torch.Tensor,
    whitening: torch.Tensor,
    sep_vectors: torch.Tensor,
    base_centr: torch.Tensor,
    spikes_centr: torch.Tensor,
    emg_calib: torch.Tensor,
    ipts_calib: torch.Tensor,
    spikes_calib: torch.Tensor,
    param_space: dict,
    base_config: Optional[AdaptConfig] = None,
    pca_components: Optional[torch.Tensor] = None,
    pca_mean: Optional[torch.Tensor] = None,
    gt_full_bin: Optional[np.ndarray] = None,
    compute_roa: bool = False,
    roa_kwargs: Optional[dict] = None,
    preprocess: bool = True,
    n_trials: int = 100,
    sampler=None,
    random_seed: Optional[int] = 1909,
    best_result_path: Optional[str] = None,
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple:
    """Search for optimal AdaptDecomp hyperparameters using Bayesian optimisation.

    Builds a fresh AdaptDecomp per trial. The loss is wh_loss + sv_loss, with NaN
    guard and trace ratio guard to avoid divergence.

    Args:
        emg (torch.Tensor): Online EMG to decompose, with shape
            (samples, channels).
        whitening (torch.Tensor): Calibration whitening matrix, with shape
            (D, D) (or (n, D) when the calibration used PCA reduction).
        sep_vectors (torch.Tensor): Calibration separation matrix, with shape
            (M, D) (or (M, n) when the calibration used PCA reduction).
        base_centr (torch.Tensor): Baseline centroids, with shape (M,).
        spikes_centr (torch.Tensor): Spike centroids, with shape (M,).
        emg_calib (torch.Tensor): Calibration EMG, with shape
            (samples_calib, channels).
        ipts_calib (torch.Tensor): Calibration source signals, with shape
            (samples_calib, M).
        spikes_calib (torch.Tensor): Calibration spike trains, with shape
            (samples_calib, M).
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
        base_config (Optional[AdaptConfig], optional): Resolved base
            AdaptConfig instance each trial is deep-copied from, before
            param_space overrides are applied. Defaults to None, which uses
            AdaptConfig().
        pca_components (Optional[torch.Tensor], optional): Fitted PCA
            components from the calibration (sklearn PCA.components_
            convention), with shape (n, D). Required whenever whitening/
            sep_vectors are dimensioned for a PCA-reduced space (i.e. the
            calibration used CBSSConfig.n_components) -- passed unchanged to
            every trial's AdaptDecomp. Defaults to None (no PCA reduction).
        pca_mean (Optional[torch.Tensor], optional): Fitted PCA mean from
            the calibration, with shape (D,). Required together with
            pca_components. Defaults to None.
        gt_full_bin (Optional[np.ndarray], optional): Ground-truth binary
            spike train for the full recording, already aligned to emg and
            matched/ordered to its motor units, with shape (samples, M).
            Required when compute_roa is True; ignored otherwise. Building
            it (matching a loader's recording to a calibration's matched
            ground-truth units) is dataset-specific and stays the caller's
            job -- AdaptDecomp has no notion of ground truth beyond the
            gt_matched_indices it threads through from a supervised
            calibration (see AdaptDecomp.from_calibration). Defaults to
            None.
        compute_roa (bool, optional): If True, compute
            rate_of_agreement_paired() against gt_full_bin for every trial
            and log it as diagnostic trial.set_user_attr()s ("roa_mean",
            "roa_per_unit"; see Returns) -- never added to the returned
            loss. On an improving trial, the per-unit RoA is also written
            onto the saved AdaptationResult.roa, so it's saved/returned
            alongside outputs (see best_result_path/Returns) without
            needing to dig through study.pkl's user_attrs. Requires
            gt_full_bin. Defaults to False.
        roa_kwargs (Optional[dict], optional): Extra keyword arguments
            forwarded to rate_of_agreement_paired() (e.g. tol_spike_ms).
            "fs" defaults to the resolved config's fs unless overridden
            here. Ignored when compute_roa is False. Defaults to None.
        preprocess (bool, optional): Whether to preprocess emg before
            extension, passed to each trial's AdaptDecomp. Defaults to
            True.
        n_trials (int, optional): Number of Optuna trials to run. Defaults
            to 100.
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses TPESampler with
            n_startup_trials=15.
        random_seed (Optional[int], optional): Seed for the default
            TPESampler (ignored when an explicit sampler is passed instead),
            so repeated searches over the same param_space draw the same
            sequence of trials and land on the same best_config. Independent
            of AdaptConfig.random_seed -- there isn't one: AdaptDecomp itself
            has no RNG dependency, so each trial is already deterministic
            given its suggested hyperparameters; this only pins which
            hyperparameters TPESampler chooses to try. Matches
            CBSSConfig.random_seed's default. Defaults to 1909.
        best_result_path (Optional[str], optional): If set, every trial
            that improves on the best loss seen so far overwrites a
            directory at this path with the winning trial's
            AdaptationResult ("result.pkl"), its resolved AdaptConfig as
            YAML ("config.yaml", via AdaptConfig.to_yaml() -- human-readable,
            unlike a pickle), and, once the search finishes, the completed
            Optuna study ("study.pkl") -- avoiding a second AdaptDecomp run
            just to recover the winning config's outputs (see Returns).
            Distinct from AdaptDecomp's own save_path/config.save_params
            (an HDF5 per-batch parameter trace, not produced here).
            Defaults to None (no saving).
        on_trial (Optional[Callable[[Dict[str, Any]], None]], optional):
            Called once per completed trial, from inside the objective
            itself, with one canonical log dict:
                {"trial_number": int, "loss": float, "params": dict,
                 "roa_mean": float, "roa_per_unit": List[float]}
            ("roa_mean"/"roa_per_unit" only present if compute_roa is True.)
            Lets a caller observe every trial as it finishes (e.g. to log it
            to wandb/mlflow/print) without this module taking a dependency
            on that tracker or the caller needing to know Optuna's
            Study/Trial API or which trial.user_attrs keys this function
            happens to set. Defaults to None.

    Raises:
        ValueError: If compute_roa is True and gt_full_bin is None.

    Returns:
        Tuple[AdaptConfig, optuna.Study]: best_config, the resolved
        base_config (AdaptConfig() if base_config was None) with the best
        trial's suggested parameters applied -- built via
        build_trial_config(), the same helper every trial uses, so it's
        already validate_literals()-checked; call best_config.to_dict()/
        .to_yaml() for a plain-dict/YAML representation. study is the
        completed Optuna study. If best_result_path is set, the winning
        trial's AdaptationResult is prepended instead:
        Tuple[AdaptationResult, AdaptConfig, optuna.Study] (outputs,
        best_config, study) -- with outputs.roa set when compute_roa is
        True. Every trial (not just the winning one) additionally carries
        user_attrs["roa_mean"] and ["roa_per_unit"] when compute_roa is
        True.

    Notes:
        The objective is built from four small composable pieces, also
        usable directly: suggest_overrides(), build_trial_config(),
        run_trial_adapt_decomp(), and compute_trial_loss(). For a search
        pooled across multiple conditions/recordings sharing one suggested
        hyperparameter set per trial, see optimize_adapt_decomp_pooled().
    """
    # Build base AdaptConfig
    run_config = base_config if base_config is not None else AdaptConfig()
    validate_literals(run_config)

    if compute_roa:
        if gt_full_bin is None:
            raise ValueError("compute_roa=True requires gt_full_bin to be set.")
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}

    # Best-so-far tracking, only active when best_result_path is set.
    best_dir = Path(best_result_path) if best_result_path is not None else None
    if best_dir is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_outputs: Optional[AdaptationResult] = None

    # Define the objective to optimise
    def objective(trial):
        nonlocal best_loss, best_outputs
        overrides = suggest_overrides(trial, param_space)
        trial_config = build_trial_config(run_config, overrides)
        adapter, outputs = run_trial_adapt_decomp(
            trial_config,
            emg=emg,
            whitening=whitening,
            sep_vectors=sep_vectors,
            base_centr=base_centr,
            spikes_centr=spikes_centr,
            emg_calib=emg_calib,
            ipts_calib=ipts_calib,
            spikes_calib=spikes_calib,
            preprocess=preprocess,
            pca_components=pca_components,
            pca_mean=pca_mean,
        )
        loss = compute_trial_loss(adapter, outputs)

        # Canonical per-trial log record -- the single source of truth for what's
        # loggable about this trial, consumed by on_trial() below and mirrored onto
        # trial.user_attrs for post-hoc access via study.trials/trials_dataframe().
        log_vars: Dict[str, Any] = {
            "trial_number": trial.number,
            "loss": loss,
            "params": overrides,
        }

        # Diagnostic only -- never folded into loss.
        if compute_roa:
            pred_spikes = adapter.spikes.detach().cpu().numpy().astype(np.float32)
            roa_vals, _, _ = rate_of_agreement_paired(gt_full_bin, pred_spikes, **roa_kwargs)
            outputs.roa = np.asarray(roa_vals, dtype=np.float32)  # travels with outputs.save()
            roa_mean = float(np.nanmean(roa_vals)) * 100
            roa_per_unit = [float(x) for x in roa_vals]
            trial.set_user_attr("roa_mean", roa_mean)
            trial.set_user_attr("roa_per_unit", roa_per_unit)
            log_vars["roa_mean"] = roa_mean
            log_vars["roa_per_unit"] = roa_per_unit

        if on_trial is not None:
            on_trial(log_vars)

        # Overwrite the saved best-so-far result whenever this trial improves on it.
        if best_dir is not None and loss < best_loss:
            _save_best_trial(best_dir, outputs, trial_config)
            best_outputs = outputs
            best_loss = loss

        return loss

    # Build the Optuna study and run the optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            n_startup_trials=15, seed=random_seed,
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    # Build the winning AdaptConfig via the same helper every trial used.
    best_config = build_trial_config(run_config, study.best_params)

    if best_dir is not None:
        _finalize_best_result(best_dir, study)
        return best_outputs, best_config, study

    # Return the best configuration and the study object for further analysis if needed
    return best_config, study


@dataclass
class PooledCondition:
    """One condition's tensors (and optional ground truth) for a pooled hyperparameter search.

    One instance per condition passed to optimize_adapt_decomp_pooled()'s
    pool. Every field but gt_full_bin is forwarded unchanged to
    run_trial_adapt_decomp() for every trial.

    Attributes:
        emg (torch.Tensor): Online EMG to decompose, with shape
            (samples, channels).
        whitening (torch.Tensor): Calibration whitening matrix, with shape
            (D, D) (or (n, D) when the calibration used PCA reduction).
        sep_vectors (torch.Tensor): Calibration separation matrix, with
            shape (M, D) (or (M, n) when the calibration used PCA reduction).
        base_centr (torch.Tensor): Baseline centroids, with shape (M,).
        spikes_centr (torch.Tensor): Spike centroids, with shape (M,).
        emg_calib (torch.Tensor): Calibration EMG, with shape
            (samples_calib, channels).
        ipts_calib (torch.Tensor): Calibration source signals, with shape
            (samples_calib, M).
        spikes_calib (torch.Tensor): Calibration spike trains, with shape
            (samples_calib, M).
        pca_components (Optional[torch.Tensor]): Fitted PCA components from
            the calibration (sklearn PCA.components_ convention), with
            shape (n, D). Defaults to None (no PCA reduction).
        pca_mean (Optional[torch.Tensor]): Fitted PCA mean from the
            calibration, with shape (D,). Defaults to None.
        preprocess (bool, optional): Whether to preprocess emg before
            extension, for every condition's AdaptDecomp. Defaults to True.
        gt_full_bin (Optional[np.ndarray]): Ground-truth binary spike train
            for the full recording, already paired to the corresponding
            AdaptDecomp's M units for the calibration window, with shape
            (samples, M). Required when optimize_adapt_decomp_pooled(compute_roa=True)
            is used; ignored otherwise. Defaults to None.
        
    """

    emg: torch.Tensor
    whitening: torch.Tensor
    sep_vectors: torch.Tensor
    base_centr: torch.Tensor
    spikes_centr: torch.Tensor
    emg_calib: torch.Tensor
    ipts_calib: torch.Tensor
    spikes_calib: torch.Tensor
    pca_components: Optional[torch.Tensor] = None
    pca_mean: Optional[torch.Tensor] = None
    preprocess: bool = True
    gt_full_bin: Optional[np.ndarray] = None


def optimize_adapt_decomp_pooled(
    *,
    pool: Dict[str, PooledCondition],
    base_config: Optional[AdaptConfig] = None,
    compute_roa: bool = False,
    roa_kwargs: Optional[dict] = None,
    param_space: dict,
    n_trials: int = 100,
    sampler=None,
    random_seed: Optional[int] = 1909,
    best_result_path: Optional[str] = None,
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[AdaptConfig, optuna.Study]:
    """Search hyperparameters shared across every condition in pool, summing per-condition loss.

    One suggested parameter set per trial (e.g. wh_learning_rate,
    sv_learning_rate) is applied unchanged to a fresh AdaptDecomp per
    condition (see run_trial_adapt_decomp) -- the objective is the SUM of
    their compute_trial_loss() values, not the mean: a setting that
    diverges on even one pooled condition should dominate the sum and be
    rejected, which summing makes explicit rather than incidental.
    Otherwise identical to optimize_adapt_decomp -- see its docstring for
    the shared building blocks (suggest_overrides/build_trial_config/
    run_trial_adapt_decomp/compute_trial_loss) and for base_config/
    param_space/sampler/best_result_path semantics, which carry over
    unchanged here. on_trial's log dict differs -- see its own Args entry
    below.

    Args:
        pool (Dict[str, PooledCondition]): Condition name -> its tensors
            (and optional ground truth, and its own preprocess flag). Every
            condition is evaluated on every trial.
        base_config (Optional[AdaptConfig], optional): Resolved base
            AdaptConfig instance each trial/condition is deep-copied from,
            before param_space overrides are applied. Defaults to None,
            which uses AdaptConfig().
        compute_roa (bool, optional): If True, compute
            rate_of_agreement_paired() per condition against
            PooledCondition.gt_full_bin and log it as diagnostic
            trial.set_user_attr()s for every trial (see Returns) -- never
            added to the returned loss. On an improving trial, the
            per-unit RoA is also written onto that condition's
            AdaptationResult.roa, so it's saved/returned alongside
            outputs (see best_result_path/Returns) without needing to dig
            through study.pkl's user_attrs. Every condition in pool must
            then set gt_full_bin. Defaults to False.
        roa_kwargs (Optional[dict], optional): Extra keyword arguments
            forwarded to rate_of_agreement_paired() (e.g. tol_spike_ms).
            "fs" defaults to the resolved config's fs unless overridden
            here. Ignored when compute_roa is False. Defaults to None.
        param_space (dict): Maps parameter name to a (kind, low, high)
            tuple. See optimize_adapt_decomp's docstring for the full
            format and DEFAULT_PARAM_SPACE.
        n_trials (int, optional): Number of Optuna trials to run. Defaults
            to 100.
        sampler (Optional[optuna.samplers.BaseSampler], optional): Optuna
            sampler. Defaults to None, which uses
            TPESampler(n_startup_trials=15).
        random_seed (Optional[int], optional): Seed for the default
            TPESampler (ignored when an explicit sampler is passed instead).
            See optimize_adapt_decomp's docstring for why this is the only
            seed needed -- AdaptDecomp has no RNG dependency of its own, so
            this only pins which hyperparameters TPESampler chooses to try.
            Matches CBSSConfig.random_seed's default. Defaults to 1909.
        best_result_path (Optional[str], optional): If set, every trial
            that improves on the pooled total loss seen so far overwrites
            a directory at this path with one AdaptationResult per
            condition ("<condition>.pkl"), the winning shared AdaptConfig
            as YAML ("config.yaml", via AdaptConfig.to_yaml()), and, once
            the search finishes, the completed Optuna study
            ("study.pkl") -- see Returns. Defaults to None (no saving).
        on_trial (Optional[Callable[[Dict[str, Any]], None]], optional):
            Called once per completed trial, from inside the objective
            itself, with one canonical log dict:
                {"trial_number": int, "loss": float, "params": dict,
                 "per_condition": {cond: {"loss": float, "roa_mean": float,
                                           "roa_per_unit": List[float]}},
                 "roa_mean": float}
            "loss" is the pooled SUM across every condition; "per_condition"
            has one entry per pool key with that condition's own loss (and
            RoA sub-keys, only present if compute_roa); "roa_mean" (top
            level) is the pooled mean of per-condition RoA means, only
            present if compute_roa is True. Defaults to None.

    Raises:
        ValueError: If compute_roa is True and any PooledCondition in pool
            has gt_full_bin=None.

    Returns:
        Tuple[AdaptConfig, optuna.Study]: best_config, the resolved
        base_config (AdaptConfig() if base_config was None) with the best
        trial's suggested parameters applied (shared across every
        condition) -- built via build_trial_config(), the same helper
        every trial uses, so it's already validate_literals()-checked;
        call best_config.to_dict()/.to_yaml() for a plain-dict/YAML
        representation. study is the completed Optuna study. If
        best_result_path is set, the winning trial's per-condition results
        are prepended instead: Tuple[Dict[str, AdaptationResult],
        AdaptConfig, optuna.Study] (outputs, best_config, study), where
        outputs maps
        condition name to its AdaptationResult -- with .roa set per
        condition when compute_roa is True (see compute_roa). Every trial
        (not just the winning one) additionally carries
        user_attrs["loss_<condition>"], ["roa_mean_<condition>"], and
        ["roa_per_unit_<condition>"] per condition, plus
        ["roa_mean_pooled"] (the mean of the per-condition RoA means), when
        compute_roa is True.

    Notes:
        Building PooledCondition.gt_full_bin (matching a loader's recording
        to a calibration's matched ground-truth units, over the full
        recording) is dataset-specific and stays the caller's job.
    """
    run_config = base_config if base_config is not None else AdaptConfig()
    validate_literals(run_config)

    if compute_roa:
        roa_kwargs = {"fs": run_config.fs, **(roa_kwargs or {})}
        missing = [cond for cond, c in pool.items() if c.gt_full_bin is None]
        if missing:
            raise ValueError(
                f"compute_roa=True requires gt_full_bin to be set for every condition in "
                f"pool; missing for: {missing}"
            )

    # Best-so-far tracking, only active when best_result_path is set.
    best_dir = Path(best_result_path) if best_result_path is not None else None
    if best_dir is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_outputs: Optional[Dict[str, AdaptationResult]] = None

    # Define the objective to optimise
    def objective(trial):
        nonlocal best_loss, best_outputs
        # ONE suggestion, shared across every condition in the pool.
        overrides = suggest_overrides(trial, param_space)

        total_loss = 0.0
        roa_means = []
        trial_outputs: Dict[str, AdaptationResult] = {}
        per_condition: Dict[str, Dict[str, Any]] = {}
        reference_config: Optional[AdaptConfig] = None
        for cond, c in pool.items():
            trial_config = build_trial_config(run_config, overrides)
            if reference_config is None:
                reference_config = trial_config  # value-identical across conditions

            adapter, outputs = run_trial_adapt_decomp(
                trial_config,
                emg=c.emg,
                whitening=c.whitening,
                sep_vectors=c.sep_vectors,
                base_centr=c.base_centr,
                spikes_centr=c.spikes_centr,
                emg_calib=c.emg_calib,
                ipts_calib=c.ipts_calib,
                spikes_calib=c.spikes_calib,
                preprocess=c.preprocess,
                pca_components=c.pca_components,
                pca_mean=c.pca_mean,
            )
            loss = compute_trial_loss(adapter, outputs)
            total_loss += loss
            trial_outputs[cond] = outputs
            trial.set_user_attr(f"loss_{cond}", loss)
            per_condition[cond] = {"loss": loss}

            # Diagnostic only -- never folded into total_loss.
            if compute_roa:
                pred_spikes = adapter.spikes.detach().cpu().numpy().astype(np.float32)
                roa_vals, _, _ = rate_of_agreement_paired(c.gt_full_bin, pred_spikes, **roa_kwargs)
                outputs.roa = np.asarray(roa_vals, dtype=np.float32)  # travels with outputs.save()
                roa_mean = float(np.nanmean(roa_vals)) * 100
                roa_per_unit = [float(x) for x in roa_vals]
                trial.set_user_attr(f"roa_mean_{cond}", roa_mean)
                trial.set_user_attr(f"roa_per_unit_{cond}", roa_per_unit)
                per_condition[cond]["roa_mean"] = roa_mean
                per_condition[cond]["roa_per_unit"] = roa_per_unit
                roa_means.append(roa_mean)

        # Canonical per-trial log record -- see on_trial's Args entry above.
        log_vars: Dict[str, Any] = {
            "trial_number": trial.number,
            "loss": total_loss,
            "params": overrides,
            "per_condition": per_condition,
        }

        if compute_roa:
            roa_mean_pooled = float(np.mean(roa_means))
            trial.set_user_attr("roa_mean_pooled", roa_mean_pooled)
            log_vars["roa_mean"] = roa_mean_pooled

        if on_trial is not None:
            on_trial(log_vars)

        # Overwrite the saved best-so-far result whenever this trial improves on it.
        if best_dir is not None and total_loss < best_loss:
            _save_best_trial(best_dir, trial_outputs, reference_config)
            best_outputs = trial_outputs
            best_loss = total_loss

        # Pooled objective: SUM, not mean, across every condition (see docstring).
        return total_loss

    # Build the Optuna study and run the optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            n_startup_trials=15, seed=random_seed,
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    # Build the winning AdaptConfig via the same helper every trial used.
    best_config = build_trial_config(run_config, study.best_params)

    if best_dir is not None:
        _finalize_best_result(best_dir, study)
        return best_outputs, best_config, study

    return best_config, study
