from typing import Optional

import wandb
import copy
import torch
import numpy as np
import typer
from dataclasses import asdict
from adapt_decomp.adaptation.config import load_yaml, load_config
from adapt_decomp.utils import load_data
from adapt_decomp.utils.loaders import PooledDatasetMemory
from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.adaptation.optimize import optimize_adapt_decomp_pooled_memory, ObjectiveName
from adapt_decomp.spikes import rate_of_agreement_paired

app = typer.Typer(help="Adaptive EMG decomposition.")


def _run_pooled(pool, config):
    """Run one AdaptDecomp per pooled dataset, sharing the same config.

    Args:
        pool: Dataset name -> PooledDatasetMemory (from load_data()/load_pooled_cbss_memory()).
        config: The resolved AdaptConfig, applied to every dataset (deep-copied
            per dataset so one run can't mutate another's).

    Returns:
        Dict: Dataset name -> its AdaptationResult.
    """
    outputs = {}
    for name, dataset in pool.items():
        adapter = AdaptDecomp.from_calibration(
            calibration=dataset.calibration, cbss_config=dataset.cbss_config,
            adapt_config=copy.deepcopy(config),
        )
        outputs[name] = adapter.process_data(dataset.emg, preprocess=dataset.preprocess)
    return outputs


def _log_pooled_outputs(outputs_by_dataset, pool, config):
    """Log per-batch time series plus per-dataset/pooled summary stats to the active wandb run.

    One dataset or many alike.

    Args:
        outputs_by_dataset: Dataset name -> AdaptationResult, from _run_pooled()
            or optimize_adapt_decomp_pooled_memory(best_result_path=...).
        pool: The same pool passed in (needs .calibration/.gt_paired_bin per dataset).
        config: The resolved AdaptConfig for this run.

    Returns:
        None
    """

    max_batches = max(len(outputs['wh_loss']) for outputs in outputs_by_dataset.values())
    for batch in range(max_batches):
        log_dict = {}
        for name, outputs in outputs_by_dataset.items():
            if batch < len(outputs['wh_loss']):
                log_dict[f'{name}/wh_loss'] = outputs['wh_loss'][batch]
                log_dict[f'{name}/sv_loss'] = outputs['sv_loss'][batch].nansum()
                log_dict[f'{name}/total_time_ms'] = outputs['total_time_ms'][batch]
        wandb.log(log_dict)

    roa_adapt_means = []
    roa_calib_means = []
    total_time_ms_means = []
    for name, outputs in outputs_by_dataset.items():
        wandb.summary[f'{name}/wh_loss'] = torch.median(outputs['wh_loss']).item()
        wandb.summary[f'{name}/sv_loss'] = torch.median(outputs['sv_loss'].nansum(dim=1)).item()
        wandb.summary[f'{name}/total_loss'] = outputs['total_loss'].item()
        total_time_ms_mean = torch.mean(outputs['total_time_ms']).item()
        wandb.summary[f'{name}/total_time_ms'] = total_time_ms_mean
        total_time_ms_means.append(total_time_ms_mean)

        gt_full_bin = pool[name].gt_paired_bin
        if gt_full_bin is not None:
            roa_adapt, _, _ = rate_of_agreement_paired(
                gt_full_bin, outputs['spikes'].numpy(), fs=config.fs, tol_spike_ms=2,
            )
            roa_mean = float(np.mean(roa_adapt))
            wandb.summary[f'{name}/roa_adapt'] = roa_mean
            roa_adapt_means.append(roa_mean)

            # Calibration-window RoA, set on calibration by select_supervised() --
            # the pooled-loading equivalent of load_example's roa_calib.
            calib_roa = pool[name].calibration.roa
            if calib_roa is not None:
                roa_calib_mean = float(np.mean(calib_roa))
                wandb.summary[f'{name}/roa_calib'] = roa_calib_mean
                roa_calib_means.append(roa_calib_mean)

    wandb.summary['wh_loss'] = float(
        sum(outputs['wh_loss_median'].item() for outputs in outputs_by_dataset.values())
    )
    wandb.summary['sv_loss'] = float(
        sum(outputs['sv_loss_median'].item() for outputs in outputs_by_dataset.values())
    )
    wandb.summary['total_loss'] = float(
        sum(outputs['total_loss'].item() for outputs in outputs_by_dataset.values())
    )
    wandb.summary['total_time_ms'] = float(np.mean(total_time_ms_means))
    if roa_adapt_means:
        wandb.summary['roa_adapt'] = float(np.mean(roa_adapt_means))
    if roa_calib_means:
        wandb.summary['roa_calib'] = float(np.mean(roa_calib_means))


def _log_trial_to_wandb(log_vars):
    """on_trial hook: log one completed trial's loss/params/mean RoA to the active wandb run.

    Args:
        log_vars: Per-trial dict from optimize_adapt_decomp_pooled_memory()
            (trial_number/loss/objective/sv_loss/wh_loss/total_loss/params, plus roa
            (guarded, inverted-for-minimisation) /roa_mean/roa_per_unit when compute_roa=True,
            i.e. whenever ground truth is available or objective="roa").

    Returns:
        None
    """
    log_dict = {
        "optuna/trial_number": log_vars["trial_number"],
        "optuna/loss": log_vars["loss"],
        "optuna/objective": log_vars["objective"],
        "optuna/sv_loss": log_vars["sv_loss"],
        "optuna/wh_loss": log_vars["wh_loss"],
        "optuna/total_loss": log_vars["total_loss"],
    }
    log_dict.update({f"optuna/param_{k}": v for k, v in log_vars["params"].items()})
    if "roa" in log_vars:
        log_dict["optuna/roa_loss"] = log_vars["roa"]
    if "roa_mean" in log_vars:
        log_dict["optuna/roa_mean"] = log_vars["roa_mean"]
    wandb.log(log_dict)


def _setup(adapt_config, data_config, wandb_project_name, wandb_config=None):
    """Load data_config into a pool, resolve config, and start/rename the active wandb run.

    Args:
        adapt_config: Path to model config YAML.
        data_config: Path to data config YAML.
        wandb_project_name: WandB project name, used only if no wandb run is active yet.
        wandb_config: Optional dict of wandb sweep overrides.

    Returns:
        Tuple[Dict[str, PooledDatasetMemory], AdaptConfig]: pool, config.
    """
    data_config = load_yaml(data_config)
    data = load_data(data_config)

    if isinstance(data, dict) and data and isinstance(next(iter(data.values())), PooledDatasetMemory):
        pool = data
    else:
        # load_example's legacy flat-dict shape -- wrapped into a one-entry pool
        # so everything downstream has exactly one code path to go through.
        pool = {"dataset": PooledDatasetMemory(
            emg=data['emg'], calibration=data['cbss_result'], cbss_config=data['cbss_config'],
            preprocess=data['preprocess'], gt_paired_bin=data.get('gt_full_bin'),
        )}

    config = load_config(adapt_config, wandb_config)
    config.ext_fact = next(iter(pool.values())).calibration.ext_fact

    run_name = (
        f'adapt_decomp_dv{config.wh_learning_rate:.0e}'
        f'_db{config.sv_learning_rate:.0e}'
    )
    if wandb.run is None:
        wandb.init(project=wandb_project_name, name=run_name, config=asdict(config))
    else:
        wandb.run.name = run_name

    return pool, config


def _run(adapt_config, data_config, wandb_project_name, wandb_config=None):
    """One plain adaptive-decomposition pass, no search -- one dataset or many alike.

    Args:
        adapt_config: Path to model config YAML.
        data_config: Path to data config YAML.
        wandb_project_name: WandB project name, used only if no wandb run is active yet.
        wandb_config: Optional dict of wandb sweep overrides.

    Returns:
        None
    """
    pool, config = _setup(adapt_config, data_config, wandb_project_name, wandb_config)
    outputs = _run_pooled(pool, config)
    _log_pooled_outputs(outputs, pool, config)
    wandb.finish()


@app.command(name="run")
def run(
    adapt_config: str = typer.Option(..., "--adapt_config", help="Path to model config YAML"),
    data_config: str = typer.Option(..., "--data_config", help="Path to data config YAML"),
    wandb_project_name: str = typer.Option(
        "adaptive_emg_decomp_dyn", "--wandb_project_name", help="WandB project name",
    ),
) -> None:
    """One plain adaptive-decomposition pass, no search -- one dataset or many alike."""
    _run(adapt_config, data_config, wandb_project_name)


def _run_optuna(adapt_config, data_config, wandb_project_name, optim_config,
                 objective=None, n_trials=None, best_result_path=None,
                 compute_roa=None, roa_kwargs=None):
    """Optuna hyperparameter search -- one dataset or many alike.

    Args:
        adapt_config: Path to model config YAML.
        data_config: Path to data config YAML.
        wandb_project_name: WandB project name, used only if no wandb run is active yet.
        optim_config: Path to Optuna search-settings YAML.
        objective: Overrides optim_config's own objective. None -> optim_config's value,
            else "total_loss".
        n_trials: Overrides optim_config's own n_trials. None -> optim_config's value,
            else 100.
        best_result_path: Directory to save the winning trial's AdaptationResult/config/
            study. None -> only trial-level summaries logged to wandb.
        compute_roa: Forces RoA scoring on/off. None -> auto: True iff every dataset in
            the pool has ground truth.
        roa_kwargs: Extra keyword arguments forwarded to rate_of_agreement_paired()
            (e.g. tol_spike_ms). None -> optimize_adapt_decomp_pooled_memory's own default.

    Returns:
        None
    """
    pool, config = _setup(adapt_config, data_config, wandb_project_name)

    optim_settings = load_yaml(optim_config)
    param_space = {k: tuple(v) for k, v in optim_settings.get("param_space", {}).items()}
    resolved_objective = objective if objective is not None else optim_settings.get("objective", "total_loss")
    resolved_n_trials = n_trials if n_trials is not None else optim_settings.get("n_trials", 100)
    resolved_n_jobs = optim_settings.get("n_jobs", 1)
    resolved_random_seed = optim_settings.get("random_seed", 1909)
    if compute_roa is None:
        compute_roa = all(dataset.gt_paired_bin is not None for dataset in pool.values())

    result = optimize_adapt_decomp_pooled_memory(
        pool=pool,
        param_space=param_space,
        objective=resolved_objective,
        n_trials=resolved_n_trials,
        n_jobs=resolved_n_jobs,
        random_seed=resolved_random_seed,
        base_config=config,
        on_trial=_log_trial_to_wandb,
        compute_roa=compute_roa,
        roa_kwargs=roa_kwargs,
        best_result_path=best_result_path,
    )

    # best_result_path set -> also got the winning trial's AdaptationResult(s);
    # otherwise just log study-level summaries.
    if best_result_path is not None:
        outputs, best_config, study = result
        _log_pooled_outputs(outputs, pool, config)
    else:
        best_config, study = result
        wandb.summary['best_loss'] = study.best_value
        if compute_roa:
            wandb.summary['best_roa_mean'] = study.best_trial.user_attrs.get('roa_mean_pooled')
    wandb.log({'best_config': best_config.to_dict()})
    wandb.finish()


@app.command(name="run_optuna")
def run_optuna(
    adapt_config: str = typer.Option(..., "--adapt_config", help="Path to model config YAML"),
    data_config: str = typer.Option(..., "--data_config", help="Path to data config YAML"),
    optim_config: str = typer.Option(
        ..., "--optim_config",
        help="Path to Optuna search-settings YAML -- param_space/objective/n_trials/n_jobs/"
             "random_seed. See configs/adapt_configs/sweep_optuna.yaml.",
    ),
    wandb_project_name: str = typer.Option(
        "adaptive_emg_decomp_dyn", "--wandb_project_name", help="WandB project name",
    ),
    objective: Optional[ObjectiveName] = typer.Option(
        None, "--objective",
        help="Overrides optim_config's own objective if set. Falls back to total_loss if "
             "neither is set.",
    ),
    n_trials: Optional[int] = typer.Option(
        None, "--n_trials",
        help="Overrides optim_config's own n_trials if set. Falls back to 100 if neither "
             "is set.",
    ),
    best_result_path: Optional[str] = typer.Option(
        None, "--best_result_path",
        help="Directory to save the winning trial's AdaptationResult/config/study. Omitted "
             "-> only trial-level summaries are logged to wandb.",
    ),
) -> None:
    """Optuna hyperparameter search -- one dataset or many alike. compute_roa/roa_kwargs
    have no CLI flags -- call _run_optuna() directly (not via this CLI command) to set them.
    """
    _run_optuna(adapt_config, data_config, wandb_project_name, optim_config,
                objective=objective, n_trials=n_trials, best_result_path=best_result_path)


@app.command(name="run_wandb")
def run_wandb(
    adapt_config: str = typer.Option(..., "--adapt_config", help="Path to model config YAML"),
    data_config: str = typer.Option(..., "--data_config", help="Path to data config YAML"),
    sweep_config: str = typer.Option(
        ..., "--sweep_config",
        help="Path to wandb sweep config YAML. See configs/adapt_configs/sweep_wandb.yaml.",
    ),
    wandb_project_name: str = typer.Option(
        "adaptive_emg_decomp_dyn", "--wandb_project_name", help="WandB project name",
    ),
    sweep_counts: Optional[int] = typer.Option(
        None, "--sweep_counts",
        help="Overrides sweep_config's own sweep_counts if set. Falls back to 20 if "
             "neither is set.",
    ),
) -> None:
    """wandb-managed sweep -- each iteration is a plain run() call, hyperparameters chosen
    by wandb itself (no nested Optuna search, by design)."""
    sweep_settings = load_yaml(sweep_config)
    yaml_sweep_counts = sweep_settings.pop("sweep_counts", None)  # popped either way --
                                                                    # wandb.sweep() must never see it
    resolved_sweep_counts = sweep_counts if sweep_counts is not None else (
        yaml_sweep_counts if yaml_sweep_counts is not None else 20
    )

    sweep_id = wandb.sweep(sweep_settings, project=wandb_project_name)

    def sweep_run():
        wandb.init(project=wandb_project_name, name="temp_run")
        _run(adapt_config, data_config, wandb_project_name, wandb_config=dict(wandb.config))

    wandb.agent(sweep_id, function=sweep_run, count=resolved_sweep_counts)


if __name__ == "__main__":
    app()
