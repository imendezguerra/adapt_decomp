import wandb
import argparse
import torch
import numpy as np
from dataclasses import asdict
from adapt_decomp.adaptation.config import load_yaml, load_config
from adapt_decomp.utils import load_data
from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.adaptation.optimize import optimize_adapt_decomp
from adapt_decomp.spikes import rate_of_agreement, rate_of_agreement_paired


def _log_outputs(outputs, data, config):
    """Log decomposition outputs and mean RoA (if ground truth is available) to
    the active wandb run.

    Only the mean RoA across units is logged, as wandb.summary -- a single
    scalar per run, directly comparable across sweep runs in wandb's
    parallel-coordinates/scatter views. Per-unit RoA is deliberately not
    logged individually: wandb has no good way to visualise a per-motor-unit
    breakdown alongside per-run hyperparameters, so it would just add noise
    to the sweep comparison this is actually used for.

    If outputs already carries a non-None .roa (set by
    optimize_adapt_decomp(..., compute_roa=True) on the winning trial), it is
    logged directly instead of being recomputed here. The recompute path
    (rate_of_agreement_paired against a freshly reordered spikes_gt) stays for
    the plain, non-optimised run path, whose AdaptationResult has no .roa.
    """
    if 'spikes_gt' in data:
        roa_calib, pair_calib, _ = rate_of_agreement(
            data['spikes_gt'].numpy()[0:data['spikes_calib'].shape[0]],
            data['spikes_calib'].numpy(),
            fs=data['fs'],
            tol_spike_ms=2,
        )
        if 'roa' in outputs:
            roa_adapt = outputs['roa']
        else:
            spikes_gt = data['spikes_gt'].numpy()[:, np.array(pair_calib)[:, 0]]
            roa_adapt, _, _ = rate_of_agreement_paired(
                spikes_gt,
                outputs['spikes'].numpy(),
                fs=data['fs'],
                tol_spike_ms=2,
            )
        wandb.summary['roa_calib'] = np.mean(roa_calib)
        wandb.summary['roa_adapt'] = np.mean(roa_adapt)

    batches = len(outputs['wh_loss'])
    for batch in range(batches):
        wandb.log({
            'wh_loss': outputs['wh_loss'][batch],
            'sv_loss': outputs['sv_loss'][batch].nansum(),
            'total_loss': outputs['total_loss'][batch],
            'total_time_ms': outputs['total_time_ms'][batch],
        })

    wandb.summary['total_time_ms'] = torch.mean(outputs['total_time_ms'])
    wandb.summary['wh_loss'] = torch.median(outputs['wh_loss'])
    wandb.summary['sv_loss'] = torch.median(outputs['sv_loss'].nansum(dim=1))
    wandb.summary['total_loss'] = torch.median(outputs['total_loss'])


def _log_trial_to_wandb(log_vars):
    """on_trial hook: log one completed trial's loss/params/mean RoA to the active wandb run.

    log_vars is the single dict optimize_adapt_decomp()/optimize_adapt_decomp_pooled()
    build once per trial (trial_number/loss/params, plus roa_mean/roa_per_unit,
    and for pooled a "per_condition" breakdown, when compute_roa=True) --
    keeps optimize.py itself free of any wandb dependency and this function
    free of any need to know Optuna's Study/Trial API.

    Deliberately logs only the top-level "roa_mean" -- never "roa_per_unit"
    (a per-motor-unit breakdown has no good wandb visualisation) or
    "per_condition" (pooled's per-condition detail). This is enough for both
    shapes unchanged: for optimize_adapt_decomp, "roa_mean" is that trial's
    mean RoA; for optimize_adapt_decomp_pooled, it's already the mean of the
    per-condition mean RoAs for that trial's (shared) hyperparameter set --
    exactly the one number needed to compare loss vs. RoA per hyperparameter
    value when defining a sweep.
    """
    log_dict = {"optuna/trial_number": log_vars["trial_number"], "optuna/loss": log_vars["loss"]}
    log_dict.update({f"optuna/param_{k}": v for k, v in log_vars["params"].items()})
    if "roa_mean" in log_vars:
        log_dict["optuna/roa_mean"] = log_vars["roa_mean"]
    wandb.log(log_dict)


def run(model_config, data_config, wandb_project_name, wandb_config=None,
        optim_config=None, n_trials=100, best_result_path=None):
    """Run adaptive decomposition, optionally with Optuna hyperparameter search.

    Args:
        model_config: Path to model config YAML.
        data_config: Path to data config YAML.
        wandb_project_name: WandB project name for logging.
        wandb_config: Optional dict of wandb sweep overrides.
        optim_config: Path to Optuna param-space YAML. When provided, runs
            single-objective hyperparameter search, with every trial's
            loss/params/RoA logged live to wandb (see _log_trial_to_wandb).
        n_trials: Number of Optuna trials (used when optim_config is set).
        best_result_path: Optional directory to also persist the winning
            trial's full AdaptationResult/config/study to disk (used when
            optim_config is set). When omitted, only trial-level summaries
            are logged to wandb -- no per-batch trace of the winning trial
            is saved or logged.
    """
    data_config = load_yaml(data_config)
    data = load_data(data_config)

    config = load_config(model_config, wandb_config)
    config.ext_fact = data['ext_fact']

    run_name = (
        f'adapt_decomp_dv{config.wh_learning_rate:.0e}'
        f'_db{config.sv_learning_rate:.0e}'
    )

    if wandb.run is None:
        wandb.init(project=wandb_project_name, name=run_name, config=asdict(config))
    else:
        wandb.run.name = run_name

    # base_config= vs adapt_config=: optimize_adapt_decomp() names this
    # parameter "base_config", AdaptDecomp's own constructor uses
    # "adapt_config" -- kept out of common_kwargs and passed explicitly per
    # branch below so each call gets the keyword its own signature expects.
    common_kwargs = dict(
        emg=data['emg'].clone(),
        whitening=data['whitening'].clone(),
        sep_vectors=data['sep_vectors'].clone(),
        base_centr=data['base_centr'].clone(),
        spikes_centr=data['spikes_centr'].clone(),
        emg_calib=data['emg_calib'].clone(),
        ipts_calib=data['ipts_calib'].clone(),
        spikes_calib=data['spikes_calib'].clone(),
        preprocess=data['preprocess'],
    )

    if optim_config is not None:
        param_space_raw = load_yaml(optim_config)
        param_space = {k: tuple(v) for k, v in param_space_raw.items()}

        # Ground truth, if the data config's loader supplies it -- reordered to match
        # the calibration's matched-unit ordering, exactly as _log_outputs already does
        # for the plain (non-optimised) path's own RoA computation.
        compute_roa = 'spikes_gt' in data
        gt_full_bin = None
        if compute_roa:
            _, pair_calib, _ = rate_of_agreement(
                data['spikes_gt'].numpy()[0:data['spikes_calib'].shape[0]],
                data['spikes_calib'].numpy(),
                fs=data['fs'],
                tol_spike_ms=2,
            )
            gt_full_bin = data['spikes_gt'].numpy()[:, np.array(pair_calib)[:, 0]]

        result = optimize_adapt_decomp(
            param_space=param_space,
            n_trials=n_trials,
            base_config=config,
            on_trial=_log_trial_to_wandb,
            compute_roa=compute_roa,
            gt_full_bin=gt_full_bin,
            best_result_path=best_result_path,
            **common_kwargs,
        )

        # best_result_path set -> optimize_adapt_decomp prepends the winning trial's
        # full AdaptationResult, so it can go through the same _log_outputs() path as
        # a plain run. Left unset -> no per-batch AdaptationResult exists; log the
        # study-level summary instead.
        if best_result_path is not None:
            outputs, best_config, study = result
            _log_outputs(outputs, data, config)
        else:
            best_config, study = result
            wandb.summary['best_loss'] = study.best_value
            if compute_roa:
                wandb.summary['best_roa_mean'] = study.best_trial.user_attrs.get('roa_mean')
        wandb.log({'best_config': best_config.to_dict()})
    else:
        adapter = AdaptDecomp(adapt_config=config, **common_kwargs)
        outputs = adapter.run()
        _log_outputs(outputs, data, config)

    wandb.finish()


def main():
    """CLI entry point for adaptive EMG decomposition."""
    parser = argparse.ArgumentParser(description="Adaptive EMG decomposition.")
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to data config YAML")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--sweep_config", type=str, default=None,
                        help="Path to wandb sweep config YAML (wandb-managed sweep)")
    parser.add_argument("--sweep_counts", type=int, default=20,
                        help="Number of wandb sweep trials")
    parser.add_argument("--optim_config", type=str, default=None,
                        help="Path to Optuna param-space YAML (single-obj optimisation)")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of Optuna trials (used with --optim_config)")
    parser.add_argument("--best_result_path", type=str, default=None,
                        help="Optional directory to save the winning Optuna trial's full "
                             "AdaptationResult/config/study (used with --optim_config). If "
                             "omitted, only trial-level summaries are logged to wandb.")
    parser.add_argument("--wandb_project_name", type=str,
                        default="adaptive_emg_decomp_dyn",
                        help="WandB project name")
    args = parser.parse_args()

    if args.sweep_config:
        sweep_config = load_yaml(args.sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project_name)

        def sweep_run():
            wandb.init(project=args.wandb_project_name, name="temp_run")
            run(args.model_config, args.data_config, args.wandb_project_name,
                wandb_config=dict(wandb.config),
                optim_config=args.optim_config, n_trials=args.n_trials,
                best_result_path=args.best_result_path)

        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_counts)
    else:
        run(
            args.model_config,
            args.data_config,
            args.wandb_project_name,
            optim_config=args.optim_config,
            n_trials=args.n_trials,
            best_result_path=args.best_result_path,
        )


if __name__ == "__main__":
    main()
