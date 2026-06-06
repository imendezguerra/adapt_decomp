import wandb
import argparse
import torch
import numpy as np
from dataclasses import asdict
from adapt_decomp.config import load_yaml, load_config
from adapt_decomp.loaders import load_data
from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.optimize import run_with_optimization
from adapt_decomp.utils import rate_of_agreement, rate_of_agreement_paired


def _log_outputs(outputs, data, config):
    """Log decomposition outputs and optional RoA metrics to the active wandb run."""
    if 'spikes_gt' in data:
        roa_calib, pair_calib, _ = rate_of_agreement(
            data['spikes_gt'].numpy()[0:data['spikes_calib'].shape[0]],
            data['spikes_calib'].numpy(),
            fs=data['fs'],
            tol_spike_ms=2,
        )
        spikes_gt = data['spikes_gt'].numpy()[:, np.array(pair_calib)[:, 0]]
        roa_adapt, _, _ = rate_of_agreement_paired(
            spikes_gt,
            outputs['spikes'].numpy(),
            fs=data['fs'],
            tol_spike_ms=2,
        )
        for i in range(len(roa_adapt)):
            wandb.log({'roa_calib': roa_calib[i], 'roa_adapt': roa_adapt[i]})
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


def run(model_config, data_config, wandb_project_name, wandb_config=None,
        optim_config=None, n_trials=100):
    """Run adaptive decomposition, optionally with Optuna hyperparameter search.

    Args:
        model_config: Path to model config YAML.
        data_config: Path to data config YAML.
        wandb_project_name: WandB project name for logging.
        wandb_config: Optional dict of wandb sweep overrides.
        optim_config: Path to Optuna param-space YAML. When provided, runs
            single-objective hyperparameter search before the final run.
        n_trials: Number of Optuna trials (used when optim_config is set).
    """
    data_config = load_yaml(data_config)
    data = load_data(data_config)

    config = load_config(model_config, wandb_config)
    config.ext_fact = data['ext_fact']

    run_name = (
        f'adapt_decomp_dv{config.max_rel_delta_v:.0e}'
        f'_db{config.max_rel_delta_b:.0e}'
    )

    if wandb.run is None:
        wandb.init(project=wandb_project_name, name=run_name, config=asdict(config))
    else:
        wandb.run.name = run_name

    common_kwargs = dict(
        emg=data['emg'].clone(),
        whitening=data['whitening'].clone(),
        sep_vectors=data['sep_vectors'].clone(),
        base_centroids=data['base_centroids'].clone(),
        spike_centroids=data['spike_centroids'].clone(),
        emg_calib=data['emg_calib'].clone(),
        ipts_calib=data['ipts_calib'].clone(),
        spikes_calib=data['spikes_calib'].clone(),
        preprocess=data['preprocess'],
        config=config,
    )

    if optim_config is not None:
        param_space_raw = load_yaml(optim_config)
        param_space = {k: tuple(v) for k, v in param_space_raw.items()}
        config.optim_loss = "single_obj"
        outputs, best_config, _ = run_with_optimization(
            param_space=param_space,
            n_trials=n_trials,
            **common_kwargs,
        )
        wandb.log({'best_config': best_config})
    else:
        adapter = AdaptDecomp(**common_kwargs)
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
                wandb_config=dict(wandb.config))

        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_counts)
    else:
        run(
            args.model_config,
            args.data_config,
            args.wandb_project_name,
            optim_config=args.optim_config,
            n_trials=args.n_trials,
        )


if __name__ == "__main__":
    main()
