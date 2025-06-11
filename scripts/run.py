import wandb
import argparse
import torch
import numpy as np
from dataclasses import asdict
from adapt_decomp.config import load_yaml
from adapt_decomp.loaders import load_data
from adapt_decomp.config import load_config
from adapt_decomp.adaptation import AdaptDecomp
from adapt_decomp.utils import rate_of_agreement, rate_of_agreement_paired

def run(model_config, data_config, wandb_project_name, wandb_config=None):
    """Run the model with the given configuration."""
    # Importing here to avoid circular imports
  
    # Load data
    data_config = load_yaml(data_config)
    data = load_data(data_config)

    # Load config
    config = load_config(model_config, wandb_config)
    config.ext_fact = data['ext_fact']

    # Run adaptive decomposition
    adapt_decomp = AdaptDecomp(
        emg = data['emg'].clone(),
        whitening = data['whitening'].clone(),
        sep_vectors = data['sep_vectors'].clone(),
        base_centr = data['base_centr'].clone(),
        spikes_centr = data['spikes_centr'].clone(),
        emg_calib = data['emg_calib'].clone(),
        ipts_calib = data['ipts_calib'].clone(),
        spikes_calib = data['spikes_calib'].clone(),
        preprocess = data['preprocess'],
        config = config
        )
    outputs = adapt_decomp.run()

    # Initialise wandb
    name = f'adapt_decomp_cov_alpha_wh_lr{config.wh_learning_rate:.5f}_sv_lr{config.sv_learning_rate:.5f}'
    if wandb.run is None:
        # wandb has not been initialised - single train run
        wandb.init(
            project=wandb_project_name,
            name=name,
            config=asdict(config)
        )
    else:
        # wandb had been initialised - during train sweep
        wandb.run.name = name

    # Compute rate of agreement if possible
    if 'spikes_gt' in data.keys():
        # RoA throughout the recording
        roa_calib, pair_calib, lag_calib = rate_of_agreement(
            data['spikes_gt'].numpy()[0:data['spikes_calib'].shape[0]],
            data['spikes_calib'].numpy(),
            fs=data['fs'],
            tol_spike_ms=2
            )
        spikes_gt = data['spikes_gt'].numpy()[:, np.array(pair_calib)[:, 0]]
        roa_adapt, pair_adapt, lag_adapt = rate_of_agreement_paired(
            spikes_gt,
            outputs['spikes'].numpy(),
            fs=data['fs'],
            tol_spike_ms=2
            )
        for i in range(len(roa_adapt)):
            wandb.log({
                'roa_calib': roa_calib[i],
                'roa_adapt': roa_adapt[i],
            })    

    # Log metrics in wandb
    batches = len(outputs['wh_loss'])
    for batch in range(batches):
        wandb.log({
            'wh_loss': outputs['wh_loss'][batch],
            'sv_loss': outputs['sv_loss'][batch].nansum(),
            'total_loss': outputs['total_loss'][batch],
            'total_time_ms': outputs['total_time_ms'][batch],
        })   
    
    # Change summary to mean
    wandb.summary['roa_calib'] = np.mean(roa_calib)
    wandb.summary['roa_adapt'] = np.mean(roa_adapt)
    wandb.summary['total_time_ms'] = torch.mean(outputs['total_time_ms'])
    wandb.summary['wh_loss'] = torch.median(outputs['wh_loss'])
    wandb.summary['sv_loss'] = torch.median(outputs['sv_loss'].nansum(dim=1))
    wandb.summary['total_loss'] = torch.median(outputs['total_loss'])
    
    # End logging
    wandb.finish()

def main():
    """Main function for running the model."""
    parser = argparse.ArgumentParser(description="Train model with dataset.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to dataset config YAML file")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML file")
    parser.add_argument("--sweep_config", type=str, required=False, help="Path to sweep config YAML file", default=None)
    parser.add_argument("--sweep_counts", type=int, required=False, help="Sweep counts for hyperparameter tuning", default=20)
    parser.add_argument("--wandb_project_name", type=str, required=False, help="WandB entity for the project", default="adaptive_emg_decomp_dyn")
    args = parser.parse_args()

    # Determine if it is a sweep or just a single train run
    if args.sweep_config:
        # Initialise sweep with wandb
        sweep_config = load_yaml(args.sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project_name)
        # Send fixed arguments to train, agent will automatically pass wandb_config
        def sweep_run():
            """Sweep run function that loads config correctly."""
            wandb.init(project=args.wandb_project_name, name="temp_run")
            run(args.model_config, args.data_config, args.wandb_project_name, dict(wandb.config))
        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_counts)
    else:
        # Train model with a single config
        run(args.model_config, args.data_config, args.wandb_project_name)

if __name__ == "__main__":
    main()