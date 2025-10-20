"""Data stucture for the algorithm"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Literal

@dataclass
class Config:
    """Configuration parameters"""
    
    # General parameters
    fs: int = 2048
    device: Literal['cpu','cuda','mps', None] = None

    # Preprocessing parameters
    lowcut: float = 20          # Low cutoff frequency for bandpass filter   
    highcut: float = 500        # High cutoff frequency for bandpass filter
    powerline: bool = True      # Flag to remove powerline noise
    powerline_freq: float = 50  # Powerline frequency for removal

    # Decomposition parameters
    ext_fact: int = 10         # Extension factor for the EMG data

    # Decomposition adaptation
    batch_ms: int = 100         # Batch size in ms
    adapt_wh: bool = True       # Flag to adapt the whitening matrix
    adapt_sv: bool = True       # Flag to adapt the separation vectors
    adapt_sd: bool = True       # Flag to adapt the spike detection
    compute_loss: bool = True   # Flag to compute the loss during the decomposition
    save_params: bool = False   # Flag to save the decomposition parameters during adaptation
    
    # Learning rates
    wh_learning_rate: float = 7e-3  # NeuroMotion: 7e−3 | Wrist: 1e−3 | Forearm: 2e−3
    sv_learning_rate: float = 3e-3  # NeuroMotion: 3e-3 | Wrist and Forearm: 5e-4

    # Source separation parameters
    sv_epochs: int = 1            # Number of epochs for the adaptation of the separation vectors 
    sv_tol: float = 1e-4          # Convergence tolerance for the separation vector adaptation
    contrast_fun: Literal['logcosh', 'cube'] = 'logcosh' # Contrast function for the ICA, options: 'cube' or 'logcosh'

    # Whitening parameters
    cov_alpha: float = 0.1        # Regularisation parameter for the covariance matrix

    # Spike detection paramteters
    spike_height_mult: int = 3    # Multiplier for the spike centroid for maximum height allowed
    spike_prev_weight: int = 5    # Weight for the previous spike centroid during adaptation
    spike_dist_ms: int = 10       # Minimum distance between spikes in ms (10 ms -> 100 Hz max firing)
    spike_dist: int = field(init=False)  # Minimum distance between spikes in samples

    def __post_init__(self) -> None:
        self.spike_dist = int(self.spike_dist_ms * self.fs / 1000)
        self.batch_size = int(self.batch_ms * self.fs / 1000)

def load_yaml(file_path: str) -> Dict:
    """Loads a YAML file into a dictionary."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_config(
    defaults_path="configs/model_configs/default_neuromotion.yml", 
    wandb_config=None
    ) -> Config:
    """Loads YAML config and applies overrides from wandb sweeps 
    parameters if provided"""
    # Load default config
    defaults = load_yaml(defaults_path)
    # Overwrite the required parameters
    if wandb_config:
        for key, value in wandb_config.items():
            if key in defaults:
                defaults[key] = value

    return Config(**defaults)
