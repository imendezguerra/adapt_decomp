"""Functions to save and load adaptive decomposition outputs"""

import h5py
from typing import Dict

def save_output(path:str, outputs:Dict) -> None:
    """
    Save the dynamic decomposition outputs to a file.

    Args:
        path (str): The path to save the file to.
        outputs (Dict): A dictionary containing the dynamic decomposition outpus containing:
        - 'ipts' (torch.Tensor): Innervated pulse trains with shape (samples, sources).
        - 'spikes' (torch.Tensor): Spikes with shape (samples, sources).
        - 'wh_loss' (torch.Tensor): Whitening loss with shape (batches)
        - 'sv_loss' (torch.Tensor): Separation loss with shape (batches, sources)
        - 'total_loss' (torch.Tensor): Total loss with shape (batches)
        - 'time_wh_ms' (torch.Tensor): Time (ms) taken for whitening with shape (batches)
        - 'time_sv_ms' (torch.Tensor): Time (ms) taken for source separation with shape (batches)
        - 'time_sd_ms' (torch.Tensor): Time (ms) taken for spike detection with shape (batches)
    Returns:
        None
    """
    with h5py.File(path, 'w') as f:
        for key in outputs:
            f.create_dataset(key, data=outputs[key])

def load_output(path:str) -> Dict:
    """
    Load the dynamic decomposition output from a file.
    
    Args:
        path (str): The path to save the file to.
    Returns:
        outputs (Dict): A dictionary containing the dynamic decomposition outpus containing:
        - 'ipts' (torch.Tensor): Innervated pulse trains with shape (samples, sources).
        - 'spikes' (torch.Tensor): Spikes with shape (samples, sources).
        - 'wh_loss' (torch.Tensor): Whitening loss with shape (batches)
        - 'sv_loss' (torch.Tensor): Separation loss with shape (batches, sources)
        - 'total_loss' (torch.Tensor): Total loss with shape (batches)
        - 'time_wh_ms' (torch.Tensor): Time (ms) taken for whitening with shape (batches)
        - 'time_sv_ms' (torch.Tensor): Time (ms) taken for source separation with shape (batches)
        - 'time_sd_ms' (torch.Tensor): Time (ms) taken for spike detection with shape (batches)
    """
    outputs = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            outputs[key] = f[key][:]
    return outputs