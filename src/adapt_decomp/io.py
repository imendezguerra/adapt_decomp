"""Functions to save and load adaptive decomposition outputs"""

import h5py
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Literal, Union
from torch.utils.data import Dataset

class H5PraramsBatchWriter:
    """Class to store adaptive deocomposition parameters in HDF5 format per batch."""
    
    def __init__(self, 
        path: Union[str, Path],
        wh_shape: Tuple,
        sv_shape: Tuple,
        sd_shape: Tuple,
        batches: int = None,
        dtype: str = 'float32', 
        compression: Literal['gzip', None] = None
    ) -> None:
        self.path = path
        self.shapes = {
            'whitening': wh_shape,
            'sep_vectors': sv_shape,
            'base_centr': sd_shape,
            'spikes_centr': sd_shape
        }
        self.batches = batches
        self.dtype = dtype
        self.compression = compression
        
        if not os.path.exists(self.path):
            self._init_file()

    def _init_file(self) -> None:
        with h5py.File(self.path, 'w') as f:
            for key in ['whitening', 'sep_vectors', 'base_centr', 'spikes_centr']: 
                f.create_dataset(
                    key,
                    shape=(0,) + self.shapes[key],
                    maxshape=(self.batches,) + self.shapes[key],
                    dtype=self.dtype,
                    chunks=True,
                    compression=self.compression
                )
    
    def _append(self, batch_data:Dict) -> None:
        with h5py.File(self.path, 'a') as f:
            for k, v in batch_data.items(): 
                f[k].resize(f[k].shape[0] + 1, axis=0)
                f[k][-1:] = np.asarray(v, dtype=self.dtype)

    def _append_batch(self, batch_data:Dict) -> None:
        with h5py.File(self.path, 'a') as f:
            for k, v in batch_data.items(): 
                batch_len = v.shape[0]
                fk_len = f[k].shape[0]
                f[k].resize(fk_len + batch_len, axis=0)
                f[k][fk_len:fk_len + batch_len] = np.asarray(v, dtype=self.dtype)

    def _save(self, data:Dict) -> None:
        with h5py.File(self.path, 'a') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)

    def _load(self) -> Dict:
        data = {}
        with h5py.File(self.path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        return data

class H5PraramsDataset(Dataset):
    """Class to load adaptive decomposition parameters from HDF5 format"""
    
    def __init__(self, path:str) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx:int) -> Dict:
        pass

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