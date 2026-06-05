"""Functions to save and load adaptive decomposition outputs"""

import h5py
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Literal, Union

class H5ParamsBatchWriter:
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
            'base_centroids': sd_shape,
            'spike_centroids': sd_shape
        }
        self.batches = batches
        self.dtype = dtype
        self.compression = compression
        
        if not os.path.exists(self.path):
            self._init_file()

    def _init_file(self) -> None:
        with h5py.File(self.path, 'w') as f:
            for key in ['whitening', 'sep_vectors', 'base_centroids', 'spike_centroids']: 
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

def save_output(path: str, outputs: Dict) -> None:
    """Save AdaptDecomp.run() outputs to HDF5.

    Args:
        path: Destination file path.
        outputs: Dict returned by AdaptDecomp.run(). Keys and shapes:

            Always present:
              'spikes'        [samples, M]  int32
              'ipts'          [samples, M]  float32
              'wh_time_ms'    [batches]     float32
              'sv_time_ms'    [batches]     float32
              'sd_time_ms'    [batches]     float32
              'total_time_ms' [batches]     float32

            Present when log_loss=True (default):
              'wh_loss'       [batches]     float32
              'sv_loss'       [batches, M]  float32
              'centroid_loss' [batches, M]  float32
              'wh_trace'      [batches]     float32
              'total_loss'    [batches]     float32

            Present when debug=True:
              'diagnostics'   dict (not saved to HDF5 by this function)
    """
    with h5py.File(path, 'w') as f:
        for key, val in outputs.items():
            if key != 'diagnostics':
                f.create_dataset(key, data=val)

def load_output(path: str) -> Dict:
    """Load AdaptDecomp outputs from HDF5.

    Args:
        path: Path to the HDF5 file written by save_output.
    Returns:
        Dict with the same keys and numpy arrays as values.
    """
    outputs = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            outputs[key] = f[key][:]
    return outputs