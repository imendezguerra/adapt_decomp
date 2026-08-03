"""CBSSResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CBSSResult:
    """Output of CBSS.decompose() and CBSS.apply(). All array fields are CPU numpy arrays."""

    sources: np.ndarray                    # [T, n_mu]
    spikes: np.ndarray                     # [T, n_mu]  int32
    spikes_dict: Dict[int, np.ndarray]
    sil: np.ndarray                        # [n_mu]
    cov: np.ndarray                        # [n_mu]
    sep_vectors: np.ndarray                # [dim, n_mu]
    whitening: np.ndarray                  # [dim, dim]
    extension_mean: np.ndarray             # [1, C*ext_fact]
    spikes_centr: np.ndarray               # [n_mu]
    base_centr: np.ndarray                 # [n_mu]
    pca_components: Optional[np.ndarray] = None   # [n_comp, C*ext_fact] or None
    pca_mean: Optional[np.ndarray] = None          # [C*ext_fact] or None
    pnr: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    muaps: Optional[np.ndarray] = None
    emg: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    gt_matched_indices: Optional[np.ndarray] = None  # [n_mu] index into GT units after supervised selection
    roa: Optional[np.ndarray] = None                 # [n_mu] RoA vs gt_matched_indices, set by select_units_supervised

    def to_dict(self) -> Dict:
        return {
            "sources": self.sources,
            "spikes": self.spikes,
            "spikes_dict": self.spikes_dict,
            "sil": self.sil,
            "cov": self.cov,
            "sep_vectors": self.sep_vectors,
            "whitening": self.whitening,
            "extension_mean": self.extension_mean,
            "spikes_centr": self.spikes_centr,
            "base_centr": self.base_centr,
            "pca_components": self.pca_components,
            "pca_mean": self.pca_mean,
            "pnr": self.pnr,
            "dr": self.dr,
            "muaps": self.muaps,
            "emg": self.emg,
            "timestamps": self.timestamps,
            "gt_matched_indices": self.gt_matched_indices,
            "roa": self.roa,
        }
