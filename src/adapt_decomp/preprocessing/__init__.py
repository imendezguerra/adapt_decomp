"""Preprocessing functions shared by cbss and adaptation, including:
    1. Channel selection
    2. Filtering
    3. Extension

PCA reduction and whitening (pca_reduction, whiten) live in
adapt_decomp.cbss instead -- they are only ever called from
cbss/core.py at calibration time; online adaptation only consumes their
fitted outputs (pca_components/pca_mean, the whitening matrix), never the
functions themselves.
"""

from adapt_decomp.preprocessing.preprocessing import (
    filter_kwargs,
    preprocess_emg,
    replace_bad_channels,
)
from adapt_decomp.preprocessing.extension import extend_data

__all__ = [
    "filter_kwargs",
    "preprocess_emg",
    "replace_bad_channels",
    "extend_data",
]
