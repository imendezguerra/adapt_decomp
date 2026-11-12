"""Preprocessing functions shared by calibration (decomposition) and adaptation."""

from adapt_decomp.preprocessing.preprocessing import (
    filter_kwargs,
    preprocess_emg,
    preprocess_emg_stateful,
    replace_bad_channels,
    select_channels,
    validate_channel_selection,
)
from adapt_decomp.preprocessing.extension import extend_data
from adapt_decomp.preprocessing.pca import pca_reduction
from adapt_decomp.preprocessing.whitening import whiten

__all__ = [
    "filter_kwargs",
    "preprocess_emg",
    "preprocess_emg_stateful",
    "replace_bad_channels",
    "select_channels",
    "validate_channel_selection",
    "extend_data",
    "pca_reduction",
    "whiten",
]
