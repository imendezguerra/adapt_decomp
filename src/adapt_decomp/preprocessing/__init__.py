"""Preprocessing functions shared by cbss and adaptation."""

from adapt_decomp.preprocessing.preprocessing import (
    filter_kwargs,
    preprocess_emg,
    preprocess_emg_stateful,
    replace_bad_channels,
    select_channels,
    validate_channel_selection,
)
from adapt_decomp.preprocessing.extension import extend_data

__all__ = [
    "filter_kwargs",
    "preprocess_emg",
    "preprocess_emg_stateful",
    "replace_bad_channels",
    "select_channels",
    "validate_channel_selection",
    "extend_data",
]
