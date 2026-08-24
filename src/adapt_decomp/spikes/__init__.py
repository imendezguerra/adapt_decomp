"""Functions for spike detection and classification."""

from adapt_decomp.spikes.detection import detect_spikes, find_peaks_multisource
from adapt_decomp.spikes.comparison import (
    rate_of_agreement_paired,
    rate_of_agreement,
    remove_duplicates,
    spikes_dict_to_binary,
)
from adapt_decomp.spikes.metrics import firings_to_spikes

__all__ = [
    "detect_spikes",
    "find_peaks_multisource",
    "rate_of_agreement_paired",
    "rate_of_agreement",
    "remove_duplicates",
    "spikes_dict_to_binary",
    "firings_to_spikes",
]