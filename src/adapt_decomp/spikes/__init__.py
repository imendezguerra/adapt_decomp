"""Functions for spike detection and classification."""

from adapt_decomp.spikes.detection import detect_spikes, find_peaks_multisource
from adapt_decomp.spikes.comparison import (
    SpikeAgreement,
    compare_paired_spikes,
    compare_spikes,
    find_duplicates,
)
from adapt_decomp.spikes.metrics import firings_to_spikes, get_sil, spikes_dict_to_binary
from adapt_decomp.spikes.deprecated import rate_of_agreement, rate_of_agreement_paired

__all__ = [
    "SpikeAgreement",
    "compare_paired_spikes",
    "compare_spikes",
    "detect_spikes",
    "find_duplicates",
    "find_peaks_multisource",
    "rate_of_agreement_paired",
    "rate_of_agreement",
    "spikes_dict_to_binary",
    "firings_to_spikes",
    "get_sil",
]
