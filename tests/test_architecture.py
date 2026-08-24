"""Cross-module consistency guards.

Unlike the rest of the suite (unit tests scoped to one module), these check
an architectural invariant CLAUDE.md documents explicitly: certain functions
must have exactly one implementation, imported everywhere they're used,
rather than independent duplicates that can silently drift apart.
"""


def test_cbss_and_data_structures_share_extend_data():
    """cbss/core.py (calibration) and adaptation/data_structures.py (online
    adaptation) must both call the exact same preprocessing.extend_data
    function, not independent duplicates, so an AdaptConfig.ext_mode that matches
    CBSSConfig.ext_mode really does keep them in sync."""
    from adapt_decomp import cbss as cbss_module
    from adapt_decomp.adaptation import data_structures as data_structures_module
    from adapt_decomp.preprocessing import extend_data

    assert cbss_module.core.extend_data is extend_data
    assert data_structures_module._extend_data is extend_data


def test_find_peaks_multisource_has_one_canonical_implementation():
    """spikes/detection.py must be the single canonical find_peaks_multisource --
    adaptation/ops.py and spikes/metrics.py both import it from there rather than
    holding independent copies (ops.py used to carry its own byte-identical copy)."""
    from adapt_decomp.spikes.detection import find_peaks_multisource
    from adapt_decomp.adaptation import ops as adaptation_ops
    from adapt_decomp.spikes import metrics as spikes_metrics

    assert adaptation_ops.find_peaks_multisource is find_peaks_multisource
    assert spikes_metrics.find_peaks_multisource is find_peaks_multisource
