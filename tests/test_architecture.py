"""Cross-module consistency guards.

Unlike the rest of the suite (unit tests scoped to one module), these check
an architectural invariant: certain functions
must have exactly one implementation, imported everywhere they're used,
rather than independent duplicates that can silently drift apart.
"""


def test_decomposition_and_data_structures_share_extend_data():
    """decomposition/base.py (calibration, every method) and adaptation/data_structures.py
    (online adaptation) must both call the exact same preprocessing.extend_data
    function, not independent duplicates, so an AdaptConfig.ext_mode that matches
    the calibration's ext_mode really does keep them in sync."""
    from adapt_decomp.adaptation import data_structures as data_structures_module
    from adapt_decomp.decomposition import base as decomposition_base
    from adapt_decomp.preprocessing import extend_data

    assert decomposition_base.extend_data is extend_data
    assert data_structures_module._extend_data is extend_data


def test_whitening_and_pca_have_one_canonical_implementation():
    """Every calibration method whitens and reduces through the base class, which uses
    preprocessing.whiten/pca_reduction; the old cbss paths only re-export them."""
    from adapt_decomp.cbss import pca as cbss_pca
    from adapt_decomp.cbss import whitening as cbss_whitening
    from adapt_decomp.decomposition import base as decomposition_base
    from adapt_decomp.preprocessing import pca_reduction, whiten

    assert decomposition_base.whiten is whiten
    assert decomposition_base.pca_reduction is pca_reduction
    assert cbss_whitening.whiten is whiten
    assert cbss_pca.pca_reduction is pca_reduction


def test_find_peaks_multisource_has_one_canonical_implementation():
    """spikes/detection.py must be the single canonical find_peaks_multisource --
    adaptation/ops.py and spikes/metrics.py both import it from there rather than
    holding independent copies (ops.py used to carry its own byte-identical copy)."""
    from adapt_decomp.adaptation import ops as adaptation_ops
    from adapt_decomp.spikes import metrics as spikes_metrics
    from adapt_decomp.spikes.detection import find_peaks_multisource

    assert adaptation_ops.find_peaks_multisource is find_peaks_multisource
    assert spikes_metrics.find_peaks_multisource is find_peaks_multisource
