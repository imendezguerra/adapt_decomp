"""Tests for AdaptDecomp.from_calibration()'s required cbss_config and the
reconciliation it performs against adapt_config.

cbss_config is treated as ground truth for every field in
core._SHARED_CBSS_ADAPT_FIELDS (ext_fact, ext_mode, spike_det_exp, and the
preprocessing/filter fields): a disagreeing adapt_config is silently
corrected, with one UserWarning naming what changed; a disagreeing
(cbss_config, calibration) pairing (ext_fact) is a caller bug and raises
immediately instead.
"""

import warnings

import numpy as np
import pytest

from adapt_decomp.adaptation import AdaptConfig, AdaptDecomp
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult


def _make_cbss_result(
    ext_fact: int = 2, n_mu: int = 2, C: int = 2, T: int = 300, spike_stride: int = 40,
) -> CBSSResult:
    """Small, valid CBSSResult -- mirrors utils/test_loaders.py's helper, with
    D derived from C*ext_fact so a real Decomposition can be built from it."""
    D = C * ext_fact
    spikes = np.zeros((T, n_mu), dtype=np.int32)
    spikes[::spike_stride] = 1
    return CBSSResult(
        sources=np.random.randn(T, n_mu).astype(np.float32),
        spikes=spikes,
        spikes_dict={i: np.where(spikes[:, i])[0] for i in range(n_mu)},
        sep_vectors=np.random.randn(D, n_mu).astype(np.float32),
        whitening=np.eye(D, dtype=np.float32),
        extension_mean=np.zeros((1, D), dtype=np.float32),
        spikes_centr=np.ones(n_mu, dtype=np.float32),
        base_centr=np.zeros(n_mu, dtype=np.float32),
        sil=np.full(n_mu, 0.9, dtype=np.float32),
        cov_isi=np.full(n_mu, 0.1, dtype=np.float32),
        ext_fact=ext_fact,
        emg=np.random.randn(T, C).astype(np.float32),
    )


def test_missing_adapt_config_is_seeded_entirely_from_cbss_config():
    """adapt_config=None must build one from every shared field's cbss_config
    value, not just ext_fact -- and must not emit the reconciliation warning
    (nothing was disagreed with)."""
    calibration = _make_cbss_result(ext_fact=3, C=2)
    cbss_config = CBSSConfig(ext_fact=3, spike_det_exp=1.5, ext_mode="toeplitz")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = AdaptDecomp.from_calibration(
            emg=np.random.randn(300, 2).astype(np.float32),
            calibration=calibration,
            cbss_config=cbss_config,
            preprocess=False,
        )

    assert not any("adapt_config disagreed" in str(w.message) for w in caught)
    assert adapter.config.ext_fact == 3
    assert adapter.config.spike_det_exp == 1.5
    assert adapter.config.ext_mode == "toeplitz"


def test_disagreeing_adapt_config_is_overwritten_and_warns():
    """cbss_config wins on every shared field it disagrees with adapt_config
    on; the caller's own adapt_config instance is left untouched (a copy is
    reconciled, not mutated in place); exactly one warning names the field."""
    calibration = _make_cbss_result(ext_fact=2, C=2)
    cbss_config = CBSSConfig(ext_fact=2, spike_det_exp=1.5)
    adapt_config = AdaptConfig(ext_fact=2, spike_det_exp=9.0, device="cpu")

    with pytest.warns(UserWarning, match="spike_det_exp"):
        adapter = AdaptDecomp.from_calibration(
            emg=np.random.randn(300, 2).astype(np.float32),
            calibration=calibration,
            cbss_config=cbss_config,
            adapt_config=adapt_config,
            preprocess=False,
        )

    assert adapter.config.spike_det_exp == 1.5   # cbss_config won
    assert adapt_config.spike_det_exp == 9.0      # caller's instance untouched


def test_cbss_config_calibration_ext_fact_mismatch_raises():
    """A mismatched (cbss_config, calibration) pairing is a caller bug -- wrong
    config handed in alongside the wrong result -- not a reconcilable drift,
    so it raises immediately rather than going through the warn-and-overwrite path."""
    calibration = _make_cbss_result(ext_fact=2)
    cbss_config = CBSSConfig(ext_fact=5)

    with pytest.raises(ValueError, match="ext_fact"):
        AdaptDecomp.from_calibration(
            emg=np.random.randn(300, 2).astype(np.float32),
            calibration=calibration,
            cbss_config=cbss_config,
        )
