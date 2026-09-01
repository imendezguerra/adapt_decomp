"""Tests for AdaptDecomp.from_calibration()'s required cbss_config and the
reconciliation it performs against adapt_config.

cbss_config is treated as ground truth for every field in
core._SHARED_CBSS_ADAPT_FIELDS (ext_fact, ext_mode, spike_det_exp, the
preprocessing/filter fields, and ch_mask/ch_map/replace_bad_channels): a
disagreeing adapt_config is silently corrected, with one UserWarning naming
what changed -- see reconcile_with_calib_config()'s own array-safe
comparison tests below for the ch_mask/ch_map-specific cases (plain !=
raises on ndarrays); a disagreeing (cbss_config, calibration) pairing
(ext_fact) is a caller bug and raises immediately instead.
"""

import warnings

import numpy as np
import pytest

from adapt_decomp.adaptation import AdaptConfig, AdaptDecomp
from adapt_decomp.adaptation.core import SharedCalibFields, reconcile_with_calib_config
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
            calibration=calibration,
            cbss_config=cbss_config,
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
            calibration=calibration,
            cbss_config=cbss_config,
            adapt_config=adapt_config,
        )

    assert adapter.config.spike_det_exp == 1.5   # cbss_config won
    assert adapt_config.spike_det_exp == 9.0      # caller's instance untouched


def _shared(**overrides) -> SharedCalibFields:
    """SharedCalibFields with sane defaults for the 12 pre-existing fields,
    plus overridable ch_mask/ch_map/replace_bad_channels."""
    base = dict(
        ext_fact=2, ext_mode="block", spike_det_exp=2.0, fs=2048,
        lowcut=20.0, highcut=500.0, filter_order=4, powerline=True,
        powerline_freq=50.0, notch_width_hz=1.0, notch_n_harmonics=3,
        notch_order=2, ch_mask=None, ch_map=None, replace_bad_channels=False,
    )
    base.update(overrides)
    return SharedCalibFields(**base)


def test_reconcile_ch_mask_array_disagreement_warns_and_overwrites():
    """A ch_mask disagreement (both non-None, different content) must be
    detected via np.array_equal (not bare !=, which raises on ndarrays) and
    overwritten from shared, like any other shared field."""
    shared = _shared(ch_mask=np.array([True, False, True]))
    adapt_config = AdaptConfig(
        ext_fact=2, ch_mask=np.array([True, True, True]), device="cpu",
    )
    with pytest.warns(UserWarning, match="ch_mask"):
        reconciled = reconcile_with_calib_config(adapt_config, shared)
    np.testing.assert_array_equal(reconciled.ch_mask, shared.ch_mask)
    # Caller's own instance is untouched.
    np.testing.assert_array_equal(adapt_config.ch_mask, np.array([True, True, True]))


def test_reconcile_ch_mask_none_on_both_sides_does_not_warn():
    """ch_mask=None on both sides must not raise the ndarray-truthiness
    error, and must not spuriously warn (nothing disagreed)."""
    shared = _shared()   # ch_mask=None, ch_map=None
    adapt_config = AdaptConfig(ext_fact=2, device="cpu")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reconciled = reconcile_with_calib_config(adapt_config, shared)

    assert not any("disagreed" in str(w.message) for w in caught)
    assert reconciled.ch_mask is None
    assert reconciled.ch_map is None


def test_reconcile_ch_mask_one_sided_none_warns():
    """One side None, the other an array, must be detected as a disagreement
    (not crash trying np.array_equal(None, array))."""
    shared = _shared(ch_mask=np.array([True, False]))
    adapt_config = AdaptConfig(ext_fact=2, ch_mask=None, device="cpu")
    with pytest.warns(UserWarning, match="ch_mask"):
        reconciled = reconcile_with_calib_config(adapt_config, shared)
    np.testing.assert_array_equal(reconciled.ch_mask, shared.ch_mask)


def test_reconcile_ch_mask_equal_arrays_do_not_warn():
    """Equal (but distinct-object) ch_mask arrays on both sides must compare
    equal via np.array_equal and not warn."""
    shared = _shared(ch_mask=np.array([True, False, True]))
    adapt_config = AdaptConfig(
        ext_fact=2, ch_mask=np.array([True, False, True]), device="cpu",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reconcile_with_calib_config(adapt_config, shared)
    assert not any("disagreed" in str(w.message) for w in caught)


def test_cbss_config_calibration_ext_fact_mismatch_raises():
    """A mismatched (cbss_config, calibration) pairing is a caller bug -- wrong
    config handed in alongside the wrong result -- not a reconcilable drift,
    so it raises immediately rather than going through the warn-and-overwrite path."""
    calibration = _make_cbss_result(ext_fact=2)
    cbss_config = CBSSConfig(ext_fact=5)

    with pytest.raises(ValueError, match="ext_fact"):
        AdaptDecomp.from_calibration(
            calibration=calibration,
            cbss_config=cbss_config,
        )
