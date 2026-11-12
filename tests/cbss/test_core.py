"""Tests for cbss/core.py's result fields."""

import numpy as np

from adapt_decomp import CBSS, CBSSConfig


def test_cbss_stores_its_fixed_contrast_exponent_per_unit(synthetic_calibration):
    """The default "square" contrast (g = u²) is G ∝ u³."""
    n_mu = synthetic_calibration.sources.shape[1]
    np.testing.assert_array_equal(synthetic_calibration.contrast_exp, np.full(n_mu, 3.0))


def test_cbss_logcosh_leaves_contrast_exp_none(synthetic_recording):
    rec = synthetic_recording
    config = CBSSConfig(fs=rec.fs, ext_fact=8, search_iter=3, random_seed=0, contrast_fun="logcosh")
    assert CBSS(config).decompose(rec.emg[: rec.n_cal]).contrast_exp is None
