"""Tests for cbss/data_structure.py: CBSSResult save/load."""

import pickle

import numpy as np
import pytest

from adapt_decomp.cbss.data_structure import CBSSResult


def test_cbss_result_save_load_roundtrip(tmp_path):
    """CBSSResult.save()/.load() should round-trip fields via pickle."""
    n_mu, T, D = 2, 20, 4
    spikes = np.zeros((T, n_mu), dtype=np.int32)
    spikes[::5] = 1
    result = CBSSResult(
        sources=np.random.randn(T, n_mu).astype(np.float32),
        spikes=spikes,
        spikes_dict={0: np.array([0, 5]), 1: np.array([1, 6])},
        sep_vectors=np.random.randn(D, n_mu).astype(np.float32),
        whitening=np.eye(D, dtype=np.float32),
        extension_mean=np.zeros((1, D), dtype=np.float32),
        spikes_centr=np.ones(n_mu, dtype=np.float32),
        base_centr=np.zeros(n_mu, dtype=np.float32),
        sil=np.full(n_mu, 0.9, dtype=np.float32),
        cov_isi=np.full(n_mu, 0.1, dtype=np.float32),
        ext_fact=2,
    )

    path = tmp_path / "cbss_result.pkl"
    result.save(path)
    loaded = CBSSResult.load(path)

    assert isinstance(loaded, CBSSResult)
    np.testing.assert_array_equal(loaded.sources, result.sources)
    np.testing.assert_array_equal(loaded.spikes, result.spikes)
    assert loaded.ext_fact == result.ext_fact
    assert loaded.spikes_dict.keys() == result.spikes_dict.keys()


def test_cbss_result_load_rejects_wrong_type(tmp_path):
    """CBSSResult.load() must raise ValueError if the pickle holds a different type."""
    path = tmp_path / "not_a_result.pkl"
    with open(path, "wb") as f:
        pickle.dump({"not": "a CBSSResult"}, f)

    with pytest.raises(ValueError):
        CBSSResult.load(path)
