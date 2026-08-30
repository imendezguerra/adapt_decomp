"""Tests for cbss/data_structure.py: CBSSResult save/load and to_adapt_tensors()."""

import pickle

import numpy as np
import pytest
import torch

from adapt_decomp.cbss.data_structure import CBSSResult


def _make_result(n_mu: int = 3, T: int = 20, D: int = 6, C: int = 2, ext_fact: int = 2) -> CBSSResult:
    """Small, valid CBSSResult for to_adapt_tensors() tests -- emg set (required),
    timestamps set (enables .fs, unused here but cheap to include)."""
    spikes = np.zeros((T, n_mu), dtype=np.int32)
    spikes[::5] = 1
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
        timestamps=np.arange(T, dtype=np.float64) / 2048.0,
    )


def test_to_adapt_tensors_transposes_sep_vectors_and_matches_shapes():
    """to_adapt_tensors() should reshape sep_vectors from [dim, n_mu] (this
    class's own storage convention) to [n_mu, dim] (AdaptDecomp/optimize.py's),
    and leave every other field's shape unchanged, as float32 tensors."""
    n_mu, T, D, C = 3, 20, 6, 2
    result = _make_result(n_mu=n_mu, T=T, D=D, C=C)

    tensors = result.to_adapt_tensors()

    assert tensors["sep_vectors"].shape == (n_mu, D)
    np.testing.assert_allclose(tensors["sep_vectors"].numpy(), result.sep_vectors.T)
    assert tensors["whitening"].shape == (D, D)
    assert tensors["emg_calib"].shape == (T, C)
    assert tensors["ipts_calib"].shape == (T, n_mu)
    assert tensors["spikes_calib"].shape == (T, n_mu)
    assert tensors["pca_components"] is None
    assert tensors["pca_mean"] is None
    for t in tensors.values():
        if t is not None:
            assert t.dtype == torch.float32


def test_to_adapt_tensors_raises_if_emg_not_set():
    """to_adapt_tensors() needs calibration EMG to build emg_calib -- same
    requirement AdaptDecomp.from_calibration() enforces for the same reason."""
    result = _make_result()
    result.emg = None
    with pytest.raises(ValueError, match="emg"):
        result.to_adapt_tensors()


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
