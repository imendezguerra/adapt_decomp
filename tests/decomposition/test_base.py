"""Tests for decomposition/base.py's result helpers."""

import numpy as np
import torch

from adapt_decomp.decomposition import Decomposition


def test_keep_units_slices_the_native_fields_too():
    """Duplicate removal keeps a method's native fields aligned with its units."""
    n, m = 10, 3
    result = {
        "sources": torch.randn(n, m),
        "spikes": torch.zeros(n, m, dtype=torch.int32),
        "sep_vectors": torch.randn(4, m),
        "spikes_dict": {i: np.array([i]) for i in range(m)},
        "sil": torch.rand(m),
        "cov_isi": torch.rand(m),
        "spikes_centr": torch.rand(m),
        "base_centr": torch.rand(m),
        "sep_vectors_native": torch.randn(4, m),
        "sources_native": torch.randn(n, m),
        "spikes_dict_native": {i: np.array([i, i + 1]) for i in range(m)},
        "native_roa": torch.tensor([0.1, 0.2, 0.3]),
        "contrast_exp": torch.tensor([2.0, 3.0, 4.0]),
    }
    native = result["sources_native"].clone()
    sv_native = result["sep_vectors_native"].clone()

    kept = Decomposition._keep_units(result, np.array([0, 2]))

    torch.testing.assert_close(kept["sources_native"], native[:, [0, 2]])
    torch.testing.assert_close(kept["sep_vectors_native"], sv_native[:, [0, 2]])
    torch.testing.assert_close(kept["native_roa"], torch.tensor([0.1, 0.3]))
    torch.testing.assert_close(kept["contrast_exp"], torch.tensor([2.0, 4.0]))
    assert list(kept["spikes_dict_native"]) == [0, 1]
    np.testing.assert_array_equal(kept["spikes_dict_native"][1], [2, 3])
