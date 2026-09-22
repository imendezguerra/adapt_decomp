"""Tests for cbss/config.py: CBSSConfig YAML save/load."""

import numpy as np
import torch

from adapt_decomp.cbss.config import CBSSConfig


def test_cbss_config_to_yaml_from_yaml_roundtrip(tmp_path):
    """CBSSConfig.to_yaml()/.from_yaml() should round-trip fields, including the
    numpy-array (ch_map) and torch.dtype coercion handled by adapt_decomp.utils'
    to_yaml_safe/dtype_from_string."""
    cfg = CBSSConfig(
        fs=2000.0,
        ext_fact=8,
        ch_map=np.arange(12).reshape(3, 4),
        dtype=torch.float64,
        device="cpu",
    )

    path = tmp_path / "cbss_config.yml"
    cfg.to_yaml(path)
    loaded = CBSSConfig.from_yaml(path)

    assert loaded.fs == cfg.fs
    assert loaded.ext_fact == cfg.ext_fact
    assert isinstance(loaded.ch_map, np.ndarray)
    np.testing.assert_array_equal(loaded.ch_map, cfg.ch_map)
    assert loaded.dtype == torch.float64
