"""Tests for cbss/config.py: CBSSConfig YAML save/load."""

import numpy as np
import pytest
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
        dtype=torch.float32,
        device="cpu",
    )

    path = tmp_path / "calib_config.yml"
    cfg.to_yaml(path)
    loaded = CBSSConfig.from_yaml(path)

    assert loaded.fs == cfg.fs
    assert loaded.ext_fact == cfg.ext_fact
    assert isinstance(loaded.ch_map, np.ndarray)
    np.testing.assert_array_equal(loaded.ch_map, cfg.ch_map)
    assert loaded.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float64, "float64", "torch.float16"])
def test_cbss_config_rejects_dtypes_other_than_float32(dtype):
    with pytest.raises(ValueError, match="float32 only"):
        CBSSConfig(dtype=dtype)


def test_cbss_config_to_dict_leaves_out_ground_truth():
    """selection_kwargs["gt_spikes"] is data, not config: to_dict() (and so to_yaml()
    and the config embedded in a saved CBSSResult) keeps every other kwarg."""
    cfg = CBSSConfig(
        selection="supervised",
        selection_kwargs={"gt_spikes": np.ones((10, 2)), "roa_th": 0.9},
    )
    assert cfg.to_dict()["selection_kwargs"] == {"roa_th": 0.9}
    assert "gt_spikes" in cfg.selection_kwargs  # the config itself is untouched
