"""Tests for scd/config.py: SCDConfig's validation (no scd install needed)."""

import pytest

from adapt_decomp import SCDConfig


def test_scd_kwargs_cannot_set_wrapper_owned_keys():
    """Filters, extension, whitening and device follow the shared config fields."""
    with pytest.raises(ValueError, match=r"\['device', 'low_pass_cutoff'\]"):
        SCDConfig(scd_kwargs={"low_pass_cutoff": 400, "device": "cuda", "max_iterations": 5})


def test_scd_config_rejects_settings_scd_cannot_honour():
    with pytest.raises(ValueError, match="spike_det_exp=2"):
        SCDConfig(spike_det_exp=3.0)
    SCDConfig(n_components=20)  # PCA runs in the shared pipeline, before SCD
