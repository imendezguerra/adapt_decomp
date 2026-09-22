"""Tests for adaptation/config.py: AdaptConfig YAML save/load."""

import pytest

from adapt_decomp.adaptation.config import AdaptConfig


def test_adapt_config_to_yaml_from_yaml_roundtrip(tmp_path):
    """AdaptConfig.to_yaml()/.from_yaml() should round-trip fields, with derived
    fields (spike_min_dist, batch_size) recomputed by __post_init__ on load."""
    cfg = AdaptConfig()
    cfg.fs = 1000
    cfg.ext_fact = 4
    cfg.wh_mode = "kl_to_cal"
    cfg.wh_learning_rate = 7e-3
    cfg.spike_min_dist_ms = 20
    cfg.batch_ms = 50
    cfg.__post_init__()

    path = tmp_path / "adapt_config.yml"
    cfg.to_yaml(path)
    loaded = AdaptConfig.from_yaml(path)

    assert loaded.fs == cfg.fs
    assert loaded.ext_fact == cfg.ext_fact
    assert loaded.wh_mode == cfg.wh_mode
    assert loaded.wh_learning_rate == cfg.wh_learning_rate
    assert loaded.spike_min_dist == int(20 * 1000 / 1000)
    assert loaded.batch_size == int(50 * 1000 / 1000)


def test_adapt_config_from_yaml_rejects_bad_literal(tmp_path):
    """A YAML file with an invalid Literal value must raise ValueError (via
    validate_literals in __post_init__), not silently construct a bad config."""
    import yaml

    path = tmp_path / "bad_config.yml"
    with open(path, "w") as f:
        yaml.safe_dump({"wh_mode": "not_a_real_mode"}, f)

    with pytest.raises(ValueError):
        AdaptConfig.from_yaml(path)
