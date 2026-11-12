"""Tests for decomposition/config.py: the method registry behind DecompositionConfig.from_dict()
and Decomposition.from_config()."""

import pytest

from adapt_decomp import CBSS, SCD, CBSSConfig, Decomposition, DecompositionConfig, SCDConfig


@pytest.mark.parametrize("config_cls", [CBSSConfig, SCDConfig])
def test_from_dict_rebuilds_the_method_named_in_to_dict(config_cls, tmp_path):
    """to_dict() records the method, so the base class rebuilds the right subclass."""
    config = config_cls(ext_fact=7)
    assert config.to_dict()["method"] == config_cls.method

    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = DecompositionConfig.from_yaml(path)

    assert type(loaded) is config_cls
    assert loaded.to_dict() == config.to_dict()


def test_dict_without_method_loads_as_cbss():
    """Configs saved before the method key existed are CBSS configs."""
    data = CBSSConfig(sil_th=0.8).to_dict()
    del data["method"]

    loaded = DecompositionConfig.from_dict(data)

    assert type(loaded) is CBSSConfig
    assert loaded.sil_th == 0.8


def test_from_dict_rejects_an_unknown_or_mismatched_method():
    with pytest.raises(ValueError, match="Unknown decomposition method"):
        DecompositionConfig.from_dict({"method": "nope"})
    with pytest.raises(ValueError, match="cannot load a config of method 'scd'"):
        CBSSConfig.from_dict(SCDConfig().to_dict())


@pytest.mark.parametrize(("config", "method_cls"), [(CBSSConfig(), CBSS), (SCDConfig(), SCD)])
def test_from_config_builds_the_registered_method(config, method_cls):
    assert type(Decomposition.from_config(config)) is method_cls
