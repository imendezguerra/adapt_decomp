"""Small, domain-agnostic helpers."""

import dataclasses
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import numpy as np
import torch


def validate_literals(obj) -> None:
    """Raise ValueError if any Literal[...]-typed dataclass field holds a value
    outside its declared choices.

    AdaptConfig/CBSSConfig fields are set via plain setattr() in several places
    (config_overrides, base_config, categorical Optuna search spaces), which
    bypasses dataclass construction entirely -- a typo'd value would otherwise
    silently fall through whatever == checks read that field downstream
    instead of raising. Call this after every such override.

    Args:
        obj: A dataclass instance (e.g. AdaptConfig or CBSSConfig) to validate.

    Returns:
        None
    """
    for f in dataclasses.fields(obj):
        if get_origin(f.type) is Literal:
            value = getattr(obj, f.name)
            choices = get_args(f.type)
            if value not in choices:
                raise ValueError(
                    f"{type(obj).__name__}.{f.name} = {value!r} is not one of {choices!r}"
                )


def to_yaml_safe(value: Any) -> Any:
    """Coerce a single config field value to a plain, YAML-safe type.

    Shared by CBSSConfig.to_dict() and AdaptConfig.to_dict() so both configs'
    to_yaml()/from_yaml() round-trip the same non-YAML-native value types
    (numpy arrays, torch dtypes/devices, Path) the same way.

    Args:
        value: A single dataclass field value.

    Returns:
        Any: value unchanged, or a YAML-safe equivalent (list, scalar, str).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, (torch.device, Path)):
        return str(value)
    return value


def dtype_from_string(value: str) -> torch.dtype:
    """Parse a torch dtype from its short name (e.g. "float32") or "torch.float32".

    Args:
        value (str): Dtype name, with or without the "torch." prefix.

    Returns:
        torch.dtype: The resolved dtype.

    Raises:
        ValueError: If value is not a known torch dtype name.
    """
    name = value.replace("torch.", "")
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {value!r}") from exc
