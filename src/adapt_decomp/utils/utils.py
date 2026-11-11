"""Small, domain-agnostic helpers."""

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, get_args, get_origin

import h5py
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


def resolve_config(embedded: Optional[Any], explicit: Optional[Any]) -> Optional[Any]:
    """Pick the config embedded in a result over an explicitly given one.

    Shared by AdaptDecomp.from_calibration(), load_calib() and load_adapt(), so a
    result's own config always wins and an explicit config (e.g. a YAML for a
    legacy result with none embedded) is only a fallback.

    Args:
        embedded (Optional[Any]): The result's own config (CBSSConfig or
            AdaptConfig), or None if it has none.
        explicit (Optional[Any]): A config of the same type given by the caller,
            or None.

    Returns:
        Optional[Any]: embedded if set, else explicit. None if neither is set.
    """
    if embedded is None:
        return explicit
    if explicit is not None:
        ours, theirs = embedded.to_dict(), explicit.to_dict()
        differ = sorted(k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k))
        if differ:
            warnings.warn(
                f"The given {type(explicit).__name__} differs from the one embedded in the "
                f"result on {differ}; the embedded one is used.",
                UserWarning,
                stacklevel=3,
            )
    return embedded


def write_h5(path: Union[str, Path], data: Dict[str, Any], attrs: Dict[str, Any]) -> None:
    """Write a nested dict of arrays to an HDF5 file, overwriting it.

    Dicts become groups (keys as str), tensors are moved to CPU numpy, and every
    other value becomes a dataset. Shared by CBSSResult.save() and
    AdaptationResult.save().

    Args:
        path (Union[str, Path]): Destination file path.
        data (Dict[str, Any]): Values to write, with no None values.
        attrs (Dict[str, Any]): Root attributes (e.g. format, embedded configs).

    Returns:
        None
    """

    def _write(group: h5py.Group, values: Dict[Any, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                _write(group.create_group(str(key)), value)
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            group.create_dataset(str(key), data=value)

    with h5py.File(path, "w") as f:
        f.attrs.update(attrs)
        _write(f, data)


def read_h5(path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Read an HDF5 file written by write_h5().

    Args:
        path (Union[str, Path]): Path to the HDF5 file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The nested data (groups as dicts
        with str keys, datasets as numpy) and the root attributes.
    """

    def _read(group: h5py.Group) -> Dict[str, Any]:
        return {
            key: _read(item) if isinstance(item, h5py.Group) else item[()]
            for key, item in group.items()
        }

    with h5py.File(path, "r") as f:
        return _read(f), dict(f.attrs)
