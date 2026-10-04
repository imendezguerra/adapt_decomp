"""Unit selection: which calibration units a search adapts and scores."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from loguru import logger

from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.utils.loaders import PooledDataset, PooledDatasetMemory


UnitSelection = Literal["unsupervised", "supervised", None]
_VALID_UNIT_SELECTIONS: Tuple[Optional[str], ...] = ("unsupervised", "supervised", None)
DEFAULT_UNIT_SELECTION_KWARGS: dict = {"cov_th": 0.3}


def validate_unit_selection(unit_selection: UnitSelection) -> None:
    """Check unit_selection is one of the known options.

    Args:
        unit_selection (UnitSelection): "unsupervised", "supervised" or None.

    Raises:
        ValueError: If unit_selection is not one of them.

    Returns:
        None
    """
    if unit_selection not in _VALID_UNIT_SELECTIONS:
        raise ValueError(
            f"Unknown unit_selection: {unit_selection!r}; expected one of {_VALID_UNIT_SELECTIONS}"
        )


def has_gt(dataset: PooledDataset) -> bool:
    """Whether a pooled dataset carries ground truth (gt_paired_bin or path_gt).

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        bool: True if its ground truth is set.
    """
    if isinstance(dataset, PooledDatasetMemory):
        return dataset.gt_paired_bin is not None
    return dataset.path_gt is not None


def select_units(
    calibration: CBSSResult,
    gt_paired_bin: Optional[np.ndarray],
    unit_selection: UnitSelection,
    unit_selection_kwargs: dict,
) -> Optional[Tuple[CBSSResult, Optional[np.ndarray]]]:
    """Drop the calibration units the search should not adapt or score.

    Only "unsupervised" subsets here (CBSSResult.unsupervised_mask, with
    gt_paired_bin's columns subset alike so RoA stays paired); "supervised"
    pools are already narrowed to GT-matched units by their loaders.

    Args:
        calibration (CBSSResult): A dataset's calibration.
        gt_paired_bin (Optional[np.ndarray]): Its paired ground truth, with
            shape (samples, M), or None.
        unit_selection (UnitSelection): "unsupervised", "supervised" or None.
        unit_selection_kwargs (dict): Thresholds for unsupervised_mask.

    Returns:
        Optional[Tuple[CBSSResult, Optional[np.ndarray]]]: The (possibly
        subset) calibration and gt_paired_bin, or None if no unit is kept.
    """
    if unit_selection != "unsupervised":
        return calibration, gt_paired_bin
    mask = calibration.unsupervised_mask(**unit_selection_kwargs)
    if not mask.any():
        return None
    return calibration.subset(mask), gt_paired_bin[:, mask] if gt_paired_bin is not None else None


def select_pool_units(
    pool: Dict[str, PooledDataset], unit_selection: UnitSelection, unit_selection_kwargs: dict
) -> Dict[str, PooledDataset]:
    """Apply unit selection once up front, leaving out datasets with no unit kept.

    In-memory entries are replaced by their selected copy; on-disk entries
    are kept as paths and selected again (identically) after every resolve().

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        unit_selection (UnitSelection): See select_units.
        unit_selection_kwargs (dict): See select_units.

    Raises:
        ValueError: If no dataset keeps any unit.

    Returns:
        Dict[str, PooledDataset]: The pool to search on.
    """
    if unit_selection != "unsupervised":
        return pool
    selected = {}
    for name, dataset in pool.items():
        _, calibration, _, _, gt_paired_bin = dataset.resolve()
        kept = select_units(calibration, gt_paired_bin, unit_selection, unit_selection_kwargs)
        n_units = calibration.sources.shape[1]
        if kept is None:
            logger.warning(
                f"{name}: no unit of {n_units} passes {unit_selection_kwargs}, left out of the pool"
            )
            continue
        logger.info(f"{name}: {kept[0].sources.shape[1]}/{n_units} units adapted and scored")
        if isinstance(dataset, PooledDatasetMemory):
            dataset = replace(dataset, calibration=kept[0], gt_paired_bin=kept[1])
        selected[name] = dataset
    if not selected:
        raise ValueError(f"No dataset in pool keeps any unit under {unit_selection_kwargs}.")
    return selected
