"""Calibration subpackage: CBSS-based EMG calibration and unit selection."""

from adapt_decomp.calibration.calibrate import calibrate_from_indices
from adapt_decomp.calibration.select import (
    select_units_unsupervised,
    select_units_supervised,
)

__all__ = [
    "calibrate_from_indices",
    "select_units_unsupervised",
    "select_units_supervised",
]
