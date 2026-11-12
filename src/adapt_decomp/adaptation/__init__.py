"""Online adaptive decomposition subpackage."""

from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.config import PRESETS, AdaptConfig
import warnings as _warnings

from adapt_decomp.adaptation.data_structures import AdaptationResult, AdaptState, Data
from adapt_decomp.adaptation.optimize import (
    OptimisationResult,
    optimize_adapt_decomp,
    optimize_adapt_decomp_pooled_memory,
    optimize_adapt_decomp_pooled_disk,
)

__all__ = [
    "AdaptDecomp",
    "AdaptationResult",
    "AdaptConfig",
    "AdaptState",
    "Data",
    "OptimisationResult",
    "PRESETS",
    "optimize_adapt_decomp",
    "optimize_adapt_decomp_pooled_memory",
    "optimize_adapt_decomp_pooled_disk",
]


def __getattr__(name: str):
    """Resolve the deprecated name Decomposition (now AdaptState), with a FutureWarning."""
    if name == "Decomposition":
        _warnings.warn(
            "adapt_decomp.adaptation.Decomposition was renamed AdaptState; Decomposition "
            "now names the calibration base class (adapt_decomp.Decomposition).",
            FutureWarning,
            stacklevel=2,
        )
        return AdaptState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
