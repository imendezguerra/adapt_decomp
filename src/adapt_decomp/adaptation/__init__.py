"""Online adaptive decomposition subpackage."""

from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.config import PRESETS, AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult, Data, Decomposition
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
    "Data",
    "Decomposition",
    "OptimisationResult",
    "PRESETS",
    "optimize_adapt_decomp",
    "optimize_adapt_decomp_pooled_memory",
    "optimize_adapt_decomp_pooled_disk",
]
