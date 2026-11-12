"""Calibration template shared by every decomposition method (CBSS, SCD, ...)."""

from adapt_decomp.decomposition.config import DecompositionConfig
from adapt_decomp.decomposition.result import DecompositionResult
from adapt_decomp.decomposition.base import Decomposition

__all__ = ["Decomposition", "DecompositionConfig", "DecompositionResult"]
