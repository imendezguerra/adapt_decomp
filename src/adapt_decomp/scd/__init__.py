"""SCD subpackage — Swarm-Contrastive Decomposition (optional scd extra) as calibration."""

from adapt_decomp.scd.config import SCDConfig
from adapt_decomp.scd.core import SCD

__all__ = ["SCD", "SCDConfig"]
