"""CBSS subpackage — self-contained convolutive BSS for HD-EMG calibration."""

from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.result import CBSSResult
from adapt_decomp.cbss.core import CBSS

__all__ = ["CBSSConfig", "CBSSResult", "CBSS"]
