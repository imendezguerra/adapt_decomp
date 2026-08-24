"""CBSS subpackage — self-contained convolutive BSS for HD-EMG calibration.

Unit selection lives on the result itself — see
CBSSResult.select_unsupervised() / CBSSResult.select_supervised().
Running CBSS on a slice of a recording and building an adapter from it is
AdaptDecomp.calibrate_from_indices() / AdaptDecomp.from_calibration()
in adapt_decomp.adaptation.core -- cbss itself has no dependency on
adaptation.
"""

from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.cbss.core import CBSS

__all__ = ["CBSSConfig", "CBSSResult", "CBSS"]
