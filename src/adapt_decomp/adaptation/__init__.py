"""Online adaptive decomposition subpackage.

Mirrors adapt_decomp.cbss's config/data_structure/core split:
- config.py: AdaptConfig, the online-adaptation configuration dataclass.
- data_structures.py: Data (extended/preprocessed EMG dataset), Decomposition
  (precalibrated model + adaptive online state), and AdaptationResult (the
  typed output of AdaptDecomp.run()).
- ops.py: per-batch tensor update primitives.
- io.py: HDF5 batch-parameter checkpointing (H5ParamsBatchWriter/load_output).
- optimize.py: Optuna-based hyperparameter search.
- core.py: AdaptDecomp itself, with three ways to build one:
  - __init__ directly, for a calibration from anything (CBSS or otherwise).
  - from_calibration(emg, calibration: CBSSResult, ...), the CBSS-specific
    factory -- unpacks an already-built CBSSResult.
  - calibrate_from_indices(emg, timestamps, calib_indices, ...), which runs
    CBSS itself on emg[calib_indices], optionally filters units via
    CBSSResult.select_unsupervised()/select_supervised(), and calls
    from_calibration() -- the one-call path from raw EMG to a ready adapter.

adaptation depends on cbss in two places, both one-directional (cbss has no
dependency on adaptation, so neither is circular): AdaptDecomp.from_calibration()/
calibrate_from_indices() work with CBSSResult/CBSSConfig/CBSS directly (see
core.py) -- a calibration from anything other than CBSS should build an
AdaptDecomp via its __init__ instead. Separately, log_cosh/contrast_fn (pure
ICA-contrast math, no cbss-specific meaning) live in adapt_decomp.cbss.ica and
are imported from there rather than duplicated -- see cbss/ica.py's module
docstring.
"""

from adapt_decomp.adaptation.core import AdaptDecomp
from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult, Data, Decomposition
from adapt_decomp.adaptation.optimize import optimize_adapt_decomp, optimize_adapt_decomp_pooled

__all__ = [
    "AdaptDecomp",
    "AdaptationResult",
    "AdaptConfig",
    "Data",
    "Decomposition",
    "optimize_adapt_decomp",
    "optimize_adapt_decomp_pooled",
]
