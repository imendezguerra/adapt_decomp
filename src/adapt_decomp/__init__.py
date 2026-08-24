__version__ = "2.0"

from adapt_decomp.cbss import CBSS, CBSSConfig, CBSSResult
from adapt_decomp.adaptation import (
    AdaptDecomp,
    AdaptationResult,
    optimize_adapt_decomp,
    optimize_adapt_decomp_pooled,
)
from adapt_decomp.utils.plots import (
    plot_sep_vectors_comp,
    plot_whitening_comp,
    plot_sep_vectors_diff
)
from adapt_decomp.utils.loaders import load_data

__all__ = [
    # CBSS calibration
    "CBSS",
    "CBSSConfig",
    "CBSSResult",
    # Online adaptation
    "AdaptDecomp",
    "AdaptationResult",
    "optimize_adapt_decomp",
    "optimize_adapt_decomp_pooled",
    # Plots 
    plot_sep_vectors_comp,
    plot_whitening_comp,
    plot_sep_vectors_diff,
    # Loaders
    load_data
]
