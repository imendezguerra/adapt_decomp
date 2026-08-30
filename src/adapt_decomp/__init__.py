__version__ = "2.0"

from adapt_decomp.cbss import CBSS, CBSSConfig, CBSSResult
from adapt_decomp.adaptation import (
    AdaptDecomp,
    AdaptationResult,
    optimize_adapt_decomp_pooled_memory,
    optimize_adapt_decomp_pooled_disk,
)
from adapt_decomp.utils.plots import (
    plot_sep_vectors_comp,
    plot_whitening_comp,
    plot_sep_vectors_diff
)
from adapt_decomp.utils.loaders import (
    load_data,
    load_pooled_cbss_memory,
    load_pooled_cbss_disk,
)

__all__ = [
    # CBSS calibration
    "CBSS",
    "CBSSConfig",
    "CBSSResult",
    # Online adaptation
    "AdaptDecomp",
    "AdaptationResult",
    "optimize_adapt_decomp_pooled_memory",
    "optimize_adapt_decomp_pooled_disk",
    # Plots
    plot_sep_vectors_comp,
    plot_whitening_comp,
    plot_sep_vectors_diff,
    # Loaders
    load_data,
    load_pooled_cbss_memory,
    load_pooled_cbss_disk,
]
