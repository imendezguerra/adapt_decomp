"""CBSS configuration dataclass."""

from dataclasses import dataclass
from typing import Literal

from adapt_decomp.decomposition.config import DecompositionConfig


@dataclass
class CBSSConfig(DecompositionConfig):
    """Configuration for the CBSS decomposition algorithm.

    Shared preprocessing, whitening, spike detection, duplicate removal and selection
    fields are documented on DecompositionConfig.

    Attributes:
        contrast_fun (Literal["logcosh", "square", "cube", "smooth_abs"]): Contrast
            function of the fixed-point ICA.
        contrast_exp (float): Exponent of the "smooth_abs" contrast. Ignored by the
            other contrasts.
        search_iter (int): Number of ICA initialisations tried, each of which can
            yield one unit.
        ica_iter (int): Maximum fixed-point iterations per initialisation.
        ica_tol (float): Fixed-point convergence tolerance.
        refinement_loop (bool): Whether to refine each converged unit by re-estimating
            its separation vector from its own spikes.
        refinement_mode (Literal["cov_isi", "sil"]): Metric the refinement loop
            improves: the silhouette (higher is better) or the coefficient of
            variation of the inter-spike intervals (lower is better).
        refine_max_iter (int): Maximum refinement iterations per unit.
        sil_th (float): Minimum silhouette for a unit to be kept.
    """

    method = "cbss"

    # ICA
    contrast_fun: Literal["logcosh", "square", "cube", "smooth_abs"] = "square"
    contrast_exp: float = 3.0
    search_iter: int = 100
    ica_iter: int = 100
    ica_tol: float = 1e-4

    # Refinement loop
    refinement_loop: bool = True
    refinement_mode: Literal["cov_isi", "sil"] = "sil"
    refine_max_iter: int = 20

    # Quality control
    sil_th: float = 0.9
