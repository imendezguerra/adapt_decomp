"""Optuna-based hyperparameter optimisation for AdaptDecomp.
"""

from adapt_decomp.utils.loaders import PooledDataset
from adapt_decomp.adaptation.optimize.units import DEFAULT_UNIT_SELECTION_KWARGS, UnitSelection
from adapt_decomp.adaptation.optimize.scoring import (
    DEFAULT_OBJECTIVES,
    DEFAULT_PARAM_SPACE,
    ObjectiveName,
)
from adapt_decomp.adaptation.optimize.resources import (
    MODEL_BUILD_OVERHEAD,
    PROCESS_BASELINE_BYTES,
    RUN_OVERHEAD,
    ResourcePlan,
    available_cores,
    available_memory,
    plan_resources,
)
from adapt_decomp.adaptation.optimize.pareto import SELECTION_RULES, FrontSelector, SelectionName
from adapt_decomp.adaptation.optimize.search import (
    DEFAULT_N_STARTUP_TRIALS,
    OptimisationResult,
    optimize_adapt_decomp,
)
from adapt_decomp.adaptation.optimize.deprecated import (
    optimize_adapt_decomp_pooled_disk,
    optimize_adapt_decomp_pooled_disk_pareto,
    optimize_adapt_decomp_pooled_memory,
    optimize_adapt_decomp_pooled_memory_pareto,
)

__all__ = [
    # Search
    "optimize_adapt_decomp",
    "OptimisationResult",
    "DEFAULT_N_STARTUP_TRIALS",
    "PooledDataset",
    # Parameter space and objectives
    "DEFAULT_PARAM_SPACE",
    "DEFAULT_OBJECTIVES",
    "ObjectiveName",
    # Unit selection
    "DEFAULT_UNIT_SELECTION_KWARGS",
    "UnitSelection",
    # Pareto-front selection
    "SELECTION_RULES",
    "FrontSelector",
    "SelectionName",
    # Resources
    "available_cores",
    "available_memory",
    "plan_resources",
    "ResourcePlan",
    "PROCESS_BASELINE_BYTES",
    "MODEL_BUILD_OVERHEAD",
    "RUN_OVERHEAD",
    # Deprecated
    "optimize_adapt_decomp_pooled_memory",
    "optimize_adapt_decomp_pooled_disk",
    "optimize_adapt_decomp_pooled_memory_pareto",
    "optimize_adapt_decomp_pooled_disk_pareto",
]
