"""Shared, domain-agnostic helpers with no natural home in one subpackage:
data loaders (loaders.py), comparison plots (plots.py), and small
dataclass/YAML helpers (utils.py) shared by CBSSConfig
(cbss/config.py) and AdaptConfig (adaptation/config.py).
"""

from adapt_decomp.utils.utils import dtype_from_string, to_yaml_safe, validate_literals
from adapt_decomp.utils.loaders import load_data, load_example, load_neuromotion
from adapt_decomp.utils.plots import (
    plot_sep_vectors_comp,
    plot_sep_vectors_diff,
    plot_whitening_comp,
)

__all__ = [
    "dtype_from_string",
    "to_yaml_safe",
    "validate_literals",
    "load_data",
    "load_example",
    "load_neuromotion",
    "plot_sep_vectors_comp",
    "plot_sep_vectors_diff",
    "plot_whitening_comp",
]
