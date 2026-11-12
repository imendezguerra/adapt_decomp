"""SCD configuration dataclass."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from adapt_decomp.decomposition.config import DecompositionConfig

# scd.Config keys the wrapper sets itself: preprocessing, extension and whitening run in
# Decomposition, and output, device and verbosity follow this config.
SCD_OWNED_KEYS = frozenset(
    {
        "start_time",
        "end_time",
        "sampling_frequency",
        "time_differentiate",
        "notch_params",
        "low_pass_cutoff",
        "high_pass_cutoff",
        "extension_factor",
        "whitening_method",
        "autocorrelation_whiten",
        "bad_channels",
        "square_sources_spike_det",
        "save_data",
        "output_source_plot",
        "output_final_source_plot",
        "verbose_mode",
        "device",
    }
)


@dataclass
class SCDConfig(DecompositionConfig):
    """Configuration for Swarm-Contrastive Decomposition (the scd package) as calibration.

    Preprocessing, extension and whitening are the shared ones documented on
    DecompositionConfig; SCD only searches for units in the whitened EMG. Requires
    the scd extra: pip install "adapt_decomp[scd]".

    Attributes:
        scd_preset (Literal["default", "surface", "intramuscular", None]): scd's
            built-in parameter set (scd.load_config) to start from. None starts from
            scd.Config()'s defaults.
        scd_kwargs (Optional[Dict[str, Any]]): scd.Config search options applied on
            top of scd_preset, e.g. max_iterations, acceptance_silhouette, peel_off,
            swarm_* or adapt_clamp. Keys in SCD_OWNED_KEYS are set by the wrapper and
            rejected here.
    """

    method = "scd"

    scd_preset: Literal["default", "surface", "intramuscular", None] = None
    scd_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        owned = sorted(SCD_OWNED_KEYS & set(self.scd_kwargs or {}))
        if owned:
            raise ValueError(
                f"SCDConfig.scd_kwargs cannot set {owned}: they follow the shared "
                "preprocessing, extension and whitening fields of this config."
            )
        if self.spike_det_exp != 2.0:
            raise ValueError(
                f"SCDConfig needs spike_det_exp=2 to match Adapt_decomp, got {self.spike_det_exp}."
            )
