"""How-to examples of hyperparameter searches, on the calibration docs/snippets/workflow.py saved.

Run it after workflow.py, from the same directory; tests/docs/test_snippets.py runs both. The
searches are kept tiny (2 trials) so they run in a few minutes; use 50 or more in practice.
"""

from pathlib import Path

from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import front_mask, optimize_adapt_decomp

OUT = Path("data/fdsi_example/outputs/docs-example")
base_config = AdaptConfig.from_preset("muniverse")
base_config.device = "cpu"

# --8<-- [start:pool]
from adapt_decomp.utils import load_pooled_cbss_memory

DATA = "data/fdsi_example/data/sub-01"
pool = load_pooled_cbss_memory(
    {
        "root": ".",
        "datasets": [
            {
                "name": "triangular-ramp40s",
                "path_emg": f"{DATA}/noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz",
                "path_calib": str(OUT / "calibration.pkl"),
                "path_calib_config": str(OUT / "calibration_config.yaml"),
                "path_gt": f"{DATA}/clean/sub-01_FDSI_triangular-ramp40s_spikes.npz",  # optional
                "start": 5 * 2048,  # optional: adapt and score from the calibration's end
            },
            # ... one entry per recording to pool
        ],
    }
)
base_config.source_fifo_from_calib = True  # each trial's EMG starts where calibration ends
# --8<-- [end:pool]

# --8<-- [start:single]
result = optimize_adapt_decomp(
    pool=pool,
    objectives="sv_loss",
    base_config=base_config,
    n_trials=2,
    random_seed=42,
    n_cores=1,
)
best = result.best_config
print(best.wh_learning_rate, best.sv_learning_rate, best.centroid_momentum)
# --8<-- [end:single]

# --8<-- [start:pareto]
result = optimize_adapt_decomp(
    pool=pool,
    objectives=("wh_loss", "sv_loss"),
    selection="min_sv_loss",  # or "knee", or "max_roa_mean" with compute_roa
    base_config=base_config,
    compute_roa=True,  # also score every trial against the ground truth
    n_trials=2,
    random_seed=42,
    n_cores=1,
    best_result_path=str(OUT / "search"),  # front members and study, saved as they're found
)
trials = result.study.trials_dataframe()
on_front = front_mask(trials[["values_wh_loss", "values_sv_loss"]].to_numpy())
print(trials.loc[on_front, ["number", "values_wh_loss", "values_sv_loss"]])
# --8<-- [end:pareto]

# --8<-- [start:no-ground-truth]
# Without ground truth, adapt and score only regularly firing units during the search
result = optimize_adapt_decomp(
    pool=pool,
    unit_selection="unsupervised",  # CoV-ISI <= 0.3 by default
    base_config=base_config,
    n_trials=2,
    random_seed=42,
    n_cores=1,
)
# --8<-- [end:no-ground-truth]

# --8<-- [start:save]
result.best_config.to_yaml(OUT / "tuned_config.yaml")  # apply it with AdaptConfig.from_yaml
# --8<-- [end:save]
