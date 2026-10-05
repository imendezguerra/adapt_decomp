"""How-to examples on one FDSI recording: calibrate, adapt, evaluate, plot and record.

Run from the repository root with the FDSI data downloaded
(python scripts/download_data.py get fdsi_benchmark-data); the how-to pages include its
sections, and tests/docs/test_snippets.py runs it.
"""

# --8<-- [start:load]
from pathlib import Path

import numpy as np

from adapt_decomp.utils import load_emg, load_gt

DATA = Path("data/fdsi_benchmark/data/sub-01")
OUT = Path("data/fdsi_benchmark/outputs/docs-example")
FS = 2048
CAL_END = 5 * FS  # calibrate on the first 5 s

emg = load_emg(DATA / "noisy" / "sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz")
gt_spikes = load_gt(  # binary spikes of every simulated motor unit
    DATA / "clean" / "sub-01_FDSI_triangular-ramp40s_spikes.npz", n_samples=emg.shape[0]
)
print(emg.shape, gt_spikes.shape)  # (samples, channels), (samples, simulated units)
# --8<-- [end:load]

# --8<-- [start:calibrate]
from adapt_decomp import CBSS, CBSSConfig

cbss_config = CBSSConfig(fs=FS, ext_fact=10, sil_th=0.9, random_seed=42, device="cpu")
calibration = CBSS(cbss_config).decompose(emg[:CAL_END], np.arange(CAL_END) / FS)
print(f"{calibration.spikes.shape[1]} units found")
# --8<-- [end:calibrate]

# --8<-- [start:select-unsupervised]
# Without ground truth: keep the units passing quality thresholds
reliable = calibration.select_unsupervised(sil_th=0.9, cov_th=0.3)
print(f"{reliable.spikes.shape[1]} units with SIL >= 0.9 and CoV-ISI <= 0.3")
# --8<-- [end:select-unsupervised]

# --8<-- [start:select-supervised]
# With ground truth (simulations): keep the units matching a simulated motor unit
calibration = calibration.select_supervised(gt_spikes[:CAL_END], roa_th=0.9, tol_spike_ms=2, fs=FS)
print(calibration.gt_matched_indices)  # the simulated motor unit each unit tracks
print(calibration.roa)  # and their rate of agreement over the calibration window
# --8<-- [end:select-supervised]

# --8<-- [start:save]
OUT.mkdir(parents=True, exist_ok=True)
calibration.save(OUT / "calibration.pkl")
cbss_config.to_yaml(OUT / "calibration_config.yaml")  # needed to rebuild the model
# --8<-- [end:save]

# --8<-- [start:adapt]
from adapt_decomp import AdaptDecomp, CBSSResult
from adapt_decomp.adaptation import AdaptConfig

calibration = CBSSResult.load(OUT / "calibration.pkl")
cbss_config = CBSSConfig.from_yaml(OUT / "calibration_config.yaml")
adapt_config = AdaptConfig.from_yaml("configs/adapt_configs/default_muniverse.yaml")
adapt_config.device = "cpu"

adapter = AdaptDecomp.from_calibration(
    calibration=calibration, cbss_config=cbss_config, adapt_config=adapt_config
)
# Keep CBSS's own output over the calibration window, adapt forwards from its end
adapted = adapter.process_from_calib_end(emg, slice(0, CAL_END))
print(adapted.spikes.shape, f"{adapted.total_time_ms.float().mean():.1f} ms per 100 ms batch")
# --8<-- [end:adapt]

# --8<-- [start:baseline]
# The same calibration without adaptation: every adapt_* flag off
fixed_config = AdaptConfig.from_yaml("configs/adapt_configs/default_fixed.yaml")
fixed_config.device = "cpu"
fixed = AdaptDecomp.from_calibration(
    calibration=calibration, cbss_config=cbss_config, adapt_config=fixed_config
).process_from_calib_end(emg, slice(0, CAL_END))
# --8<-- [end:baseline]

# --8<-- [start:evaluate]
from adapt_decomp.spikes import get_sil, rate_of_agreement_paired

gt_paired = gt_spikes[:, calibration.gt_matched_indices]  # (samples, units), in unit order
after_cal = slice(CAL_END, None)
for name, result in (("no adaptation", fixed), ("adapted", adapted)):
    roa, _, _ = rate_of_agreement_paired(
        gt_paired[after_cal], result.spikes.numpy()[after_cal], fs=FS, tol_spike_ms=2
    )
    print(f"{name}: mean RoA after calibration {100 * roa.mean():.1f} %")
# --8<-- [end:evaluate]

# --8<-- [start:evaluate-window]
# Any window, e.g. the ramp of this triangular contraction (10 s to 80 s)
ramp = slice(10 * FS, 80 * FS)
roa_ramp, _, _ = rate_of_agreement_paired(
    gt_paired[ramp], adapted.spikes.numpy()[ramp], fs=FS, tol_spike_ms=2
)
# --8<-- [end:evaluate-window]

# --8<-- [start:sil]
sil = get_sil(
    adapted.sources,
    adapted.spikes,
    adapt_config.spike_min_dist,
    peak_power=adapt_config.spike_det_exp,
).numpy()
print(f"{(sil >= 0.9).sum()} of {sil.size} units with SIL >= 0.9")
# --8<-- [end:sil]

# --8<-- [start:plot-sources]
import matplotlib.pyplot as plt

from adapt_decomp.utils.plots import plot_sources

axs = plot_sources(
    sources={"no adaptation": fixed.sources.numpy(), "adapted": adapted.sources.numpy()},
    timestamps=np.arange(emg.shape[0]) / FS,
    spikes={"adapted": adapted.spikes.numpy()},
    time_range=(40, 50),  # seconds
)
plt.savefig(OUT / "sources.png", dpi=100)
# --8<-- [end:plot-sources]

# --8<-- [start:online]
# The same model, preprocessing each raw batch itself as it would online
adapter = AdaptDecomp.from_calibration(
    calibration=calibration, cbss_config=cbss_config, adapt_config=adapt_config
)
streamed = adapter.process_data(emg[: 15 * FS], processing_mode="online")
# --8<-- [end:online]

# --8<-- [start:provenance]
import sys
from datetime import datetime

from adapt_decomp.utils import build_metadata, write_metadata

started = datetime.now().astimezone()  # before the work, in practice
metadata = build_metadata(
    command=["python", *sys.argv],
    started=started,
    finished=datetime.now().astimezone(),
    run_name="docs-example",
    reproduce=["python docs/snippets/workflow.py"],  # after the checkout lines it adds
    patch_dir=OUT / "patches",  # where an uncommitted diff is saved, if any
    extra={"results": {"roa_after_cal_mean": float(roa.mean()), "n_units": int(sil.size)}},
)
write_metadata(OUT / "results.meta.yaml", metadata)
# --8<-- [end:provenance]
