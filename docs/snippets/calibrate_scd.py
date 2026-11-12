"""How-to example: calibrate one FDSI recording with SCD instead of CBSS, then adapt.

Needs the scd extra (pip install "adapt_decomp[scd]"). Run it from the directory holding
data/, with the example recording downloaded (adapt-decomp data get fdsi_example-data); the
calibration guide and how-to pages include its sections.
"""

from pathlib import Path

import numpy as np

from adapt_decomp.utils import load_emg

DATA = Path("data/fdsi_example/data/sub-01")
FS = 2048
CAL_END = 5 * FS  # calibrate on the first 5 s

emg = load_emg(DATA / "noisy" / "sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz")

# --8<-- [start:calibrate]
from adapt_decomp import SCD, SCDConfig

# The same preprocessing, extension and whitening as CBSS; SCD's swarm search finds the units
scd_config = SCDConfig(
    fs=FS,
    ext_fact=10,
    random_seed=42,
    device="cpu",
    scd_preset="surface",  # scd's surface-EMG parameter set
    scd_kwargs={"max_iterations": 50},  # any other scd.Config search option
)
calibration = SCD(scd_config).decompose(emg[:CAL_END], np.arange(CAL_END) / FS)
print(f"{calibration.spikes.shape[1]} units found")
print(calibration.native_roa)  # agreement of SCD's own spikes with the reconstructed ones
# --8<-- [end:calibrate]

# --8<-- [start:adapt]
from adapt_decomp import AdaptDecomp
from adapt_decomp.adaptation import AdaptConfig

calibration = calibration.select_unsupervised(native_roa_th=0.9)  # drop peel-off-dependent units
adapt_config = AdaptConfig.from_preset("muniverse")
adapt_config.device = "cpu"
adapt_config.source_fifo_from_calib = True  # the EMG below starts where calibration ends
adapted = AdaptDecomp.from_calibration(calibration, adapt_config=adapt_config).process_data(
    emg[CAL_END:]
)
print(adapted.spikes.shape)
# --8<-- [end:adapt]
