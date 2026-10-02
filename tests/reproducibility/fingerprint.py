"""Shared code path for the cross-platform reproducibility test and its reference.

Runs the adaptation from notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb
(section 2.2: NeuroMotion simulation, default_neuromotion.yaml, CPU, no parameter
saving) and reduces the outputs to a small "fingerprint" that is compared against
tests/reproducibility/reference/neuromotion_adapt.npz.
"""

import platform
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
NM_DATA = ROOT / "data" / "neuromotion" / "data"
PATH_EMG = NM_DATA / "data_sim.hdf5"
PATH_DECOMP = NM_DATA / "calibration" / "decomp_sim.mat"
PATH_CONFIG = ROOT / "configs" / "adapt_configs" / "default_neuromotion.yaml"
PATH_REFERENCE = Path(__file__).resolve().parent / "reference" / "neuromotion_adapt.npz"

SEED = 1909
TOL_SPIKE_MS = 2  # same tolerance as the notebook's whole-recording RoA


def data_available() -> bool:
    """Return whether the NeuroMotion tutorial data has been downloaded."""
    return PATH_EMG.exists() and PATH_DECOMP.exists()


def make_deterministic() -> None:
    """Pin every source of run-to-run variation torch/numpy expose on CPU."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    # Reduction order depends on the thread count -- fix it so results don't
    # depend on how many cores the machine has.
    torch.set_num_threads(1)


def run_tutorial_adaptation() -> Dict[str, np.ndarray]:
    """Run the tutorial's adaptation and return its fingerprint.

    Returns:
        Dict[str, np.ndarray]: "spikes" (samples, units) binary spike trains,
        "roa_gt" (units,) rate of agreement with the simulated ground truth,
        "wh_loss" (batches,), "sv_loss" (batches, units), "whitening_norm" and
        "sep_vectors_norm" (scalars, Frobenius norms of the final parameters),
        and "fs".
    """
    from adapt_decomp.adaptation import AdaptConfig, AdaptDecomp
    from adapt_decomp.spikes import rate_of_agreement_paired
    from adapt_decomp.utils import load_example

    make_deterministic()
    data = load_example(PATH_EMG, PATH_DECOMP, False)
    cbss_result = data["cbss_result"]

    config = AdaptConfig.from_yaml(PATH_CONFIG)
    config.device = "cpu"
    config.ext_fact = cbss_result.ext_fact
    config.compute_loss = True
    config.save_params = False

    adapter = AdaptDecomp.from_calibration(
        calibration=cbss_result,
        cbss_config=data["cbss_config"],
        adapt_config=config,
    )
    outputs = adapter.process_data(data["emg"], preprocess=data["preprocess"])

    spikes = outputs["spikes"].numpy()
    roa_gt, _, _ = rate_of_agreement_paired(
        data["gt_full_bin"], spikes, fs=data["fs"], tol_spike_ms=TOL_SPIKE_MS
    )
    return {
        "spikes": spikes,
        "roa_gt": np.asarray(roa_gt, dtype=np.float64),
        "wh_loss": outputs["wh_loss"].numpy(),
        "sv_loss": outputs["sv_loss"].numpy(),
        "whitening_norm": np.float64(torch.linalg.norm(adapter.decomp.whitening.cpu())),
        "sep_vectors_norm": np.float64(torch.linalg.norm(adapter.decomp.sep_vectors.cpu())),
        "fs": np.int64(data["fs"]),
    }


def environment_info() -> str:
    """Describe the platform/library versions a fingerprint was produced with."""
    return (
        f"{platform.platform()} | python {platform.python_version()} | "
        f"torch {torch.__version__} | numpy {np.__version__}"
    )


def save_fingerprint(fingerprint: Dict[str, np.ndarray], path: Path) -> None:
    """Save a fingerprint as a compressed npz (spikes stored as indices per unit)."""
    spikes = fingerprint["spikes"]
    times, units = np.nonzero(spikes)
    order = np.lexsort((times, units))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        spike_times=times[order].astype(np.int32),
        spike_units=units[order].astype(np.int16),
        n_samples=np.int64(spikes.shape[0]),
        n_units=np.int64(spikes.shape[1]),
        environment=np.array(environment_info()),
        **{k: v for k, v in fingerprint.items() if k != "spikes"},
    )


def load_fingerprint(path: Path) -> Dict[str, np.ndarray]:
    """Load a fingerprint saved by save_fingerprint (spikes rebuilt as binary trains)."""
    with np.load(path) as npz:
        fingerprint = {k: npz[k] for k in npz.files}
    spikes = np.zeros(
        (int(fingerprint.pop("n_samples")), int(fingerprint.pop("n_units"))), np.int32
    )
    spikes[fingerprint.pop("spike_times"), fingerprint.pop("spike_units")] = 1
    fingerprint["spikes"] = spikes
    return fingerprint
