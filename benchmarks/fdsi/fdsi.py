"""The FDSI dataset: raw-data paths and loaders, triangular phases and per-unit metrics, for the
benchmark stages and the dataset tour."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from adapt_decomp import CBSSResult
from adapt_decomp.spikes import rate_of_agreement_paired
from adapt_decomp.utils import load_emg, load_gt

PHASES = ("first_iso", "ramp", "last_iso")


@dataclass(frozen=True)
class Recording:
    """One noisy FDSI recording.

    Attributes:
        sub (str): Subject id, e.g. "sub-01".
        cond (str): Condition name, e.g. "triangular-ramp40s".
        snr (int): SNR level in dB.
    """

    sub: str
    cond: str
    snr: int

    @property
    def stub(self) -> str:
        """Canonical "<sub>_FDSI_<cond>_snr<N>dB" stub shared by data, calibration and results."""
        return f"{self.sub}_FDSI_{self.cond}_snr{self.snr}dB"


# Paths


def emg_path(data_root: Path, rec: Recording) -> Path:
    """Path to one recording's noisy EMG .npz (adapt_decomp.utils.load_emg's format)."""
    return data_root / rec.sub / "noisy" / f"{rec.stub}_emg.npz"


def gt_spikes_path(data_root: Path, sub: str, cond: str) -> Path:
    """Path to one condition's ground-truth spikes .npz (shared by its SNR levels).

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        Path: The ground-truth .npz path (adapt_decomp.utils.load_gt's format).
    """
    return data_root / sub / "clean" / f"{sub}_FDSI_{cond}_spikes.npz"


# Loaders


def load_raw_emg(data_root: Path, rec: Recording) -> np.ndarray:
    """Load one recording's noisy EMG, with shape (samples, channels)."""
    return load_emg(emg_path(data_root, rec))


def load_clean_emg(data_root: Path, sub: str, cond: str) -> np.ndarray:
    """Load one condition's noiseless simulated EMG (shared by its SNR levels).

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: EMG with shape (samples, channels).
    """
    return load_emg(data_root / sub / "clean" / f"{sub}_FDSI_{cond}_emg.npz")


def load_angle_profile(data_root: Path, sub: str, cond: str) -> np.ndarray:
    """Load one condition's wrist-angle trace, sample-aligned with its EMG.

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: Wrist angle in degrees with shape (samples,).
    """
    path = data_root / sub / "clean" / f"{sub}_FDSI_{cond}_angle.npz"
    return np.asarray(np.load(path)["angle"], dtype=np.float32)


def load_effort_profile(data_root: Path, sub: str, cond: str) -> np.ndarray:
    """Load one condition's target contraction-effort trace, sample-aligned with its EMG.

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: Effort as a fraction of MVC (0-1) with shape (samples,).
    """
    path = data_root / sub / "clean" / f"{sub}_FDSI_{cond}_effort.npz"
    return np.asarray(np.load(path)["effort"], dtype=np.float32)


def load_recording_metadata(data_root: Path, sub: str, cond: str) -> Dict:
    """Load one condition's simulation metadata (muscle model, grid, motor-unit pool).

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        Dict: The parsed metadata.json.
    """
    path = data_root / sub / "clean" / f"{sub}_FDSI_{cond}_metadata.json"
    with path.open() as f:
        return json.load(f)


def load_noise_metadata(data_root: Path, rec: Recording) -> Dict:
    """Load one noisy recording's noise metadata (target and realised SNR, seed)."""
    with (data_root / rec.sub / "noisy" / f"{rec.stub}_noise_metadata.json").open() as f:
        return json.load(f)


def load_gt_matched(
    data_root: Path, rec: Recording, gt_unit: np.ndarray, n_samples: int
) -> np.ndarray:
    """Ground-truth spikes of one recording's calibrated units, in their order.

    Args:
        data_root (Path): Raw data root (<dataset>/data).
        rec (Recording): The recording.
        gt_unit (np.ndarray): The simulated motor unit each calibrated unit matches, (units,).
        n_samples (int): Number of samples to densify the ground truth to.

    Returns:
        np.ndarray: Binary ground truth with shape (n_samples, units).
    """
    gt_full_bin = load_gt(gt_spikes_path(data_root, rec.sub, rec.cond), n_samples=n_samples)
    return gt_full_bin[:, np.asarray(gt_unit, dtype=np.int64)]


# Metrics


def get_triangular_phases(n_full: int, cal_end: int, iso_dur: int) -> Dict[str, slice]:
    """Phase slices of a triangular contraction.

    Args:
        n_full (int): Recording length in samples.
        cal_end (int): Calibration window length in samples (folded into "first_iso").
        iso_dur (int): Isometric bookend length in samples.

    Returns:
        Dict[str, slice]: "first_iso", "ramp" and "last_iso" slices covering
        [0, n_full) without gaps or overlap.
    """
    return {
        "first_iso": slice(0, cal_end + iso_dur),
        "ramp": slice(cal_end + iso_dur, n_full - iso_dur),
        "last_iso": slice(n_full - iso_dur, n_full),
    }


def compute_roa_subset(
    gt_full_bin: np.ndarray, pred_full_spikes: np.ndarray, s: slice, fs: int, tol_spike_ms: float
) -> np.ndarray:
    """Per-unit RoA restricted to a temporal slice.

    Args:
        gt_full_bin (np.ndarray): Binary ground truth with shape (samples, units).
        pred_full_spikes (np.ndarray): Binary predicted spikes with shape (samples, units).
        s (slice): Temporal slice.
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float): Spike-alignment tolerance in ms.

    Returns:
        np.ndarray: Per-unit RoA (0-1) with shape (units,), NaN for an empty slice.
    """
    sub_gt, sub_pred = gt_full_bin[s], pred_full_spikes[s]
    if sub_gt.shape[0] == 0:
        return np.full(gt_full_bin.shape[1], np.nan)
    roa, _, _ = rate_of_agreement_paired(sub_gt, sub_pred, fs=fs, tol_spike_ms=tol_spike_ms)
    return roa


def _labels(rec: Recording, n_units: int) -> Dict[str, list]:
    """The columns that identify a per-unit table's recording."""
    labels = {"recording": rec.stub, "sub": rec.sub, "condition": rec.cond, "snr": rec.snr}
    return {k: [v] * n_units for k, v in labels.items()}


def calibration_unit_metrics(calibration: CBSSResult, rec: Recording) -> pd.DataFrame:
    """Per-unit metrics of one calibration: its ground-truth match, RoA, SIL and CoV-ISI.

    Args:
        calibration (CBSSResult): The calibration, narrowed by select_supervised.
        rec (Recording): Its recording.

    Returns:
        pd.DataFrame: One row per calibrated unit: recording, sub, condition, snr,
        unit, gt_unit (the matched simulation motor unit's index, its column in the
        ground-truth spike trains), roa_calib (0-1, over the calibration window),
        sil_calib and cov_isi_calib.
    """
    n_units = calibration.spikes.shape[1]
    return pd.DataFrame(
        {
            **_labels(rec, n_units),
            "unit": np.arange(n_units),
            "gt_unit": np.asarray(calibration.gt_matched_indices, dtype=np.int64),
            "roa_calib": np.asarray(calibration.roa, dtype=float),
            "sil_calib": np.asarray(calibration.sil, dtype=float),
            "cov_isi_calib": np.asarray(calibration.cov_isi, dtype=float),
        }
    )


def unit_metrics(
    spikes: np.ndarray,
    sil: np.ndarray,
    gt_full_bin: np.ndarray,
    gt_unit: np.ndarray,
    rec: Recording,
    branch: str,
    *,
    cal_end: int,
    iso_dur: int,
    fs: int,
    tol_spike_ms: float,
) -> pd.DataFrame:
    """Per-unit metrics of one applied config: RoA over several windows, SIL and spike counts.

    Args:
        spikes (np.ndarray): Binary spikes over the whole recording, (samples, units).
        sil (np.ndarray): Per-unit SIL over the whole recording, (units,).
        gt_full_bin (np.ndarray): Matched binary ground truth, (samples, units).
        gt_unit (np.ndarray): The simulated motor unit each unit matches, (units,).
        rec (Recording): The recording; phase RoA is computed for triangular
            conditions only.
        branch (str): The applied config.
        cal_end (int): Calibration window length in samples.
        iso_dur (int): Isometric bookend length in samples.
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float): Spike-alignment tolerance in ms.

    Returns:
        pd.DataFrame: One row per unit: branch, recording, sub, condition, snr, unit,
        gt_unit, roa_full, roa_after_cal, roa_<phase> for each of PHASES (NaN for
        non-triangular conditions), sil, n_spikes and n_spikes_gt (RoA as a 0-1 fraction).
    """
    spikes = np.asarray(spikes, dtype=np.float32)
    n_samples, n_units = spikes.shape
    nan = np.full(n_units, np.nan)

    # RoA over the whole recording, after the calibration window and per phase
    windows = {"full": slice(0, n_samples), "after_cal": slice(cal_end, n_samples)}
    if "triangular" in rec.cond:
        windows.update(get_triangular_phases(n_samples, cal_end, iso_dur))
    roa = {
        name: compute_roa_subset(gt_full_bin, spikes, s, fs, tol_spike_ms)
        for name, s in windows.items()
    }

    return pd.DataFrame(
        {
            "branch": [branch] * n_units,
            **_labels(rec, n_units),
            "unit": np.arange(n_units),
            "gt_unit": np.asarray(gt_unit, dtype=np.int64),
            "roa_full": roa["full"],
            "roa_after_cal": roa["after_cal"],
            **{f"roa_{phase}": roa.get(phase, nan) for phase in PHASES},
            "sil": np.asarray(sil, dtype=float),
            "n_spikes": spikes.sum(axis=0).astype(np.int64),
            "n_spikes_gt": gt_full_bin.sum(axis=0).astype(np.int64),
        }
    )
