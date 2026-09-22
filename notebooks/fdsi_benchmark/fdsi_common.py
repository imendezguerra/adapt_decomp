"""Benchmark-specific glue for the notebooks/fdsi_benchmark/ showcase; not part of the
installed adapt_decomp package. Path builders and RoA/SIL/phase aggregation (reading
already-computed results back off disk into tidy DataFrames for plotting) for the FDSI
benchmark's fixed data/ (raw) and outputs/{calibration,adaptation}/ (generated) layout. The CBSS/AdaptDecomp calls
that produce those results live in the notebooks themselves, not here.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adapt_decomp import CBSSResult, AdaptationResult
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.spikes import get_sil, rate_of_agreement_paired
from adapt_decomp.utils import load_emg, load_gt

LR_MODE_TOKEN = {'fixed': 'lr_fixed', 'rel_error': 'lr_relerror'}
PHASE_ORDER = ['first_iso', 'ramp', 'last_iso']
PHASE_LABELS = {'first_iso': 'First iso', 'ramp': 'Triangular ramp', 'last_iso': 'Last iso'}


# -- Path builders ---------------------------------------------------------------------------

def recording_stub(sub: str, cond: str, snr: int) -> str:
    """Canonical '<sub>_FDSI_<cond>_snr<N>dB' stub shared by data/calibration/adaptation.

    Args:
        sub (str): Subject id, e.g. 'sub-01'.
        cond (str): Condition name, e.g. 'triangular-ramp40s'.
        snr (int): SNR level in dB.

    Returns:
        str: The recording stub.
    """
    return f'{sub}_FDSI_{cond}_snr{snr}dB'


def calibration_paths(cal_dir: Path, sub: str, cond: str, snr: int) -> Tuple[Path, Path]:
    """Cached calibration result/config paths for one recording.

    Args:
        cal_dir (Path): Calibration cache root (<dataset>/outputs/calibration).
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.

    Returns:
        Tuple[Path, Path]: (cbss_result.pkl, cbss_config.yaml).
    """
    d = cal_dir / sub
    stub = recording_stub(sub, cond, snr)
    return d / f'{stub}_cbss.pkl', d / f'{stub}_cbss_config.yaml'


def fixed_paths(adapt_dir: Path, sub: str, cond: str, snr: int) -> Tuple[Path, Path]:
    """Cached fixed-adaptation (no-adaptation baseline) result/config paths.

    Args:
        adapt_dir (Path): Adaptation cache root (<dataset>/outputs/adaptation).
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.

    Returns:
        Tuple[Path, Path]: (adapt_fixed.pkl, adapt_fixed_config.yaml).
    """
    d = adapt_dir / sub / 'fixed'
    stub = recording_stub(sub, cond, snr)
    return d / f'{stub}_adapt_fixed.pkl', d / f'{stub}_adapt_fixed_config.yaml'


def adapted_paths(adapt_dir: Path, sub: str, cond: str, snr: int, lr_mode: str,
                   sampler_name: str) -> Tuple[Path, Path]:
    """Cached applied-adaptation result/config paths for one winning config.

    Args:
        adapt_dir (Path): Adaptation cache root (<dataset>/outputs/adaptation).
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.
        lr_mode (str): 'fixed' or 'rel_error'.
        sampler_name (str): Search identifier, e.g. 'tpe' or 'tpe_pareto'.

    Returns:
        Tuple[Path, Path]: (adapt_<lr_token>_<sampler>.pkl, ..._config.yaml).
    """
    lr_token = LR_MODE_TOKEN[lr_mode]
    d = adapt_dir / sub / lr_token / sampler_name
    stub = recording_stub(sub, cond, snr)
    return (d / f'{stub}_adapt_{lr_token}_{sampler_name}.pkl',
            d / f'{stub}_adapt_{lr_token}_{sampler_name}_config.yaml')


def gt_spikes_path(data_dir: Path, sub: str, cond: str) -> Path:
    """Path to one condition's clean-recording ground-truth spikes .npz.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        Path: The ground-truth .npz path (adapt_decomp.utils.load_gt's own format).
    """
    return data_dir / sub / 'clean' / f'{sub}_FDSI_{cond}_spikes.npz'


# -- Raw data (thin FDSI-path wrapper around adapt_decomp.utils.load_emg) --------------------

def load_raw_emg(data_dir: Path, sub: str, cond: str, snr: int) -> np.ndarray:
    """Load raw noisy EMG for one FDSI recording.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.

    Returns:
        np.ndarray: EMG with shape (samples, channels).
    """
    path = data_dir / sub / 'noisy' / f'{recording_stub(sub, cond, snr)}_emg.npz'
    return load_emg(path)


def load_angle_profile(data_dir: Path, sub: str, cond: str) -> np.ndarray:
    """Load one condition's wrist-angle dynamics trace.

    Clean, condition-level signal (no SNR suffix, no adapt_decomp.utils loader
    counterpart -- read directly off its own npz key) -- sample-aligned to this
    condition's EMG at every SNR level.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: Wrist angle in degrees, shape (samples,).
    """
    path = data_dir / sub / 'clean' / f'{sub}_FDSI_{cond}_angle.npz'
    return np.asarray(np.load(path)['angle'], dtype=np.float32)


def load_effort_profile(data_dir: Path, sub: str, cond: str) -> np.ndarray:
    """Load one condition's target contraction-effort trace.

    Clean, condition-level signal, same file layout as load_angle_profile -- read
    directly off its own npz key.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: Effort as a fraction of MVC (0-1), shape (samples,).
    """
    path = data_dir / sub / 'clean' / f'{sub}_FDSI_{cond}_effort.npz'
    return np.asarray(np.load(path)['effort'], dtype=np.float32)


def load_clean_emg(data_dir: Path, sub: str, cond: str) -> np.ndarray:
    """Load the noiseless simulated EMG for one condition.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        np.ndarray: EMG with shape (samples, channels).
    """
    path = data_dir / sub / 'clean' / f'{sub}_FDSI_{cond}_emg.npz'
    return load_emg(path)


def load_recording_metadata(data_dir: Path, sub: str, cond: str) -> Dict:
    """Load one condition's clean-recording metadata (muscle model, grid, MU pool, ...).

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.

    Returns:
        Dict: Parsed metadata.json contents.
    """
    path = data_dir / sub / 'clean' / f'{sub}_FDSI_{cond}_metadata.json'
    with open(path) as f:
        return json.load(f)


def load_noise_metadata(data_dir: Path, sub: str, cond: str, snr: int) -> Dict:
    """Load one noisy recording's noise-injection metadata (target/realised SNR, seed).

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.

    Returns:
        Dict: Parsed noise_metadata.json contents.
    """
    path = data_dir / sub / 'noisy' / f'{recording_stub(sub, cond, snr)}_noise_metadata.json'
    with open(path) as f:
        return json.load(f)


# -- Phase-window helpers ----------------------------------------------------------------------

def get_triangular_phases(n_full: int, cal_end: int, iso_dur: int) -> Dict[str, slice]:
    """Fixed phase-boundary slices for a triangular contraction.

    Args:
        n_full (int): Full recording length in samples.
        cal_end (int): Calibration-window length in samples (folds into 'first_iso').
        iso_dur (int): Isometric bookend duration in samples.

    Returns:
        Dict[str, slice]: One slice per phase ('first_iso', 'ramp', 'last_iso').
    """
    return {
        'first_iso': slice(0, cal_end + iso_dur),
        'ramp':      slice(cal_end + iso_dur, n_full - iso_dur),
        'last_iso':  slice(n_full - iso_dur, n_full),
    }


def compute_roa_subset(gt_full_bin: np.ndarray, pred_full_spikes: np.ndarray, s: slice,
                        fs: int, tol_spike_ms: float) -> np.ndarray:
    """Per-unit RoA restricted to a temporal slice.

    Args:
        gt_full_bin (np.ndarray): Binary ground-truth matrix, shape (samples, units).
        pred_full_spikes (np.ndarray): Binary predicted spikes, shape (samples, units).
        s (slice): Temporal slice to restrict to.
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float): Spike-alignment tolerance in ms.

    Returns:
        np.ndarray: Per-unit RoA (0-1), NaN-filled if the slice is empty.
    """
    sub_gt, sub_pred = gt_full_bin[s], pred_full_spikes[s]
    if sub_gt.shape[0] == 0:
        return np.full(gt_full_bin.shape[1], np.nan)
    roa, _, _ = rate_of_agreement_paired(sub_gt, sub_pred, fs=fs, tol_spike_ms=tol_spike_ms)
    return roa


# -- Attaching roa/sil to a cached AdaptationResult -------------------------------------------------------

def load_gt_full_bin(data_dir: Path, sub: str, cond: str, cbss_result: CBSSResult,
                      n_samples: int) -> Optional[np.ndarray]:
    """Ground-truth spikes for one recording, matched/sliced to this calibration's units.

    Args:
        data_dir (Path): Data root (<dataset>/data).
        sub (str): Subject id.
        cond (str): Condition name.
        cbss_result (CBSSResult): This recording's calibration.
        n_samples (int): Number of samples to densify the ground truth to.

    Returns:
        Optional[np.ndarray]: Binary ground-truth matrix with shape (n_samples, units),
            or None if cbss_result.gt_matched_indices is None (no supervised match).
    """
    if cbss_result.gt_matched_indices is None:
        return None
    gt_full_bin = load_gt(gt_spikes_path(data_dir, sub, cond), n_samples=n_samples)
    return gt_full_bin[:, cbss_result.gt_matched_indices]


def compute_roa_for_result(outputs: AdaptationResult, gt_full_bin: Optional[np.ndarray],
                            fs: int, tol_spike_ms: float) -> Optional[np.ndarray]:
    """Per-unit RoA for one cached result against its matched ground truth.

    Args:
        outputs (AdaptationResult): The cached adaptation result.
        gt_full_bin (Optional[np.ndarray]): Binary ground-truth matrix with shape
            (samples, units), or None if there's no supervised GT match (see
            load_gt_full_bin).
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float): Spike-alignment tolerance in ms.

    Returns:
        Optional[np.ndarray]: Per-unit RoA with shape (units,), or None if gt_full_bin
            is None.
    """
    if gt_full_bin is None:
        return None
    pred_spikes = outputs.spikes.numpy().astype(np.float32)
    roa, _, _ = rate_of_agreement_paired(gt_full_bin, pred_spikes, fs=fs, tol_spike_ms=tol_spike_ms)
    return roa

# -- Aggregation helpers -----------------------------------------------------------------------

def aggregate_roa_from_disk(adapt_dir: Path, cal_dir: Path, data_dir: Path, subjects: List[str],
                             conditions: List[str], snr_levels: List[int],
                             configs: List[Tuple[str, Optional[str], Optional[str]]],
                             fs: int, tol_spike_ms: float) -> pd.DataFrame:
    """Read per-unit RoA already attached to every cached AdaptationResult found on disk.

    Args:
        adapt_dir (Path): Adaptation cache root.
        cal_dir (Path): Unused; kept for call-site compatibility.
        data_dir (Path): Unused; kept for call-site compatibility.
        subjects (List[str]): Subject ids to scan.
        conditions (List[str]): Condition names to scan.
        snr_levels (List[int]): SNR levels to scan.
        configs (List[Tuple[str, Optional[str], Optional[str]]]): (label, lr_mode_or_None,
            sampler_or_None) tuples, e.g. [('fixed', None, None), ('lr_fixed_tpe', 'fixed', 'tpe')].
        fs (int): Unused; kept for call-site compatibility.
        tol_spike_ms (float): Unused; kept for call-site compatibility.

    Returns:
        pd.DataFrame: Long-format table, one row per (sub, condition, snr, config, unit),
            columns 'sub', 'condition', 'snr', 'config', 'unit', 'roa'.
    """
    rows = []
    for sub in subjects:
        for cond in conditions:
            for snr in snr_levels:
                for label, lr_mode, sampler_name in configs:
                    result_path = (fixed_paths(adapt_dir, sub, cond, snr)[0] if lr_mode is None
                                   else adapted_paths(adapt_dir, sub, cond, snr, lr_mode, sampler_name)[0])
                    if not result_path.exists():
                        continue
                    outputs = AdaptationResult.load(result_path)
                    if outputs.roa is None:
                        continue  # no GT match, or not yet backfilled
                    for u, r in enumerate(outputs.roa):
                        rows.append({'sub': sub, 'condition': cond, 'snr': snr,
                                     'config': label, 'unit': u, 'roa': float(r)})
    return pd.DataFrame(rows)


def aggregate_phase_roa_from_disk(adapt_dir: Path, cal_dir: Path, data_dir: Path,
                                   subjects: List[str], triangular_conditions: List[str],
                                   snr_levels: List[int],
                                   configs: List[Tuple[str, Optional[str], Optional[str]]],
                                   fs: int, tol_spike_ms: float, cal_end: int,
                                   iso_dur: int) -> pd.DataFrame:
    """Recompute per-unit, per-phase RoA fresh from every cached AdaptationResult on disk.

    Args:
        adapt_dir (Path): Adaptation cache root.
        cal_dir (Path): Calibration cache root.
        data_dir (Path): Data root.
        subjects (List[str]): Subject ids to scan.
        triangular_conditions (List[str]): Triangular condition names to scan.
        snr_levels (List[int]): SNR levels to scan.
        configs (List[Tuple[str, Optional[str], Optional[str]]]): (label, lr_mode_or_None,
            sampler_or_None) tuples, see aggregate_roa_from_disk.
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float): Spike-alignment tolerance in ms.
        cal_end (int): Calibration-window length in samples.
        iso_dur (int): Isometric bookend duration in samples.

    Returns:
        pd.DataFrame: Long-format table, one row per (sub, condition, snr, config, phase, unit),
            columns 'sub', 'condition', 'snr', 'config', 'phase', 'unit', 'roa_pct'. 'phase' is
            a categorical ordered by PHASE_ORDER.
    """
    rows = []
    for sub in subjects:
        for cond in triangular_conditions:
            for snr in snr_levels:
                cal_path, _ = calibration_paths(cal_dir, sub, cond, snr)
                if not cal_path.exists():
                    continue
                cbss_result = CBSSResult.load(cal_path)
                if cbss_result.gt_matched_indices is None:
                    continue
                emg_full = load_raw_emg(data_dir, sub, cond, snr)
                n_full = emg_full.shape[0]
                gt_full_bin = load_gt(gt_spikes_path(data_dir, sub, cond), n_samples=n_full)
                gt_full_bin = gt_full_bin[:, cbss_result.gt_matched_indices]
                phases = get_triangular_phases(n_full, cal_end, iso_dur)

                for label, lr_mode, sampler_name in configs:
                    result_path = (fixed_paths(adapt_dir, sub, cond, snr)[0] if lr_mode is None
                                   else adapted_paths(adapt_dir, sub, cond, snr, lr_mode, sampler_name)[0])
                    if not result_path.exists():
                        continue
                    pred_spikes = AdaptationResult.load(result_path).spikes.numpy().astype(np.float32)
                    for phase_name, s in phases.items():
                        roa_subset = compute_roa_subset(gt_full_bin, pred_spikes, s, fs, tol_spike_ms)
                        for u, r in enumerate(roa_subset):
                            rows.append({'sub': sub, 'condition': cond, 'snr': snr, 'config': label,
                                         'phase': phase_name, 'unit': u, 'roa_pct': float(r) * 100})
    df = pd.DataFrame(rows)
    if len(df):
        df['phase'] = pd.Categorical(df['phase'], categories=PHASE_ORDER, ordered=True)
    return df


# -- Full-recording SIL -------------------------------------------------------------------------

def aggregate_sil_from_disk(adapt_dir: Path, cal_dir: Path, subjects: List[str],
                             conditions: List[str], snr_levels: List[int],
                             configs: List[Tuple[str, Optional[str], Optional[str]]]) -> pd.DataFrame:
    """Read per-unit SIL already attached to every cached AdaptationResult found on disk.

    Args:
        adapt_dir (Path): Adaptation cache root.
        cal_dir (Path): Unused; kept for call-site compatibility.
        subjects (List[str]): Subject ids to scan.
        conditions (List[str]): Condition names to scan.
        snr_levels (List[int]): SNR levels to scan.
        configs (List[Tuple[str, Optional[str], Optional[str]]]): (label, lr_mode_or_None,
            sampler_or_None) tuples, see aggregate_roa_from_disk.

    Returns:
        pd.DataFrame: Long-format table, one row per (sub, condition, snr, config, unit),
            columns 'sub', 'condition', 'snr', 'config', 'unit', 'sil'.
    """
    rows = []
    for sub in subjects:
        for cond in conditions:
            for snr in snr_levels:
                for label, lr_mode, sampler_name in configs:
                    result_path = (fixed_paths(adapt_dir, sub, cond, snr)[0] if lr_mode is None
                                   else adapted_paths(adapt_dir, sub, cond, snr, lr_mode, sampler_name)[0])
                    if not result_path.exists():
                        continue
                    outputs = AdaptationResult.load(result_path)
                    if outputs.sil is None:
                        continue  # not yet backfilled
                    for unit, s in enumerate(outputs.sil):
                        rows.append({'sub': sub, 'condition': cond, 'snr': snr,
                                     'config': label, 'unit': unit, 'sil': float(s)})
    return pd.DataFrame(rows)
