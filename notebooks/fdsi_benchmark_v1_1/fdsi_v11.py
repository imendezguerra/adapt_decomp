"""Glue for notebooks/fdsi_benchmark_v1_1/: the FDSI benchmark re-run with adapt_decomp v1.1.

Deliberately the same scope as ../fdsi_benchmark/fdsi_common.py, which it imports rather
than copies: branch/path/label bookkeeping and disk-reload aggregation into tidy DataFrames.
Every CBSS, AdaptDecomp and optimize_adapt_decomp call lives in the notebooks themselves, so
that reading a notebook top to bottom shows the whole adapt_decomp API it exercises.

The v1.0 and v1.1 runs share one data root and one calibration cache; only the adaptation
results differ, v1.0's under OUTPUTS_ROOT/adaptation/ and v1.1's under
OUTPUTS_ROOT/adaptation_v1_1/. Every aggregator here takes an adapt_dirs mapping of
version -> that root, runs one fdsi_common aggregator per version and stacks the tables with
a 'version' column.

v1.1 here means: optimize_adapt_decomp (multivariate TPE, centroid_momentum searched,
per-unit-mean sv_loss, CoV-ISI unit selection, worker processes), Pareto fronts selected by
min-sv or knee, and adaptation that starts from the end of the calibration window
(AdaptDecomp.process_from_calib_end). In 06_*'s Pareto variants, the "no sel" branches drop
the unit selection, the "cm 0.95" ones also fix centroid_momentum at v1.0's value, and the
"sum" ones sum sv_loss over units instead of averaging it.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fdsi_benchmark"))
import fdsi_common as fc

from adapt_decomp import AdaptationResult, CBSSResult
from adapt_decomp.adaptation import AdaptConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_DIR = REPO_ROOT / "configs" / "adapt_configs"
LR_MODE = "fixed"  # v1.1 searches lr_fixed only -- the branch v1.0 promoted as its headline

VERSIONS = ("v1.0", "v1.1")

# (label, version, lr_mode or None for the fixed baseline, sampler dir or None)
Config = Tuple[str, str, Optional[str], Optional[str]]


# -- Branch registry -------------------------------------------------------------------------


class Branch(NamedTuple):
    """One applied v1.1 config and the v1.0 config it is compared with.

    Attributes:
        label (str): v1.1 config label stem, e.g. "sv_loss".
        sampler (str): v1.1 sampler dir, which is also its result dir.
        promoted (str): v1.1 promoted config file in configs/adapt_configs/.
        v10_label (str): v1.0 counterpart's label stem.
        v10_sampler (str): v1.0 counterpart's sampler dir.
        v10_promoted (str): v1.0 counterpart's promoted config file.
    """

    label: str
    sampler: str
    promoted: str
    v10_label: str
    v10_sampler: str
    v10_promoted: str


# The v1.1 Pareto search is applied twice, once per front selection; v1.0 had no knee
# selection, so both Pareto branches compare with v1.0's (min-sv) Pareto winner.
BRANCHES: Dict[str, Branch] = {
    "sv_loss": Branch(
        "sv_loss", "mtpe_sv_mean", "optim_muniverse_fdsi_v11_sv.yaml",
        "sv_loss", "tpe_sv_median", "optim_muniverse_fdsi_sv.yaml",
    ),
    "pareto_min_sv": Branch(
        "pareto min-sv", "mtpe_pareto_min_sv", "optim_muniverse_fdsi_v11_pareto_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_knee": Branch(
        "pareto knee", "mtpe_pareto_knee", "optim_muniverse_fdsi_v11_pareto_knee.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "roa": Branch(
        "roa", "mtpe_roa", "optim_muniverse_fdsi_v11_roa.yaml",
        "roa", "tpe_roa", "optim_muniverse_fdsi_roa.yaml",
    ),
    "pareto_sum_min_sv": Branch(
        "pareto min-sv, sum", "mtpe_pareto_sum_min_sv",
        "optim_muniverse_fdsi_v11_pareto_sum_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_min_sv": Branch(
        "pareto min-sv, no sel", "mtpe_pareto_nosel_min_sv",
        "optim_muniverse_fdsi_v11_pareto_nosel_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_knee": Branch(
        "pareto knee, no sel", "mtpe_pareto_nosel_knee",
        "optim_muniverse_fdsi_v11_pareto_nosel_knee.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_sum_min_sv": Branch(
        "pareto min-sv, no sel, sum", "mtpe_pareto_nosel_sum_min_sv",
        "optim_muniverse_fdsi_v11_pareto_nosel_sum_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_sum_knee": Branch(
        "pareto knee, no sel, sum", "mtpe_pareto_nosel_sum_knee",
        "optim_muniverse_fdsi_v11_pareto_nosel_sum_knee.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_cm095_min_sv": Branch(
        "pareto min-sv, no sel, cm 0.95", "mtpe_pareto_nosel_cm095_min_sv",
        "optim_muniverse_fdsi_v11_pareto_nosel_cm095_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_cm095_knee": Branch(
        "pareto knee, no sel, cm 0.95", "mtpe_pareto_nosel_cm095_knee",
        "optim_muniverse_fdsi_v11_pareto_nosel_cm095_knee.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_cm095_sum_min_sv": Branch(
        "pareto min-sv, no sel, cm 0.95, sum", "mtpe_pareto_nosel_cm095_sum_min_sv",
        "optim_muniverse_fdsi_v11_pareto_nosel_cm095_sum_min_sv.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_nosel_cm095_sum_knee": Branch(
        "pareto knee, no sel, cm 0.95, sum", "mtpe_pareto_nosel_cm095_sum_knee",
        "optim_muniverse_fdsi_v11_pareto_nosel_cm095_sum_knee.yaml",
        "pareto min-sv", "tpe_pareto", "optim_muniverse_fdsi_pareto.yaml",
    ),
}

# v1.1 search dir per search; the one Pareto study serves both Pareto branches.
SEARCH_DIRS = {"sv_loss": "mtpe_sv_mean", "pareto": "mtpe_pareto", "roa": "mtpe_roa"}

# 06_*'s variants of the Pareto search, each serving a min-sv branch and, for the no-sel ones,
# a knee branch. Kept out of SEARCH_DIRS, which 05 pairs one-to-one with a v1.0 search.
PARETO_VARIANT_SEARCH_DIRS = {
    "pareto_sum": "mtpe_pareto_sum",
    "pareto_nosel": "mtpe_pareto_nosel",
    "pareto_nosel_sum": "mtpe_pareto_nosel_sum",
    "pareto_nosel_cm095": "mtpe_pareto_nosel_cm095",
    "pareto_nosel_cm095_sum": "mtpe_pareto_nosel_cm095_sum",
}


# -- Path builders ---------------------------------------------------------------------------


def result_paths(
    adapt_dir: Path, sub: str, cond: str, snr: int, lr_mode: Optional[str], sampler: Optional[str]
) -> Tuple[Path, Path]:
    """fc.fixed_paths for the fixed baseline (lr_mode None), else fc.adapted_paths.

    Args:
        adapt_dir (Path): Adaptation results root of either version.
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.
        lr_mode (Optional[str]): 'fixed', or None for the fixed baseline.
        sampler (Optional[str]): Search/sampler dir name, None for the fixed baseline.

    Returns:
        Tuple[Path, Path]: (result .pkl, config .yaml).
    """
    if lr_mode is None:
        return fc.fixed_paths(adapt_dir, sub, cond, snr)
    return fc.adapted_paths(adapt_dir, sub, cond, snr, lr_mode, sampler)


def promoted_path(branch: str, version: str = "v1.1") -> Path:
    """Where a branch's winning config is promoted to, for either version.

    Args:
        branch (str): Key of BRANCHES.
        version (str, optional): "v1.0" or "v1.1". Defaults to "v1.1".

    Returns:
        Path: The file in configs/adapt_configs/.
    """
    b = BRANCHES[branch]
    return PROMOTED_DIR / (b.promoted if version == "v1.1" else b.v10_promoted)


# -- Config labels ---------------------------------------------------------------------------


def configs(*branches: str, fixed: bool = True) -> List[Config]:
    """Config tuples (label, version, lr_mode, sampler) for the given branches, both versions.

    Args:
        *branches (str): Keys of BRANCHES.
        fixed (bool, optional): Prepend both fixed baselines. Defaults to True.

    Returns:
        List[Config]: Each branch's v1.0 counterpart, then its v1.1 config; a v1.0 config
        shared by several branches (Pareto) appears once.
    """
    out: List[Config] = []
    if fixed:
        out += [("fixed (v1.0)", "v1.0", None, None), ("fixed (v1.1)", "v1.1", None, None)]
    for key in branches:
        b = BRANCHES[key]
        out.append((f"{b.v10_label} (v1.0)", "v1.0", LR_MODE, b.v10_sampler))
        out.append((f"{b.label} (v1.1)", "v1.1", LR_MODE, b.sampler))
    return list(dict.fromkeys(out))


def pairs(*branches: str, fixed: bool = True) -> List[Tuple[str, str]]:
    """(v1.0 label, v1.1 label) per branch, for paired_summary.

    Args:
        *branches (str): Keys of BRANCHES.
        fixed (bool, optional): Prepend the fixed baselines' pair. Defaults to True.

    Returns:
        List[Tuple[str, str]]: (before, after) config labels.
    """
    out = [("fixed (v1.0)", "fixed (v1.1)")] if fixed else []
    return out + [
        (f"{BRANCHES[k].v10_label} (v1.0)", f"{BRANCHES[k].label} (v1.1)") for k in branches
    ]


# -- Aggregation across versions -------------------------------------------------------------


def _per_version(
    adapt_dirs: Dict[str, Path],
    config_list: List[Config],
    aggregate: Callable[[Path, List[Tuple[str, Optional[str], Optional[str]]]], pd.DataFrame],
) -> pd.DataFrame:
    """Run one fdsi_common-style aggregator per version root and stack the tables.

    Args:
        adapt_dirs (Dict[str, Path]): version -> adaptation results root.
        config_list (List[Config]): From configs().
        aggregate (Callable): (adapt_dir, [(label, lr_mode, sampler), ...]) -> DataFrame.

    Returns:
        pd.DataFrame: The stacked table with a 'version' column, empty if nothing is cached.
    """
    tables = []
    for version in VERSIONS:
        if version not in adapt_dirs:
            continue
        subset = [(label, lr, s) for label, v, lr, s in config_list if v == version]
        if not subset:
            continue
        table = aggregate(adapt_dirs[version], subset)
        if len(table):
            tables.append(table.assign(version=version))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def roa_table(
    adapt_dirs: Dict[str, Path], cal_dir: Path, data_dir: Path, subjects: List[str],
    conditions: List[str], snr_levels: List[int], config_list: List[Config],
    fs: int, tol_spike_ms: float,
) -> pd.DataFrame:
    """Per-unit full-recording RoA (%) for both versions.

    Args:
        adapt_dirs (Dict[str, Path]): version -> adaptation results root.
        cal_dir (Path): Shared calibration cache.
        data_dir (Path): Raw recordings root.
        subjects (List[str]): Subject ids.
        conditions (List[str]): Condition names.
        snr_levels (List[int]): SNR levels in dB.
        config_list (List[Config]): From configs().
        fs (int): Sampling frequency (Hz).
        tol_spike_ms (float): RoA spike-alignment tolerance (ms).

    Returns:
        pd.DataFrame: Columns sub, condition, snr, config, unit, roa, roa_pct, version.
    """
    df = _per_version(
        adapt_dirs, config_list,
        lambda d, c: fc.aggregate_roa_from_disk(
            d, cal_dir, data_dir, subjects, conditions, snr_levels, c, fs, tol_spike_ms),
    )
    if len(df):
        df["roa_pct"] = df["roa"] * 100
    return df


def sil_table(
    adapt_dirs: Dict[str, Path], cal_dir: Path, subjects: List[str], conditions: List[str],
    snr_levels: List[int], config_list: List[Config],
) -> pd.DataFrame:
    """Per-unit full-recording SIL for both versions.

    Args:
        adapt_dirs (Dict[str, Path]): version -> adaptation results root.
        cal_dir (Path): Shared calibration cache.
        subjects (List[str]): Subject ids.
        conditions (List[str]): Condition names.
        snr_levels (List[int]): SNR levels in dB.
        config_list (List[Config]): From configs().

    Returns:
        pd.DataFrame: Columns sub, condition, snr, config, unit, sil, version.
    """
    return _per_version(
        adapt_dirs, config_list,
        lambda d, c: fc.aggregate_sil_from_disk(d, cal_dir, subjects, conditions, snr_levels, c),
    )


def phase_table(
    adapt_dirs: Dict[str, Path], cal_dir: Path, data_dir: Path, subjects: List[str],
    triangular: List[str], snr_levels: List[int], config_list: List[Config],
    fs: int, tol_spike_ms: float, cal_end: int, iso_dur: int,
) -> pd.DataFrame:
    """Per-unit, per-phase RoA (%) on the triangular conditions, for both versions.

    Args:
        adapt_dirs (Dict[str, Path]): version -> adaptation results root.
        cal_dir (Path): Shared calibration cache.
        data_dir (Path): Raw recordings root.
        subjects (List[str]): Subject ids.
        triangular (List[str]): Triangular condition names.
        snr_levels (List[int]): SNR levels in dB.
        config_list (List[Config]): From configs().
        fs (int): Sampling frequency (Hz).
        tol_spike_ms (float): RoA spike-alignment tolerance (ms).
        cal_end (int): Calibration window length in samples.
        iso_dur (int): Isometric bookend length in samples.

    Returns:
        pd.DataFrame: Columns sub, condition, snr, config, phase, unit, roa_pct, version.
    """
    df = _per_version(
        adapt_dirs, config_list,
        lambda d, c: fc.aggregate_phase_roa_from_disk(
            d, cal_dir, data_dir, subjects, triangular, snr_levels, c, fs, tol_spike_ms,
            cal_end, iso_dur),
    )
    if len(df):
        df["phase"] = pd.Categorical(df["phase"], categories=fc.PHASE_ORDER, ordered=True)
    return df


def window_roa_table(
    adapt_dirs: Dict[str, Path], cal_dir: Path, data_dir: Path, subjects: List[str],
    conditions: List[str], snr_levels: List[int], config_list: List[Config],
    window: slice, fs: int, tol_spike_ms: float,
) -> pd.DataFrame:
    """Per-unit RoA (%) restricted to a sample window, every condition, both versions.

    window=slice(cal_end, None) scores only after the calibration window, where both versions
    adapted -- v1.1 outputs CBSS's own spikes inside it, v1.0 re-adapted over it -- which is
    the only window where the two are directly comparable.

    Args:
        adapt_dirs (Dict[str, Path]): version -> adaptation results root.
        cal_dir (Path): Shared calibration cache.
        data_dir (Path): Raw recordings root.
        subjects (List[str]): Subject ids.
        conditions (List[str]): Condition names.
        snr_levels (List[int]): SNR levels in dB.
        config_list (List[Config]): From configs().
        window (slice): Samples to score.
        fs (int): Sampling frequency (Hz).
        tol_spike_ms (float): RoA spike-alignment tolerance (ms).

    Returns:
        pd.DataFrame: Columns sub, condition, snr, config, unit, roa_pct, version.
    """

    def aggregate(adapt_dir: Path, subset) -> pd.DataFrame:
        rows = []
        for sub in subjects:
            for cond in conditions:
                for snr in snr_levels:
                    cal_path, _ = fc.calibration_paths(cal_dir, sub, cond, snr)
                    if not cal_path.exists():
                        continue
                    cbss_result, gt_full_bin = CBSSResult.load(cal_path), None
                    for label, lr_mode, sampler in subset:
                        path, _ = result_paths(adapt_dir, sub, cond, snr, lr_mode, sampler)
                        if not path.exists():
                            continue
                        spikes = AdaptationResult.load(path).spikes.numpy().astype(np.float32)
                        if gt_full_bin is None:
                            gt_full_bin = fc.load_gt_full_bin(
                                data_dir, sub, cond, cbss_result, spikes.shape[0])
                            if gt_full_bin is None:
                                break
                        roa = fc.compute_roa_subset(
                            gt_full_bin, spikes, window, fs, tol_spike_ms)
                        rows += [
                            {"sub": sub, "condition": cond, "snr": snr, "config": label,
                             "unit": u, "roa_pct": float(r) * 100}
                            for u, r in enumerate(roa)
                        ]
        return pd.DataFrame(rows)

    return _per_version(adapt_dirs, config_list, aggregate)


# -- v1.0 vs v1.1 statistics -----------------------------------------------------------------


def paired_delta(
    df: pd.DataFrame, before: str, after: str, value: str = "roa_pct",
    keys=("sub", "condition", "snr", "unit"),
) -> pd.DataFrame:
    """Pair two configs' per-unit values (same calibrations -> same units) and difference them.

    Args:
        df (pd.DataFrame): Long table with 'config' and value columns.
        before (str): Config label of the reference (e.g. a v1.0 branch).
        after (str): Config label compared against it.
        value (str, optional): Column to compare. Defaults to "roa_pct".
        keys (tuple, optional): Columns identifying a unit.

    Returns:
        pd.DataFrame: keys + 'before', 'after', 'delta' (after - before), units in both only.
    """
    keys = list(keys)
    left = df[df["config"] == before][[*keys, value]].rename(columns={value: "before"})
    right = df[df["config"] == after][[*keys, value]].rename(columns={value: "after"})
    paired = left.merge(right, on=keys)
    return paired.assign(delta=paired["after"] - paired["before"])


def paired_summary(
    df: pd.DataFrame, pairs: Sequence[Tuple[str, str]], value: str = "roa_pct", tol: float = 0.5
) -> pd.DataFrame:
    """Paired per-unit comparison of each (before, after) config pair.

    Args:
        df (pd.DataFrame): Long table with 'config' and value columns.
        pairs (Sequence[Tuple[str, str]]): (before, after) config labels.
        value (str, optional): Column to compare. Defaults to "roa_pct".
        tol (float, optional): |delta| at or below this counts as unchanged. Defaults to 0.5
            (percentage points of RoA).

    Returns:
        pd.DataFrame: One row per pair present in df: n_units, mean/median delta, % units
        improved/unchanged/worse, and the two-sided Wilcoxon signed-rank p-value (NaN when
        every delta is zero).
    """
    rows = []
    for before, after in pairs:
        paired = paired_delta(df, before, after, value)
        if not len(paired):
            continue
        delta = paired["delta"].to_numpy()
        p = wilcoxon(delta).pvalue if np.any(delta != 0) else np.nan
        rows.append({
            "before": before, "after": after, "n_units": len(delta),
            "mean_delta": delta.mean(), "median_delta": np.median(delta),
            "pct_improved": (delta > tol).mean() * 100,
            "pct_unchanged": (np.abs(delta) <= tol).mean() * 100,
            "pct_worse": (delta < -tol).mean() * 100,
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows).set_index(["before", "after"]) if rows else pd.DataFrame()


# -- Study aggregation -----------------------------------------------------------------------


def pooled_loss_shares(study, units: Dict[str, int]) -> pd.DataFrame:
    """Each pooled recording's share of the pooled sv_loss, per completed trial.

    Args:
        study (optuna.Study): A pooled search (v1.0 or v1.1), with per-dataset
            "sv_loss_<name>" user_attrs.
        units (Dict[str, int]): Units the search adapted/scored per recording.

    Returns:
        pd.DataFrame: Columns trial, recording, units, sv_loss, share; diverged trials
        (1e10 sentinel) left out.
    """
    rows = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        losses = {name: trial.user_attrs.get(f"sv_loss_{name}") for name in units}
        if any(v is None or v >= 1e10 for v in losses.values()):
            continue
        total = sum(losses.values())
        rows += [
            {"trial": trial.number, "recording": name, "units": units[name],
             "sv_loss": loss, "share": loss / total if total > 0 else np.nan}
            for name, loss in losses.items()
        ]
    return pd.DataFrame(rows)


def trial_durations(study) -> pd.Series:
    """Wall time of each completed trial, in seconds.

    Args:
        study (optuna.Study): Any study.

    Returns:
        pd.Series: Seconds, indexed by trial number.
    """
    return pd.Series(
        {
            t.number: (t.datetime_complete - t.datetime_start).total_seconds()
            for t in study.trials
            if t.state.name == "COMPLETE" and t.datetime_start and t.datetime_complete
        },
        name="seconds",
    )


def hyperparameter_table(
    branches: Sequence[str],
    params=("wh_learning_rate", "sv_learning_rate", "centroid_momentum"),
) -> pd.DataFrame:
    """Promoted v1.0 vs v1.1 winners, one row per (branch, version).

    Args:
        branches (Sequence[str]): Keys of BRANCHES.
        params (tuple, optional): AdaptConfig fields to show.

    Returns:
        pd.DataFrame: Indexed by config label; rows whose promoted file doesn't exist yet
        are skipped.
    """
    rows = {}
    for key in branches:
        b = BRANCHES[key]
        for label, version in ((f"{b.v10_label} (v1.0)", "v1.0"), (f"{b.label} (v1.1)", "v1.1")):
            path = promoted_path(key, version)
            if path.exists():
                config = AdaptConfig.from_yaml(path)
                rows[label] = {p: getattr(config, p) for p in params}
    return pd.DataFrame.from_dict(rows, orient="index")
