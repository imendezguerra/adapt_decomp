"""Glue for notebooks/fdsi_benchmark_v1_1/: the FDSI benchmark re-run with adapt_decomp v1.1.

Builds on ../fdsi_benchmark/fdsi_common.py (imported, not copied): the same data, calibration
cache and on-disk layout, with v1.1 results written under OUTPUTS_ROOT/adaptation_v1_1/ so the
cached v1.0 results under OUTPUTS_ROOT/adaptation/ stay untouched and every fdsi_common path
builder/aggregator works on either root. v1.1 here means: optimize_adapt_decomp (multivariate
TPE, centroid_momentum searched, per-unit-mean sv_loss, CoV-ISI unit selection, worker
processes) and adaptation from the end of the calibration window (process_from_calib_end).
"""

import pickle
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fdsi_benchmark"))
import fdsi_common as fc

from adapt_decomp import AdaptationResult, AdaptDecomp, CBSSConfig, CBSSResult
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import (
    OptimisationResult,
    _select_min_sv_loss,
    optimize_adapt_decomp,
)
from adapt_decomp.spikes import get_sil
from adapt_decomp.utils.loaders import PooledDatasetMemory

try:
    from tqdm.notebook import tqdm
except ImportError:  # pragma: no cover
    from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_DIR = REPO_ROOT / "configs" / "adapt_configs"
LR_MODE = "fixed"  # lr_fixed only in v1.1, the branch v1.0 promoted as its headline config
DEVICE = "cpu"  # deterministic, and MPS lacks the QR the sv update needs

# One search setting for every v1.1 search: sequential trials (reproducible), the pool's three
# recordings spread over three worker processes, v1.0's seed, and CoV-ISI unit selection.
SEARCH = dict(
    n_jobs=1,
    n_workers=3,
    random_seed=42,
    unit_selection="unsupervised",
    unit_selection_kwargs={"cov_th": 0.3},
)


class Branch(NamedTuple):
    """One applied v1.1 config and the v1.0 config it is compared with."""

    label: str  # v1.1 config label stem, e.g. "sv_loss"
    sampler: str  # v1.1 sampler dir (lr_fixed), also its result dir
    promoted: str  # v1.1 promoted config file in configs/adapt_configs/
    v10_label: str  # v1.0 counterpart's label stem
    v10_sampler: str  # v1.0 counterpart's sampler dir
    v10_promoted: str  # v1.0 counterpart's promoted config file


# The v1.1 Pareto search is applied twice, once per front selection; v1.0 had no knee
# selection, so both Pareto branches compare with v1.0's (min-sv) Pareto winner.
BRANCHES: Dict[str, Branch] = {
    "sv_loss": Branch(
        "sv_loss",
        "mtpe_sv_mean",
        "optim_muniverse_fdsi_v11_sv.yaml",
        "sv_loss",
        "tpe_sv_median",
        "optim_muniverse_fdsi_sv.yaml",
    ),
    "pareto_min_sv": Branch(
        "pareto min-sv",
        "mtpe_pareto_min_sv",
        "optim_muniverse_fdsi_v11_pareto_min_sv.yaml",
        "pareto min-sv",
        "tpe_pareto",
        "optim_muniverse_fdsi_pareto.yaml",
    ),
    "pareto_knee": Branch(
        "pareto knee",
        "mtpe_pareto_knee",
        "optim_muniverse_fdsi_v11_pareto_knee.yaml",
        "pareto min-sv",
        "tpe_pareto",
        "optim_muniverse_fdsi_pareto.yaml",
    ),
    "roa": Branch(
        "roa",
        "mtpe_roa",
        "optim_muniverse_fdsi_v11_roa.yaml",
        "roa",
        "tpe_roa",
        "optim_muniverse_fdsi_roa.yaml",
    ),
}
# v1.1 search dir per search (the Pareto study serves both Pareto branches).
SEARCH_DIRS = {"sv_loss": "mtpe_sv_mean", "pareto": "mtpe_pareto", "roa": "mtpe_roa"}

# (label, version, lr_mode or None for the fixed baseline, sampler dir or None)
Config = Tuple[str, str, Optional[str], Optional[str]]


# -- Benchmark grid --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The FDSI benchmark grid and directories, read once from fdsi_benchmark_grid.yaml.

    Attributes:
        data_dir (Path): Raw recordings, <sub>/{clean,noisy}/.
        cal_dir (Path): Cached v1.0 calibrations (shared by both versions).
        adapt_dir_v10 (Path): Cached v1.0 adaptation results.
        adapt_dir_v11 (Path): v1.1 adaptation results, same layout.
        fs (int): Sampling frequency (Hz).
        ext_fact (int): Extension factor of every calibration.
        tol_spike_ms (float): RoA spike-alignment tolerance (ms).
        cal_end (int): Calibration window length; every calibration is [0, cal_end).
        iso_dur (int): Isometric bookend length for the triangular phase analysis.
        subjects, conditions, snr_levels: The 5 x 5 x 4 = 100-recording grid.
        pool_conditions (List[str]): Conditions pooled in the searches (optim_sub, optim_snr).
        optim_sub (str): Subject of the pooled recordings.
        optim_snr (int): SNR of the pooled recordings.
        n_trials (int): Trials per search.
    """

    data_dir: Path
    cal_dir: Path
    adapt_dir_v10: Path
    adapt_dir_v11: Path
    fs: int
    ext_fact: int
    tol_spike_ms: float
    cal_end: int
    iso_dur: int
    subjects: List[str]
    conditions: List[str]
    snr_levels: List[int]
    pool_conditions: List[str]
    optim_sub: str
    optim_snr: int
    n_trials: int

    @classmethod
    def from_yaml(cls, path: Path) -> "Grid":
        """Read the grid YAML (paths in it are relative to the notebook's directory).

        Args:
            path (Path): configs/data_configs/fdsi_benchmark_grid.yaml.

        Returns:
            Grid: The benchmark grid.
        """
        with open(path) as f:
            grid = yaml.safe_load(f)
        outputs_root = Path(grid["outputs_root"])
        return cls(
            data_dir=Path(grid["data_root"]),
            cal_dir=outputs_root / "calibration",
            adapt_dir_v10=outputs_root / "adaptation",
            adapt_dir_v11=outputs_root / "adaptation_v1_1",
            fs=grid["fs"],
            ext_fact=grid["ext_fact"],
            tol_spike_ms=grid["tol_spike_ms"],
            cal_end=int(grid["cal_duration_s"] * grid["fs"]),
            iso_dur=int(grid["iso_duration_s"] * grid["fs"]),
            subjects=grid["subjects"],
            conditions=grid["conditions"],
            snr_levels=grid["snr_levels"],
            pool_conditions=grid["pool_conditions"],
            optim_sub=grid["optim_sub"],
            optim_snr=grid["optim_snr"],
            n_trials=grid["n_trials"],
        )

    @property
    def triangular(self) -> List[str]:
        """Triangular conditions, for the phase analysis."""
        return [c for c in self.conditions if "triangular" in c]

    @property
    def holdout_conditions(self) -> List[str]:
        """Conditions never seen by the searches."""
        return [c for c in self.conditions if c not in self.pool_conditions]

    def opt_dir(self, version: str, search_dir: str) -> Path:
        """A search's best_result_path (lr_fixed), for "v1.0" or "v1.1"."""
        root = self.adapt_dir_v10 if version == "v1.0" else self.adapt_dir_v11
        return root / "optimisation" / fc.LR_MODE_TOKEN[LR_MODE] / search_dir

    def adapt_dir(self, version: str) -> Path:
        """The adaptation results root of "v1.0" or "v1.1"."""
        return self.adapt_dir_v10 if version == "v1.0" else self.adapt_dir_v11

    def recordings(self) -> List[Tuple[str, str, int]]:
        """Every (sub, condition, snr) of the grid."""
        return [(s, c, n) for s in self.subjects for c in self.conditions for n in self.snr_levels]


def result_paths(
    adapt_dir: Path, sub: str, cond: str, snr: int, lr_mode: Optional[str], sampler: Optional[str]
) -> Tuple[Path, Path]:
    """fc.fixed_paths for the fixed baseline (lr_mode None), else fc.adapted_paths.

    Args:
        adapt_dir (Path): Adaptation results root of either version.
        sub, cond, snr: The recording.
        lr_mode (Optional[str]): 'fixed', or None for the fixed baseline.
        sampler (Optional[str]): Search/sampler dir name, None for the fixed baseline.

    Returns:
        Tuple[Path, Path]: (result .pkl, config .yaml).
    """
    if lr_mode is None:
        return fc.fixed_paths(adapt_dir, sub, cond, snr)
    return fc.adapted_paths(adapt_dir, sub, cond, snr, lr_mode, sampler)


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


# -- Search ----------------------------------------------------------------------------------


def search_base_config(path: Path) -> AdaptConfig:
    """v1.0's base config (default_muniverse.yaml, lr_fixed) set up for a v1.1 search.

    Args:
        path (Path): configs/adapt_configs/default_muniverse.yaml.

    Returns:
        AdaptConfig: lr_mode="fixed", the source FIFO seeded from the calibration's tail (the
        search adapts from the calibration end), per-unit-mean sv_loss, CPU.
    """
    config = AdaptConfig.from_yaml(path)
    config.lr_mode = LR_MODE
    config.source_fifo_from_calib = True
    config.sv_loss_reduction = "mean"
    config.device = DEVICE
    return config


def calib_end_pool(
    pool: Dict[str, PooledDatasetMemory], cal_end: int
) -> Dict[str, PooledDatasetMemory]:
    """Restrict every pooled recording to the samples after its calibration window.

    The manual split of docs/adaptation.md: with search_base_config's source_fifo_from_calib,
    each trial adapts forwards from the calibration's end and scores (losses and RoA) only
    after it. The one approximation: emg[cal_end:] is filtered on its own, so the filters
    restart at cal_end (a few-ms transient), unlike process_from_calib_end.

    Args:
        pool (Dict[str, PooledDatasetMemory]): From load_data(fdsi_pool_memory_example.yaml);
            every calibration is [0, cal_end) of its recording.
        cal_end (int): Calibration window length in samples.

    Returns:
        Dict[str, PooledDatasetMemory]: The same entries with emg/gt_paired_bin from cal_end.
    """
    return {
        name: replace(
            dataset,
            emg=dataset.emg[cal_end:],
            gt_paired_bin=None
            if dataset.gt_paired_bin is None
            else dataset.gt_paired_bin[cal_end:],
        )
        for name, dataset in pool.items()
    }


def front_member_config(opt_dir: Path, trial) -> AdaptConfig:
    """A Pareto front member's resolved config, as written by the search.

    Args:
        opt_dir (Path): The Pareto search's best_result_path.
        trial (optuna.trial.FrozenTrial): A member of its front.

    Returns:
        AdaptConfig: trial_<n>/config.yaml.
    """
    return AdaptConfig.from_yaml(opt_dir / f"trial_{trial.number}" / "config.yaml")


def load_study(opt_dir: Path):
    """Unpickle a search's study.pkl (either version).

    Args:
        opt_dir (Path): The search's best_result_path.

    Returns:
        optuna.Study: The study, or None if it isn't on disk.
    """
    path = opt_dir / "study.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_search(
    objectives,
    pool: Dict[str, PooledDatasetMemory],
    base_config: AdaptConfig,
    opt_dir: Path,
    n_trials: int,
    run: bool,
    tol_spike_ms: float,
) -> OptimisationResult:
    """Load a complete cached v1.1 search, or run it with SEARCH's settings.

    Args:
        objectives: optimize_adapt_decomp's objectives (one name, or a tuple for Pareto).
        pool (Dict[str, PooledDatasetMemory]): From calib_end_pool.
        base_config (AdaptConfig): From search_base_config.
        opt_dir (Path): best_result_path; cleared before a new run.
        n_trials (int): Trials to run, and the number a cached study must have completed.
        run (bool): Whether to run the search when no complete cached study exists.
        tol_spike_ms (float): RoA tolerance (RoA is logged every trial, ground truth exists).

    Raises:
        FileNotFoundError: If no complete cached study exists and run is False.

    Returns:
        OptimisationResult: best_config (single objective: the best trial; Pareto: the front's
        min-sv member), study (with its search wall time in user_attrs["wall_time_s"]), and
        pareto_front. outputs is None when loaded from cache.
    """
    study = load_study(opt_dir)
    if study is not None and sum(t.state.name == "COMPLETE" for t in study.trials) >= n_trials:
        if isinstance(objectives, str):
            return OptimisationResult(AdaptConfig.from_yaml(opt_dir / "config.yaml"), study)
        front = study.best_trials
        return OptimisationResult(
            front_member_config(opt_dir, _select_min_sv_loss(front)), study, front
        )
    if not run:
        raise FileNotFoundError(f"No complete cached search in {opt_dir} and run=False.")

    shutil.rmtree(opt_dir, ignore_errors=True)
    start = time.perf_counter()
    result = optimize_adapt_decomp(
        pool=pool,
        objectives=objectives,
        base_config=base_config,
        n_trials=n_trials,
        compute_roa=True,
        roa_kwargs={"tol_spike_ms": tol_spike_ms},
        best_result_path=str(opt_dir),
        **SEARCH,
    )
    result.study.set_user_attr("wall_time_s", time.perf_counter() - start)
    with open(opt_dir / "study.pkl", "wb") as f:
        pickle.dump(result.study, f)
    return result


def promote(config: AdaptConfig, branch: str) -> Path:
    """Save a v1.1 winner as configs/adapt_configs/<BRANCHES[branch]'s promoted v1.1 name>.

    Args:
        config (AdaptConfig): The winning config.
        branch (str): Key of BRANCHES.

    Returns:
        Path: Where it was written.
    """
    path = PROMOTED_DIR / BRANCHES[branch].promoted
    config.to_yaml(path)
    return path


def hyperparameter_table(
    branches: Sequence[str], params=("wh_learning_rate", "sv_learning_rate", "centroid_momentum")
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
        for label, file in (
            (f"{b.v10_label} (v1.0)", b.v10_promoted),
            (f"{b.label} (v1.1)", b.promoted),
        ):
            if (PROMOTED_DIR / file).exists():
                config = AdaptConfig.from_yaml(PROMOTED_DIR / file)
                rows[label] = {p: getattr(config, p) for p in params}
    return pd.DataFrame.from_dict(rows, orient="index")


# -- Application -----------------------------------------------------------------------------


def load_calibrations(g: Grid) -> Dict[Tuple[str, str, int], Optional[CBSSResult]]:
    """Every recording's cached v1.0 calibration (None where 01_calibration skipped it).

    Args:
        g (Grid): The benchmark grid.

    Returns:
        Dict[Tuple[str, str, int], Optional[CBSSResult]]: (sub, cond, snr) -> calibration.
    """
    calibrations = {}
    for sub, cond, snr in g.recordings():
        path, _ = fc.calibration_paths(g.cal_dir, sub, cond, snr)
        calibrations[(sub, cond, snr)] = CBSSResult.load(path) if path.exists() else None
    return calibrations


def fixed_config(g: Grid) -> AdaptConfig:
    """v1.0's no-adaptation baseline config (every adapt_* flag off), on CPU.

    Args:
        g (Grid): The benchmark grid.

    Returns:
        AdaptConfig: The baseline config.
    """
    return AdaptConfig(
        ext_fact=g.ext_fact, adapt_wh=False, adapt_sv=False, adapt_sd=False, device=DEVICE
    )


def apply_from_calib_end(
    g: Grid,
    sub: str,
    cond: str,
    snr: int,
    cbss_result: Optional[CBSSResult],
    adapt_config: AdaptConfig,
    sampler: Optional[str],
    run: bool,
) -> Optional[AdaptationResult]:
    """Load one recording's cached v1.1 result, or compute it from the calibration's end.

    Same steps as v1.0's run_adapted/run_fixed (SIL and RoA attached, result + config saved),
    except process_from_calib_end replaces process_data: CBSS's own output over the
    calibration window [0, cal_end), adaptation forwards from cal_end.

    Args:
        g (Grid): The benchmark grid.
        sub, cond, snr: The recording.
        cbss_result (Optional[CBSSResult]): Its calibration, or None if skipped.
        adapt_config (AdaptConfig): The config to apply.
        sampler (Optional[str]): v1.1 sampler dir (lr_fixed), or None for the fixed baseline.
        run (bool): Whether to compute a result that isn't cached.

    Returns:
        Optional[AdaptationResult]: The result, or None if uncached/uncomputable.
    """
    if cbss_result is None:
        return None
    lr_mode = None if sampler is None else LR_MODE
    result_path, config_path = result_paths(g.adapt_dir_v11, sub, cond, snr, lr_mode, sampler)
    if result_path.exists():
        return AdaptationResult.load(result_path)
    if not run:
        return None

    emg_full = fc.load_raw_emg(g.data_dir, sub, cond, snr)
    _, cal_config_path = fc.calibration_paths(g.cal_dir, sub, cond, snr)
    adapter = AdaptDecomp.from_calibration(
        calibration=cbss_result,
        cbss_config=CBSSConfig.from_yaml(cal_config_path),
        adapt_config=adapt_config,
    )
    outputs = adapter.process_from_calib_end(emg_full, slice(0, g.cal_end))

    outputs.sil = get_sil(
        outputs.sources,
        outputs.spikes,
        adapt_config.spike_min_dist,
        peak_power=adapt_config.spike_det_exp,
    ).numpy()
    gt_full_bin = fc.load_gt_full_bin(
        g.data_dir, sub, cond, cbss_result, n_samples=outputs.spikes.shape[0]
    )
    outputs.roa = fc.compute_roa_for_result(outputs, gt_full_bin, g.fs, g.tol_spike_ms)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.save(result_path)
    adapt_config.to_yaml(config_path)
    return outputs


def apply_all(
    g: Grid, calibrations: Dict, adapt_config: AdaptConfig, sampler: Optional[str], run: bool
) -> Dict[Tuple[str, str, int], Optional[AdaptationResult]]:
    """apply_from_calib_end over the whole grid, with a progress bar.

    Args:
        g (Grid): The benchmark grid.
        calibrations (Dict): From load_calibrations.
        adapt_config (AdaptConfig): The config to apply.
        sampler (Optional[str]): v1.1 sampler dir, or None for the fixed baseline.
        run (bool): Whether to compute uncached results.

    Returns:
        Dict[Tuple[str, str, int], Optional[AdaptationResult]]: (sub, cond, snr) -> result.
    """
    results = {}
    for key, cal in tqdm(calibrations.items(), desc=f"Apply {sampler or 'fixed'} (v1.1)"):
        results[key] = apply_from_calib_end(g, *key, cal, adapt_config, sampler, run)
    n_done = sum(r is not None for r in results.values())
    print(f"{sampler or 'fixed'} (v1.1): {n_done}/{len(results)} recordings available.")
    return results


# -- Aggregation across versions -------------------------------------------------------------


def _per_version(
    g: Grid,
    config_list: List[Config],
    aggregate: Callable[[Path, List[Tuple[str, Optional[str], Optional[str]]]], pd.DataFrame],
) -> pd.DataFrame:
    """Run one fdsi_common-style aggregator per version root and stack the tables.

    Args:
        g (Grid): The benchmark grid.
        config_list (List[Config]): From configs().
        aggregate (Callable): (adapt_dir, [(label, lr_mode, sampler), ...]) -> DataFrame.

    Returns:
        pd.DataFrame: The stacked table, with a 'version' column.
    """
    tables = []
    for version in ("v1.0", "v1.1"):
        subset = [(label, lr, s) for label, v, lr, s in config_list if v == version]
        if subset:
            table = aggregate(g.adapt_dir(version), subset)
            if len(table):
                tables.append(table.assign(version=version))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def roa_table(g: Grid, config_list: List[Config]) -> pd.DataFrame:
    """Per-unit full-recording RoA (%), both versions (fc.aggregate_roa_from_disk)."""
    df = _per_version(
        g,
        config_list,
        lambda d, c: fc.aggregate_roa_from_disk(
            d,
            g.cal_dir,
            g.data_dir,
            g.subjects,
            g.conditions,
            g.snr_levels,
            c,
            g.fs,
            g.tol_spike_ms,
        ),
    )
    if len(df):
        df["roa_pct"] = df["roa"] * 100
    return df


def sil_table(g: Grid, config_list: List[Config]) -> pd.DataFrame:
    """Per-unit full-recording SIL, both versions (fc.aggregate_sil_from_disk)."""
    return _per_version(
        g,
        config_list,
        lambda d, c: fc.aggregate_sil_from_disk(
            d, g.cal_dir, g.subjects, g.conditions, g.snr_levels, c
        ),
    )


def phase_table(g: Grid, config_list: List[Config]) -> pd.DataFrame:
    """Per-unit, per-phase RoA (%) on triangular conditions (fc.aggregate_phase_roa_from_disk)."""
    df = _per_version(
        g,
        config_list,
        lambda d, c: fc.aggregate_phase_roa_from_disk(
            d,
            g.cal_dir,
            g.data_dir,
            g.subjects,
            g.triangular,
            g.snr_levels,
            c,
            g.fs,
            g.tol_spike_ms,
            g.cal_end,
            g.iso_dur,
        ),
    )
    if len(df):
        df["phase"] = pd.Categorical(df["phase"], categories=fc.PHASE_ORDER, ordered=True)
    return df


def window_roa_table(g: Grid, config_list: List[Config], window: slice) -> pd.DataFrame:
    """Per-unit RoA (%) restricted to a sample window, every condition, both versions.

    E.g. window=slice(g.cal_end, None) scores only after the calibration window, where both
    versions adapted (v1.1 outputs CBSS's own spikes inside it, v1.0 re-adapted over it).

    Args:
        g (Grid): The benchmark grid.
        config_list (List[Config]): From configs().
        window (slice): Samples to score.

    Returns:
        pd.DataFrame: Columns sub, condition, snr, config, unit, roa_pct, version.
    """

    def aggregate(adapt_dir: Path, subset) -> pd.DataFrame:
        rows = []
        for sub, cond, snr in g.recordings():
            cal_path, _ = fc.calibration_paths(g.cal_dir, sub, cond, snr)
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
                        g.data_dir, sub, cond, cbss_result, spikes.shape[0]
                    )
                    if gt_full_bin is None:
                        break
                roa = fc.compute_roa_subset(gt_full_bin, spikes, window, g.fs, g.tol_spike_ms)
                rows += [
                    {
                        "sub": sub,
                        "condition": cond,
                        "snr": snr,
                        "config": label,
                        "unit": u,
                        "roa_pct": float(r) * 100,
                    }
                    for u, r in enumerate(roa)
                ]
        return pd.DataFrame(rows)

    return _per_version(g, config_list, aggregate)


# -- v1.0 vs v1.1 statistics -----------------------------------------------------------------


def paired_delta(
    df: pd.DataFrame,
    before: str,
    after: str,
    value: str = "roa_pct",
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
        rows.append(
            {
                "before": before,
                "after": after,
                "n_units": len(delta),
                "mean_delta": delta.mean(),
                "median_delta": np.median(delta),
                "pct_improved": (delta > tol).mean() * 100,
                "pct_unchanged": (np.abs(delta) <= tol).mean() * 100,
                "pct_worse": (delta < -tol).mean() * 100,
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows).set_index(["before", "after"]) if rows else pd.DataFrame()


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
            {
                "trial": trial.number,
                "recording": name,
                "units": units[name],
                "sv_loss": loss,
                "share": loss / total if total > 0 else np.nan,
            }
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
