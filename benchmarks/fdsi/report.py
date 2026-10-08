"""Tables and comparisons behind the notebooks: one version's results (results/<version>/)
with readable labels, paired per-unit comparisons, per-recording unit counts, version
comparisons and search summaries. Reads only the results in git, never the outputs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from adapt_decomp.adaptation.optimize import front_mask

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
ROA_COLUMNS: Tuple[str, ...] = (
    "roa_calib",
    "roa_full",
    "roa_after_cal",
    "roa_first_iso",
    "roa_ramp",
    "roa_last_iso",
)
# The headline RoA of every unit: the whole recording, the calibration window included
# (roa_after_cal leaves out the window CBSS fitted itself)
ROA_METRIC = "roa_full"
ROA_PCT = f"{ROA_METRIC}_pct"
PHASES: Dict[str, str] = {
    "roa_first_iso_pct": "First isometric",
    "roa_ramp_pct": "Ramp",
    "roa_last_iso_pct": "Last isometric",
}
SPLITS: Tuple[str, ...] = (
    "pool recordings",
    "pool conditions, other recordings",
    "held-out conditions",
)
UNIT_KEYS: Tuple[str, ...] = ("sub", "condition", "snr", "unit")  # one unit, within a version
FIXED_LABEL = "No adaptation"


# Loading


def split_of(pool: Dict[str, Any], sub: str, cond: str, snr: int) -> str:
    """How a recording relates to the searches' pool (config's pool section): one of SPLITS."""
    if cond not in pool["conditions"]:
        return SPLITS[2]
    if sub == pool["subject"] and int(snr) == int(pool["snr"]):
        return SPLITS[0]
    return SPLITS[1]


def _add_recording_columns(table: pd.DataFrame, pool: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Add split (with a pool), contraction and percent RoA columns to a per-unit table."""
    table = table.copy()
    if pool is not None:
        table["split"] = [
            split_of(pool, sub, cond, snr)
            for sub, cond, snr in zip(table["sub"], table["condition"], table["snr"])
        ]
    table["contraction"] = np.where(
        table["condition"].str.contains("triangular"), "triangular", "staircase"
    )
    for column in ROA_COLUMNS:
        if column in table:
            table[f"{column}_pct"] = table[column] * 100
    return table


def load_results(version: str, results_root: Path = RESULTS_ROOT) -> Dict[str, Any]:
    """One version's results, with readable labels and derived columns.

    Args:
        version (str): The version, a folder of results_root, e.g. "v1.1.0".
        results_root (Path, optional): Defaults to benchmarks/fdsi/results.

    Returns:
        Dict[str, Any]: "run" (its run.yaml), "labels" (branch -> label), and the tables it
        has: "units" and "calibration_units" (with "config" from labels, "split" when run.yaml
        records the pool, "contraction" and a "<roa column>_pct" per RoA column) and "trials"
        (with "config").
    """
    folder = Path(results_root) / version
    with open(folder / "run.yaml", encoding="utf-8") as f:
        run = yaml.safe_load(f)
    pool = (run.get("config") or {}).get("pool")
    results: Dict[str, Any] = {"run": run, "labels": run["labels"]}
    for name in ("units", "calibration_units", "trials"):
        if (folder / f"{name}.csv").exists():
            results[name] = pd.read_csv(folder / f"{name}.csv")
    if "units" in results:
        units = results["units"].assign(config=lambda t: t["branch"].map(run["labels"]))
        results["units"] = _add_recording_columns(units, pool)
    if "calibration_units" in results:
        results["calibration_units"] = _add_recording_columns(results["calibration_units"], pool)
    if "trials" in results:
        results["trials"] = results["trials"].assign(
            config=lambda t: t["search"].map(run["labels"])
        )
    return results


# Summaries


def summary_by_config(
    units: pd.DataFrame,
    value: str = ROA_PCT,
    threshold: float = 90.0,
    by: Sequence[str] = ("config",),
) -> pd.DataFrame:
    """Per-unit value summarised per config: count, mean, median, std and % at or above threshold.

    Args:
        units (pd.DataFrame): Per-unit table with the by columns and value.
        value (str, optional): Column to summarise. Defaults to ROA_PCT.
        threshold (float, optional): Threshold of the "pct_ge_threshold" column.
            Defaults to 90.0.
        by (Sequence[str], optional): Grouping columns. Defaults to ("config",).

    Returns:
        pd.DataFrame: Indexed by the by columns: n_units, mean, median, std and
        pct_ge_threshold.
    """
    grouped = units.groupby(list(by), sort=False, observed=True)[value]
    return pd.DataFrame(
        {
            "n_units": grouped.count(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "std": grouped.std(),
            "pct_ge_threshold": grouped.apply(lambda v: (v >= threshold).mean() * 100),
        }
    )


def units_per_recording(
    units: pd.DataFrame,
    sil_col: str = "sil",
    roa_col: str = ROA_METRIC,
    sil_th: float = 0.9,
    roa_th: float = 0.9,
    by: Sequence[str] = ("config", "recording"),
) -> pd.DataFrame:
    """Units per recording, and how many reach a SIL and a RoA threshold.

    Args:
        units (pd.DataFrame): Per-unit table (units or calibration_units).
        sil_col (str, optional): SIL column. Defaults to "sil" ("sil_calib" for
            calibration_units).
        roa_col (str, optional): RoA column, 0-1. Defaults to ROA_METRIC
            ("roa_calib" for calibration_units).
        sil_th (float, optional): SIL threshold. Defaults to 0.9.
        roa_th (float, optional): RoA threshold, 0-1. Defaults to 0.9.
        by (Sequence[str], optional): Columns identifying a recording (and
            config). Defaults to ("config", "recording").

    Returns:
        pd.DataFrame: One row per group: the by columns, n_units, n_sil_ge
        (units with SIL >= sil_th), pct_sil_ge and n_roa_ge (units with RoA >= roa_th).
    """
    grouped = units.groupby(list(by), sort=False, observed=True)
    counts = grouped.agg(
        n_units=(sil_col, "size"),
        n_sil_ge=(sil_col, lambda v: int((v >= sil_th).sum())),
        n_roa_ge=(roa_col, lambda v: int((v >= roa_th).sum())),
    )
    counts["pct_sil_ge"] = counts["n_sil_ge"] / counts["n_units"] * 100
    return counts[["n_units", "n_sil_ge", "pct_sil_ge", "n_roa_ge"]].reset_index()


def paired_delta(
    table: pd.DataFrame,
    before: str,
    after: str,
    value: str = ROA_PCT,
    keys: Sequence[str] = UNIT_KEYS,
) -> pd.DataFrame:
    """Pair two configs' per-unit values and difference them.

    Args:
        table (pd.DataFrame): Per-unit table with "config", the keys and value.
        before (str): Config label of the reference.
        after (str): Config label compared with it.
        value (str, optional): Column to compare. Defaults to ROA_PCT.
        keys (Sequence[str], optional): Columns identifying a unit. Defaults to UNIT_KEYS;
            ("recording", "gt_unit") pairs units across versions, which calibrated each
            recording themselves.

    Returns:
        pd.DataFrame: The keys, "before", "after" and "delta" (after - before), for the
        units present in both.
    """
    keys = list(keys)
    left = table[table["config"] == before][[*keys, value]].rename(columns={value: "before"})
    right = table[table["config"] == after][[*keys, value]].rename(columns={value: "after"})
    paired = left.merge(right, on=keys)
    return paired.assign(delta=paired["after"] - paired["before"])


def paired_summary(
    table: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    value: str = ROA_PCT,
    tol: float = 0.5,
    keys: Sequence[str] = UNIT_KEYS,
) -> pd.DataFrame:
    """Paired per-unit comparison of each (before, after) config pair.

    Args:
        table (pd.DataFrame): Per-unit table with "config", the UNIT_KEYS and value.
        pairs (Sequence[Tuple[str, str]]): (before, after) config labels.
        value (str, optional): Column to compare. Defaults to ROA_PCT.
        tol (float, optional): |delta| at or below this counts as unchanged.
            Defaults to 0.5 (percentage points of RoA).
        keys (Sequence[str], optional): Columns identifying a unit (see paired_delta).
            Defaults to UNIT_KEYS.

    Returns:
        pd.DataFrame: One row per pair present in table, indexed by (before,
        after): n_units, the mean and median of before, after and their delta over
        the paired units, pct_improved, pct_unchanged, pct_worse and the two-sided
        Wilcoxon signed-rank p-value (NaN when every delta is zero).
    """
    rows = []
    for before, after in pairs:
        paired = paired_delta(table, before, after, value, keys).dropna(subset=["delta"])
        if paired.empty:
            continue
        delta = paired["delta"].to_numpy()
        rows.append(
            {
                "before": before,
                "after": after,
                "n_units": len(delta),
                "mean_before": paired["before"].mean(),
                "mean_after": paired["after"].mean(),
                "mean_delta": delta.mean(),
                "median_before": paired["before"].median(),
                "median_after": paired["after"].median(),
                "median_delta": np.median(delta),
                "pct_improved": (delta > tol).mean() * 100,
                "pct_unchanged": (np.abs(delta) <= tol).mean() * 100,
                "pct_worse": (delta < -tol).mean() * 100,
                "wilcoxon_p": wilcoxon(delta).pvalue if np.any(delta != 0) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index(["before", "after"]) if rows else pd.DataFrame()


def phase_long(units: pd.DataFrame) -> pd.DataFrame:
    """Per-phase RoA of the triangular contractions, in plot_phase_bar's format.

    Args:
        units (pd.DataFrame): Per-unit table with config, condition, snr and the
            roa_<phase>_pct columns.

    Returns:
        pd.DataFrame: One row per (unit, phase) of a triangular condition:
        config, condition, snr, phase (one of PHASES' values) and roa_pct.
    """
    triangular = units[units["condition"].str.contains("triangular")]
    long = triangular.melt(
        id_vars=["config", "condition", "snr"],
        value_vars=list(PHASES),
        var_name="phase",
        value_name="roa_pct",
    )
    long["phase"] = long["phase"].map(PHASES)
    return long.dropna(subset=["roa_pct"])


def version_summary(
    versions: Dict[str, pd.DataFrame], value: str = ROA_PCT, threshold: float = 90.0
) -> pd.DataFrame:
    """Every config of several versions, one row each (summary_by_config).

    Args:
        versions (Dict[str, pd.DataFrame]): Version -> its units table (load_results).
        value (str, optional): Per-unit column to summarise. Defaults to ROA_PCT.
        threshold (float, optional): Threshold of the "pct_ge_threshold" column.
            Defaults to 90.0.

    Returns:
        pd.DataFrame: Indexed by (version, config), in the order given: n_units, mean,
        median, std and pct_ge_threshold.
    """
    both = pd.concat([u.assign(version=v) for v, u in versions.items()], ignore_index=True)
    return summary_by_config(both, value, threshold, by=("version", "config"))


def heatmap_delta(units: pd.DataFrame, before: str, value: str = ROA_PCT) -> pd.DataFrame:
    """Each config's per-unit gain over a reference config, for plot_metric_heatmap.

    Args:
        units (pd.DataFrame): Per-unit table with config, the UNIT_KEYS and value.
        before (str): The reference config's label (e.g. FIXED_LABEL).
        value (str, optional): Column to compare. Defaults to ROA_PCT.

    Returns:
        pd.DataFrame: One row per (config, unit), configs other than before:
        config, condition, snr and "delta" (config - before).
    """
    tables = []
    for config in units["config"].dropna().unique():
        if config == before:
            continue
        paired = paired_delta(units, before, config, value)
        tables.append(paired.assign(config=config))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# Searches


def complete_trials(searches: pd.DataFrame, search: Optional[str] = None) -> pd.DataFrame:
    """The COMPLETE trials of every search, or of one.

    Args:
        searches (pd.DataFrame): The searches table.
        search (Optional[str], optional): A search name. Defaults to None (all).

    Returns:
        pd.DataFrame: Its COMPLETE rows.
    """
    trials = searches[searches["state"] == "COMPLETE"]
    return trials if search is None else trials[trials["search"] == search]


def objective_columns(trials: pd.DataFrame) -> List[str]:
    """A search's objective columns: value_<objective> or values_<objective>.

    Args:
        trials (pd.DataFrame): One search's trials.

    Returns:
        List[str]: The value columns with at least one value.
    """
    return [c for c in trials.columns if c.startswith("value") and trials[c].notna().any()]


def front_numbers(
    trials: pd.DataFrame, objectives: Sequence[str] = ("wh_loss", "sv_loss")
) -> List[int]:
    """Numbers of a search's trials on its Pareto front.

    Args:
        trials (pd.DataFrame): One search's trials, with values_<objective> columns.
        objectives (Sequence[str], optional): The objectives, all minimised.
            Defaults to ("wh_loss", "sv_loss").

    Returns:
        List[int]: Trial numbers of the non-dominated COMPLETE trials.
    """
    complete = trials[trials["state"] == "COMPLETE"]
    values = complete[[f"values_{objective}" for objective in objectives]].to_numpy()
    return [int(n) for n in complete["number"][front_mask(values)]]


def best_so_far(trials: pd.DataFrame, value: str) -> pd.DataFrame:
    """The best (lowest) objective value reached by each trial number, for convergence plots.

    Args:
        trials (pd.DataFrame): One search's trials.
        value (str): The objective column to minimise.

    Returns:
        pd.DataFrame: number, value and best_so_far, over COMPLETE trials in order.
    """
    complete = trials[trials["state"] == "COMPLETE"].sort_values("number")
    return pd.DataFrame(
        {
            "number": complete["number"].to_numpy(),
            "value": complete[value].to_numpy(),
            "best_so_far": complete[value].cummin().to_numpy(),
        }
    )
