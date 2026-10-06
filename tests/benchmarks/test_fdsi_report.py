"""Tests for benchmarks/fdsi/report.py on synthetic tables (no real data)."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.fdsi import report
from benchmarks.fdsi.spec import FIXED_BRANCH, load_spec

SMOKE_SPEC = "benchmarks/fdsi/benchmark_smoke.yaml"


@pytest.fixture
def spec(tmp_path):
    return load_spec(SMOKE_SPEC, outputs_root=tmp_path / "outputs")


def _units(spec, n_units: int = 3, seed: int = 0) -> pd.DataFrame:
    """A units table like collect's: every branch x recording x unit."""
    rng = np.random.default_rng(seed)
    rows = []
    for branch in spec.branches:
        for rec in spec.recordings():
            for unit in range(n_units):
                rows.append(
                    {
                        "branch": branch,
                        "recording": rec.stub,
                        "sub": rec.sub,
                        "condition": rec.cond,
                        "snr": rec.snr,
                        "unit": unit,
                        "gt_unit": 10 + unit,
                        "roa_calib": 0.95,
                        "roa_full": 0.5 if branch == FIXED_BRANCH else 0.9,
                        "roa_after_cal": rng.uniform(0.3, 1),
                        "roa_first_iso": rng.uniform(0.3, 1)
                        if "triangular" in rec.cond
                        else np.nan,
                        "roa_ramp": rng.uniform(0.3, 1) if "triangular" in rec.cond else np.nan,
                        "roa_last_iso": rng.uniform(0.3, 1) if "triangular" in rec.cond else np.nan,
                        "sil": [0.95, 0.85, 0.92][unit % 3],
                        "n_spikes": 100,
                        "n_spikes_gt": 100,
                    }
                )
    return pd.DataFrame(rows)


def _write_tables(spec) -> None:
    """Every collected table, small and synthetic."""
    units = _units(spec)
    recs = spec.recordings()
    trials = pd.DataFrame(
        {
            "search": "pareto_sum",
            "number": [0, 1, 2, 3],
            "state": ["COMPLETE"] * 4,
            "values_wh_loss": [1.0, 2.0, 3.0, 2.5],
            "values_sv_loss": [3.0, 2.0, 1.0, 2.5],
            "user_attrs_roa_mean_pooled": [40.0, 50.0, 60.0, 30.0],
        }
    )
    tables = {
        "calibrations": pd.DataFrame(
            {
                "recording": [r.stub for r in recs],
                "sub": [r.sub for r in recs],
                "condition": [r.cond for r in recs],
                "snr": [r.snr for r in recs],
                "status": "done",
                "n_units": 3,
            }
        ),
        "calibration_units": units[units["branch"] == FIXED_BRANCH]
        .drop(columns=["branch"])
        .rename(columns={"sil": "sil_calib"}),
        "searches": trials,
        "best_configs": pd.DataFrame({"search": ["pareto_sum"], "chosen_trial": [2]}),
        "recordings": units.groupby(["branch", "recording", "sub", "condition", "snr"])
        .size()
        .rename("n_units")
        .reset_index()
        .assign(status="done", mean_batch_ms=20.0, run_time_s=5.0),
        "units": units,
        "provenance": pd.DataFrame(
            {
                "stage": ["calibrate", "calibrate", "apply"],
                "id": ["a", "b", "c"],
                "status": ["done", "skipped", "done"],
                "run_time_s": [3600.0, 1800.0, 60.0],
                "hostname": ["node1", "node2", "node1"],
                "cpu": ["Xeon", "Xeon", "Xeon"],
                "commit": ["abc", "abc", "abc"],
                "dirty": [False, False, True],
            }
        ),
    }
    spec.tables_dir.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(spec.tables_dir / f"{name}.csv", index=False)


def test_config_labels_follow_the_spec():
    spec = load_spec(SMOKE_SPEC)
    assert report.config_order(spec) == [
        "No adaptation",
        "sv_loss, mean",
        "sv_loss, sum",
        "Pareto min-sv, mean",
        "Pareto min-sv, sum",
        "RoA (oracle)",
    ]


def test_split_of_separates_pool_recordings_pool_conditions_and_held_out_ones(spec):
    assert report.split_of(spec, "sub-01", "triangular-ramp40s", 30) == report.SPLITS[0]
    assert report.split_of(spec, "sub-02", "triangular-ramp40s", 30) == report.SPLITS[1]
    assert report.split_of(spec, "sub-01", "triangular-ramp40s", 20) == report.SPLITS[1]
    assert report.split_of(spec, "sub-01", "staircase", 30) == report.SPLITS[2]


def test_load_tables_adds_labels_and_derived_columns(spec):
    _write_tables(spec)
    tables = report.load_tables(spec)

    assert set(tables) == set(report.TABLES)
    units = tables["units"]
    assert set(units["config"]) == set(report.config_order(spec))
    assert set(units["split"]) <= set(report.SPLITS)
    assert set(units["contraction"]) == {"staircase", "triangular"}
    np.testing.assert_allclose(units[report.ROA_PCT], units[report.ROA_METRIC] * 100)
    assert set(tables["searches"]["config"]) == {"Pareto min-sv, sum"}


def test_load_tables_with_an_empty_table(spec):
    _write_tables(spec)
    pd.DataFrame().to_csv(spec.tables_dir / "searches.csv", index=False)
    assert report.load_tables(spec)["searches"].empty


def test_load_tables_names_the_collect_command_when_missing(spec):
    with pytest.raises(FileNotFoundError, match=r"benchmarks.fdsi collect"):
        report.load_tables(spec)


def test_summary_and_units_per_recording(spec):
    _write_tables(spec)
    units = report.load_tables(spec)["units"]

    summary = report.summary_by_config(units)
    assert summary.loc["No adaptation", "mean"] == pytest.approx(50.0)
    assert summary.loc["sv_loss, mean", "pct_ge_threshold"] == pytest.approx(100.0)

    counts = report.units_per_recording(units)
    assert (counts["n_units"] == 3).all()
    assert (counts["n_sil_ge"] == 2).all()  # SILs 0.95, 0.85, 0.92
    assert counts["pct_sil_ge"].iloc[0] == pytest.approx(200 / 3)
    calib = report.units_per_recording(
        report.load_tables(spec)["calibration_units"],
        sil_col="sil_calib",
        roa_col="roa_calib",
        by=["recording"],
    )
    assert (calib["n_roa_ge"] == 3).all()


def test_paired_summary_pairs_each_unit_across_configs(spec):
    _write_tables(spec)
    tables = report.load_tables(spec)
    units = tables["units"]

    within = report.paired_summary(units, [("No adaptation", "sv_loss, mean")])
    row = within.loc[("No adaptation", "sv_loss, mean")]
    assert row["mean_delta"] == pytest.approx(40.0)
    assert row["mean_after"] - row["mean_before"] == pytest.approx(row["mean_delta"])
    assert row["median_before"] <= row["median_after"]
    assert row["pct_improved"] == pytest.approx(100.0)


def test_phase_long_and_heatmap_delta(spec):
    _write_tables(spec)
    units = report.load_tables(spec)["units"]

    phases = report.phase_long(units)
    assert set(phases["phase"]) == set(report.PHASES.values())
    assert phases["condition"].str.contains("triangular").all()
    assert list(phases.columns) == ["config", "condition", "snr", "phase", "roa_pct"]

    delta = report.heatmap_delta(units, "No adaptation")
    assert "No adaptation" not in set(delta["config"])
    assert delta["delta"].mean() == pytest.approx(40.0)


def test_search_helpers(spec):
    _write_tables(spec)
    trials = report.complete_trials(report.load_tables(spec)["searches"], "pareto_sum")

    assert report.objective_columns(trials) == ["values_wh_loss", "values_sv_loss"]
    assert report.front_numbers(trials) == [0, 1, 2]  # trial 3 is dominated by trial 1
    convergence = report.best_so_far(trials, "values_sv_loss")
    assert list(convergence["best_so_far"]) == [3.0, 2.0, 1.0, 1.0]


def test_provenance_and_cost_summaries(spec):
    _write_tables(spec)
    tables = report.load_tables(spec)

    provenance = report.provenance_summary(tables["provenance"])
    assert provenance.loc["calibrate", "n_tasks"] == 2
    assert provenance.loc["calibrate", "compute_h"] == pytest.approx(1.5)
    assert provenance.loc["calibrate", "skipped"] == 1
    assert bool(provenance.loc["apply", "any_dirty"]) is True

    cost = report.cost_summary(tables["recordings"], batch_ms=100)
    assert cost["real_time_factor"].iloc[0] == pytest.approx(0.2)
