"""End-to-end run of the FDSI benchmark pipeline (benchmarks/fdsi/pipeline.py): calibrate,
search, apply and collect, on synthetic recordings written in the FDSI layout
(tests/synthetic.py), with a one-recording pool and one one-trial search. Covers the stages'
plumbing (skipping done tasks, re-scoring saved results, bit-exact re-runs); the algorithms
themselves are checked against ground truth in tests/test_pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.fdsi import fdsi, pipeline
from tests.synthetic import make_synthetic_recording

pytestmark = pytest.mark.slow


@pytest.fixture
def cfg(tmp_path):
    """The quick config over two synthetic recordings in tmp_path: a triangular one, also the
    search pool, and a held-out staircase one."""
    cfg = pipeline.load_config(quick=True)
    cfg["data_root"] = str(tmp_path / "data")
    cfg["outputs"], cfg["results"] = tmp_path / "outputs", tmp_path / "results"
    cfg["grid"]["conditions"] = ["triangular-ramp40s", "staircase"]
    cfg["calibration"]["cbss_config"].update(ext_fact=8, search_iter=15)
    cfg["pool"]["conditions"] = ["triangular-ramp40s"]  # one recording: the search runs in-process
    cfg["search"].update(n_trials=1, n_jobs=1, n_cores=1)
    cfg["searches"] = {"sv_mean": cfg["searches"]["sv_mean"]}

    for seed, rec in enumerate(pipeline.recordings(cfg)):
        synthetic = make_synthetic_recording(
            seed=seed, fs=cfg["grid"]["fs"], cal_s=cfg["grid"]["cal_duration_s"]
        )
        firings = np.empty(synthetic.spikes.shape[1], dtype=object)  # FDSI's per-unit firings
        firings[:] = [np.flatnonzero(unit) for unit in synthetic.spikes.T]
        data_root = tmp_path / "data"
        for path, values in (
            (fdsi.emg_path(data_root, rec), {"emg": synthetic.emg}),
            (fdsi.gt_spikes_path(data_root, rec.sub, rec.cond), {"spikes": firings}),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **values)
    return cfg


def _mtimes(root):
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_every_stage_runs_skips_rescores_and_reproduces(cfg):
    for stage in pipeline.STAGES:
        pipeline.run(cfg, stage)

    # A second run computes nothing: every output exists
    before = _mtimes(cfg["outputs"])
    for stage in ("calibrate", "search", "apply"):
        pipeline.run(cfg, stage)
    assert _mtimes(cfg["outputs"]) == before

    results = cfg["results"]
    units = pd.read_csv(results / "units.csv")
    assert set(units["branch"]) == {"fixed", "sv_mean"}
    assert set(units["condition"]) == {"triangular-ramp40s", "staircase"}
    assert units["roa_after_cal"].between(0, 1).all() and units["roa_calib"].notna().all()
    # The searched config tracks the drifting MUAPs the fixed one loses
    roa = units.groupby("branch")["roa_after_cal"].mean()
    assert roa["sv_mean"] > 0.9 > roa["fixed"]
    trials = pd.read_csv(results / "trials.csv")
    assert list(trials["search"]) == ["sv_mean"] and trials["chosen"].all()
    assert (results / "configs" / "sv_mean.yaml").exists()

    # Scores come back from the saved spikes and sources alone (e.g. a downloaded archive)
    rec = pipeline.recordings(cfg)[0]
    pipeline.result_path(cfg, "sv_mean", rec, "_units.csv").unlink()
    pipeline.collect(cfg)
    pd.testing.assert_frame_equal(pd.read_csv(results / "units.csv"), units)

    # A re-run of an applied config reproduces its spike trains bit for bit
    saved = np.load(pipeline.result_path(cfg, "sv_mean", rec))["spikes"]
    pipeline.result_path(cfg, "sv_mean", rec).unlink()
    pipeline.result_path(cfg, "sv_mean", rec, "_units.csv").unlink()
    pipeline.apply_one(cfg, "sv_mean", rec)
    np.testing.assert_array_equal(
        np.load(pipeline.result_path(cfg, "sv_mean", rec))["spikes"], saved
    )
