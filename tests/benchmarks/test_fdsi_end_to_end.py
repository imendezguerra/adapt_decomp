"""End-to-end run of the FDSI benchmark stages: calibrate, search, apply, collect and verify,
on synthetic recordings written in the FDSI layout (tests/synthetic.py), with a one-recording
pool and one one-trial search. Covers the stages' plumbing (caching, metadata, tables, bit-exact
re-runs); the algorithms themselves are checked against ground truth in tests/test_pipeline.py."""

import numpy as np
import pandas as pd
import pytest
import yaml

from adapt_decomp.utils import read_metadata
from benchmarks.fdsi import stages
from benchmarks.fdsi.spec import REPO_ROOT, load_spec
from tests.synthetic import make_synthetic_recording

SMOKE_SPEC = REPO_ROOT / "benchmarks" / "fdsi" / "benchmark_smoke.yaml"

pytestmark = pytest.mark.slow


@pytest.fixture
def spec(tmp_path):
    """The smoke spec over two synthetic recordings in tmp_path: a triangular one, also the
    search pool, and a held-out staircase one."""
    raw = yaml.safe_load(SMOKE_SPEC.read_text())
    raw["data_root"] = str(tmp_path / "data")
    raw["outputs_root"] = str(tmp_path / "outputs")
    raw["grid"]["conditions"] = ["triangular-ramp40s", "staircase"]
    raw["calibration"]["cbss_config"].update(ext_fact=8, search_iter=15)
    raw["pool"]["conditions"] = ["triangular-ramp40s"]  # one recording: the search runs in-process
    raw["search"].update(n_trials=1, n_jobs=1)
    raw["searches"] = {"sv_mean": raw["searches"]["sv_mean"]}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    loaded = load_spec(path)

    for seed, rec in enumerate(loaded.recordings()):
        synthetic = make_synthetic_recording(
            seed=seed, fs=loaded.grid.fs, cal_s=loaded.grid.cal_duration_s
        )
        firings = np.empty(synthetic.spikes.shape[1], dtype=object)  # FDSI's per-unit firings
        firings[:] = [np.flatnonzero(unit) for unit in synthetic.spikes.T]
        for path, values in (
            (loaded.emg_path(rec), {"emg": synthetic.emg}),
            (loaded.gt_path(rec), {"spikes": firings}),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **values)
    return loaded


def test_every_stage_runs_caches_and_reproduces(spec):
    command = ["python", "-m", "benchmarks.fdsi", "test"]

    # Run every stage, then again: the second pass only checks the cache
    for stage in ("calibrate", "search", "apply"):
        counts = stages.run_tasks(spec, spec.tasks(stage), command=command)
        assert counts == {"missing": len(spec.tasks(stage))}
    for stage in ("calibrate", "search", "apply"):
        counts = stages.run_tasks(spec, spec.tasks(stage), command=command)
        assert set(counts) <= {"done", "skipped"}

    # Every output has its metadata, and the applied results their tables
    task = spec.tasks("apply")[-1]
    meta = read_metadata(spec.meta_path(task))
    assert meta["task"]["status"] == "done"
    assert meta["digest"]["n_units"] > 0
    assert meta["git"]["commit"]
    assert meta["reproduce"].splitlines()[-1] == spec.task_command(task)

    paths = stages.collect(spec, command=command)
    units = pd.read_csv(paths["units"])
    assert set(units["branch"]) == {"fixed", "sv_mean"}
    assert set(units["condition"]) == {"triangular-ramp40s", "staircase"}
    assert units["roa_after_cal"].between(0, 1).all()
    # The searched config tracks the drifting MUAPs the fixed one loses
    roa = units.groupby("branch")["roa_after_cal"].mean()
    assert roa["sv_mean"] > 0.9 > roa["fixed"]
    assert (pd.read_csv(paths["provenance"])["status"] == "done").all()
    assert (spec.tables_dir / "collect.meta.yaml").exists()

    # A re-run of an applied config reproduces its spike trains bit for bit
    report = stages.verify(
        spec, "apply", [task.index], spec.outputs_root.parent / "verify", command=command
    )
    assert list(report["result"]) == ["exact"]
