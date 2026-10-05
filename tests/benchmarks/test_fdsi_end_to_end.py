"""End-to-end run of the FDSI benchmark stages on real data: calibrate, search, apply, collect
and verify, on the pool's three recordings with one one-trial search. Needs the FDSI data
(adapt-decomp-data get fdsi_benchmark-data)."""

import pandas as pd
import pytest
import yaml

from adapt_decomp.utils import read_metadata
from benchmarks.fdsi import stages
from benchmarks.fdsi.spec import REPO_ROOT, load_spec

SMOKE_SPEC = REPO_ROOT / "benchmarks" / "fdsi" / "benchmark_smoke.yaml"
DATA_ROOT = REPO_ROOT / "data" / "fdsi_benchmark" / "data"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DATA_ROOT.exists(), reason="FDSI data not downloaded"),
]


@pytest.fixture
def spec(tmp_path, allow_cores):
    """The smoke spec cut down to the pool's recordings and one one-trial search, in tmp_path."""
    raw = yaml.safe_load(SMOKE_SPEC.read_text())
    raw["outputs_root"] = str(tmp_path / "outputs")
    raw["grid"]["conditions"] = raw["pool"]["conditions"]
    raw["search"].update(n_trials=1, n_jobs=1)
    raw["searches"] = {"sv_mean": raw["searches"]["sv_mean"]}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    allow_cores(3)  # one search trial runs its three recordings in parallel
    return load_spec(path)


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
    assert units["roa_after_cal"].between(0, 1).all()
    assert (pd.read_csv(paths["provenance"])["status"] == "done").all()
    assert (spec.tables_dir / "collect.meta.yaml").exists()

    # A re-run of an applied config reproduces its spike trains bit for bit
    report = stages.verify(
        spec, "apply", [task.index], spec.outputs_root.parent / "verify", command=command
    )
    assert list(report["result"]) == ["exact"]
