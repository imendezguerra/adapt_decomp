"""Tests for benchmarks/fdsi/spec.py and the task runner of stages.py: spec validation, task
enumeration and selection, cache keys, status and the stale/force/skip logic. No real data."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from adapt_decomp.adaptation.optimize import DEFAULT_PARAM_SPACE
from adapt_decomp.utils import read_metadata, write_metadata
from benchmarks.fdsi import fdsi, stages
from benchmarks.fdsi import spec as spec_module
from benchmarks.fdsi.spec import FIXED_BRANCH, load_spec, select_tasks


def _raw_spec(tmp_path: Path) -> dict:
    """A small valid spec over a fake data root: 2 subjects x 2 conditions x 2 SNRs, 2 searches."""
    return {
        "name": "test_run",
        "data_root": str(tmp_path / "data"),
        "outputs_root": str(tmp_path / "outputs"),
        "grid": {
            "subjects": ["sub-01", "sub-02"],
            "conditions": ["triangular-ramp40s", "staircase"],
            "snr_levels": [30, 20],
            "fs": 2048,
            "cal_duration_s": 5.0,
            "iso_duration_s": 5.0,
            "tol_spike_ms": 2.0,
        },
        "calibration": {
            "cbss_config": {"fs": 2048, "ext_fact": 10, "random_seed": 42, "device": "cpu"},
            "supervised_roa_th": 0.9,
        },
        "pool": {"subject": "sub-01", "snr": 30, "conditions": ["triangular-ramp40s"]},
        "search": {
            "base_config": "src/adapt_decomp/adaptation/presets/muniverse.yaml",
            "overrides": {"lr_mode": "fixed", "device": "cpu"},
            "n_trials": 5,
            "n_jobs": 2,
            "random_seed": 42,
        },
        "searches": {
            "sv": {"objectives": ["sv_loss"]},
            "pareto_sum": {
                "objectives": ["wh_loss", "sv_loss"],
                "selection": "min_sv_loss",
                "sv_loss_reduction": "sum",
            },
        },
        "apply": {"fixed_config": "src/adapt_decomp/adaptation/presets/fixed.yaml"},
    }


def _write(tmp_path: Path, raw: dict, name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))  # keep the searches' order
    return path


def _fake_data(spec, seed: int = 0) -> None:
    """Tiny EMG and ground-truth files for every recording (only their bytes matter here)."""
    rng = np.random.default_rng(seed)
    for rec in spec.recordings():
        for path in (spec.emg_path(rec), spec.gt_path(rec)):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, emg=rng.standard_normal((8, 2)))


@pytest.fixture
def spec(tmp_path):
    loaded = load_spec(_write(tmp_path, _raw_spec(tmp_path)))
    _fake_data(loaded)
    return loaded


# Loading and validation


@pytest.mark.parametrize("name", ["benchmark.yaml", "benchmark_smoke.yaml"])
def test_shipped_specs_load(name):
    spec = load_spec(f"benchmarks/fdsi/{name}")
    n_recordings = len(spec.recordings())
    assert len(spec.tasks("calibrate")) == n_recordings
    assert len(spec.tasks("search")) == len(spec.searches) == 5
    assert len(spec.tasks("apply")) == 6 * n_recordings
    assert spec.branches[0] == FIXED_BRANCH


def test_full_spec_is_the_planned_benchmark():
    spec = load_spec("benchmarks/fdsi/benchmark.yaml")
    assert len(spec.recordings()) == 100
    assert spec.search_n_cores == 12  # n_jobs=4 x 3 pooled recordings, one thread per run
    assert spec.cal_end == 5 * 2048
    for name in spec.searches:
        settings = spec.search_settings(name)
        assert settings["unit_selection"] is None
        assert settings["n_trials"] == 50 and settings["n_jobs"] == 4
        assert "centroid_momentum" in settings["param_space"]


def _mutate(raw: dict, path: tuple, value) -> dict:
    """A deep copy of raw with the value at path set (or deleted when value is ...)."""
    raw = copy.deepcopy(raw)
    target = raw
    for key in path[:-1]:
        target = target[key]
    if value is ...:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return raw


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("extra",), 1, "Unknown key"),
        (("grid", "fps"), 1, "Unknown key"),
        (("pool",), ..., "Missing key"),
        (("searches", "sv", "objective"), "sv_loss", "Unknown key"),
        (("searches", "fixed"), {"objectives": ["sv_loss"]}, "Invalid search name"),
        (("searches", "a/b"), {"objectives": ["sv_loss"]}, "Invalid search name"),
        (("pool", "conditions"), ["staircase", "triangular-ramp5s"], "not in the grid"),
        (("pool", "snr"), 15, "not in the grid"),
        (("searches", "sv", "objectives"), ["bogus"], "bogus"),
        (("searches", "sv", "unit_selection"), "bogus", "unit_selection"),
        (("searches", "pareto_sum", "selection"), "bogus", "selection"),
        (("search", "overrides", "not_a_field"), 1, "Unknown AdaptConfig field"),
        (("calibration", "cbss_config", "not_a_field"), 1, "Unknown CBSSConfig field"),
        (("apply", "fixed_config"), "configs/missing.yaml", "not found"),
        (("search", "n_jobs"), 0, "at least 1"),
    ],
)
def test_invalid_specs_raise_naming_the_problem(tmp_path, path, value, match):
    raw = _mutate(_raw_spec(tmp_path), path, value)
    with pytest.raises(ValueError, match=match):
        load_spec(_write(tmp_path, raw))


def test_missing_spec_file_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_spec(tmp_path / "missing.yaml")


# Settings


def test_search_settings_merge_shared_and_own_settings(spec):
    sv, pareto = spec.search_settings("sv"), spec.search_settings("pareto_sum")

    assert sv["param_space"] == DEFAULT_PARAM_SPACE
    assert sv["objectives"] == ("sv_loss",)
    assert sv["n_cores"] == pareto["n_cores"] == 2  # n_jobs=2 x 1 pooled recording
    assert pareto["selection"] == "min_sv_loss"
    assert spec.search_base_config("sv").lr_mode == "fixed"
    assert spec.search_base_config("sv").sv_loss_reduction == "mean"  # the base file's
    assert spec.search_base_config("pareto_sum").sv_loss_reduction == "sum"
    fixed = spec.fixed_config()
    assert fixed.device == "cpu" and not (fixed.adapt_wh or fixed.adapt_sv or fixed.adapt_sd)


# Tasks


def test_tasks_are_enumerated_in_a_fixed_order(spec):
    calibrate, search, apply = (spec.tasks(s) for s in ("calibrate", "search", "apply"))

    assert [t.id for t in calibrate[:2]] == [
        "sub-01_FDSI_triangular-ramp40s_snr30dB",
        "sub-01_FDSI_triangular-ramp40s_snr20dB",
    ]
    assert [t.id for t in search] == ["sv", "pareto_sum"]
    assert len(apply) == 3 * 8
    assert apply[0].id == f"{FIXED_BRANCH}/sub-01_FDSI_triangular-ramp40s_snr30dB"
    assert apply[8].branch == "sv"
    assert [t.index for t in apply] == list(range(len(apply)))
    assert spec.task_for("apply", apply[9].id) == apply[9]
    with pytest.raises(ValueError, match="Unknown stage"):
        spec.tasks("bogus")


def test_select_tasks_by_index_chunk_or_all(spec, monkeypatch):
    for var in ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(var, raising=False)
    tasks = spec.tasks("apply")

    assert select_tasks(tasks, task_index=5) == [tasks[5]]
    assert select_tasks(tasks, run_all=True) == tasks
    assert select_tasks(tasks, array_index=2, chunk_size=4) == tasks[8:12]
    assert select_tasks(tasks, array_index=4, chunk_size=5) == tasks[20:]  # last chunk is short

    monkeypatch.setenv("PBS_ARRAYID", "1")  # Torque, used when PBS Pro's variable is unset
    assert select_tasks(tasks, chunk_size=3) == tasks[3:6]
    monkeypatch.setenv("PBS_ARRAY_INDEX", "0")  # PBS Pro takes precedence
    assert select_tasks(tasks, chunk_size=3) == tasks[0:3]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "No task selected"),
        ({"task_index": 99}, "out of range"),
        ({"array_index": 30}, "beyond"),
        ({"task_index": 1, "array_index": 1}, "only one"),
        ({"task_index": 1, "run_all": True}, "only one"),
        ({"array_index": 0, "chunk_size": 0}, "at least 1"),
    ],
)
def test_select_tasks_rejects_ambiguous_or_out_of_range_selections(
    spec, monkeypatch, kwargs, match
):
    for var in ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match=match):
        select_tasks(spec.tasks("apply"), **kwargs)


# Cache keys


def _keys(spec) -> dict:
    return {
        (t.stage, t.id): spec.task_key(t)
        for s in ("calibrate", "search", "apply")
        for t in spec.tasks(s)
    }


def test_keys_are_stable_and_ignore_where_outputs_go(tmp_path, spec):
    reloaded = load_spec(spec.path)
    moved = spec.with_outputs_root(tmp_path / "elsewhere")
    assert _keys(spec) == _keys(reloaded) == _keys(moved)
    assert len(set(_keys(spec).values())) == len(_keys(spec))  # every task has its own key


def test_changing_a_search_invalidates_only_that_search_and_its_applications(tmp_path, spec):
    raw = _mutate(_raw_spec(tmp_path), ("searches", "sv", "n_trials"), 7)
    changed = {
        k
        for k, v in _keys(load_spec(_write(tmp_path, raw, "b.yaml"))).items()
        if v != _keys(spec)[k]
    }

    assert changed == {("search", "sv")} | {
        ("apply", t.id) for t in spec.tasks("apply") if t.branch == "sv"
    }


def test_changing_the_calibration_invalidates_everything(tmp_path, spec):
    raw = _mutate(_raw_spec(tmp_path), ("calibration", "cbss_config", "random_seed"), 7)
    before, after = _keys(spec), _keys(load_spec(_write(tmp_path, raw, "b.yaml")))
    assert all(before[k] != after[k] for k in before)


@pytest.mark.parametrize(
    "stage, downstream",
    [
        ("calibrate", {"calibrate", "search", "apply"}),
        ("search", {"search", "apply"}),
        ("apply", {"apply"}),
    ],
)
def test_bumping_a_stage_version_invalidates_it_and_everything_downstream(
    spec, monkeypatch, stage, downstream
):
    before = _keys(spec)
    monkeypatch.setitem(spec_module.STAGE_VERSIONS, stage, spec_module.STAGE_VERSIONS[stage] + 1)
    after = _keys(load_spec(spec.path))

    changed_stages = {s for (s, _), key in before.items() if after[(s, _)] != key}
    assert changed_stages == downstream
    if stage == "search":  # the fixed baseline doesn't depend on any search
        assert all(
            before[("apply", t.id)] == after[("apply", t.id)]
            for t in spec.tasks("apply")
            if t.branch == FIXED_BRANCH
        )


def test_changing_an_input_file_invalidates_its_recording_and_the_searches_using_it(spec):
    before = _keys(spec)
    rec = spec.pool_recordings()[0]
    np.savez(spec.emg_path(rec), emg=np.ones((9, 2)))  # new content, size and mtime
    after = _keys(load_spec(spec.path))

    changed = {k for k in before if before[k] != after[k]}
    assert ("calibrate", rec.stub) in changed
    assert {("search", "sv"), ("search", "pareto_sum")} <= changed
    assert ("calibrate", spec.recordings()[1].stub) not in changed
    assert (("apply", f"{FIXED_BRANCH}/{spec.recordings()[1].stub}")) not in changed


def test_missing_input_files_point_to_the_download(spec):
    spec.emg_path(spec.recordings()[0]).unlink()
    with pytest.raises(FileNotFoundError, match=r"adapt-decomp-data get"):
        load_spec(spec.path).task_key(spec.tasks("calibrate")[0])


# Status and the task runner


def _mark(spec, task, key: str, status: str = "done") -> None:
    write_metadata(spec.meta_path(task), {"task": {"key": key, "status": status}})


def test_task_status_missing_done_stale_skipped(spec):
    task = spec.tasks("calibrate")[0]
    assert spec.task_status(task) == ("missing", None)
    _mark(spec, task, spec.task_key(task))
    assert spec.task_status(task)[0] == "done"
    _mark(spec, task, "another key")
    assert spec.task_status(task)[0] == "stale"
    _mark(spec, task, spec.task_key(task), "skipped")
    assert spec.task_status(task)[0] == "skipped"


@pytest.fixture
def fake_calibrate(monkeypatch):
    """Replace the calibrate stage by a counter, to test run_task without real data."""
    calls = []

    def _calibrate(spec, task):
        calls.append(task.id)
        return {"status": "done", "digest": {"n_units": 3}}

    monkeypatch.setitem(stages.STAGE_RUNNERS, "calibrate", _calibrate)
    return calls


def test_run_task_runs_once_then_skips_and_writes_full_metadata(spec, fake_calibrate):
    task = spec.tasks("calibrate")[3]
    command = ["python", "-m", "benchmarks.fdsi", "calibrate", "--all"]

    assert stages.run_task(spec, task, command=command) == "missing"
    assert stages.run_task(spec, task, command=command) == "done"
    assert fake_calibrate == [task.id]

    meta = read_metadata(spec.meta_path(task))
    assert meta["task"]["key"] == spec.task_key(task)
    assert meta["task"]["status"] == "done" and meta["task"]["index"] == 3
    assert meta["digest"] == {"n_units": 3}
    assert meta["command"] == "python -m benchmarks.fdsi calibrate --all"
    reproduce = meta["reproduce"].splitlines()
    assert (
        reproduce[-1]
        == f"python -m benchmarks.fdsi calibrate --spec {spec.spec_ref} --task-index 3"
    )
    assert "adapt-decomp-data get fdsi_benchmark-data" in reproduce
    for section in ("started_at", "run_time_s", "host", "os", "hardware", "python", "packages"):
        assert section in meta


def test_run_task_refuses_a_stale_output_unless_forced(spec, fake_calibrate):
    task = spec.tasks("calibrate")[0]
    _mark(spec, task, "made from other settings")

    with pytest.raises(ValueError, match="--force"):
        stages.run_task(spec, task, command=["python"])
    assert stages.run_task(spec, task, command=["python"], force=True) == "stale"
    assert spec.task_status(task)[0] == "done"


def test_apply_requires_its_upstream_and_skips_when_the_calibration_was_skipped(spec):
    rec = spec.recordings()[0]
    fixed, adapted = (
        spec.task_for("apply", f"{FIXED_BRANCH}/{rec.stub}"),
        spec.task_for("apply", f"sv/{rec.stub}"),
    )
    with pytest.raises(ValueError, match="Upstream calibrate"):
        stages.apply(spec, fixed)

    cal_task = spec.task_for("calibrate", rec.stub)
    _mark(spec, cal_task, spec.task_key(cal_task), "skipped")
    assert stages.apply(spec, fixed)["status"] == "skipped"
    with pytest.raises(ValueError, match="Upstream search"):
        stages.apply(spec, adapted)


def test_paths_and_command_of_a_task(spec):
    rec = spec.recordings()[0]
    paths = spec.result_paths("sv", rec)
    assert paths["result"] == spec.outputs_root / "results" / "sv" / "sub-01" / f"{rec.stub}.pkl"
    assert spec.meta_path(spec.task_for("apply", f"sv/{rec.stub}")) == paths["meta"]
    assert spec.search_paths("sv")["meta"] == spec.outputs_root / "searches" / "sv.meta.yaml"
    assert fdsi.recording_stub(rec.sub, rec.cond, rec.snr) == rec.stub
