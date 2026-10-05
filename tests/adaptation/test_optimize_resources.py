"""Tests for adaptation/optimize/'s resource planning (resources.py) and
batched trial loop (workers.py). Planning and checks are pure logic;
end-to-end searches are marked slow.
"""

import numpy as np
import optuna
import psutil
import pytest
from loguru import logger

from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    PROCESS_BASELINE_BYTES,
    ResourcePlan,
    optimize_adapt_decomp,
    plan_resources,
    resources,
    search,
    workers,
)
from adapt_decomp.adaptation.optimize.resources import (
    _dataset_shape,
    _predict_peak_bytes,
    _shard,
    plan_search_resources,
)
from adapt_decomp.adaptation.optimize.scoring import suggest_overrides
from adapt_decomp.adaptation.optimize.search import _make_study
from adapt_decomp.utils.loaders import PooledDatasetMemory, emg_shape
from tests.adaptation.test_optimize import _make_pooled_disk_dataset

GB = 1024**3


def _memory_pool(make_optimize_kwargs, names):
    """In-memory pool of identical synthetic datasets, plus its base config."""
    pool = {}
    for name in names:
        common, _ = make_optimize_kwargs()
        base_config = common.pop("base_config")
        pool[name] = PooledDatasetMemory(**common)
    return pool, base_config


@pytest.mark.parametrize(
    "n_datasets, n_trials_at_once, n_cores, expected",
    [
        (3, 1, 24, ResourcePlan(1, 3)),
        (3, 15, 24, ResourcePlan(8, 3)),
        (3, 4, 24, ResourcePlan(4, 3)),
        (12, 1, 24, ResourcePlan(1, 12)),
        (50, 15, 24, ResourcePlan(1, 24)),
        (3, 1, 1, ResourcePlan(1, 1)),
        (3, 15, 2, ResourcePlan(1, 2)),
    ],
)
def test_plan_resources_fills_cores_with_dataset_runs_first(
    n_datasets, n_trials_at_once, n_cores, expected
):
    assert plan_resources(n_datasets, n_trials_at_once, n_cores) == expected


def test_shard_balances_datasets_longest_first(make_optimize_kwargs):
    pool, _ = _memory_pool(make_optimize_kwargs, ("a", "b", "c"))
    for name, n in (("a", 600), ("b", 400), ("c", 300)):
        pool[name].emg = pool[name].emg.repeat(n // 100 + 1, 1)[:n]
    assert _shard(pool, 2) == [["a"], ["b", "c"]]
    assert _shard(pool, 5) == [["a"], ["b"], ["c"]]


def test_predict_peak_bytes_takes_each_workers_largest_run_per_group():
    run_bytes = {"long": 10 * GB, "short": 2 * GB, "mid": 4 * GB}
    resident = {"long": 1 * GB, "short": 0, "mid": 0}
    peak = _predict_peak_bytes(run_bytes, resident, [["long"], ["short", "mid"]], n_groups=2)
    assert peak == 2 * (2 * PROCESS_BASELINE_BYTES + 1 * GB + 10 * GB + 4 * GB)


def test_emg_shape_reads_the_npz_header(tmp_path):
    path = tmp_path / "rec.npz"
    np.savez(path, emg=np.zeros((1234, 7), dtype=np.float64))
    assert emg_shape(path) == (1234, 7)
    with pytest.raises(ValueError, match="emg_loader"):
        emg_shape(path, "mat")


def test_disk_entry_shape_counts_only_its_start_stop_samples(tmp_path):
    """The memory check sizes a disk entry's run by the samples it adapts."""
    from dataclasses import replace

    entry = _make_pooled_disk_dataset(tmp_path, "a")
    n_samples, *rest = _dataset_shape(entry)
    window = replace(entry, start=10, stop=n_samples - 5)
    assert _dataset_shape(window) == (n_samples - 15, *rest)


def test_disk_and_memory_pools_plan_the_same_runs(tmp_path, make_optimize_kwargs, allow_cores):
    allow_cores(4)
    disk = {"a": _make_pooled_disk_dataset(tmp_path, "a")}
    memory, _ = _memory_pool(make_optimize_kwargs, ("a",))
    plan_disk, _, _ = plan_search_resources(disk, 1, 1, 4)
    plan_memory, _, _ = plan_search_resources(memory, 1, 1, 4)
    assert plan_disk == plan_memory == ResourcePlan(1, 1)


def testplan_search_resources_rejects_bad_counts_and_too_many_cores(make_optimize_kwargs):
    pool, _ = _memory_pool(make_optimize_kwargs, ("a",))
    with pytest.raises(ValueError, match="at least 1"):
        plan_search_resources(pool, 1, 0, 1)
    with pytest.raises(ValueError, match="--cpus-per-task"):
        plan_search_resources(pool, 1, 1, 2)  # conftest makes 1 core available


def test_plan_search_resources_resolves_default_cores_through_available_cores(
    make_optimize_kwargs,
):
    pool, _ = _memory_pool(make_optimize_kwargs, ("a",))
    # n_cores=None resolves to conftest's patched available_cores
    assert plan_search_resources(pool, 1, 1, None)[2] == 1


def testplan_search_resources_guides_the_user_when_memory_does_not_fit(
    make_optimize_kwargs, allow_cores, monkeypatch
):
    allow_cores(8)
    pool, _ = _memory_pool(make_optimize_kwargs, ("a", "b"))
    current = psutil.Process().memory_info().rss
    # Room for exactly one worker: n_cores=1 fits, more does not
    budget = current + PROCESS_BASELINE_BYTES + 10 * 1024**2
    monkeypatch.setattr(resources, "available_memory", lambda: (budget, budget))
    with pytest.raises(ValueError, match=r"n_cores=1 fits\..*lower n_cores.*--mem"):
        plan_search_resources(pool, 4, 1, 8)

    monkeypatch.setattr(resources, "available_memory", lambda: (current, current))
    with pytest.raises(ValueError, match="Even n_cores=1 does not fit"):
        plan_search_resources(pool, 4, 1, 8)


def testplan_search_resources_warns_when_free_memory_is_short_or_batches_wait(
    make_optimize_kwargs, allow_cores, monkeypatch
):
    allow_cores(2)
    pool, _ = _memory_pool(make_optimize_kwargs, ("a",))
    monkeypatch.setattr(resources, "available_memory", lambda: (1000 * GB, 1))
    messages = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        plan, _, _ = plan_search_resources(pool, 4, 4, 2)
    finally:
        logger.remove(handler)
    assert plan == ResourcePlan(2, 1)
    text = " ".join(messages)
    assert "Close other memory-heavy programs" in text
    assert "batches run in waves. Raise n_cores or lower n_jobs" in text


@pytest.mark.slow
def test_batched_start_up_trials_match_a_one_at_a_time_search(make_optimize_kwargs):
    pool, base_config = _memory_pool(make_optimize_kwargs, ("a",))
    seen = []
    optimize_adapt_decomp(
        pool=pool, base_config=base_config, n_trials=4, random_seed=7, on_trial=seen.append
    )

    reference = _make_study(("sv_loss",), None, 7, n_jobs=1)
    expected = []
    for _ in range(4):
        trial = reference.ask()
        expected.append(suggest_overrides(trial, DEFAULT_PARAM_SPACE))
        reference.tell(trial, 1.0)
    assert [log["params"] for log in seen] == expected


@pytest.mark.slow
def test_worker_processes_reproduce_the_in_process_search(make_optimize_kwargs, allow_cores):
    """n_cores changes only where trials run, not what the sampler suggests."""
    allow_cores(4)
    pool, base_config = _memory_pool(make_optimize_kwargs, ("dataset_a", "dataset_b"))

    def run(n_cores):
        seen = []
        optimize_adapt_decomp(
            pool=pool, base_config=base_config, n_trials=3, n_cores=n_cores, on_trial=seen.append
        )
        return sorted(seen, key=lambda log: log["trial_number"])

    for in_process, workers in zip(run(1), run(4)):
        assert workers["params"] == in_process["params"]
        for name in pool:
            assert workers["per_dataset"][name]["sv_loss"] == pytest.approx(
                in_process["per_dataset"][name]["sv_loss"], rel=1e-4
            )


@pytest.mark.slow
def test_failed_trial_is_marked_failed_and_raises(make_optimize_kwargs, monkeypatch):
    pool, base_config = _memory_pool(make_optimize_kwargs, ("a",))

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(workers, "score_dataset", boom)
    studies = []
    real_make_study = search._make_study
    monkeypatch.setattr(
        search, "_make_study", lambda *a: studies.append(real_make_study(*a)) or studies[-1]
    )
    with pytest.raises(RuntimeError, match="boom"):
        optimize_adapt_decomp(pool=pool, base_config=base_config, n_trials=2)
    assert studies[0].trials[0].state == optuna.trial.TrialState.FAIL
