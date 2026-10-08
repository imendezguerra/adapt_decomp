"""Tests for adapt_decomp.utils.system: scheduler-aware cores and memory."""

import pytest

from adapt_decomp.utils import system

ALL_SCHEDULER_VARS = (*system.SCHEDULER_CORE_VARS, "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")


@pytest.fixture
def no_scheduler(monkeypatch):
    """Run as if outside any scheduler, on a 16-core machine."""
    for var in ALL_SCHEDULER_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(system.joblib, "cpu_count", lambda only_physical_cores=False: 16)


def test_available_cores_without_a_scheduler_is_the_physical_cores(no_scheduler):
    assert system.available_cores() == 16


@pytest.mark.parametrize("var", system.SCHEDULER_CORE_VARS)
def test_available_cores_honours_slurm_pbs_pro_and_torque(no_scheduler, monkeypatch, var):
    monkeypatch.setenv(var, "3")
    assert system.available_cores() == 3


def test_available_cores_never_exceeds_the_machine(no_scheduler, monkeypatch):
    monkeypatch.setenv("NCPUS", "64")
    assert system.available_cores() == 16


def test_cgroup_memory_limit_takes_the_tightest_v1_or_v2_limit(tmp_path):
    v2, v1 = tmp_path / "memory.max", tmp_path / "memory.limit_in_bytes"
    v2.write_text("max\n")
    v1.write_text("8589934592\n")
    assert system.cgroup_memory_limit((v2, v1)) == 8589934592  # v2 unlimited, v1 set (PBS Pro)

    v2.write_text("4294967296\n")
    assert system.cgroup_memory_limit((v2, v1)) == 4294967296
    assert system.cgroup_memory_limit((tmp_path / "missing",)) is None


def test_available_memory_is_capped_by_the_cgroup_limit(no_scheduler, monkeypatch):
    monkeypatch.setattr(system, "cgroup_memory_limit", lambda: 1024)
    limit, available = system.available_memory()
    assert limit == 1024
    assert available <= 1024
