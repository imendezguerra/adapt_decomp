"""Tests for adapt_decomp.utils.system: scheduler-aware cores and memory, and describe_system()."""

import platform

import pytest
import yaml

import adapt_decomp
from adapt_decomp.utils import system

ALL_SCHEDULER_VARS = (
    *system.SCHEDULER_CORE_VARS,
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "PBS_JOBID",
    "PBS_ARRAY_INDEX",
    "PBS_ARRAYID",
)


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


def test_describe_system_has_every_section_and_is_yaml_safe(no_scheduler):
    description = system.describe_system()

    assert set(description) == {"host", "os", "hardware", "python", "packages"}
    assert set(description["hardware"]) == {"cpu", "gpu", "memory"}
    assert isinstance(description["hardware"]["cpu"]["model"], str)
    assert description["hardware"]["cpu"]["model"]
    assert description["hardware"]["memory"]["total_gb"] > 0
    assert description["python"]["version"] == platform.python_version()
    assert description["packages"]["adapt_decomp"] == adapt_decomp.__version__
    assert description["host"]["scheduler"] is None
    yaml.safe_dump(description)  # every value is a plain YAML type


def test_describe_system_reads_the_pbs_job(no_scheduler, monkeypatch):
    monkeypatch.setenv("PBS_JOBID", "1234[7].pbs")
    monkeypatch.setenv("PBS_ARRAY_INDEX", "7")
    monkeypatch.setenv("NCPUS", "12")

    host = system.describe_system()["host"]

    assert host["scheduler"] == "pbs"
    assert host["job_id"] == "1234[7].pbs"
    assert host["array_index"] == "7"
    assert host["ncpus"] == "12"
