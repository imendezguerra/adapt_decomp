"""System queries: the cores and memory this process may use, scheduler- and cgroup-aware."""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import psutil

_GB = 1024**3

# Cores granted by the scheduler: SLURM, PBS Pro, Torque
SCHEDULER_CORE_VARS: Tuple[str, ...] = ("SLURM_CPUS_PER_TASK", "NCPUS", "PBS_NUM_PPN")

# Memory limit of the job's or container's cgroup: v2, then v1 (e.g. PBS Pro's cgroup hook)
CGROUP_MEMORY_LIMIT_FILES: Tuple[Path, ...] = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)


def available_cores() -> int:
    """Physical cores this process may use, honouring CPU affinity, containers and schedulers.

    Returns:
        int: Usable physical cores, at least 1, capped by SLURM_CPUS_PER_TASK,
        NCPUS (PBS Pro) or PBS_NUM_PPN (Torque) when set.
    """
    cores = joblib.cpu_count(only_physical_cores=True)
    for var in SCHEDULER_CORE_VARS:
        value = os.environ.get(var)
        if value:
            cores = min(cores, int(value))
    return max(1, cores)


def cgroup_memory_limit(files: Sequence[Path] = CGROUP_MEMORY_LIMIT_FILES) -> Optional[int]:
    """The tightest cgroup memory limit among files, in bytes.

    Args:
        files (Sequence[Path], optional): Limit files to read. Defaults to
            CGROUP_MEMORY_LIMIT_FILES (cgroup v2, then v1).

    Returns:
        Optional[int]: The smallest limit found, or None if no file sets one
        ("max" is cgroup v2's no-limit value).
    """
    limits = []
    for path in files:
        if path.exists():
            value = path.read_text().strip()
            if value and value != "max":
                limits.append(int(value))
    return min(limits) if limits else None


def available_memory() -> Tuple[int, int]:
    """Memory limit and currently free memory for this process, in bytes.

    The limit is the SLURM job's or the cgroup's (v2 or v1, as set by
    containers and PBS Pro) when set, else the machine's total RAM.

    Returns:
        Tuple[int, int]: (limit, available), available never above limit.
    """
    vm = psutil.virtual_memory()
    limit = vm.total
    if "SLURM_MEM_PER_NODE" in os.environ:
        limit = min(limit, int(os.environ["SLURM_MEM_PER_NODE"]) * 1024**2)
    elif "SLURM_MEM_PER_CPU" in os.environ:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        limit = min(limit, int(os.environ["SLURM_MEM_PER_CPU"]) * cpus * 1024**2)
    cgroup = cgroup_memory_limit()
    if cgroup is not None:
        limit = min(limit, cgroup)
    return limit, min(limit, vm.available)
