"""System queries and settings: the cores and memory this process may use (scheduler- and
cgroup-aware), and deterministic CUDA kernels for seeded runs."""

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import joblib
import psutil
import torch

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


# 8 x 4 MiB cuBLAS workspaces: deterministic CUDA matmuls (PyTorch's recommended value)
CUBLAS_WORKSPACE = ":4096:8"


def set_cublas_workspace() -> None:
    """Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA matmuls, unless the user set it.

    PyTorch reads it once, when the process first uses cuBLAS, so this runs on
    `import adapt_decomp`. Warns if CUDA was initialised before, as it may then be too late.
    """
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        warnings.warn(
            "CUDA was initialised before adapt_decomp was imported, so seeded GPU runs may not "
            f"repeat: import adapt_decomp first or set CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE}.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE


@contextmanager
def deterministic_algorithms(enabled: bool) -> Iterator[None]:
    """Use torch's deterministic algorithms inside the block, then restore the previous mode.

    Args:
        enabled (bool): Whether to switch them on; False leaves torch untouched.

    Yields:
        None
    """
    if not enabled:
        yield
        return
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    # warn_only: an op with no deterministic CUDA kernel warns, naming itself, instead of failing
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2:]
