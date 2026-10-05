"""System queries: usable cores and memory (scheduler-aware) and a description of the machine."""

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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

# Packages whose versions are recorded with every result
RECORDED_PACKAGES: Tuple[str, ...] = (
    "adapt_decomp",
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "optuna",
    "scikit-learn",
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


def cpu_model() -> str:
    """The CPU's brand string, e.g. "Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz".

    Returns:
        str: From /proc/cpuinfo (Linux), the registry (Windows) or sysctl
        (macOS), falling back to platform.processor() or the machine type.
    """
    system = platform.system()
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    elif system == "Windows":
        import winreg

        # A missing key only means the fallback below applies
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif system == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def _scheduler_info() -> Dict[str, Any]:
    """The batch scheduler this process runs under, with its job and array ids.

    Returns:
        Dict[str, Any]: scheduler ("pbs", "slurm" or None), job_id,
        array_index and ncpus (None where not set).
    """
    env = os.environ
    if "PBS_JOBID" in env:
        return {
            "scheduler": "pbs",
            "job_id": env["PBS_JOBID"],
            "array_index": env.get("PBS_ARRAY_INDEX", env.get("PBS_ARRAYID")),
            "ncpus": env.get("NCPUS", env.get("PBS_NUM_PPN")),
        }
    if "SLURM_JOB_ID" in env:
        return {
            "scheduler": "slurm",
            "job_id": env["SLURM_JOB_ID"],
            "array_index": env.get("SLURM_ARRAY_TASK_ID"),
            "ncpus": env.get("SLURM_CPUS_PER_TASK"),
        }
    return {"scheduler": None, "job_id": None, "array_index": None, "ncpus": None}


def _package_versions(packages: Sequence[str] = RECORDED_PACKAGES) -> Dict[str, Optional[str]]:
    """Installed versions of packages, None for those not installed.

    Args:
        packages (Sequence[str], optional): Distribution names. Defaults to
            RECORDED_PACKAGES.

    Returns:
        Dict[str, Optional[str]]: name -> version; adapt_decomp's comes from
        its own __version__, which an editable install's metadata can lag.
    """
    import adapt_decomp

    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        if name == "adapt_decomp":
            versions[name] = adapt_decomp.__version__
            continue
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def describe_system() -> Dict[str, Any]:
    """Describe the host, OS, hardware, Python and key packages of this process.

    Returns:
        Dict[str, Any]: YAML-safe sections "host", "os", "hardware" (cpu, gpu,
        memory), "python" and "packages" (versions plus torch's BLAS backends
        and the thread-count environment variables).
    """
    # Memory
    vm = psutil.virtual_memory()
    limit, available = available_memory()

    # GPUs
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"name": props.name, "memory_gb": round(props.total_memory / _GB, 2)})

    return {
        "host": {"hostname": platform.node(), **_scheduler_info()},
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "hardware": {
            "cpu": {
                "model": cpu_model(),
                "architecture": platform.machine(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "cores_available": available_cores(),
                "torch_threads": torch.get_num_threads(),
            },
            "gpu": {
                "devices": gpus,
                "cuda_version": torch.version.cuda,
                "mps_available": torch.backends.mps.is_available(),
            },
            "memory": {
                "total_gb": round(vm.total / _GB, 2),
                "limit_gb": round(limit / _GB, 2),
                "available_gb": round(available / _GB, 2),
            },
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.path.realpath(sys.executable),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "packages": {
            **_package_versions(),
            "torch_mkl": torch.backends.mkl.is_available(),
            "torch_openmp": torch.backends.openmp.is_available(),
            "thread_env": {
                var: os.environ.get(var)
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    }
