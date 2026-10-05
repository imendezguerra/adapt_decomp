"""Resources: cores, memory and how a search's runs map onto them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from loguru import logger

from adapt_decomp.utils.loaders import PooledDataset, PooledDatasetMemory, emg_shape, load_calib
from adapt_decomp.utils.system import available_cores, available_memory

_GB = 1024**3

# Memory of one worker process before any run: Python, torch and adapt_decomp (measured 0.48 GB)
PROCESS_BASELINE_BYTES: int = 512 * 1024**2

# Peak memory of one run is the larger of building the model and preprocessing the recording.
# Both factors are properties of the code, not of the dataset (measured on FDSI recordings).
# Building: the calibration windows Decomposition stacks at once (32 x fifo x D float32)
MODEL_BUILD_OVERHEAD: float = 4.8
# Preprocessing: the extended recording (samples x channels x ext_fact float32)
RUN_OVERHEAD: float = 2.0
_KL_CAL_WINDOWS = 32  # Decomposition._compute_mean_sigma_kl_cal's chunk of windows

_MEMORY_GUIDANCE = (
    "To fit, lower n_cores (fewer runs at once), use a smaller pool or shorter recordings, "
    "use a disk pool (PooledDatasetDisk) so workers do not keep datasets resident, or "
    "request more memory (e.g. SLURM --mem or PBS -l select=1:mem=...)."
)


@dataclass(frozen=True)
class ResourcePlan:
    """How a search's runs map onto worker processes.

    Attributes:
        n_groups (int): Trials that can run at once, one copy of the pool each.
        workers_per_group (int): Worker processes each pool copy is sharded over.
    """

    n_groups: int
    workers_per_group: int


def plan_resources(n_datasets: int, n_trials_at_once: int, n_cores: int) -> ResourcePlan:
    """Fill the cores with dataset runs first; leftover cores become torch threads per run.

    Args:
        n_datasets (int): Datasets in the pool, each run once per trial.
        n_trials_at_once (int): Most trials ever run together.
        n_cores (int): Cores the search may use.

    Returns:
        ResourcePlan: At most n_cores // n_datasets groups, each with up to
        n_datasets workers.
    """
    n_groups = max(1, min(n_trials_at_once, n_cores // n_datasets))
    workers_per_group = max(1, min(n_datasets, n_cores // n_groups))
    return ResourcePlan(n_groups, workers_per_group)


def _nbytes(value: Any) -> int:
    """Memory held by an array or tensor (its whole storage, for a tensor view).

    Args:
        value (Any): Any object.

    Returns:
        int: Bytes for a numpy array or torch tensor, else 0.
    """
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    return 0


def _dataset_shape(dataset: PooledDataset) -> Tuple[int, int, int, int]:
    """(samples adapted, channels, ext_fact, units) of a pool entry, without loading a disk entry's EMG.

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        Tuple[int, int, int, int]: Its shape, with every calibration unit counted.
    """
    if isinstance(dataset, PooledDatasetMemory):
        calibration, cbss_config = dataset.calibration, dataset.cbss_config
        n_samples, n_channels = dataset.emg.shape
    else:
        calibration, cbss_config = load_calib(
            dataset.path_calib, dataset.path_calib_config, dataset.calib_loader
        )
        n_samples, n_channels = emg_shape(dataset.path_emg, dataset.emg_loader)
        n_samples = len(range(n_samples)[dataset.start : dataset.stop])  # the samples adapted
    return int(n_samples), int(n_channels), int(cbss_config.ext_fact), calibration.sources.shape[1]


def _run_bytes(
    n_samples: int, n_channels: int, ext_fact: int, n_units: int, fifo_length: Optional[int]
) -> int:
    """Peak memory of one run of a dataset with this shape.

    Args:
        n_samples (int): Samples adapted.
        n_channels (int): EMG channels.
        ext_fact (int): Extension factor.
        n_units (int): Units adapted.
        fifo_length (Optional[int]): AdaptConfig.fifo_length, None for the default 2 x D.

    Returns:
        int: Bytes.
    """
    n_dims = n_channels * ext_fact
    fifo_samples = max(n_dims, fifo_length if fifo_length else 2 * n_dims)
    model_build = MODEL_BUILD_OVERHEAD * _KL_CAL_WINDOWS * fifo_samples * n_dims * 4
    preprocessing = RUN_OVERHEAD * n_samples * n_dims * 4
    return int(max(model_build, preprocessing)) + n_samples * n_units * 8


def _resident_bytes(dataset: PooledDataset) -> int:
    """Memory a worker keeps for a pool entry for the whole search.

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        int: Bytes of its EMG, ground truth and calibration arrays; 0 for a
        disk entry, which is loaded per run.
    """
    if not isinstance(dataset, PooledDatasetMemory):
        return 0
    calibration_bytes = sum(_nbytes(v) for v in vars(dataset.calibration).values())
    return _nbytes(dataset.emg) + _nbytes(dataset.gt_paired_bin) + calibration_bytes


def _predict_peak_bytes(
    run_bytes: Dict[str, int],
    resident_bytes: Dict[str, int],
    shards: List[List[str]],
    n_groups: int,
) -> int:
    """Peak memory of all workers, each running the largest dataset of its shard.

    Args:
        run_bytes (Dict[str, int]): Dataset name -> peak memory of one run.
        resident_bytes (Dict[str, int]): Dataset name -> memory kept by its worker.
        shards (List[List[str]]): Dataset names per worker of one group.
        n_groups (int): Worker groups, each holding a copy of the pool.

    Returns:
        int: Bytes.
    """
    per_group = sum(
        PROCESS_BASELINE_BYTES
        + sum(resident_bytes[name] for name in shard)
        + max(run_bytes[name] for name in shard)
        for shard in shards
    )
    return n_groups * per_group


def _dataset_size(dataset: PooledDataset) -> int:
    """Rough cost of running one pool entry, to balance shards.

    Args:
        dataset (PooledDataset): Pool entry.

    Returns:
        int: Its EMG sample count (in memory) or file size in bytes (on disk).
    """
    if isinstance(dataset, PooledDatasetMemory):
        return len(dataset.emg)
    return Path(dataset.path_emg).stat().st_size


def _shard(pool: Dict[str, PooledDataset], n_workers: int) -> List[List[str]]:
    """Split pool's dataset names over n_workers shards, longest datasets first.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        n_workers (int): Number of shards (capped at len(pool)).

    Returns:
        List[List[str]]: Dataset names per shard, balanced by _dataset_size.
    """
    shards: List[List[str]] = [[] for _ in range(max(1, min(n_workers, len(pool))))]
    loads = [0] * len(shards)
    for name in sorted(pool, key=lambda n: _dataset_size(pool[n]), reverse=True):
        i = loads.index(min(loads))
        shards[i].append(name)
        loads[i] += _dataset_size(pool[name])
    return shards


def plan_search_resources(
    pool: Dict[str, PooledDataset],
    n_trials_at_once: int,
    n_jobs: int,
    n_cores: Optional[int],
    fifo_length: Optional[int] = None,
) -> Tuple[ResourcePlan, List[List[str]], int]:
    """Plan the workers for a search and check they fit, before any worker starts.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        n_trials_at_once (int): Most trials ever run together.
        n_jobs (int): Trials suggested together after the start-up trials.
        n_cores (Optional[int]): Cores the search may use; None for all
            available (available_cores()).
        fifo_length (Optional[int], optional): The base config's fifo_length.
            Defaults to None (2 x D).

    Raises:
        ValueError: If n_jobs or n_cores is below 1, n_cores exceeds the
            available cores, or the predicted peak memory exceeds the limit.

    Returns:
        Tuple[ResourcePlan, List[List[str]], int]: The plan, its shards and
        n_cores (resolved when None).
    """
    cores = available_cores()
    n_cores = n_cores if n_cores is not None else cores
    if n_jobs < 1 or n_cores < 1:
        raise ValueError(
            f"n_jobs and n_cores must be at least 1, got n_jobs={n_jobs}, n_cores={n_cores}."
        )
    if n_cores > cores:
        raise ValueError(
            f"n_cores={n_cores} exceeds the {cores} physical cores available to this process. "
            "Lower n_cores, or request more CPUs (e.g. SLURM --cpus-per-task or PBS "
            "-l select=1:ncpus=...)."
        )

    # Memory per dataset, from each dataset's own shape
    run_bytes = {
        name: _run_bytes(*_dataset_shape(dataset), fifo_length) for name, dataset in pool.items()
    }
    resident_bytes = {name: _resident_bytes(dataset) for name, dataset in pool.items()}

    def peak(cores_used: int) -> Tuple[ResourcePlan, List[List[str]], int]:
        plan = plan_resources(len(pool), n_trials_at_once, cores_used)
        shards = _shard(pool, plan.workers_per_group)
        return plan, shards, _predict_peak_bytes(run_bytes, resident_bytes, shards, plan.n_groups)

    # Compare the predicted peak with the hard limit and with what is free now
    plan, shards, peak_bytes = peak(n_cores)
    limit, available = available_memory()
    budget = limit - psutil.Process().memory_info().rss
    if peak_bytes > budget:
        fitting = [c for c in range(n_cores - 1, 0, -1) if peak(c)[2] <= budget]
        suggestion = f"n_cores={fitting[0]} fits. " if fitting else "Even n_cores=1 does not fit. "
        raise ValueError(
            f"Predicted peak memory {peak_bytes / _GB:.1f} GB exceeds the {budget / _GB:.1f} GB "
            f"left under the {limit / _GB:.1f} GB limit. {suggestion}{_MEMORY_GUIDANCE}"
        )
    if peak_bytes > available:
        logger.warning(
            f"Predicted peak memory {peak_bytes / _GB:.1f} GB exceeds the {available / _GB:.1f} GB "
            f"currently free, so the search may swap or be killed. Close other memory-heavy "
            f"programs. {_MEMORY_GUIDANCE}"
        )
    if n_jobs > plan.n_groups:
        logger.warning(
            f"n_jobs={n_jobs} trials per batch but only {plan.n_groups} fit at once on "
            f"n_cores={n_cores}, so batches run in waves. Raise n_cores or lower n_jobs "
            "(the search is the same either way, only slower)."
        )
    logger.info(
        f"{plan.n_groups} trial(s) at once x {plan.workers_per_group} worker(s) on "
        f"{n_cores} cores; predicted peak memory {peak_bytes / _GB:.1f} GB of "
        f"{available / _GB:.1f} GB free"
    )
    return plan, shards, n_cores
