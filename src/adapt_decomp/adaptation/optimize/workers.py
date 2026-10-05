"""Worker processes, each holding a shard of the pool for the whole search, and
the batch loop that runs trials on them.
"""

from __future__ import annotations

import multiprocessing
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna
import torch
from loguru import logger

from adapt_decomp.adaptation.optimize.resources import ResourcePlan
from adapt_decomp.adaptation.optimize.scoring import score_dataset
from adapt_decomp.utils.loaders import PooledDataset

_WORKER_POOL: Dict[str, PooledDataset] = {}


def _init_worker(shard: Dict[str, PooledDataset]) -> None:
    """Worker initializer: keep this worker's shard resident.

    Args:
        shard (Dict[str, PooledDataset]): The pool entries this worker runs.

    Returns:
        None
    """
    _WORKER_POOL.update(shard)


def _score_in_worker(
    name: str, overrides: dict, stage_path: Optional[Path], settings: dict, n_threads: int
) -> Dict[str, Any]:
    """score_dataset on a pool entry resident in this worker, with n_threads torch threads.

    Args:
        name (str): Pool key of the entry, which must be in this worker's shard.
        overrides (dict): This trial's suggested parameter overrides.
        stage_path (Optional[Path]): See score_dataset.
        settings (dict): score_dataset's keyword-only arguments.
        n_threads (int): Torch threads for this run.

    Returns:
        Dict[str, Any]: See score_dataset.
    """
    torch.set_num_threads(n_threads)
    return score_dataset(_WORKER_POOL[name], overrides, stage_path, **settings)


def start_workers(
    pool: Dict[str, PooledDataset], shards: List[List[str]]
) -> Dict[str, ProcessPoolExecutor]:
    """Start one single-process executor per shard, each holding its shard for the whole search.

    Each shard is sent to its worker once, at start-up; trials then only
    send overrides.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        shards (List[List[str]]): Dataset names per worker, from _shard.

    Returns:
        Dict[str, ProcessPoolExecutor]: Dataset name -> the executor owning it.
    """
    context = multiprocessing.get_context("spawn")
    owners = {}
    for shard in shards:
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=_init_worker,
            initargs=({name: pool[name] for name in shard},),
        )
        owners.update(dict.fromkeys(shard, executor))
    return owners


def score_pool(
    pool: Dict[str, PooledDataset],
    group: Optional[Dict[str, ProcessPoolExecutor]],
    overrides: dict,
    stage_dir: Optional[Path],
    trial_number: int,
    settings: dict,
    n_threads: int,
) -> Dict[str, Dict[str, Any]]:
    """Score one trial's overrides on every dataset, in pool order.

    Args:
        pool (Dict[str, PooledDataset]): Dataset name -> pool entry.
        group (Optional[Dict[str, ProcessPoolExecutor]]): One worker group,
            dataset name -> its owner executor (from start_workers), or None
            to run in-process.
        overrides (dict): This trial's suggested parameter overrides.
        stage_dir (Optional[Path]): Scratch directory for this trial's
            "<trial_number>_<dataset>.pkl" outputs, or None.
        trial_number (int): This trial's Optuna trial.number, unique even
            when trials run concurrently, so staged files never collide.
        settings (dict): score_dataset's keyword-only arguments.
        n_threads (int): Torch threads per run.

    Returns:
        Dict[str, Dict[str, Any]]: Dataset name -> its losses.
    """

    def stage_path(name: str) -> Optional[Path]:
        return stage_dir / f"{trial_number}_{name}.pkl" if stage_dir is not None else None

    if group is None:
        torch.set_num_threads(n_threads)
        return {
            name: score_dataset(dataset, overrides, stage_path(name), **settings)
            for name, dataset in pool.items()
        }
    futures = {
        name: group[name].submit(
            _score_in_worker, name, overrides, stage_path(name), settings, n_threads
        )
        for name in pool
    }
    return {name: future.result() for name, future in futures.items()}


def run_trials(
    study: optuna.Study,
    objective: Callable[[optuna.trial.Trial, int, int], Union[float, Tuple[float, ...]]],
    n_trials: int,
    n_jobs: int,
    n_startup: int,
    n_cores: int,
    plan: ResourcePlan,
    callbacks: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]],
) -> None:
    """Run trials in batches: ask a batch, run it on free worker groups, tell it in trial order.

    The sampler's random start-up trials form one batch, the rest batches of
    n_jobs; a batch larger than plan.n_groups runs in waves. The suggested
    parameters therefore depend on n_jobs and the seed, not on n_cores.

    Args:
        study (optuna.Study): The study to fill.
        objective (Callable): (trial, group index, torch threads per run)
            -> the trial's objective value(s).
        n_trials (int): Trials to run.
        n_jobs (int): Trials suggested together after the start-up trials.
        n_startup (int): The sampler's random start-up trials, 0 if unknown.
        n_cores (int): Cores the search may use.
        plan (ResourcePlan): Worker groups to run on.
        callbacks (List[Callable]): Called with (study, trial) after each tell.

    Returns:
        None
    """
    free_groups: queue.Queue = queue.Queue()
    for group in range(plan.n_groups):
        free_groups.put(group)

    def run(trial: optuna.trial.Trial, n_threads: int) -> Union[float, Tuple[float, ...]]:
        group = free_groups.get()
        try:
            return objective(trial, group, n_threads)
        finally:
            free_groups.put(group)

    n_done = 0
    with ThreadPoolExecutor(max_workers=plan.n_groups) as executor:
        while n_done < n_trials:
            # Batch size from the search, threads from the cores left per running trial
            size = min(n_startup - n_done if n_done < n_startup else n_jobs, n_trials - n_done)
            running = min(size, plan.n_groups)
            n_threads = max(1, n_cores // (running * plan.workers_per_group))

            # Ask the batch, run it, tell it in trial order
            trials = [study.ask() for _ in range(size)]
            futures = [executor.submit(run, trial, n_threads) for trial in trials]
            for trial, future in zip(trials, futures):
                try:
                    values = future.result()
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise
                frozen = study.tell(trial, values)
                logger.info(
                    f"Trial {frozen.number} finished with values {frozen.values} and "
                    f"parameters {frozen.params}"
                )
                for callback in callbacks:
                    callback(study, frozen)
            n_done += size
