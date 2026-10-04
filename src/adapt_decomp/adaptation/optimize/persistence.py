"""Best results on disk: promoting and evicting front members, and study snapshots."""

from __future__ import annotations

import pickle
import shutil
import threading
from pathlib import Path
from typing import Iterable

import optuna

from adapt_decomp.adaptation.config import AdaptConfig


def promote_trial(
    temp_dir: Path,
    member_dir: Path,
    trial_number: int,
    dataset_names: Iterable[str],
    trial_config: AdaptConfig,
) -> None:
    """Copy one trial's staged per-dataset results + config into member_dir.

    Args:
        temp_dir (Path): Directory holding this trial's
            "<trial_number>_<dataset>.pkl" scratch files.
        member_dir (Path): Destination: best_result_path itself for a
            single-objective search, or its "trial_<trial_number>"
            subdirectory for a Pareto front member.
        trial_number (int): This trial's Optuna trial.number.
        dataset_names (Iterable[str]): Every dataset name to copy over.
        trial_config (AdaptConfig): This trial's resolved configuration,
            written alongside the results as "config.yaml".

    Returns:
        None
    """
    member_dir.mkdir(parents=True, exist_ok=True)
    for dataset in dataset_names:
        shutil.copy2(temp_dir / f"{trial_number}_{dataset}.pkl", member_dir / f"{dataset}.pkl")
    trial_config.to_yaml(member_dir / "config.yaml")


def evict_front_member(best_dir: Path, trial_number: int) -> None:
    """Delete a previously-saved front member's subdirectory.

    Called when a later trial dominates a resident member -- ignore_errors
    so a member that was never actually saved (shouldn't happen, but not
    worth crashing the search over) is a no-op rather than an exception.

    Args:
        best_dir (Path): Front's root directory.
        trial_number (int): The dominated trial's Optuna trial.number.

    Returns:
        None
    """
    shutil.rmtree(best_dir / f"trial_{trial_number}", ignore_errors=True)


def save_study_snapshot(best_dir: Path, study: optuna.Study, lock: threading.Lock) -> None:
    """Pickle study to best_dir/"study.pkl", overwriting any previous snapshot.

    Called after every told trial (see run_trials), so a crashed run's
    study.pkl still reflects every trial that finished before the crash. Study/InMemoryStorage are
    picklable mid-run by design (strip thread-local/lock state in
    __getstate__, rebuild it in __setstate__).

    Args:
        best_dir (Path): Directory to write "study.pkl" into, already
            created by the caller.
        study (optuna.Study): The study so far.
        lock (threading.Lock): Guards the write against concurrent writers.

    Returns:
        None
    """
    with lock:
        with open(best_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)
