"""Benchmark stages. Each task calls the adapt_decomp API directly (CBSS, optimize_adapt_decomp,
AdaptDecomp), so reading a stage top to bottom shows how that step is done with the library;
run_task adds the caching and the metadata around it.
"""

import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from loguru import logger

from adapt_decomp import CBSS, AdaptationResult, AdaptDecomp, CBSSConfig, CBSSResult
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import SELECTION_RULES, optimize_adapt_decomp
from adapt_decomp.spikes import get_sil
from adapt_decomp.utils import (
    build_metadata,
    load_gt,
    load_pooled_cbss_memory,
    read_metadata,
    write_metadata,
)
from benchmarks.fdsi import fdsi
from benchmarks.fdsi.spec import (
    FIXED_BRANCH,
    REPO_ROOT,
    THREADS_PER_RUN,
    BenchmarkSpec,
    Recording,
    Task,
    file_sha256,
)

# Lines that set up a fresh checkout before a task's own command
REPRODUCE_SETUP = (
    "conda env create -f environment.yaml",
    "conda activate adapt_decomp",
    "pip install -e .",
    "adapt-decomp-data get fdsi_benchmark-data",
)

# v1.0's applied configs (lr_mode="fixed"): branch -> search dir, None for the fixed baseline
V10_BRANCHES: Dict[str, Optional[str]] = {
    "fixed": None,
    "sv_loss": "tpe_sv_median",
    "pareto": "tpe_pareto",
    "roa": "tpe_roa",
}

# Relative tolerance of verify on search trial values when they are not bit-identical
VERIFY_RTOL = 1e-6


# Shared helpers


def _now() -> datetime:
    """The current local time, timezone-aware."""
    return datetime.now().astimezone()


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file through a temporary name, so a reader never sees a partial file.

    Args:
        path (Path): Destination.
        write (Callable[[Path], None]): Writes the content to the path it is given.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def _remove(paths: Sequence[Path]) -> None:
    """Delete files or directories that exist (outputs a rerun replaces).

    Args:
        paths (Sequence[Path]): Files or directories.

    Returns:
        None
    """
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _require_upstream(spec: BenchmarkSpec, tasks: Sequence[Task]) -> List[str]:
    """Check every upstream task is done (or skipped) with the key this spec expects.

    Args:
        spec (BenchmarkSpec): The spec.
        tasks (Sequence[Task]): The upstream tasks.

    Raises:
        ValueError: If one is missing or stale, naming the stage to run first.

    Returns:
        List[str]: Their statuses ("done" or "skipped"), in order.
    """
    statuses = []
    for task in tasks:
        status, _ = spec.task_status(task)
        if status not in ("done", "skipped"):
            raise ValueError(
                f"Upstream {task.stage} task {task.id!r} is {status}; run "
                f"'python -m benchmarks.fdsi {task.stage} --spec {spec.spec_ref}' first."
            )
        statuses.append(status)
    return statuses


def _sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def _float_or_none(value: Any) -> Optional[float]:
    """A tensor/array scalar or float as a float, None stays None."""
    return None if value is None else float(value)


# Task runner


def run_task(
    spec: BenchmarkSpec, task: Task, *, command: Sequence[str], force: bool = False
) -> str:
    """Run one task unless its output is current, then write its metadata.

    Args:
        spec (BenchmarkSpec): The spec.
        task (Task): The task.
        command (Sequence[str]): The command as invoked, recorded in the metadata.
        force (bool, optional): Recompute a stale output. Defaults to False.

    Raises:
        ValueError: If the output is stale and force is False.

    Returns:
        str: The status before running: "done"/"skipped" (nothing run),
        "missing" or "stale" (computed).
    """
    status, _ = spec.task_status(task)
    if status in ("done", "skipped"):
        logger.info(f"{task.stage} {task.id}: {status}, nothing to do")
        return status
    if status == "stale" and not force:
        raise ValueError(
            f"{task.stage} {task.id}: its output was made from other settings or inputs. "
            "Pass --force to recompute it."
        )

    # Run the stage and record it on THREADS_PER_RUN threads, restoring the caller's afterwards
    key = spec.task_key(task)
    logger.info(f"{task.stage} {task.id}: running ({status})")
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(THREADS_PER_RUN)
    try:
        started = _now()
        outcome = STAGE_RUNNERS[task.stage](spec, task)
        _write_task_metadata(spec, task, key, outcome, started, _now(), command)
    finally:
        torch.set_num_threads(torch_threads)
    return status


def _write_task_metadata(
    spec: BenchmarkSpec,
    task: Task,
    key: str,
    outcome: Dict[str, Any],
    started: datetime,
    finished: datetime,
    command: Sequence[str],
) -> None:
    """Record when, where and from what a task ran, and how to redo it.

    Args:
        spec (BenchmarkSpec): The spec.
        task (Task): The task.
        key (str): Its cache key.
        outcome (Dict[str, Any]): Its stage's metadata sections, with "status".
        started (datetime): When it started.
        finished (datetime): When it finished.
        command (Sequence[str]): The command as invoked.

    Returns:
        None
    """
    reproduce = [*REPRODUCE_SETUP]
    if task.stage != "calibrate":
        reproduce.append("# its upstream outputs (see inputs.upstream) must exist first")
    reproduce.append(spec.task_command(task))
    sections = dict(outcome)
    metadata = build_metadata(
        command=command,
        started=started,
        finished=finished,
        run_name=f"{spec.name}-{task.stage}-{task.index}",
        reproduce=reproduce,
        repo_dir=REPO_ROOT,
        patch_dir=spec.patch_dir,
        extra={
            "task": {
                "stage": task.stage,
                "index": task.index,
                "id": task.id,
                "key": key,
                "status": sections.pop("status"),
                "spec": spec.spec_ref,
            },
            **sections,
        },
    )
    write_metadata(spec.meta_path(task), metadata)


def _run_task_in_worker(
    spec: BenchmarkSpec, task: Task, command: Sequence[str], force: bool
) -> str:
    """run_task for a joblib worker process (keyword-only arguments don't pickle by position)."""
    return run_task(spec, task, command=command, force=force)


def run_tasks(
    spec: BenchmarkSpec,
    tasks: Sequence[Task],
    *,
    command: Sequence[str],
    force: bool = False,
    n_workers: int = 1,
) -> Dict[str, int]:
    """Run tasks one after another, or over n_workers processes (one thread each).

    Args:
        spec (BenchmarkSpec): The spec.
        tasks (Sequence[Task]): The tasks.
        command (Sequence[str]): The command as invoked.
        force (bool, optional): Recompute stale outputs. Defaults to False.
        n_workers (int, optional): Worker processes; searches always run one at a
            time, since each already spreads over spec.search_n_cores. Defaults to 1.

    Returns:
        Dict[str, int]: How many tasks had each status before running.
    """
    if any(task.stage == "search" for task in tasks):
        n_workers = 1
    if n_workers > 1:
        statuses = Parallel(n_jobs=n_workers)(
            delayed(_run_task_in_worker)(spec, task, command, force) for task in tasks
        )
    else:
        statuses = [run_task(spec, task, command=command, force=force) for task in tasks]
    return {status: statuses.count(status) for status in sorted(set(statuses))}


# Stages


def calibrate(spec: BenchmarkSpec, task: Task) -> Dict[str, Any]:
    """Calibrate one recording with CBSS and keep the units matching its ground truth.

    Args:
        spec (BenchmarkSpec): The spec.
        task (Task): A calibrate task.

    Returns:
        Dict[str, Any]: Metadata sections: status ("done", or "skipped" when no
        unit matches the ground truth), inputs, digest and results.
    """
    rec = task.recording
    paths = spec.calibration_paths(rec)
    inputs = {
        name: {
            "path": Path(os.path.relpath(path, REPO_ROOT)).as_posix(),
            "sha256": file_sha256(path),
        }
        for name, path in (("emg", spec.emg_path(rec)), ("gt", spec.gt_path(rec)))
    }

    # Load the calibration window and its ground truth
    emg_full = fdsi.load_raw_emg(spec.data_root, rec.sub, rec.cond, rec.snr)
    emg_calib, ts_calib = emg_full[: spec.cal_end], np.arange(spec.cal_end) / spec.fs
    gt_bin = load_gt(spec.gt_path(rec), n_samples=spec.cal_end)

    # Calibrate
    cbss_config = spec.cbss_config()
    result = CBSS(cbss_config).decompose(emg_calib, ts_calib)

    # Keep only the units matching the ground truth. select_supervised raises ValueError for
    # misaligned ground truth, a missing fs or no matched unit; the first two are set here.
    try:
        result = result.select_supervised(
            gt_bin,
            roa_th=spec.calibration["supervised_roa_th"],
            tol_spike_ms=spec.tol_spike_ms,
            fs=spec.fs,
        )
    except ValueError as exc:
        logger.warning(f"calibrate {task.id}: skipped, {exc}")
        _remove([paths["result"], paths["config"], paths["units"]])
        return {"status": "skipped", "reason": str(exc), "inputs": inputs}

    # Per-unit ground-truth match, RoA, SIL and CoV-ISI over the calibration window
    units = fdsi.with_recording_labels(
        fdsi.calibration_unit_metrics(result), rec.sub, rec.cond, rec.snr
    )

    # Save
    _atomic(paths["result"], result.save)
    _atomic(paths["config"], cbss_config.to_yaml)
    _atomic(paths["units"], lambda p: units.to_csv(p, index=False))
    return {
        "status": "done",
        "inputs": inputs,
        "outputs": {k: paths[k].name for k in ("result", "config", "units")},
        "digest": {
            "n_units": int(result.spikes.shape[1]),
            "spikes_sha256": fdsi.spikes_digest(result.spikes),
            "sep_vectors_sha256": _sha256_bytes(np.ascontiguousarray(result.sep_vectors).tobytes()),
        },
        "results": {
            "n_units": int(result.spikes.shape[1]),
            "roa_calib": [round(float(r), 6) for r in result.roa],
        },
    }


def search(spec: BenchmarkSpec, task: Task) -> Dict[str, Any]:
    """Search one config with optimize_adapt_decomp on the pool's recordings.

    Args:
        spec (BenchmarkSpec): The spec.
        task (Task): A search task.

    Returns:
        Dict[str, Any]: Metadata sections: status, inputs, outputs, digest and results.
    """
    name = task.branch
    pool_tasks = [spec.task_for("calibrate", rec.stub) for rec in spec.pool_recordings()]
    if "skipped" in _require_upstream(spec, pool_tasks):
        raise ValueError(f"search {name}: a pool recording has no calibrated unit.")
    settings = spec.search_settings(name)
    base_config = spec.search_base_config(name)

    # Pool: this run's calibrations, from the end of their calibration window
    data_config = {
        "root": ".",
        "preprocess": True,
        "datasets": [
            {
                "name": rec.cond,
                "path_emg": str(spec.emg_path(rec)),
                "path_calib": str(spec.calibration_paths(rec)["result"]),
                "path_calib_config": str(spec.calibration_paths(rec)["config"]),
                "path_gt": str(spec.gt_path(rec)),
                "fs": spec.fs,
                "start": spec.cal_end,
            }
            for rec in spec.pool_recordings()
        ],
    }
    pool = load_pooled_cbss_memory(data_config)

    # Search, writing into a temporary directory swapped in when complete
    out_dir = spec.search_paths(name)["dir"]
    tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp")
    _remove([tmp_dir])
    tmp_dir.mkdir(parents=True)
    started = _now()
    result = optimize_adapt_decomp(
        pool=pool,
        objectives=settings["objectives"],
        param_space=settings["param_space"],
        base_config=base_config,
        compute_roa=settings["compute_roa"],
        roa_kwargs={"tol_spike_ms": spec.tol_spike_ms},
        unit_selection=settings["unit_selection"],
        unit_selection_kwargs=settings["unit_selection_kwargs"],
        selection=settings["selection"],
        n_trials=settings["n_trials"],
        n_jobs=settings["n_jobs"],
        n_cores=settings["n_cores"],
        random_seed=settings["random_seed"],
        best_result_path=str(tmp_dir / "best"),
    )
    study = result.study
    study.set_user_attr("wall_time_s", (_now() - started).total_seconds())
    chosen = (
        study.best_trial
        if result.pareto_front is None
        else SELECTION_RULES[settings["selection"]](result.pareto_front)
    )

    # Save the study, its trials, the chosen config and the settings that produced them
    with open(tmp_dir / "study.pkl", "wb") as f:
        pickle.dump(study, f)
    study.trials_dataframe().to_csv(tmp_dir / "trials.csv", index=False)
    result.best_config.to_yaml(tmp_dir / "best_config.yaml")
    base_config.to_yaml(tmp_dir / "base_config.yaml")
    sweep = {
        "param_space": {k: list(v) for k, v in settings["param_space"].items()},
        "objectives": list(settings["objectives"]),
        "selection": settings["selection"],
        "unit_selection": settings["unit_selection"],
        "unit_selection_kwargs": settings["unit_selection_kwargs"],
        "n_trials": settings["n_trials"],
        "n_jobs": settings["n_jobs"],
        "n_cores": settings["n_cores"],
        "random_seed": settings["random_seed"],
    }
    with open(tmp_dir / "search.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(sweep, f, sort_keys=False)
    _remove([out_dir])
    os.replace(tmp_dir, out_dir)

    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    trial_values = [[t.number, t.params, t.values] for t in complete]
    return {
        "status": "done",
        "inputs": {
            "upstream": {"calibrate": {t.id: spec.calibration_key(t.recording) for t in pool_tasks}}
        },
        "outputs": {"dir": out_dir.name, "files": sorted(p.name for p in out_dir.iterdir())},
        "digest": {
            "n_complete": len(complete),
            "trials_sha256": _sha256_bytes(json.dumps(trial_values, sort_keys=True).encode()),
        },
        "results": {
            "chosen_trial": chosen.number,
            "chosen_params": chosen.params,
            "chosen_values": list(chosen.values),
            "front_size": None if result.pareto_front is None else len(result.pareto_front),
            "wall_time_s": round(study.user_attrs["wall_time_s"], 1),
        },
    }


def apply(spec: BenchmarkSpec, task: Task) -> Dict[str, Any]:
    """Apply the fixed baseline or a search's winner to one recording, then score it.

    Args:
        spec (BenchmarkSpec): The spec.
        task (Task): An apply task.

    Returns:
        Dict[str, Any]: Metadata sections: status ("skipped" when the calibration
        was), inputs, outputs, digest and results.
    """
    rec, branch = task.recording, task.branch
    paths = spec.result_paths(branch, rec)
    cal_task = spec.task_for("calibrate", rec.stub)
    upstream = [cal_task] if branch == FIXED_BRANCH else [cal_task, spec.task_for("search", branch)]
    statuses = _require_upstream(spec, upstream)
    inputs = {"upstream": {t.stage: {t.id: spec.task_key(t)} for t in upstream}}
    if statuses[0] == "skipped":
        _remove([paths["result"], paths["config"], paths["metrics"]])
        return {"status": "skipped", "reason": "its calibration was skipped", "inputs": inputs}

    # The calibration and the config to apply
    cal_paths = spec.calibration_paths(rec)
    cbss_result = CBSSResult.load(cal_paths["result"])
    cbss_config = CBSSConfig.from_yaml(cal_paths["config"])
    adapt_config = (
        spec.fixed_config()
        if branch == FIXED_BRANCH
        else AdaptConfig.from_yaml(spec.search_paths(branch)["dir"] / "best_config.yaml")
    )

    # Adapt from the end of the calibration window, the source FIFO seeded from its tail
    emg_full = fdsi.load_raw_emg(spec.data_root, rec.sub, rec.cond, rec.snr)
    adapt_config.source_fifo_from_calib = True
    adapter = AdaptDecomp.from_calibration(
        calibration=cbss_result, cbss_config=cbss_config, adapt_config=adapt_config
    )
    outputs = adapter.process_data(emg_full[spec.cal_end :])

    # Prepend CBSS's own output over the calibration window, so the whole recording is scored
    outputs.spikes = torch.cat(
        [torch.as_tensor(cbss_result.spikes, dtype=outputs.spikes.dtype), outputs.spikes]
    )
    outputs.sources = torch.cat(
        [torch.as_tensor(cbss_result.sources, dtype=outputs.sources.dtype), outputs.sources]
    )

    # Score against the ground truth
    outputs.sil = get_sil(
        outputs.sources,
        outputs.spikes,
        adapt_config.spike_min_dist,
        peak_power=adapt_config.spike_det_exp,
    ).numpy()
    gt_full_bin = fdsi.load_gt_full_bin(
        spec.data_root, rec.sub, rec.cond, cbss_result, n_samples=outputs.spikes.shape[0]
    )
    outputs.roa = fdsi.compute_roa_for_result(outputs, gt_full_bin, spec.fs, spec.tol_spike_ms)
    metrics = fdsi.with_recording_labels(
        fdsi.unit_metrics(
            outputs,
            gt_full_bin,
            cbss_result,
            rec.cond,
            spec.cal_end,
            spec.iso_dur,
            spec.fs,
            spec.tol_spike_ms,
        ),
        rec.sub,
        rec.cond,
        rec.snr,
        branch=branch,
    )

    # Save
    _atomic(paths["result"], outputs.save)
    _atomic(paths["config"], adapt_config.to_yaml)
    _atomic(paths["metrics"], lambda p: metrics.to_csv(p, index=False))
    spikes = outputs.spikes.numpy()
    return {
        "status": "done",
        "inputs": inputs,
        "outputs": {k: paths[k].name for k in ("result", "config", "metrics")},
        "digest": {
            "n_units": int(spikes.shape[1]),
            "n_spikes": int(spikes.sum()),
            "spikes_sha256": fdsi.spikes_digest(spikes),
        },
        "results": {
            "roa_after_cal_mean": _float_or_none(np.nanmean(metrics["roa_after_cal"]))
            if len(metrics)
            else None,
            "wh_loss_total": _float_or_none(outputs.wh_loss_total),
            "sv_loss_total": _float_or_none(outputs.sv_loss_total),
            "total_loss": _float_or_none(outputs.total_loss),
            "n_batches": int(outputs.total_time_ms.shape[0]),
            "mean_batch_ms": float(outputs.total_time_ms.float().mean()),
        },
    }


STAGE_RUNNERS: Dict[str, Callable[[BenchmarkSpec, Task], Dict[str, Any]]] = {
    "calibrate": calibrate,
    "search": search,
    "apply": apply,
}


# Tables


def _meta_row(task: Task, status: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One provenance row of a task: status, timing, machine and git state.

    Args:
        task (Task): The task.
        status (str): Its status.
        meta (Optional[Dict[str, Any]]): Its metadata, None when missing.

    Returns:
        Dict[str, Any]: Flat row.
    """
    meta = meta or {}
    git = meta.get("git") or {}
    hardware = meta.get("hardware", {})
    return {
        "stage": task.stage,
        "index": task.index,
        "id": task.id,
        "status": status,
        "key": meta.get("task", {}).get("key"),
        "started_at": meta.get("started_at"),
        "run_time_s": meta.get("run_time_s"),
        "hostname": meta.get("host", {}).get("hostname"),
        "job_id": meta.get("host", {}).get("job_id"),
        "cpu": hardware.get("cpu", {}).get("model"),
        "torch_threads": hardware.get("cpu", {}).get("torch_threads"),
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "patch": git.get("patch"),
    }


def collect(spec: BenchmarkSpec, *, command: Sequence[str]) -> Dict[str, Path]:
    """Gather every current output into tables; read-only, tolerant of a partial run.

    Args:
        spec (BenchmarkSpec): The spec.
        command (Sequence[str]): The command as invoked.

    Returns:
        Dict[str, Path]: Table name -> CSV in spec.tables_dir: calibrations (per
        recording), calibration_units (per calibrated unit, the metrics of
        fdsi.calibration_unit_metrics), searches (every trial), best_configs,
        recordings, units (per unit, the metrics of fdsi.unit_metrics) and
        provenance (one row per task).
    """
    started = _now()
    rows: Dict[str, List[Any]] = {
        name: []
        for name in (
            "calibrations",
            "calibration_units",
            "searches",
            "best_configs",
            "recordings",
            "units",
        )
    }
    provenance = []

    # Calibrations: one row per recording, and their per-unit metrics
    for task in spec.tasks("calibrate"):
        status, meta = spec.task_status(task)
        provenance.append(_meta_row(task, status, meta))
        rec = task.recording
        units = (
            pd.read_csv(spec.calibration_paths(rec)["units"])
            if status == "done"
            else pd.DataFrame()
        )
        if len(units):
            rows["calibration_units"].append(units)
        rows["calibrations"].append(
            {
                "recording": rec.stub,
                "sub": rec.sub,
                "condition": rec.cond,
                "snr": rec.snr,
                "status": status,
                "n_units": len(units) if status == "done" else None,
                "roa_calib_mean": units["roa_calib"].mean() if len(units) else None,
                "n_units_sil_ge_0.9": int((units["sil_calib"] >= 0.9).sum())
                if len(units)
                else None,
                "run_time_s": (meta or {}).get("run_time_s"),
            }
        )

    # Searches: every trial, and the chosen config of each
    for task in spec.tasks("search"):
        status, meta = spec.task_status(task)
        provenance.append(_meta_row(task, status, meta))
        if status != "done":
            continue
        out_dir = spec.search_paths(task.branch)["dir"]
        trials = pd.read_csv(out_dir / "trials.csv")
        trials.insert(0, "search", task.branch)
        rows["searches"].append(trials)
        settings = spec.search_settings(task.branch)
        best = AdaptConfig.from_yaml(out_dir / "best_config.yaml")
        rows["best_configs"].append(
            {
                "search": task.branch,
                "objectives": ",".join(settings["objectives"]),
                "selection": settings["selection"] if len(settings["objectives"]) > 1 else None,
                "sv_loss_reduction": best.sv_loss_reduction,
                "chosen_trial": meta["results"]["chosen_trial"],
                **{p: getattr(best, p) for p in settings["param_space"]},
            }
        )

    # Applied configs: one row per recording, and their per-unit metrics
    for task in spec.tasks("apply"):
        status, meta = spec.task_status(task)
        provenance.append(_meta_row(task, status, meta))
        rec = task.recording
        results = (meta or {}).get("results", {}) if status == "done" else {}
        rows["recordings"].append(
            {
                "branch": task.branch,
                "recording": rec.stub,
                "sub": rec.sub,
                "condition": rec.cond,
                "snr": rec.snr,
                "status": status,
                "n_units": (meta or {}).get("digest", {}).get("n_units"),
                **results,
                "run_time_s": (meta or {}).get("run_time_s"),
            }
        )
        if status == "done":
            rows["units"].append(pd.read_csv(spec.result_paths(task.branch, rec)["metrics"]))

    # Write the tables
    tables = {
        "calibrations": pd.DataFrame(rows["calibrations"]),
        "calibration_units": pd.concat(rows["calibration_units"], ignore_index=True)
        if rows["calibration_units"]
        else pd.DataFrame(),
        "searches": pd.concat(rows["searches"], ignore_index=True)
        if rows["searches"]
        else pd.DataFrame(),
        "best_configs": pd.DataFrame(rows["best_configs"]),
        "recordings": pd.DataFrame(rows["recordings"]),
        "units": pd.concat(rows["units"], ignore_index=True) if rows["units"] else pd.DataFrame(),
        "provenance": pd.DataFrame(provenance),
    }
    for name, table in tables.items():
        _atomic(
            spec.tables_dir / f"{name}.csv", lambda p, table=table: table.to_csv(p, index=False)
        )
    counts = tables["provenance"].groupby(["stage", "status"]).size()
    for (stage, status), n in counts.items():
        logger.info(f"collect: {stage} {status}: {n}")

    _write_table_metadata(spec, "collect", tables, started, command)
    return {name: spec.tables_dir / f"{name}.csv" for name in tables}


def _write_table_metadata(
    spec: BenchmarkSpec,
    name: str,
    tables: Dict[str, pd.DataFrame],
    started: datetime,
    command: Sequence[str],
) -> None:
    """Write the metadata of the tables collect or import-v10 wrote to spec.tables_dir.

    Args:
        spec (BenchmarkSpec): The spec.
        name (str): "collect" or "import-v10".
        tables (Dict[str, pd.DataFrame]): Table name -> the table written as "<name>.csv".
        started (datetime): When the command started.
        command (Sequence[str]): The command as invoked.

    Returns:
        None
    """
    metadata = build_metadata(
        command=command,
        started=started,
        finished=_now(),
        run_name=f"{spec.name}-{name}",
        reproduce=[
            *REPRODUCE_SETUP,
            f"# the run's outputs must exist under {spec.outputs_root.name}/",
            f"python -m benchmarks.fdsi {name} --spec {spec.spec_ref}",
        ],
        repo_dir=REPO_ROOT,
        patch_dir=spec.patch_dir,
        extra={
            "task": {"stage": name, "spec": spec.spec_ref},
            "digest": {
                f"{table_name}.csv": {
                    "rows": len(table),
                    "sha256": _sha256_bytes((spec.tables_dir / f"{table_name}.csv").read_bytes()),
                }
                for table_name, table in tables.items()
            },
        },
    )
    write_metadata(spec.tables_dir / f"{name}.meta.yaml", metadata)


def _v10_unit_metrics(
    spec: BenchmarkSpec, branch: str, sampler: Optional[str], rec: Recording
) -> Optional[pd.DataFrame]:
    """Per-unit metrics of one cached v1.0 result, None if it isn't cached.

    Args:
        spec (BenchmarkSpec): The spec.
        branch (str): Key of V10_BRANCHES.
        sampler (Optional[str]): Its v1.0 search dir.
        rec (Recording): The recording.

    Returns:
        Optional[pd.DataFrame]: As fdsi.unit_metrics, with branch, recording, sub,
        condition and snr; gt_unit pairs its units with this run's.
    """
    root = spec.v10_outputs_root
    result_path = fdsi.v10_result_path(root, rec.sub, rec.cond, rec.snr, sampler)
    cal_path = fdsi.v10_calibration_path(root, rec.sub, rec.cond, rec.snr)
    if not (result_path.exists() and cal_path.exists()):
        return None
    outputs = AdaptationResult.load(result_path)
    cbss_result = CBSSResult.load(cal_path)
    gt_full_bin = fdsi.load_gt_full_bin(
        spec.data_root, rec.sub, rec.cond, cbss_result, n_samples=outputs.spikes.shape[0]
    )
    metrics = fdsi.unit_metrics(
        outputs,
        gt_full_bin,
        cbss_result,
        rec.cond,
        spec.cal_end,
        spec.iso_dur,
        spec.fs,
        spec.tol_spike_ms,
    )
    return fdsi.with_recording_labels(metrics, rec.sub, rec.cond, rec.snr, branch=branch)


def import_v10(spec: BenchmarkSpec, *, command: Sequence[str], n_workers: int = 1) -> Path:
    """Compute the same per-unit metrics from the cached v1.0 results, for comparison.

    Args:
        spec (BenchmarkSpec): The spec (needs v10_outputs_root).
        command (Sequence[str]): The command as invoked.
        n_workers (int, optional): Worker processes. Defaults to 1.

    Raises:
        ValueError: If the spec has no v10_outputs_root.

    Returns:
        Path: spec.tables_dir / "v1_0_units.csv".
    """
    if spec.v10_outputs_root is None:
        raise ValueError("The spec has no v10_outputs_root to import v1.0 results from.")
    started = _now()
    jobs = [
        (branch, sampler, rec)
        for branch, sampler in V10_BRANCHES.items()
        for rec in spec.recordings()
    ]
    tables = Parallel(n_jobs=n_workers)(
        delayed(_v10_unit_metrics)(spec, branch, sampler, rec) for branch, sampler, rec in jobs
    )
    found = [table for table in tables if table is not None]
    logger.info(f"import-v10: {len(found)}/{len(jobs)} v1.0 results found")
    path = spec.tables_dir / "v1_0_units.csv"
    units = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    _atomic(path, lambda p: units.to_csv(p, index=False))
    _write_table_metadata(spec, "import-v10", {"v1_0_units": units}, started, command)
    return path


# Verification


def _copy_upstream(spec: BenchmarkSpec, scratch: BenchmarkSpec, task: Task) -> None:
    """Copy the upstream outputs a task reads into a scratch outputs root.

    Args:
        spec (BenchmarkSpec): The original spec.
        scratch (BenchmarkSpec): The same spec writing to the scratch root.
        task (Task): The task to re-run there.

    Returns:
        None
    """
    if task.stage == "calibrate":
        return
    recs = spec.pool_recordings() if task.stage == "search" else [task.recording]
    files = [path for rec in recs for path in spec.calibration_paths(rec).values()]
    if task.stage == "apply" and task.branch != FIXED_BRANCH:
        search_paths = spec.search_paths(task.branch)
        files += [search_paths["meta"], search_paths["dir"] / "best_config.yaml"]
    for path in files:
        if path.exists():
            target = scratch.outputs_root / path.relative_to(spec.outputs_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _compare_search(spec: BenchmarkSpec, scratch: BenchmarkSpec, name: str) -> str:
    """Compare two runs of a search trial by trial when their digests differ.

    Args:
        spec (BenchmarkSpec): The original spec.
        scratch (BenchmarkSpec): The re-run's spec.
        name (str): Search name.

    Returns:
        str: "within tolerance" if every trial's params and values agree within
        VERIFY_RTOL, else "MISMATCH".
    """
    original = pd.read_csv(spec.search_paths(name)["dir"] / "trials.csv")
    rerun = pd.read_csv(scratch.search_paths(name)["dir"] / "trials.csv")
    columns = [c for c in original.columns if c.startswith(("params_", "value"))]
    if len(original) != len(rerun) or columns != [c for c in rerun.columns if c in columns]:
        return "MISMATCH"
    same = np.allclose(
        original[columns].to_numpy(float),
        rerun[columns].to_numpy(float),
        rtol=VERIFY_RTOL,
        equal_nan=True,
    )
    return "within tolerance" if same else "MISMATCH"


def verify(
    spec: BenchmarkSpec,
    stage: str,
    indices: Sequence[int],
    scratch_root: Path,
    *,
    command: Sequence[str],
) -> pd.DataFrame:
    """Re-run tasks into a scratch root and compare their outputs with the originals.

    Calibrations and applied configs must reproduce bit for bit (their spike
    trains' SHA-256); a search must reproduce every trial (exactly, or within
    VERIFY_RTOL when its digest differs).

    Args:
        spec (BenchmarkSpec): The spec whose outputs are checked.
        stage (str): One of STAGES.
        indices (Sequence[int]): Task indices of that stage.
        scratch_root (Path): Where the re-runs write (upstream outputs are copied there).
        command (Sequence[str]): The command as invoked.

    Raises:
        ValueError: If a task to verify isn't done.

    Returns:
        pd.DataFrame: One row per task: stage, id, result ("exact", "within
        tolerance" or "MISMATCH") and both digests.
    """
    scratch = spec.with_outputs_root(scratch_root)
    tasks = spec.tasks(stage)
    rows = []
    for index in indices:
        task = tasks[index]
        status, meta = spec.task_status(task)
        if status != "done":
            raise ValueError(
                f"{stage} task {task.id!r} is {status}, so there is nothing to verify."
            )
        _copy_upstream(spec, scratch, task)
        run_task(scratch, task, command=command, force=True)
        rerun = read_metadata(scratch.meta_path(task))
        if rerun["digest"] == meta["digest"]:
            result = "exact"
        elif stage == "search":
            result = _compare_search(spec, scratch, task.branch)
        else:
            result = "MISMATCH"
        rows.append(
            {
                "stage": stage,
                "id": task.id,
                "result": result,
                "original": json.dumps(meta["digest"], sort_keys=True),
                "rerun": json.dumps(rerun["digest"], sort_keys=True),
            }
        )
    return pd.DataFrame(rows)
