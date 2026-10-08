"""The FDSI benchmark's stages: calibrate every recording, search the adaptation's
hyperparameters on a pool, apply each search's winner (and no adaptation) to every recording,
then collect the scores into benchmarks/fdsi/results/<version>/.

Every stage is one function per task (calibrate_one, search_one, apply_one) calling the
adapt_decomp API directly, and returns at once when its output exists: delete an output to
recompute it. run() runs a stage's tasks, all of them or one array chunk, and is what the
notebooks, `python -m benchmarks.fdsi` and the PBS jobs call.
"""

import copy
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from loguru import logger
from threadpoolctl import threadpool_limits

import adapt_decomp
from adapt_decomp import CBSS, AdaptDecomp, CBSSConfig, CBSSResult
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    SELECTION_RULES,
    optimize_adapt_decomp,
)
from adapt_decomp.spikes import get_sil
from adapt_decomp.utils import load_gt
from adapt_decomp.utils.download import ARCHIVES
from adapt_decomp.utils.loaders import PooledDatasetMemory
from benchmarks.fdsi import fdsi
from benchmarks.fdsi.fdsi import Recording

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
STAGES = ("calibrate", "search", "apply", "collect")
FIXED = "fixed"  # the no-adaptation baseline, applied next to every search's winner
SELECTION_LABELS = {"min_sv_loss": "min-sv", "knee": "knee", "max_roa_mean": "max-RoA"}


# Config


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """base with override's values, merged recursively into nested dicts."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _path(value: Union[str, Path]) -> Path:
    """A path from the config: absolute, or relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Union[str, Path] = DEFAULT_CONFIG, quick: bool = False) -> Dict[str, Any]:
    """Load the benchmark config.

    Args:
        path (Union[str, Path], optional): The config YAML. Defaults to config.yaml.
        quick (bool, optional): Apply its quick section, a small run to check the pipeline,
            written under "<version>-quick". Defaults to False.

    Returns:
        Dict[str, Any]: The config, plus "outputs" (<outputs_root>/<version>) and "results"
        (<results_root>/<version>) as absolute paths.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    quick_section = cfg.pop("quick", {})
    if quick:
        cfg = _merge(cfg, quick_section)
        cfg["version"] = f"{cfg['version']}-quick"
    cfg["outputs"] = _path(cfg["outputs_root"]) / cfg["version"]
    cfg["results"] = _path(cfg["results_root"]) / cfg["version"]
    return cfg


def recordings(cfg: Dict[str, Any]) -> List[Recording]:
    """Every recording of the grid, subject by subject."""
    grid = cfg["grid"]
    return [
        Recording(sub, cond, snr)
        for sub in grid["subjects"]
        for cond in grid["conditions"]
        for snr in grid["snr_levels"]
    ]


def pool_recordings(cfg: Dict[str, Any]) -> List[Recording]:
    """The recordings every search runs on."""
    pool = cfg["pool"]
    return [Recording(pool["subject"], cond, pool["snr"]) for cond in pool["conditions"]]


def branches(cfg: Dict[str, Any]) -> List[str]:
    """The applied configs: the no-adaptation baseline, then each search's winner."""
    return [FIXED, *cfg["searches"]]


def tasks(cfg: Dict[str, Any], stage: str) -> List[Any]:
    """A stage's tasks, in a fixed order: recordings, search names or (branch, recording)."""
    if stage == "calibrate":
        return recordings(cfg)
    if stage == "search":
        return list(cfg["searches"])
    if stage == "apply":
        return [(branch, rec) for branch in branches(cfg) for rec in recordings(cfg)]
    return ["collect"]


def cal_end(cfg: Dict[str, Any]) -> int:
    """Calibration window length in samples; adaptation starts here."""
    return int(cfg["grid"]["cal_duration_s"] * cfg["grid"]["fs"])


def cbss_config(cfg: Dict[str, Any]) -> CBSSConfig:
    """The calibration's CBSSConfig."""
    return CBSSConfig(**cfg["calibration"]["cbss_config"])


def _adapt_config(path: Union[str, Path], overrides: Dict[str, Any]) -> AdaptConfig:
    """An AdaptConfig from a YAML file, with some fields replaced."""
    return AdaptConfig(**{**AdaptConfig.from_yaml(_path(path)).to_dict(), **overrides})


def search_settings(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One search's settings: the shared search section overridden by its own entry.

    Returns:
        Dict[str, Any]: The merged settings, with param_space resolved (None ->
        DEFAULT_PARAM_SPACE), initial_params as {param: value} dicts (a config file gives its
        values of the searched parameters) and base_config the AdaptConfig every trial
        starts from (the file, the overrides, then sv_loss_reduction).
    """
    settings = _merge(cfg["search"], cfg["searches"][name])
    space = settings.get("param_space")
    space = DEFAULT_PARAM_SPACE if space is None else {k: tuple(v) for k, v in space.items()}
    overrides = dict(settings.get("overrides") or {})
    if settings.get("sv_loss_reduction") is not None:
        overrides["sv_loss_reduction"] = settings["sv_loss_reduction"]
    settings.update(
        param_space=space,
        initial_params=[
            dict(p) if isinstance(p, dict) else {k: getattr(_adapt_config(p, {}), k) for k in space}
            for p in settings.get("initial_params") or []
        ],
        base_config=_adapt_config(settings["base_config"], overrides),
    )
    return settings


def applied_config(cfg: Dict[str, Any], branch: str) -> AdaptConfig:
    """The AdaptConfig applied to every recording: the fixed baseline or a search's winner,
    from the search's outputs or else the published results (to apply it without searching)."""
    if branch == FIXED:
        config = _adapt_config(cfg["fixed_config"], {"device": "cpu"})
    elif (search_dir(cfg, branch) / "best_config.yaml").exists():
        config = AdaptConfig.from_yaml(search_dir(cfg, branch) / "best_config.yaml")
    else:
        config = AdaptConfig.from_yaml(cfg["results"] / "configs" / f"{branch}.yaml")
    config.source_fifo_from_calib = True  # adapt from the end of the calibration window
    return config


def config_label(cfg: Dict[str, Any], branch: str) -> str:
    """A readable label of an applied config, e.g. "Pareto min-sv, sum"."""
    if branch == FIXED:
        return "No adaptation"
    search = _merge(cfg["search"], cfg["searches"][branch])
    objectives = list(search["objectives"])
    if objectives == ["roa"]:
        return "RoA (oracle)"
    name = objectives[0]
    if len(objectives) > 1:
        selection = search.get("selection", "min_sv_loss")
        name = f"Pareto {SELECTION_LABELS.get(selection, selection)}"
    reduction = search.get("sv_loss_reduction")
    return f"{name}, {reduction}" if reduction else name


# Output paths


def calibration_path(cfg: Dict[str, Any], rec: Recording, suffix: str = ".pkl") -> Path:
    """A calibration's CBSSResult (.pkl) or per-unit metrics (_units.csv)."""
    return cfg["outputs"] / "calibration" / rec.sub / f"{rec.stub}_cbss{suffix}"


def search_dir(cfg: Dict[str, Any], name: str) -> Path:
    """A search's trials.csv and best_config.yaml."""
    return cfg["outputs"] / "searches" / name


def result_path(cfg: Dict[str, Any], branch: str, rec: Recording, suffix: str = ".npz") -> Path:
    """An applied config's spikes and sources (.npz) or per-unit metrics (_units.csv)."""
    return cfg["outputs"] / "results" / branch / rec.sub / f"{rec.stub}{suffix}"


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file through a temporary name, so an interrupted write leaves no output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


class _one_thread:
    """Run torch and numpy on one thread, so results don't depend on the machine."""

    def __enter__(self) -> None:
        self.torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.limits = threadpool_limits(1)

    def __exit__(self, *exc: Any) -> None:
        self.limits.restore_original_limits()
        torch.set_num_threads(self.torch_threads)


# Stages


def calibrate_one(cfg: Dict[str, Any], rec: Recording) -> None:
    """Calibrate one recording with CBSS and keep the units matching its ground truth.

    Writes the CBSSResult and its per-unit metrics; a recording without any matching unit
    gets an empty metrics table, and nothing to apply.
    """
    units_path = calibration_path(cfg, rec, "_units.csv")
    if units_path.exists():
        return
    grid, n_cal = cfg["grid"], cal_end(cfg)

    # Load the calibration window and its ground truth
    data_root = _path(cfg["data_root"])
    emg_calib = fdsi.load_raw_emg(data_root, rec)[:n_cal]
    gt_bin = load_gt(fdsi.gt_spikes_path(data_root, rec.sub, rec.cond), n_samples=n_cal)

    # Calibrate, then keep the units matching a ground-truth unit with RoA >= supervised_roa_th
    with _one_thread():
        result = CBSS(cbss_config(cfg)).decompose(emg_calib, np.arange(n_cal) / grid["fs"])
    try:
        result = result.select_supervised(
            gt_bin,
            roa_th=cfg["calibration"]["supervised_roa_th"],
            tol_spike_ms=grid["tol_spike_ms"],
            fs=grid["fs"],
        )
    except ValueError as exc:  # no matched unit
        logger.warning(f"calibrate {rec.stub}: no unit kept, {exc}")
        _atomic(units_path, lambda p: pd.DataFrame().to_csv(p, index=False))
        return

    _atomic(calibration_path(cfg, rec), result.save)
    units = fdsi.calibration_unit_metrics(result, rec)
    _atomic(units_path, lambda p: units.to_csv(p, index=False))


def _calibration(cfg: Dict[str, Any], rec: Recording) -> Optional[CBSSResult]:
    """A recording's calibration, None if it kept no unit."""
    if not calibration_path(cfg, rec, "_units.csv").exists():
        raise FileNotFoundError(f"{rec.stub} is not calibrated yet: run the calibrate stage.")
    path = calibration_path(cfg, rec)
    return CBSSResult.load(path) if path.exists() else None


def search_one(cfg: Dict[str, Any], name: str) -> None:
    """Run one hyperparameter search on the pool, adapting and scoring each recording from the
    end of its calibration window. Writes every trial and the chosen trial's config."""
    out_dir = search_dir(cfg, name)
    if out_dir.exists():
        return
    settings = search_settings(cfg, name)
    data_root, n_cal = _path(cfg["data_root"]), cal_end(cfg)

    # The pool: each recording's calibration, its EMG and matched ground truth after the window
    pool = {}
    for rec in pool_recordings(cfg):
        calibration = _calibration(cfg, rec)
        if calibration is None:
            raise ValueError(f"search {name}: pool recording {rec.stub} has no calibrated unit.")
        emg = fdsi.load_raw_emg(data_root, rec)
        gt = fdsi.load_gt_matched(data_root, rec, calibration.gt_matched_indices, len(emg))
        pool[rec.cond] = PooledDatasetMemory(
            emg=torch.from_numpy(emg[n_cal:].copy()),
            calibration=calibration,
            cbss_config=cbss_config(cfg),
            gt_paired_bin=gt[n_cal:],
        )

    # Search: n_cores only sets the speed (None: every core available)
    result = optimize_adapt_decomp(
        pool=pool,
        objectives=tuple(settings["objectives"]),
        param_space=settings["param_space"],
        base_config=settings["base_config"],
        compute_roa=settings.get("compute_roa", True),
        roa_kwargs={"tol_spike_ms": cfg["grid"]["tol_spike_ms"]},
        unit_selection=settings.get("unit_selection"),
        unit_selection_kwargs=settings.get("unit_selection_kwargs"),
        selection=settings.get("selection", "min_sv_loss"),
        n_trials=settings["n_trials"],
        n_jobs=settings.get("n_jobs", 1),
        n_cores=settings.get("n_cores"),
        random_seed=settings["random_seed"],
        initial_params=settings["initial_params"],
    )
    chosen = (
        result.study.best_trial
        if result.pareto_front is None
        else SELECTION_RULES[settings.get("selection", "min_sv_loss")](result.pareto_front)
    )

    # Save every trial (the chosen one marked) and the chosen config, all at once
    trials = result.study.trials_dataframe()
    trials["chosen"] = trials["number"] == chosen.number
    tmp_dir = out_dir.with_name(f".{out_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    trials.to_csv(tmp_dir / "trials.csv", index=False)
    result.best_config.to_yaml(tmp_dir / "best_config.yaml")
    os.replace(tmp_dir, out_dir)


def apply_one(cfg: Dict[str, Any], branch: str, rec: Recording) -> None:
    """Apply the fixed baseline or a search's winner to one recording, from the end of its
    calibration window, then score it (score_one)."""
    if result_path(cfg, branch, rec, "_units.csv").exists():
        return
    calibration = _calibration(cfg, rec)
    if calibration is None:
        return
    n_cal = cal_end(cfg)
    emg = fdsi.load_raw_emg(_path(cfg["data_root"]), rec)

    with _one_thread():
        adapter = AdaptDecomp.from_calibration(
            calibration=calibration,
            cbss_config=cbss_config(cfg),
            adapt_config=applied_config(cfg, branch),
        )
        outputs = adapter.process_data(emg[n_cal:])

    # The whole recording: CBSS's own spikes and sources over the calibration window, then these
    spikes = np.concatenate([np.asarray(calibration.spikes), outputs.spikes.numpy()])
    sources = np.concatenate([np.asarray(calibration.sources), outputs.sources.numpy()])

    def save(path: Path) -> None:
        with open(path, "wb") as f:  # a file object, so numpy keeps the temporary name
            np.savez_compressed(
                f,
                spikes=spikes.astype(np.int8),
                sources=sources.astype(np.float32),
                gt_unit=np.asarray(calibration.gt_matched_indices, dtype=np.int64),
            )

    _atomic(result_path(cfg, branch, rec), save)
    score_one(cfg, branch, rec)


def score_one(cfg: Dict[str, Any], branch: str, rec: Recording) -> pd.DataFrame:
    """Score one applied config's saved spikes and sources against the ground truth (RoA over
    several windows, SIL), writing and returning its per-unit metrics."""
    grid = cfg["grid"]
    saved = np.load(result_path(cfg, branch, rec))
    spikes, sources, gt_unit = saved["spikes"], saved["sources"], saved["gt_unit"]
    config = applied_config(cfg, branch)
    sil = get_sil(
        torch.from_numpy(sources),
        torch.from_numpy(spikes.astype(np.float32)),
        config.spike_min_dist,
        peak_power=config.spike_det_exp,
    ).numpy()
    gt = fdsi.load_gt_matched(_path(cfg["data_root"]), rec, gt_unit, len(spikes))
    units = fdsi.unit_metrics(
        spikes,
        sil,
        gt,
        gt_unit,
        rec,
        branch,
        cal_end=cal_end(cfg),
        iso_dur=int(grid["iso_duration_s"] * grid["fs"]),
        fs=grid["fs"],
        tol_spike_ms=grid["tol_spike_ms"],
    )
    _atomic(result_path(cfg, branch, rec, "_units.csv"), lambda p: units.to_csv(p, index=False))
    return units


def _concat(frames: List[pd.DataFrame], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenate tables; an empty one (with columns) when there are none yet."""
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _git_commit() -> Optional[str]:
    """The repository's commit, with "-dirty" if it has uncommitted changes; None outside git."""
    try:
        run_git = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return run_git("rev-parse", "HEAD") + ("-dirty" if run_git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(cfg: Dict[str, Any]) -> Path:
    """Gather the outputs into cfg["results"]: units.csv (per unit and applied config),
    calibration_units.csv, trials.csv (every search's trials), configs/<search>.yaml and
    run.yaml (what produced them). Scores any saved result without metrics (e.g. downloaded).
    """
    results = cfg["results"]
    calib = [
        pd.read_csv(calibration_path(cfg, rec, "_units.csv"))
        for rec in recordings(cfg)
        if calibration_path(cfg, rec, "_units.csv").exists()
    ]
    calib = _concat([c for c in calib if len(c)])

    units = []
    for branch, rec in tasks(cfg, "apply"):
        if result_path(cfg, branch, rec, "_units.csv").exists():
            units.append(pd.read_csv(result_path(cfg, branch, rec, "_units.csv")))
        elif result_path(cfg, branch, rec).exists():
            units.append(score_one(cfg, branch, rec))
    units = _concat(units, columns=["recording", "unit", "gt_unit"])
    roa_calib = calib.reindex(columns=["recording", "unit", "roa_calib"])
    units = units.merge(roa_calib, on=["recording", "unit"], how="left")
    columns = list(units.columns.drop("roa_calib"))
    columns.insert(columns.index("gt_unit") + 1, "roa_calib")

    (results / "configs").mkdir(parents=True, exist_ok=True)
    trials = []
    for name in cfg["searches"]:
        if search_dir(cfg, name).exists():
            trials.append(pd.read_csv(search_dir(cfg, name) / "trials.csv"))
            trials[-1].insert(0, "search", name)
            shutil.copy(
                search_dir(cfg, name) / "best_config.yaml", results / "configs" / f"{name}.yaml"
            )
    trials = _concat(trials)

    units[columns].to_csv(results / "units.csv", index=False)
    calib.to_csv(results / "calibration_units.csv", index=False)
    trials.to_csv(results / "trials.csv", index=False)
    run = {
        "version": cfg["version"],
        "adapt_decomp": adapt_decomp.__version__,
        "commit": _git_commit(),
        "date": date.today().isoformat(),
        "data_doi": ARCHIVES["fdsi_benchmark-data"][0],
        "labels": {branch: config_label(cfg, branch) for branch in branches(cfg)},
        "config": {k: v for k, v in cfg.items() if k not in ("outputs", "results")},
    }
    with open(results / "run.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run, f, sort_keys=False)
    logger.info(
        f"collect: {len(calib)} calibrated units, {units['recording'].nunique()} recordings, "
        f"{len(trials)} trials -> {results}"
    )
    return results


# Running


def _run_task(cfg: Dict[str, Any], stage: str, task: Any) -> None:
    """Run one task of a stage."""
    if stage == "calibrate":
        calibrate_one(cfg, task)
    elif stage == "search":
        search_one(cfg, task)
    elif stage == "apply":
        apply_one(cfg, *task)
    else:
        collect(cfg)


def run(
    cfg: Dict[str, Any],
    stage: str,
    index: Optional[int] = None,
    chunk: int = 1,
    n_workers: int = 1,
) -> None:
    """Run a stage's tasks whose outputs don't exist yet.

    Args:
        cfg (Dict[str, Any]): The config (load_config).
        stage (str): One of STAGES.
        index (Optional[int], optional): Run only the index-th chunk of tasks (an array job's
            index). Defaults to None (every task).
        chunk (int, optional): Tasks per index. Defaults to 1.
        n_workers (int, optional): Processes running calibrate or apply tasks at once (one
            thread each); searches run one at a time, each on every core it is given.
            Defaults to 1.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}, expected one of {STAGES}")
    selected = tasks(cfg, stage)
    if index is not None:
        selected = selected[index * chunk : (index + 1) * chunk]
    logger.info(f"{stage}: {len(selected)} task(s) under {cfg['outputs']}")
    if n_workers > 1 and stage in ("calibrate", "apply"):
        Parallel(n_jobs=n_workers)(delayed(_run_task)(cfg, stage, t) for t in selected)
    else:
        for task in selected:
            _run_task(cfg, stage, task)
