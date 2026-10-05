"""The FDSI benchmark spec: loading and validation, tasks, output paths, cache keys and status."""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml

from adapt_decomp import CBSSConfig
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import DEFAULT_PARAM_SPACE
from adapt_decomp.adaptation.optimize.pareto import validate_selection
from adapt_decomp.adaptation.optimize.scoring import validate_objectives
from adapt_decomp.adaptation.optimize.units import validate_unit_selection
from adapt_decomp.utils import read_metadata
from benchmarks.fdsi import fdsi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = "benchmarks/fdsi/benchmark.yaml"

STAGES: Tuple[str, ...] = ("calibrate", "search", "apply")
FIXED_BRANCH = "fixed"  # the no-adaptation baseline, applied next to every search's winner
THREADS_PER_RUN = 1  # torch threads of every run, so results don't depend on the machine

# Version of each stage's outputs, hashed into its cache keys: bump it when a stage's code
# changes what it writes, so the outputs written before become stale (and so does every
# output downstream of them). calibrate 2 / apply 2: per-unit ground-truth match and RoA.
STAGE_VERSIONS: Dict[str, int] = {"calibrate": 2, "search": 1, "apply": 2}

# Array-index environment variables, in lookup order: PBS Pro, Torque, SLURM
ARRAY_INDEX_VARS: Tuple[str, ...] = ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID")

# Allowed (and required) keys of each spec section
TOP_KEYS: Set[str] = {
    "name",
    "data_root",
    "outputs_root",
    "v10_outputs_root",
    "grid",
    "calibration",
    "pool",
    "search",
    "searches",
    "apply",
}
GRID_KEYS: Set[str] = {
    "subjects",
    "conditions",
    "snr_levels",
    "fs",
    "cal_duration_s",
    "iso_duration_s",
    "tol_spike_ms",
}
CALIBRATION_KEYS: Set[str] = {"cbss_config", "supervised_roa_th"}
POOL_KEYS: Set[str] = {"subject", "snr", "conditions"}
# The keys of configs/sweep_configs/sweep_optuna.yaml, plus the base config, its overrides
# and sv_loss_reduction (an AdaptConfig override, set per search)
SEARCH_KEYS: Set[str] = {
    "base_config",
    "overrides",
    "param_space",
    "objectives",
    "selection",
    "unit_selection",
    "unit_selection_kwargs",
    "n_trials",
    "n_jobs",
    "random_seed",
    "compute_roa",
    "sv_loss_reduction",
}
SEARCH_DEFAULTS: Dict[str, Any] = {
    "overrides": {},
    "param_space": None,
    "selection": "min_sv_loss",
    "unit_selection": None,
    "unit_selection_kwargs": None,
    "n_trials": 50,
    "n_jobs": 1,
    "random_seed": 1909,
    "compute_roa": True,
}
APPLY_KEYS: Set[str] = {"fixed_config"}
_SEARCH_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class Recording:
    """One FDSI recording.

    Attributes:
        sub (str): Subject id.
        cond (str): Condition name.
        snr (int): SNR level in dB.
    """

    sub: str
    cond: str
    snr: int

    @property
    def stub(self) -> str:
        """Canonical "<sub>_FDSI_<cond>_snr<N>dB" stub."""
        return fdsi.recording_stub(self.sub, self.cond, self.snr)


@dataclass(frozen=True)
class Task:
    """One unit of work of a stage: a recording to calibrate, a search, or a recording to apply.

    Attributes:
        stage (str): One of STAGES.
        index (int): Position in the stage's task list (the array index unit).
        id (str): Stable identifier, e.g. "pareto_sum/sub-01_FDSI_staircase_snr30dB".
        recording (Optional[Recording]): The recording (calibrate and apply).
        branch (Optional[str]): The search name (search and apply) or FIXED_BRANCH.
    """

    stage: str
    index: int
    id: str
    recording: Optional[Recording] = None
    branch: Optional[str] = None


def _check_keys(section: Any, allowed: Set[str], required: Set[str], where: str) -> None:
    """Check a spec section is a mapping with only allowed keys and every required one.

    Args:
        section (Any): The parsed section.
        allowed (Set[str]): Keys it may have.
        required (Set[str]): Keys it must have.
        where (str): Section name for error messages.

    Raises:
        ValueError: If section is not a mapping, or has unknown or missing keys.

    Returns:
        None
    """
    if not isinstance(section, dict):
        raise ValueError(f"Spec section {where!r} must be a mapping, got {type(section).__name__}")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in spec section {where!r}")
    missing = required - set(section)
    if missing:
        raise ValueError(f"Missing key(s) {sorted(missing)} in spec section {where!r}")


def _resolve_path(value: Union[str, Path]) -> Path:
    """Resolve a spec path: absolute as given, relative to the repository root otherwise.

    Args:
        value (Union[str, Path]): The path in the spec.

    Returns:
        Path: The absolute path.
    """
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _digest(payload: Dict[str, Any]) -> str:
    """SHA-256 of a payload's canonical JSON.

    Args:
        payload (Dict[str, Any]): JSON-serialisable content (non-JSON leaves are str()-ed).

    Returns:
        str: Hex digest.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, computed once per process per (path, size, mtime).

    Args:
        path (Path): The file.

    Raises:
        FileNotFoundError: If path doesn't exist, with a hint to download the data.

    Returns:
        str: Hex digest.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Download the data with "
            "'python scripts/download_data.py get fdsi_benchmark-data'."
        )
    stat = path.stat()
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=None)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash a file's bytes; size and mtime_ns only key the cache.

    Args:
        path (str): The file.
        size (int): Its size in bytes.
        mtime_ns (int): Its modification time in ns.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_array_index(array_index: Optional[int]) -> Optional[int]:
    """The array index given, else the scheduler's (PBS Pro, Torque, SLURM), else None.

    Args:
        array_index (Optional[int]): Explicit array index, or None.

    Returns:
        Optional[int]: The array index, or None if neither is set.
    """
    if array_index is not None:
        return array_index
    for var in ARRAY_INDEX_VARS:
        value = os.environ.get(var)
        if value:
            return int(value)
    return None


def select_tasks(
    tasks: Sequence[Task],
    *,
    task_index: Optional[int] = None,
    array_index: Optional[int] = None,
    chunk_size: int = 1,
    run_all: bool = False,
) -> List[Task]:
    """Pick the tasks one invocation runs: one task, one array chunk, or all of them.

    Args:
        tasks (Sequence[Task]): The stage's tasks.
        task_index (Optional[int], optional): Run only this task. Defaults to None.
        array_index (Optional[int], optional): Run tasks [array_index * chunk_size,
            (array_index + 1) * chunk_size); None falls back to the scheduler's
            array index (see resolve_array_index). Defaults to None.
        chunk_size (int, optional): Tasks per array index. Defaults to 1.
        run_all (bool, optional): Run every task. Defaults to False.

    Raises:
        ValueError: If not exactly one of task_index, an array index or run_all
            is given, or the index is out of range.

    Returns:
        List[Task]: The selected tasks.
    """
    if run_all:
        if task_index is not None or array_index is not None:
            raise ValueError("Pass only one of --all, --task-index and --array-index.")
        return list(tasks)
    if task_index is not None:
        if array_index is not None:
            raise ValueError("Pass only one of --task-index and --array-index.")
        if not 0 <= task_index < len(tasks):
            raise ValueError(f"Task index {task_index} out of range [0, {len(tasks)}).")
        return [tasks[task_index]]

    array_index = resolve_array_index(array_index)
    if array_index is None:
        raise ValueError(
            "No task selected: pass --task-index, --array-index or --all, or run inside a "
            f"scheduler array job (one of {list(ARRAY_INDEX_VARS)} set)."
        )
    if chunk_size < 1:
        raise ValueError(f"--chunk-size must be at least 1, got {chunk_size}.")
    start = array_index * chunk_size
    if not 0 <= start < len(tasks):
        raise ValueError(
            f"Array index {array_index} x chunk size {chunk_size} is beyond the {len(tasks)} tasks."
        )
    return list(tasks[start : start + chunk_size])


@dataclass
class BenchmarkSpec:
    """A parsed and validated benchmark spec (see benchmarks/fdsi/benchmark.yaml).

    Attributes:
        path (Path): The spec file.
        name (str): Run name, used in branch names and metadata.
        data_root (Path): Raw data root (<dataset>/data).
        outputs_root (Path): Where every output of this run goes.
        v10_outputs_root (Optional[Path]): Cached v1.0 outputs, read by import-v10.
        grid (Dict[str, Any]): Recordings and the shared signal settings.
        calibration (Dict[str, Any]): CBSSConfig fields and the supervised RoA threshold.
        pool (Dict[str, Any]): The recordings every search is run on.
        search (Dict[str, Any]): Settings shared by every search.
        searches (Dict[str, Dict[str, Any]]): Search name -> its own settings.
        apply (Dict[str, Any]): The fixed baseline's config file.
    """

    path: Path
    name: str
    data_root: Path
    outputs_root: Path
    v10_outputs_root: Optional[Path]
    grid: Dict[str, Any]
    calibration: Dict[str, Any]
    pool: Dict[str, Any]
    search: Dict[str, Any]
    searches: Dict[str, Dict[str, Any]]
    apply: Dict[str, Any]
    _keys: Dict[Tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)

    # Grid

    @property
    def fs(self) -> int:
        """Sampling frequency in Hz."""
        return int(self.grid["fs"])

    @property
    def cal_end(self) -> int:
        """Calibration window length in samples; adaptation starts here."""
        return int(self.grid["cal_duration_s"] * self.fs)

    @property
    def iso_dur(self) -> int:
        """Isometric bookend length of a triangular contraction, in samples."""
        return int(self.grid["iso_duration_s"] * self.fs)

    @property
    def tol_spike_ms(self) -> float:
        """Spike-alignment tolerance of every RoA, in ms."""
        return float(self.grid["tol_spike_ms"])

    def recordings(self) -> List[Recording]:
        """Every recording of the grid, subject-major.

        Returns:
            List[Recording]: subjects x conditions x snr_levels.
        """
        return [
            Recording(sub, cond, int(snr))
            for sub in self.grid["subjects"]
            for cond in self.grid["conditions"]
            for snr in self.grid["snr_levels"]
        ]

    def pool_recordings(self) -> List[Recording]:
        """The recordings every search is run on.

        Returns:
            List[Recording]: One per pool condition.
        """
        return [
            Recording(self.pool["subject"], cond, int(self.pool["snr"]))
            for cond in self.pool["conditions"]
        ]

    @property
    def branches(self) -> List[str]:
        """Applied configs: the fixed baseline, then each search's winner."""
        return [FIXED_BRANCH, *self.searches]

    @property
    def search_n_cores(self) -> int:
        """Cores of every search: n_jobs x pool size, so each run gets one thread."""
        return int(self.search["n_jobs"]) * len(self.pool["conditions"])

    def tasks(self, stage: str) -> List[Task]:
        """A stage's tasks, in a fixed order.

        Args:
            stage (str): One of STAGES.

        Raises:
            ValueError: If stage is unknown.

        Returns:
            List[Task]: calibrate: one per recording; search: one per search;
            apply: one per (branch, recording), branch-major.
        """
        if stage == "calibrate":
            return [Task(stage, i, rec.stub, rec) for i, rec in enumerate(self.recordings())]
        if stage == "search":
            return [Task(stage, i, name, branch=name) for i, name in enumerate(self.searches)]
        if stage == "apply":
            pairs = [(branch, rec) for branch in self.branches for rec in self.recordings()]
            return [
                Task(stage, i, f"{branch}/{rec.stub}", rec, branch)
                for i, (branch, rec) in enumerate(pairs)
            ]
        raise ValueError(f"Unknown stage: {stage!r}. Expected one of {list(STAGES)}.")

    def task_for(self, stage: str, task_id: str) -> Task:
        """Look a task up by its id.

        Args:
            stage (str): One of STAGES.
            task_id (str): The task's id.

        Raises:
            ValueError: If no task of stage has that id.

        Returns:
            Task: The task.
        """
        for task in self.tasks(stage):
            if task.id == task_id:
                return task
        raise ValueError(f"No {stage} task with id {task_id!r}.")

    # Configs

    def cbss_config(self) -> CBSSConfig:
        """The calibration's CBSSConfig.

        Raises:
            ValueError: If calibration.cbss_config names an unknown CBSSConfig field.

        Returns:
            CBSSConfig: Built from calibration.cbss_config.
        """
        values = self.calibration["cbss_config"]
        unknown = set(values) - {f.name for f in fields(CBSSConfig) if f.init}
        if unknown:
            raise ValueError(
                f"Unknown CBSSConfig field(s) {sorted(unknown)} in calibration.cbss_config"
            )
        return CBSSConfig(**values)

    def search_settings(self, name: str) -> Dict[str, Any]:
        """One search's settings: the shared ones, overridden by its own.

        Args:
            name (str): Search name.

        Returns:
            Dict[str, Any]: Every SEARCH_KEYS setting (defaults filled in), with
            param_space resolved (None -> DEFAULT_PARAM_SPACE, as tuples),
            objectives as a tuple, overrides merged (shared, then the search's,
            then its sv_loss_reduction) and n_cores added.
        """
        entry = self.searches[name]
        settings = {**SEARCH_DEFAULTS, **self.search, **entry}
        overrides = {**self.search.get("overrides", {}), **entry.get("overrides", {})}
        if "sv_loss_reduction" in settings:
            overrides["sv_loss_reduction"] = settings.pop("sv_loss_reduction")
        param_space = settings["param_space"]
        settings.update(
            overrides=overrides,
            objectives=tuple(settings["objectives"]),
            param_space=DEFAULT_PARAM_SPACE
            if param_space is None
            else {k: tuple(v) for k, v in param_space.items()},
            n_cores=self.search_n_cores,
        )
        return settings

    def search_base_config(self, name: str) -> AdaptConfig:
        """The AdaptConfig every trial of a search starts from.

        Args:
            name (str): Search name.

        Raises:
            ValueError: If an override names an unknown AdaptConfig field.

        Returns:
            AdaptConfig: The base config file with the search's overrides applied.
        """
        settings = self.search_settings(name)
        return _adapt_config(settings["base_config"], settings["overrides"], f"searches.{name}")

    def fixed_config(self) -> AdaptConfig:
        """The fixed (no-adaptation) baseline's AdaptConfig, on the CPU.

        Returns:
            AdaptConfig: apply.fixed_config with device="cpu".
        """
        return _adapt_config(self.apply["fixed_config"], {"device": "cpu"}, "apply")

    # Output paths

    @property
    def patch_dir(self) -> Path:
        """Where uncommitted diffs are saved, one file per distinct diff."""
        return self.outputs_root / "provenance" / "patches"

    @property
    def tables_dir(self) -> Path:
        """Where collect and import-v10 write their tables."""
        return self.outputs_root / "tables"

    def emg_path(self, rec: Recording) -> Path:
        """A recording's noisy EMG file."""
        return fdsi.emg_path(self.data_root, rec.sub, rec.cond, rec.snr)

    def gt_path(self, rec: Recording) -> Path:
        """A recording's ground-truth spikes file."""
        return fdsi.gt_spikes_path(self.data_root, rec.sub, rec.cond)

    def calibration_paths(self, rec: Recording) -> Dict[str, Path]:
        """A recording's calibration outputs.

        Args:
            rec (Recording): The recording.

        Returns:
            Dict[str, Path]: "result" (CBSSResult pickle), "config" (CBSSConfig
            YAML), "units" (per-unit CSV) and "meta" (metadata YAML).
        """
        stem = self.outputs_root / "calibration" / rec.sub / f"{rec.stub}_cbss"
        return {
            "result": stem.with_name(f"{stem.name}.pkl"),
            "config": stem.with_name(f"{stem.name}_config.yaml"),
            "units": stem.with_name(f"{stem.name}_units.csv"),
            "meta": stem.with_name(f"{stem.name}.meta.yaml"),
        }

    def search_paths(self, name: str) -> Dict[str, Path]:
        """A search's outputs.

        Args:
            name (str): Search name.

        Returns:
            Dict[str, Path]: "dir" (study.pkl, trials.csv, best_config.yaml,
            base_config.yaml, search.yaml, best/) and "meta" (metadata YAML).
        """
        out_dir = self.outputs_root / "searches" / name
        return {"dir": out_dir, "meta": out_dir.with_name(f"{name}.meta.yaml")}

    def result_paths(self, branch: str, rec: Recording) -> Dict[str, Path]:
        """An applied config's outputs on one recording.

        Args:
            branch (str): FIXED_BRANCH or a search name.
            rec (Recording): The recording.

        Returns:
            Dict[str, Path]: "result" (AdaptationResult pickle), "config"
            (AdaptConfig YAML), "metrics" (per-unit CSV) and "meta" (metadata YAML).
        """
        out_dir = self.outputs_root / "results" / branch / rec.sub
        return {
            "result": out_dir / f"{rec.stub}.pkl",
            "config": out_dir / f"{rec.stub}_config.yaml",
            "metrics": out_dir / f"{rec.stub}_metrics.csv",
            "meta": out_dir / f"{rec.stub}.meta.yaml",
        }

    def meta_path(self, task: Task) -> Path:
        """A task's metadata file, written last as its completion marker.

        Args:
            task (Task): The task.

        Returns:
            Path: The metadata YAML.
        """
        if task.stage == "calibrate":
            return self.calibration_paths(task.recording)["meta"]
        if task.stage == "search":
            return self.search_paths(task.branch)["meta"]
        return self.result_paths(task.branch, task.recording)["meta"]

    def task_command(self, task: Task) -> str:
        """The command that runs exactly this task, from the repository root.

        Args:
            task (Task): The task.

        Returns:
            str: e.g. "python -m benchmarks.fdsi apply --spec <spec> --task-index 17".
        """
        return f"python -m benchmarks.fdsi {task.stage} --spec {self.spec_ref} --task-index {task.index}"

    @property
    def spec_ref(self) -> str:
        """The spec path as given on the command line: relative to the repository root when inside it."""
        path = self.path.resolve()
        if path.is_relative_to(REPO_ROOT):
            return path.relative_to(REPO_ROOT).as_posix()
        return path.as_posix()

    def with_outputs_root(self, outputs_root: Union[str, Path]) -> "BenchmarkSpec":
        """A copy of this spec writing to another outputs root (e.g. verify's scratch).

        Args:
            outputs_root (Union[str, Path]): The new outputs root.

        Returns:
            BenchmarkSpec: The copy, with an empty key cache.
        """
        return replace(self, outputs_root=Path(outputs_root))

    # Cache keys and status

    def task_key(self, task: Task) -> str:
        """The cache key of a task: a hash of everything its output depends on.

        Covers the resolved settings, the content of its input files and its
        upstream tasks' keys, never paths or the git commit, so ordinary commits
        and moving the outputs don't invalidate anything.

        Args:
            task (Task): The task.

        Returns:
            str: Hex SHA-256.
        """
        if task.stage == "calibrate":
            return self.calibration_key(task.recording)
        if task.stage == "search":
            return self.search_key(task.branch)
        return self.apply_key(task.branch, task.recording)

    def calibration_key(self, rec: Recording) -> str:
        """Cache key of a recording's calibration."""
        cache = ("calibrate", rec.stub)
        if cache not in self._keys:
            self._keys[cache] = _digest(
                {
                    "stage": "calibrate",
                    "version": STAGE_VERSIONS["calibrate"],
                    "id": rec.stub,
                    "cbss_config": self.cbss_config().to_dict(),
                    "supervised_roa_th": self.calibration["supervised_roa_th"],
                    "tol_spike_ms": self.tol_spike_ms,
                    "fs": self.fs,
                    "cal_end": self.cal_end,
                    "inputs": {
                        "emg": file_sha256(self.emg_path(rec)),
                        "gt": file_sha256(self.gt_path(rec)),
                    },
                }
            )
        return self._keys[cache]

    def search_key(self, name: str) -> str:
        """Cache key of a search."""
        cache = ("search", name)
        if cache not in self._keys:
            settings = self.search_settings(name)
            self._keys[cache] = _digest(
                {
                    "stage": "search",
                    "version": STAGE_VERSIONS["search"],
                    "id": name,
                    "settings": {k: v for k, v in settings.items() if k != "base_config"},
                    "base_config": self.search_base_config(name).to_dict(),
                    "tol_spike_ms": self.tol_spike_ms,
                    "cal_end": self.cal_end,
                    "pool": {rec.cond: self.calibration_key(rec) for rec in self.pool_recordings()},
                }
            )
        return self._keys[cache]

    def apply_key(self, branch: str, rec: Recording) -> str:
        """Cache key of an applied config on one recording."""
        cache = ("apply", f"{branch}/{rec.stub}")
        if cache not in self._keys:
            config = (
                {"fixed_config": self.fixed_config().to_dict()}
                if branch == FIXED_BRANCH
                else {"search": self.search_key(branch)}
            )
            self._keys[cache] = _digest(
                {
                    "stage": "apply",
                    "version": STAGE_VERSIONS["apply"],
                    "id": f"{branch}/{rec.stub}",
                    "config": config,
                    "calibration": self.calibration_key(rec),
                    "fs": self.fs,
                    "cal_end": self.cal_end,
                    "iso_dur": self.iso_dur,
                    "tol_spike_ms": self.tol_spike_ms,
                }
            )
        return self._keys[cache]

    def task_status(self, task: Task) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Whether a task's output is current.

        Args:
            task (Task): The task.

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: The status and its metadata
            (None when missing). Status is "missing" (no metadata), "stale" (made
            with another key), or the recorded "done" or "skipped".
        """
        meta_path = self.meta_path(task)
        if not meta_path.exists():
            return "missing", None
        meta = read_metadata(meta_path)
        if meta.get("task", {}).get("key") != self.task_key(task):
            return "stale", meta
        return meta["task"]["status"], meta


def _adapt_config(path: Union[str, Path], overrides: Dict[str, Any], where: str) -> AdaptConfig:
    """Build an AdaptConfig from a YAML file plus overrides.

    Args:
        path (Union[str, Path]): Config file, relative to the repository root.
        overrides (Dict[str, Any]): AdaptConfig fields to replace.
        where (str): Spec section for error messages.

    Raises:
        ValueError: If the file is missing or an override names an unknown field.

    Returns:
        AdaptConfig: The config.
    """
    config_path = _resolve_path(path)
    if not config_path.exists():
        raise ValueError(f"Config file {path!r} in spec section {where!r} not found.")
    valid = {f.name for f in fields(AdaptConfig) if f.init}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown AdaptConfig field(s) {sorted(unknown)} in spec section {where!r}"
        )
    return AdaptConfig(**{**AdaptConfig.from_yaml(config_path).to_dict(), **overrides})


def load_spec(
    path: Union[str, Path] = DEFAULT_SPEC, outputs_root: Optional[Union[str, Path]] = None
) -> BenchmarkSpec:
    """Load and validate a benchmark spec.

    Args:
        path (Union[str, Path], optional): The spec YAML, relative to the
            repository root or absolute. Defaults to DEFAULT_SPEC.
        outputs_root (Optional[Union[str, Path]], optional): Replace the spec's
            outputs_root (e.g. for a scratch run). Defaults to None.

    Raises:
        ValueError: If the file is missing, a section has unknown or missing
            keys, the pool is outside the grid, a search name is invalid, or a
            search's objectives, selection or unit selection is invalid.

    Returns:
        BenchmarkSpec: The validated spec.
    """
    spec_path = _resolve_path(path)
    if not spec_path.exists():
        raise ValueError(f"Spec file not found: {path}")
    with spec_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Sections
    _check_keys(raw, TOP_KEYS, TOP_KEYS - {"v10_outputs_root"}, "spec")
    _check_keys(raw["grid"], GRID_KEYS, GRID_KEYS, "grid")
    _check_keys(raw["calibration"], CALIBRATION_KEYS, CALIBRATION_KEYS, "calibration")
    _check_keys(raw["pool"], POOL_KEYS, POOL_KEYS, "pool")
    _check_keys(raw["search"], SEARCH_KEYS, {"base_config"}, "search")
    _check_keys(raw["apply"], APPLY_KEYS, APPLY_KEYS, "apply")
    if not isinstance(raw["searches"], dict) or not raw["searches"]:
        raise ValueError(
            "Spec section 'searches' must map at least one search name to its settings"
        )
    for name, entry in raw["searches"].items():
        if not _SEARCH_NAME.match(str(name)) or name == FIXED_BRANCH:
            raise ValueError(
                f"Invalid search name {name!r}: use letters, digits, '_' or '-', not "
                f"{FIXED_BRANCH!r}"
            )
        _check_keys(entry or {}, SEARCH_KEYS - {"base_config"}, set(), f"searches.{name}")

    # The pool must be part of the grid
    grid, pool = raw["grid"], raw["pool"]
    if pool["subject"] not in grid["subjects"] or pool["snr"] not in grid["snr_levels"]:
        raise ValueError(f"Pool subject/snr {pool['subject']!r}/{pool['snr']!r} is not in the grid")
    outside = set(pool["conditions"]) - set(grid["conditions"])
    if outside:
        raise ValueError(f"Pool condition(s) {sorted(outside)} are not in the grid")

    spec = BenchmarkSpec(
        path=spec_path,
        name=str(raw["name"]),
        data_root=_resolve_path(raw["data_root"]),
        outputs_root=_resolve_path(
            outputs_root if outputs_root is not None else raw["outputs_root"]
        ),
        v10_outputs_root=_resolve_path(raw["v10_outputs_root"])
        if raw.get("v10_outputs_root")
        else None,
        grid=grid,
        calibration=raw["calibration"],
        pool=pool,
        search=raw["search"],
        searches={name: entry or {} for name, entry in raw["searches"].items()},
        apply=raw["apply"],
    )

    # Every config builds and every search is well-formed
    spec.cbss_config()
    spec.fixed_config()
    for name in spec.searches:
        settings = spec.search_settings(name)
        if "objectives" not in settings:
            raise ValueError(f"Search {name!r} has no objectives")
        validate_objectives(settings["objectives"])
        if len(settings["objectives"]) > 1:
            validate_selection(settings["selection"], settings["objectives"])
        validate_unit_selection(settings["unit_selection"])
        for key in ("n_trials", "n_jobs"):
            if int(settings[key]) < 1:
                raise ValueError(f"Search {name!r}: {key} must be at least 1")
        spec.search_base_config(name)
    return spec
