"""The FDSI benchmark spec: loading and validation, tasks, output paths, cache keys and status."""

import hashlib
import json
import os
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml

from adapt_decomp import CBSSConfig
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import DEFAULT_PARAM_SPACE
from adapt_decomp.adaptation.optimize.pareto import validate_selection
from adapt_decomp.adaptation.optimize.scoring import validate_objectives
from adapt_decomp.adaptation.optimize.search import validate_initial_params
from adapt_decomp.adaptation.optimize.units import validate_unit_selection
from adapt_decomp.utils import read_metadata
from benchmarks.fdsi import fdsi
from benchmarks.fdsi.fdsi import Recording

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = "benchmarks/fdsi/benchmark.yaml"

STAGES: Tuple[str, ...] = ("calibrate", "search", "apply")
FIXED_BRANCH = "fixed"  # the no-adaptation baseline, applied next to every search's winner
# Torch threads of a calibrate or apply run; a search sets its own (threads_per_run)
TORCH_THREADS = 1

# Version of each stage's outputs, hashed into its cache keys: bump it when a stage's code
# changes what it writes, so the outputs written before become stale (and so does every
# output downstream of them). apply 4: its metadata reports roa_full_mean.
STAGE_VERSIONS: Dict[str, int] = {"calibrate": 2, "search": 1, "apply": 4}

# The tables collect writes to <outputs_root>/tables/, in order
TABLES: Tuple[str, ...] = (
    "calibrations",
    "calibration_units",
    "searches",
    "best_configs",
    "recordings",
    "units",
    "provenance",
)

# Array-index environment variables, in lookup order: PBS Pro, Torque, SLURM
ARRAY_INDEX_VARS: Tuple[str, ...] = ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID")

# Spec sections: each field is a key of the section, those without a default are required, and
# load_spec rejects any other key.


@dataclass(frozen=True)
class Grid:
    """The recordings and the signal settings every stage shares.

    Attributes:
        subjects (List[str]): Subject ids.
        conditions (List[str]): Condition names.
        snr_levels (List[int]): SNR levels in dB.
        fs (int): Sampling frequency in Hz.
        cal_duration_s (float): Calibration window, from the start of each recording, in s.
        iso_duration_s (float): Isometric bookends of the triangular contractions, in s.
        tol_spike_ms (float): Spike-alignment tolerance of every RoA, in ms.
    """

    subjects: List[str]
    conditions: List[str]
    snr_levels: List[int]
    fs: int
    cal_duration_s: float
    iso_duration_s: float
    tol_spike_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "fs", int(self.fs))
        object.__setattr__(self, "tol_spike_ms", float(self.tol_spike_ms))

    @property
    def cal_end(self) -> int:
        """Calibration window length in samples; adaptation starts here."""
        return int(self.cal_duration_s * self.fs)

    @property
    def iso_dur(self) -> int:
        """Isometric bookend length of a triangular contraction, in samples."""
        return int(self.iso_duration_s * self.fs)


@dataclass(frozen=True)
class Calibration:
    """CBSS's settings (CBSSConfig fields) and the supervised selection's RoA threshold."""

    cbss_config: Dict[str, Any]
    supervised_roa_th: float


@dataclass(frozen=True)
class Pool:
    """The recordings every search runs on: one subject and SNR level, several conditions."""

    subject: str
    snr: int
    conditions: List[str]


@dataclass(frozen=True)
class Search:
    """One search: optimize_adapt_decomp's settings and the AdaptConfig its trials start from.

    The spec's search section sets them for every search, and each entry of searches
    overrides some (not base_config or threads_per_run, shared by all). load_spec resolves
    each search's objectives (a tuple), param_space (None -> DEFAULT_PARAM_SPACE),
    initial_params ({param: value} dicts; a config file gives its values of the searched
    parameters) and overrides (the shared ones, then the search's, then sv_loss_reduction).
    """

    base_config: str
    objectives: Tuple[str, ...] = ()
    overrides: Dict[str, Any] = field(default_factory=dict)
    sv_loss_reduction: Optional[str] = None
    param_space: Optional[Dict[str, Any]] = None
    initial_params: Optional[List[Any]] = None
    selection: str = "min_sv_loss"
    unit_selection: Optional[str] = None
    unit_selection_kwargs: Optional[Dict[str, Any]] = None
    n_trials: int = 50
    n_jobs: int = 1
    threads_per_run: int = 1  # speed only: outside the cache key
    random_seed: int = 1909
    compute_roa: bool = True


@dataclass(frozen=True)
class Apply:
    """The no-adaptation baseline's AdaptConfig file, applied next to every search's winner."""

    fixed_config: str


SHARED_ONLY = {"base_config", "threads_per_run"}  # Search fields a search can't override
_SEARCH_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


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


def _section(cls: type, raw: Any, where: str) -> Any:
    """Build a section's dataclass from its parsed YAML.

    Args:
        cls (type): The section's dataclass.
        raw (Any): The parsed section.
        where (str): Section name for error messages.

    Raises:
        ValueError: As _check_keys, for keys outside cls's fields or required ones missing.

    Returns:
        Any: The cls instance.
    """
    required = {
        f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
    }
    _check_keys(raw, {f.name for f in fields(cls)}, required, where)
    return cls(**raw)


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
            "'adapt-decomp-data get fdsi_benchmark-data'."
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
        grid (Grid): The recordings and the shared signal settings.
        calibration (Calibration): CBSS's settings and the supervised RoA threshold.
        pool (Pool): The recordings every search is run on.
        searches (Dict[str, Search]): Search name -> its resolved settings.
        apply (Apply): The fixed baseline's config file.
    """

    path: Path
    name: str
    data_root: Path
    outputs_root: Path
    grid: Grid
    calibration: Calibration
    pool: Pool
    searches: Dict[str, Search]
    apply: Apply
    _keys: Dict[Tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)

    # Recordings and tasks

    def recordings(self) -> List[Recording]:
        """Every recording of the grid, subject-major.

        Returns:
            List[Recording]: subjects x conditions x snr_levels.
        """
        return [
            Recording(sub, cond, int(snr))
            for sub in self.grid.subjects
            for cond in self.grid.conditions
            for snr in self.grid.snr_levels
        ]

    def pool_recordings(self) -> List[Recording]:
        """The recordings every search is run on.

        Returns:
            List[Recording]: One per pool condition.
        """
        return [
            Recording(self.pool.subject, cond, int(self.pool.snr)) for cond in self.pool.conditions
        ]

    @property
    def branches(self) -> List[str]:
        """Applied configs: the fixed baseline, then each search's winner."""
        return [FIXED_BRANCH, *self.searches]

    @property
    def search_n_cores(self) -> int:
        """Cores of every search job: n_jobs x pool size x threads_per_run (the largest n_jobs).

        Each run of a guided trial gets threads_per_run torch threads; the random start-up
        trials, suggested together, run that many more at once on a thread each instead.
        """
        return len(self.pool.conditions) * max(
            s.n_jobs * s.threads_per_run for s in self.searches.values()
        )

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
        values = self.calibration.cbss_config
        unknown = set(values) - {f.name for f in fields(CBSSConfig) if f.init}
        if unknown:
            raise ValueError(
                f"Unknown CBSSConfig field(s) {sorted(unknown)} in calibration.cbss_config"
            )
        return CBSSConfig(**values)

    def search_base_config(self, name: str) -> AdaptConfig:
        """The AdaptConfig every trial of a search starts from.

        Args:
            name (str): Search name.

        Raises:
            ValueError: If an override names an unknown AdaptConfig field.

        Returns:
            AdaptConfig: The base config file with the search's overrides applied.
        """
        search = self.searches[name]
        return _adapt_config(search.base_config, search.overrides, f"searches.{name}")

    def fixed_config(self) -> AdaptConfig:
        """The fixed (no-adaptation) baseline's AdaptConfig, on the CPU.

        Returns:
            AdaptConfig: apply.fixed_config with device="cpu".
        """
        return _adapt_config(self.apply.fixed_config, {"device": "cpu"}, "apply")

    # Output paths

    @property
    def patch_dir(self) -> Path:
        """Where uncommitted diffs are saved, one file per distinct diff."""
        return self.outputs_root / "provenance" / "patches"

    @property
    def tables_dir(self) -> Path:
        """Where collect writes its tables."""
        return self.outputs_root / "tables"

    def emg_path(self, rec: Recording) -> Path:
        """A recording's noisy EMG file."""
        return fdsi.emg_path(self.data_root, rec)

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
                    "supervised_roa_th": self.calibration.supervised_roa_th,
                    "tol_spike_ms": self.grid.tol_spike_ms,
                    "fs": self.grid.fs,
                    "cal_end": self.grid.cal_end,
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
            settings = asdict(self.searches[name])
            self._keys[cache] = _digest(
                {
                    "stage": "search",
                    "version": STAGE_VERSIONS["search"],
                    "id": name,
                    "settings": {k: v for k, v in settings.items() if k not in SHARED_ONLY},
                    "base_config": self.search_base_config(name).to_dict(),
                    "tol_spike_ms": self.grid.tol_spike_ms,
                    "cal_end": self.grid.cal_end,
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
                    "fs": self.grid.fs,
                    "cal_end": self.grid.cal_end,
                    "iso_dur": self.grid.iso_dur,
                    "tol_spike_ms": self.grid.tol_spike_ms,
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


def _resolve_search(shared: Search, name: str, entry: Dict[str, Any]) -> Search:
    """One search's settings: the shared ones overridden by its own, resolved (see Search).

    Args:
        shared (Search): The spec's search section.
        name (str): The search's name.
        entry (Dict[str, Any]): Its own settings.

    Raises:
        ValueError: If an initial_params config file is missing.

    Returns:
        Search: The resolved settings.
    """
    merged = replace(
        shared, **{**entry, "overrides": {**shared.overrides, **entry.get("overrides", {})}}
    )
    param_space = (
        DEFAULT_PARAM_SPACE
        if merged.param_space is None
        else {k: tuple(v) for k, v in merged.param_space.items()}
    )
    initial_params = [
        dict(p)
        if isinstance(p, dict)
        else {  # a config file: its values of the searched parameters
            k: getattr(_adapt_config(p, {}, f"searches.{name}.initial_params"), k)
            for k in param_space
        }
        for p in merged.initial_params or []
    ]
    overrides = dict(merged.overrides)
    if merged.sv_loss_reduction is not None:
        overrides["sv_loss_reduction"] = merged.sv_loss_reduction
    return replace(
        merged,
        objectives=tuple(merged.objectives),
        overrides=overrides,
        sv_loss_reduction=None,
        param_space=param_space,
        initial_params=initial_params,
    )


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
            search's objectives, selection, unit selection or initial
            parameters are invalid.

    Returns:
        BenchmarkSpec: The validated spec.
    """
    spec_path = _resolve_path(path)
    if not spec_path.exists():
        raise ValueError(f"Spec file not found: {path}")
    with spec_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Sections
    top = {
        "name",
        "data_root",
        "outputs_root",
        "grid",
        "calibration",
        "pool",
        "search",
        "searches",
        "apply",
    }
    _check_keys(raw, top, top, "spec")
    grid = _section(Grid, raw["grid"], "grid")
    pool = _section(Pool, raw["pool"], "pool")
    shared = _section(Search, raw["search"], "search")
    if not isinstance(raw["searches"], dict) or not raw["searches"]:
        raise ValueError(
            "Spec section 'searches' must map at least one search name to its settings"
        )
    searches = {}
    for name, entry in raw["searches"].items():
        if not _SEARCH_NAME.match(str(name)) or name == FIXED_BRANCH:
            raise ValueError(
                f"Invalid search name {name!r}: use letters, digits, '_' or '-', not "
                f"{FIXED_BRANCH!r}"
            )
        own = {f.name for f in fields(Search)} - SHARED_ONLY
        _check_keys(entry or {}, own, set(), f"searches.{name}")
        searches[name] = _resolve_search(shared, name, entry or {})

    # The pool must be part of the grid
    if pool.subject not in grid.subjects or pool.snr not in grid.snr_levels:
        raise ValueError(f"Pool subject/snr {pool.subject!r}/{pool.snr!r} is not in the grid")
    outside = set(pool.conditions) - set(grid.conditions)
    if outside:
        raise ValueError(f"Pool condition(s) {sorted(outside)} are not in the grid")

    spec = BenchmarkSpec(
        path=spec_path,
        name=str(raw["name"]),
        data_root=_resolve_path(raw["data_root"]),
        outputs_root=_resolve_path(
            outputs_root if outputs_root is not None else raw["outputs_root"]
        ),
        grid=grid,
        calibration=_section(Calibration, raw["calibration"], "calibration"),
        pool=pool,
        searches=searches,
        apply=_section(Apply, raw["apply"], "apply"),
    )

    # Every config builds and every search is well-formed
    spec.cbss_config()
    spec.fixed_config()
    for name, search in spec.searches.items():
        if not search.objectives:
            raise ValueError(f"Search {name!r} has no objectives")
        validate_objectives(search.objectives)
        if len(search.objectives) > 1:
            validate_selection(search.selection, search.objectives)
        validate_unit_selection(search.unit_selection)
        validate_initial_params(search.initial_params, search.param_space)
        for key in ("n_trials", "n_jobs", "threads_per_run"):
            if int(getattr(search, key)) < 1:
                raise ValueError(f"Search {name!r}: {key} must be at least 1")
        spec.search_base_config(name)
    return spec
