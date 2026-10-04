"""Command line of the FDSI benchmark: one command per stage, run from the repository root.

python -m benchmarks.fdsi tasks apply --count
python -m benchmarks.fdsi calibrate --all --n-workers 8
python -m benchmarks.fdsi apply --array-index 3 --chunk-size 4   # or $PBS_ARRAY_INDEX
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer

from benchmarks.fdsi import stages
from benchmarks.fdsi.spec import DEFAULT_SPEC, BenchmarkSpec, load_spec, select_tasks

app = typer.Typer(
    help="FDSI benchmark: calibrate, search, apply, then collect. Run from the repository root.",
    no_args_is_help=True,
    add_completion=False,
)

SPEC_HELP = "Benchmark spec YAML, relative to the repository root"


def _command() -> List[str]:
    """The command as invoked, for the metadata.

    Returns:
        List[str]: ["python", "-m", "benchmarks.fdsi", *arguments].
    """
    return ["python", "-m", "benchmarks.fdsi", *sys.argv[1:]]


def _run_stage(
    stage: str,
    spec_path: str,
    task_index: Optional[int],
    array_index: Optional[int],
    chunk_size: int,
    run_all: bool,
    n_workers: int,
    force: bool,
) -> None:
    """Select a stage's tasks and run them.

    Args:
        stage (str): "calibrate", "search" or "apply".
        spec_path (str): The spec YAML.
        task_index (Optional[int]): One task.
        array_index (Optional[int]): One array chunk (None: the scheduler's index).
        chunk_size (int): Tasks per array index.
        run_all (bool): Every task.
        n_workers (int): Worker processes for run_all.
        force (bool): Recompute stale outputs.

    Returns:
        None
    """
    spec = load_spec(spec_path)
    tasks = select_tasks(
        spec.tasks(stage),
        task_index=task_index,
        array_index=array_index,
        chunk_size=chunk_size,
        run_all=run_all,
    )
    counts = stages.run_tasks(spec, tasks, command=_command(), force=force, n_workers=n_workers)
    typer.echo(f"{stage}: {len(tasks)} task(s); before running: {counts}")


TASK_INDEX = typer.Option(None, "--task-index", help="Run only this task")
ARRAY_INDEX = typer.Option(
    None, "--array-index", help="Run one array chunk (default: $PBS_ARRAY_INDEX)"
)
CHUNK_SIZE = typer.Option(1, "--chunk-size", help="Tasks per array index")
RUN_ALL = typer.Option(False, "--all", help="Run every task of the stage")
N_WORKERS = typer.Option(1, "--n-workers", help="Worker processes for --all (one thread each)")
FORCE = typer.Option(False, "--force", help="Recompute outputs made from other settings")


@app.command()
def tasks(
    stage: str = typer.Argument(..., help="calibrate, search or apply"),
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    count: bool = typer.Option(False, "--count", help="Print only the number of tasks"),
    status: bool = typer.Option(False, "--status", help="Also show whether each is current"),
    ncpus: bool = typer.Option(False, "--ncpus", help="Print only the cores one task uses"),
) -> None:
    """List a stage's tasks (index and id), their count, status or cores."""
    benchmark = load_spec(spec)
    stage_tasks = benchmark.tasks(stage)
    if count:
        typer.echo(len(stage_tasks))
        return
    if ncpus:
        typer.echo(benchmark.search_n_cores if stage == "search" else 1)
        return
    statuses = []
    for task in stage_tasks:
        line = f"{task.index:4d}  {task.id}"
        if status:
            task_status, _ = benchmark.task_status(task)
            statuses.append(task_status)
            line += f"  {task_status}"
        typer.echo(line)
    if status:
        typer.echo({s: statuses.count(s) for s in sorted(set(statuses))})


@app.command()
def calibrate(
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    task_index: Optional[int] = TASK_INDEX,
    array_index: Optional[int] = ARRAY_INDEX,
    chunk_size: int = CHUNK_SIZE,
    run_all: bool = RUN_ALL,
    n_workers: int = N_WORKERS,
    force: bool = FORCE,
) -> None:
    """Calibrate recordings with CBSS, keeping the units matching their ground truth."""
    _run_stage("calibrate", spec, task_index, array_index, chunk_size, run_all, n_workers, force)


@app.command()
def search(
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    task_index: Optional[int] = TASK_INDEX,
    array_index: Optional[int] = ARRAY_INDEX,
    chunk_size: int = CHUNK_SIZE,
    run_all: bool = RUN_ALL,
    force: bool = FORCE,
) -> None:
    """Run the hyperparameter searches on the pool (each uses n_jobs x pool-size cores)."""
    _run_stage("search", spec, task_index, array_index, chunk_size, run_all, 1, force)


@app.command()
def apply(
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    task_index: Optional[int] = TASK_INDEX,
    array_index: Optional[int] = ARRAY_INDEX,
    chunk_size: int = CHUNK_SIZE,
    run_all: bool = RUN_ALL,
    n_workers: int = N_WORKERS,
    force: bool = FORCE,
) -> None:
    """Apply the fixed baseline and each search's winner to every recording."""
    _run_stage("apply", spec, task_index, array_index, chunk_size, run_all, n_workers, force)


@app.command()
def collect(spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP)) -> None:
    """Gather every current output into CSV tables (read-only, works on a partial run)."""
    benchmark = load_spec(spec)
    paths = stages.collect(benchmark, command=_command())
    typer.echo(f"Tables written to {benchmark.tables_dir}: {sorted(paths)}")


@app.command("import-v10")
def import_v10(
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    n_workers: int = typer.Option(1, "--n-workers", help="Worker processes"),
) -> None:
    """Compute the same per-unit metrics from the cached v1.0 results."""
    path = stages.import_v10(load_spec(spec), command=_command(), n_workers=n_workers)
    typer.echo(f"v1.0 per-unit metrics written to {path}")


@app.command()
def verify(
    stage: str = typer.Argument(..., help="calibrate, search or apply"),
    task_indices: str = typer.Option(..., "--tasks", help="Comma-separated task indices"),
    spec: str = typer.Option(DEFAULT_SPEC, "--spec", help=SPEC_HELP),
    scratch: Optional[str] = typer.Option(
        None, "--scratch", help="Re-run outputs root (default: <outputs_root>_verify)"
    ),
) -> None:
    """Re-run tasks into a scratch root and check they reproduce the cached outputs."""
    benchmark: BenchmarkSpec = load_spec(spec)
    scratch_root = (
        Path(scratch)
        if scratch is not None
        else benchmark.outputs_root.with_name(f"{benchmark.outputs_root.name}_verify")
    )
    indices = [int(i) for i in task_indices.split(",") if i.strip()]
    report = stages.verify(benchmark, stage, indices, scratch_root, command=_command())
    typer.echo(report[["stage", "id", "result"]].to_string(index=False))
    if (report["result"] == "MISMATCH").any():
        raise typer.Exit(code=1)
