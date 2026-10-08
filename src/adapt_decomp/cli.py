"""The adapt-decomp command: calibrate, adapt and tune from the shell.

Each command wraps the public function it is named after, with options named after that
function's arguments: decompose (CBSS.decompose), process_data (AdaptDecomp.process_data),
calibrate_and_process (AdaptDecomp.calibrate_and_process) and optimize_adapt_decomp. data
downloads the datasets, and wandb_sweep runs a wandb sweep (with the wandb extra).
"""

import copy
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import optuna
import typer
import yaml

from adapt_decomp.adaptation import PRESETS, AdaptationResult, AdaptConfig, AdaptDecomp
from adapt_decomp.adaptation import optimize_adapt_decomp as _optimize_adapt_decomp
from adapt_decomp.adaptation.optimize import DEFAULT_PARAM_SPACE
from adapt_decomp.adaptation.optimize.units import has_gt
from adapt_decomp.cbss import CBSS, CBSSConfig, CBSSResult
from adapt_decomp.spikes import rate_of_agreement_paired
from adapt_decomp.utils import download
from adapt_decomp.utils.loaders import PooledDataset, load_data, load_emg, load_gt

app = typer.Typer(
    help="Adaptive EMG decomposition: calibrate, adapt and tune hyperparameters.",
    no_args_is_help=True,
)
app.add_typer(download.app, name="data")

CALIBRATION_FILE = "calibration.pkl"
CALIBRATION_CONFIG_FILE = "calibration_config.yaml"
WANDB_MISSING = 'wandb is not installed; install it with: pip install "adapt-decomp[wandb]"'


# ------------------------------------------------------------------
# Choices and shared options
# ------------------------------------------------------------------


# The configs shipped with the package (AdaptConfig.from_preset)
Preset = Enum("Preset", {name: name for name in PRESETS}, type=str)


class FileFormat(str, Enum):
    """On-disk formats of the EMG and ground-truth files (load_emg, load_gt)."""

    npz = "npz"
    neuromotion = "neuromotion"


class ProcessingMode(str, Enum):
    """AdaptDecomp.process_data's processing_mode."""

    offline = "offline"
    online = "online"


class AdaptFrom(str, Enum):
    """AdaptDecomp.calibrate_and_process's adapt_from."""

    calib_end = "calib_end"
    emg_start = "emg_start"


class Objective(str, Enum):
    """optimize_adapt_decomp's objectives."""

    sv_loss = "sv_loss"
    wh_loss = "wh_loss"
    total_loss = "total_loss"
    roa = "roa"


EmgArgument = Annotated[
    Path,
    typer.Argument(
        exists=True, dir_okay=False, help="EMG recording, (samples, channels); see --emg_loader."
    ),
]
CbssConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--cbss_config",
        exists=True,
        dir_okay=False,
        help="CBSSConfig YAML (CBSSConfig.to_yaml). Omit for CBSSConfig()'s defaults.",
    ),
]
PresetOption = Annotated[
    Optional[Preset],
    typer.Option("--preset", help="Adaptation preset. Use this or --adapt_config."),
]
AdaptConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--adapt_config",
        exists=True,
        dir_okay=False,
        help="AdaptConfig YAML (AdaptConfig.to_yaml). Use this or --preset; omit both for "
        "AdaptConfig()'s defaults.",
    ),
]
SourceFifoOption = Annotated[
    Optional[bool],
    typer.Option(
        "--source_fifo_from_calib/--no-source_fifo_from_calib",
        help="Seed the source FIFO with the calibration's last sources: on when the EMG "
        "starts where the calibration window ends. Omit to keep the config's.",
        show_default=False,
    ),
]
GtOption = Annotated[
    Optional[Path],
    typer.Option(
        "--gt",
        exists=True,
        dir_okay=False,
        help="Ground-truth spikes: keep only the units that match one (select_supervised).",
    ),
]
RoaThOption = Annotated[
    float, typer.Option("--roa_th", help="Minimum rate of agreement of a kept unit, with --gt.")
]
EmgLoaderOption = Annotated[
    FileFormat, typer.Option("--emg_loader", help="Format of the EMG file.")
]
GtLoaderOption = Annotated[
    FileFormat, typer.Option("--gt_loader", help="Format of the ground-truth file.")
]
ProcessingModeOption = Annotated[
    ProcessingMode,
    typer.Option(
        "--processing_mode",
        help="offline preprocesses the whole recording first; online each raw batch.",
    ),
]
DataConfigOption = Annotated[
    Path,
    typer.Option(
        "--data_config",
        exists=True,
        dir_okay=False,
        help="Recordings and their calibrations, as a pool YAML "
        "(load_pooled_cbss_memory's format; loader: load_pooled_cbss_disk loads them per run).",
    ),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    """Read a YAML file into a dict; an empty dict for no file.

    Args:
        path (Optional[Path]): The file, or None.

    Returns:
        Dict[str, Any]: Its contents.
    """
    if path is None:
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _cbss_config(path: Optional[Path]) -> CBSSConfig:
    """Load a CBSSConfig from YAML, or CBSSConfig() for no file.

    Args:
        path (Optional[Path]): CBSSConfig YAML, or None.

    Returns:
        CBSSConfig: The config.
    """
    return CBSSConfig.from_yaml(path) if path is not None else CBSSConfig()


def _adapt_config(
    preset: Optional[Preset],
    adapt_config: Optional[Path],
    source_fifo_from_calib: Optional[bool] = None,
) -> AdaptConfig:
    """Load the adaptation config from --preset or --adapt_config.

    Args:
        preset (Optional[Preset]): A preset name, or None.
        adapt_config (Optional[Path]): An AdaptConfig YAML, or None.
        source_fifo_from_calib (Optional[bool]): Overrides the config's, unless None.

    Raises:
        typer.BadParameter: If both preset and adapt_config are given.

    Returns:
        AdaptConfig: The preset's, the file's, or AdaptConfig() for neither.
    """
    if preset is not None and adapt_config is not None:
        raise typer.BadParameter("Give --preset or --adapt_config, not both.")
    if preset is not None:
        config = AdaptConfig.from_preset(preset.value)
    elif adapt_config is not None:
        config = AdaptConfig.from_yaml(adapt_config)
    else:
        config = AdaptConfig()
    if source_fifo_from_calib is not None:
        config.source_fifo_from_calib = source_fifo_from_calib
    return config


def _timestamps(n_samples: int, fs: float) -> np.ndarray:
    """Sample times of a recording, in s, from its first sample.

    Args:
        n_samples (int): Samples in the recording.
        fs (float): Sampling frequency, in Hz.

    Returns:
        np.ndarray: Times with shape (n_samples,).
    """
    return np.arange(n_samples) / fs


def _save_calibration(calibration: CBSSResult, cbss_config: CBSSConfig, out_dir: Path) -> None:
    """Save a calibration and its config side by side, as process_data reads them.

    Args:
        calibration (CBSSResult): The calibration.
        cbss_config (CBSSConfig): The config that produced it.
        out_dir (Path): Folder to write calibration.pkl and calibration_config.yaml into.

    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration.save(out_dir / CALIBRATION_FILE)
    cbss_config.to_yaml(out_dir / CALIBRATION_CONFIG_FILE)


def _describe(result: AdaptationResult) -> str:
    """Summarise an adaptation's output in one line.

    Args:
        result (AdaptationResult): The output.

    Returns:
        str: Units, samples and time per batch.
    """
    n_samples, n_units = result.spikes.shape
    return (
        f"{n_units} units over {n_samples} samples, "
        f"{result.total_time_ms.float().mean():.1f} ms per batch"
    )


def _load_pool(data_config: Path) -> Dict[str, PooledDataset]:
    """Load a pool YAML into optimize_adapt_decomp's pool.

    Args:
        data_config (Path): The pool YAML. Its loader defaults to load_pooled_cbss_memory.

    Raises:
        typer.BadParameter: If its loader is not a pooled one.

    Returns:
        Dict[str, PooledDataset]: Dataset name -> its pool entry.
    """
    config = _read_yaml(data_config)
    config.setdefault("loader", "load_pooled_cbss_memory")
    if config["loader"] not in ("load_pooled_cbss_memory", "load_pooled_cbss_disk"):
        raise typer.BadParameter(
            f"loader: {config['loader']} is not supported; use load_pooled_cbss_memory or "
            "load_pooled_cbss_disk.",
            param_hint="--data_config",
        )
    return load_data(config)


def _wandb():
    """Import wandb, or exit with the command that installs it.

    Raises:
        typer.Exit: If wandb is not installed.

    Returns:
        module: The wandb module.
    """
    try:
        import wandb
    except ImportError:
        typer.echo(WANDB_MISSING, err=True)
        raise typer.Exit(code=1) from None
    return wandb


def _log_trial(wandb, log_vars: Dict[str, Any]) -> None:
    """Log one completed search trial to the active wandb run (optimize_adapt_decomp's on_trial).

    Args:
        wandb (module): The wandb module.
        log_vars (Dict[str, Any]): The trial's log dict; see optimize_adapt_decomp's on_trial.

    Returns:
        None
    """
    if "loss" in log_vars:
        scored = {"optuna/loss": log_vars["loss"]}
    else:
        scored = {
            f"optuna/value_{o}": v for o, v in zip(log_vars["objectives"], log_vars["values"])
        }
    log = {
        "optuna/trial_number": log_vars["trial_number"],
        **scored,
        **{f"optuna/{key}": log_vars[key] for key in ("sv_loss", "wh_loss", "total_loss")},
        **{f"optuna/param_{key}": value for key, value in log_vars["params"].items()},
    }
    if "roa_mean" in log_vars:
        log["optuna/roa_mean"] = log_vars["roa_mean"]
    wandb.log(log)


def _log_outputs(
    wandb, outputs: Dict[str, AdaptationResult], roa: Dict[str, Optional[np.ndarray]]
) -> None:
    """Log each recording's per-batch losses and timings, and the run's totals, to wandb.

    Args:
        wandb (module): The wandb module.
        outputs (Dict[str, AdaptationResult]): Recording name -> its adaptation output.
        roa (Dict[str, Optional[np.ndarray]]): Recording name -> each unit's rate of
            agreement with the ground truth, or None without ground truth.

    Returns:
        None
    """
    n_batches = max(len(result.wh_loss) for result in outputs.values())
    for batch in range(n_batches):
        wandb.log(
            {
                f"{name}/{key}": value
                for name, result in outputs.items()
                if batch < len(result.wh_loss)
                for key, value in (
                    ("wh_loss", result.wh_loss[batch]),
                    ("sv_loss", result.sv_loss[batch].nansum()),
                    ("total_time_ms", result.total_time_ms[batch]),
                )
            }
        )
    for name, result in outputs.items():
        wandb.summary[f"{name}/wh_loss"] = result.wh_loss_total.item()
        wandb.summary[f"{name}/sv_loss"] = result.sv_loss_total.item()
        wandb.summary[f"{name}/total_time_ms"] = result.total_time_ms.float().mean().item()
        if roa.get(name) is not None:
            wandb.summary[f"{name}/roa"] = float(np.mean(roa[name]))
    for key, field in (
        ("wh_loss", "wh_loss_total"),
        ("sv_loss", "sv_loss_total"),
        ("total_loss", "total_loss"),
    ):
        wandb.summary[key] = float(
            sum(getattr(result, field).item() for result in outputs.values())
        )
    roa_means = [float(np.mean(values)) for values in roa.values() if values is not None]
    if roa_means:
        wandb.summary["roa"] = float(np.mean(roa_means))


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@app.command(name="decompose")
def decompose(
    emg: EmgArgument,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out_dir",
            help=f"Folder to write {CALIBRATION_FILE} and {CALIBRATION_CONFIG_FILE} into.",
        ),
    ],
    cbss_config: CbssConfigOption = None,
    start: Annotated[
        int, typer.Option("--start", help="First sample of the calibration window.")
    ] = 0,
    stop: Annotated[
        Optional[int],
        typer.Option("--stop", help="One past its last sample. Omit for the end of the recording."),
    ] = None,
    gt: GtOption = None,
    roa_th: RoaThOption = 0.9,
    emg_loader: EmgLoaderOption = FileFormat.npz,
    gt_loader: GtLoaderOption = FileFormat.npz,
) -> None:
    """Calibrate: find the motor units in a window of a recording (CBSS.decompose)."""
    config = _cbss_config(cbss_config)
    emg_data = load_emg(emg, emg_loader.value)
    window = slice(start, stop)
    calibration = CBSS(config).decompose(
        emg_data[window], _timestamps(emg_data.shape[0], config.fs)[window]
    )
    if gt is not None:
        gt_spikes = load_gt(gt, emg_data.shape[0], gt_loader.value)
        calibration = calibration.select_supervised(gt_spikes[window], roa_th=roa_th, fs=config.fs)
    _save_calibration(calibration, config, out_dir)
    typer.echo(f"{calibration.spikes.shape[1]} units, saved in {out_dir}")


@app.command(name="process_data")
def process_data(
    emg: EmgArgument,
    calibration: Annotated[
        Path,
        typer.Option(
            "--calibration", exists=True, dir_okay=False, help="CBSSResult (CBSSResult.save)."
        ),
    ],
    calibration_config: Annotated[
        Path,
        typer.Option(
            "--calibration_config",
            exists=True,
            dir_okay=False,
            help="The CBSSConfig YAML that produced --calibration.",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="File to write the AdaptationResult into (.pkl).")
    ],
    preset: PresetOption = None,
    adapt_config: AdaptConfigOption = None,
    start: Annotated[
        int, typer.Option("--start", help="First sample adapted, e.g. the calibration's end.")
    ] = 0,
    stop: Annotated[
        Optional[int],
        typer.Option("--stop", help="One past the last sample adapted. Omit for the end."),
    ] = None,
    source_fifo_from_calib: SourceFifoOption = None,
    processing_mode: ProcessingModeOption = ProcessingMode.offline,
    emg_loader: EmgLoaderOption = FileFormat.npz,
) -> None:
    """Adapt a calibration over a recording, batch by batch (AdaptDecomp.process_data)."""
    adapter = AdaptDecomp.from_calibration(
        calibration=CBSSResult.load(calibration),
        cbss_config=CBSSConfig.from_yaml(calibration_config),
        adapt_config=_adapt_config(preset, adapt_config, source_fifo_from_calib),
    )
    result = adapter.process_data(
        load_emg(emg, emg_loader.value)[start:stop], processing_mode=processing_mode.value
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    typer.echo(f"{_describe(result)}, saved in {out}")


@app.command(name="calibrate_and_process")
def calibrate_and_process(
    emg: EmgArgument,
    calib_stop: Annotated[
        int,
        typer.Option("--calib_stop", help="One past the last sample of the calibration window."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out_dir",
            help=f"Folder to write adapted.pkl, {CALIBRATION_FILE} and {CALIBRATION_CONFIG_FILE} into.",
        ),
    ],
    calib_start: Annotated[
        int, typer.Option("--calib_start", help="First sample of the calibration window.")
    ] = 0,
    cbss_config: CbssConfigOption = None,
    preset: PresetOption = None,
    adapt_config: AdaptConfigOption = None,
    adapt_from: Annotated[
        AdaptFrom,
        typer.Option(
            "--adapt_from",
            help="calib_end adapts after the calibration window, with CBSS's output before it; "
            "emg_start adapts the whole recording.",
        ),
    ] = AdaptFrom.calib_end,
    gt: GtOption = None,
    roa_th: RoaThOption = 0.9,
    processing_mode: ProcessingModeOption = ProcessingMode.offline,
    emg_loader: EmgLoaderOption = FileFormat.npz,
    gt_loader: GtLoaderOption = FileFormat.npz,
) -> None:
    """Calibrate on a window, then adapt over the recording (AdaptDecomp.calibrate_and_process)."""
    config = _cbss_config(cbss_config)
    emg_data = load_emg(emg, emg_loader.value)
    window = slice(calib_start, calib_stop)
    run_config = config
    if gt is not None:
        gt_spikes = load_gt(gt, emg_data.shape[0], gt_loader.value)
        run_config = dataclasses.replace(
            config,
            selection="supervised",
            selection_kwargs={"gt_spikes": gt_spikes[window], "roa_th": roa_th},
        )
    result, calibration = AdaptDecomp.calibrate_and_process(
        emg_data,
        timestamps=_timestamps(emg_data.shape[0], config.fs),
        calib_indices=window,
        cbss_config=run_config,
        adapt_config=_adapt_config(preset, adapt_config),
        processing_mode=processing_mode.value,
        adapt_from=adapt_from.value,
    )
    _save_calibration(calibration, config, out_dir)  # config: without the ground truth
    result.save(out_dir / "adapted.pkl")
    typer.echo(f"{_describe(result)}, saved in {out_dir}")


@app.command(name="optimize_adapt_decomp")
def optimize_adapt_decomp(
    data_config: DataConfigOption,
    best_result_path: Annotated[
        Path,
        typer.Option(
            "--best_result_path",
            help="Folder to write the search's results into: best_config.yaml, study.pkl and "
            "the best trial's outputs.",
        ),
    ],
    search_config: Annotated[
        Optional[Path],
        typer.Option(
            "--search_config",
            exists=True,
            dir_okay=False,
            help="Search settings YAML: param_space, objectives, selection, unit_selection, "
            "unit_selection_kwargs, n_trials, n_jobs, n_cores, random_seed, initial_params, "
            "sampler. Omit for optimize_adapt_decomp's defaults.",
        ),
    ] = None,
    preset: PresetOption = None,
    adapt_config: AdaptConfigOption = None,
    source_fifo_from_calib: SourceFifoOption = None,
    objectives: Annotated[
        Optional[List[Objective]],
        typer.Option(
            "--objectives",
            help="Overrides the search config's. Repeat it for a Pareto search: "
            "--objectives wh_loss --objectives sv_loss.",
            show_default=False,
        ),
    ] = None,
    n_trials: Annotated[
        Optional[int], typer.Option("--n_trials", help="Overrides the search config's.")
    ] = None,
    n_cores: Annotated[
        Optional[int], typer.Option("--n_cores", help="Overrides the search config's.")
    ] = None,
    compute_roa: Annotated[
        Optional[bool],
        typer.Option(
            "--compute_roa/--no-compute_roa",
            help="Score every trial against the ground truth. Omit for on when every "
            "recording has ground truth.",
            show_default=False,
        ),
    ] = None,
    wandb_project: Annotated[
        Optional[str],
        typer.Option(
            "--wandb_project", help="Log every trial to this wandb project (wandb extra)."
        ),
    ] = None,
) -> None:
    """Tune the adaptation's hyperparameters on a pool of recordings (optimize_adapt_decomp)."""
    wandb = _wandb() if wandb_project is not None else None
    pool = _load_pool(data_config)
    settings = _read_yaml(search_config)
    base_config = _adapt_config(preset, adapt_config, source_fifo_from_calib)

    param_space = settings.get("param_space")
    if param_space is not None:
        param_space = {name: tuple(spec) for name, spec in param_space.items()}
    random_seed = settings.get("random_seed", 1909)
    sampler_kwargs = settings.get("sampler")
    if compute_roa is None:
        compute_roa = all(has_gt(dataset) for dataset in pool.values())

    if wandb is not None:
        wandb.init(project=wandb_project, config={**base_config.to_dict(), "search": settings})

    result = _optimize_adapt_decomp(
        pool=pool,
        objectives=tuple(o.value for o in objectives)
        if objectives
        else settings.get("objectives", "sv_loss"),
        param_space=param_space,
        base_config=base_config,
        compute_roa=compute_roa,
        unit_selection=settings.get("unit_selection"),
        unit_selection_kwargs=settings.get("unit_selection_kwargs"),
        selection=settings.get("selection", "min_sv_loss"),
        n_trials=n_trials if n_trials is not None else settings.get("n_trials", 100),
        n_jobs=settings.get("n_jobs", 1),
        n_cores=n_cores if n_cores is not None else settings.get("n_cores"),
        sampler=optuna.samplers.TPESampler(seed=random_seed, **sampler_kwargs)
        if sampler_kwargs
        else None,
        random_seed=random_seed,
        initial_params=settings.get("initial_params"),
        best_result_path=str(best_result_path),
        on_trial=(lambda log_vars: _log_trial(wandb, log_vars)) if wandb is not None else None,
    )
    result.best_config.to_yaml(best_result_path / "best_config.yaml")

    if wandb is not None:
        if result.outputs is not None:
            roa = {name: output.roa for name, output in result.outputs.items()}
            _log_outputs(wandb, result.outputs, roa)
        wandb.summary["best_config"] = result.best_config.to_dict()
        wandb.finish()

    chosen = {
        name: getattr(result.best_config, name) for name in param_space or DEFAULT_PARAM_SPACE
    }
    typer.echo(f"Best setting: {chosen}, saved in {best_result_path / 'best_config.yaml'}")


@app.command(name="wandb_sweep")
def wandb_sweep(
    data_config: DataConfigOption,
    sweep_config: Annotated[
        Path,
        typer.Option(
            "--sweep_config",
            exists=True,
            dir_okay=False,
            help="wandb sweep config YAML, plus sweep_counts, the number of runs.",
        ),
    ],
    wandb_project: Annotated[str, typer.Option("--wandb_project", help="wandb project.")],
    preset: PresetOption = None,
    adapt_config: AdaptConfigOption = None,
    source_fifo_from_calib: SourceFifoOption = None,
    sweep_counts: Annotated[
        Optional[int],
        typer.Option("--sweep_counts", help="Overrides the sweep config's (default 20)."),
    ] = None,
) -> None:
    """Search with a wandb sweep: each run adapts the pool with the parameters wandb chose."""
    wandb = _wandb()
    pool = _load_pool(data_config)
    base_config = _adapt_config(preset, adapt_config, source_fifo_from_calib)
    sweep_settings = _read_yaml(sweep_config)
    counts = sweep_settings.pop("sweep_counts", 20)  # wandb.sweep must not see it
    config_fields = {field.name for field in dataclasses.fields(AdaptConfig) if field.init}

    def run() -> None:
        wandb.init(project=wandb_project)
        overrides = {key: value for key, value in wandb.config.items() if key in config_fields}
        config = dataclasses.replace(base_config, **overrides)
        outputs, roa = {}, {}
        for name, dataset in pool.items():
            emg, calibration, cbss_config, preprocess, gt_paired = dataset.resolve()
            outputs[name] = AdaptDecomp.from_calibration(
                calibration=calibration, cbss_config=cbss_config, adapt_config=copy.deepcopy(config)
            ).process_data(emg, preprocess=preprocess)
            roa[name] = (
                rate_of_agreement_paired(
                    gt_paired, outputs[name].spikes.numpy(), fs=config.fs, tol_spike_ms=2
                )[0]
                if gt_paired is not None
                else None
            )
        _log_outputs(wandb, outputs, roa)
        wandb.finish()

    sweep_id = wandb.sweep(sweep_settings, project=wandb_project)
    wandb.agent(sweep_id, function=run, count=sweep_counts if sweep_counts is not None else counts)


if __name__ == "__main__":
    app()
