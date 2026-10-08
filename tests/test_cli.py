"""The adapt-decomp command (adapt_decomp.cli) on the synthetic recording."""

import sys

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from adapt_decomp.adaptation import AdaptationResult, AdaptConfig
from adapt_decomp.cbss import CBSSConfig, CBSSResult
from adapt_decomp.cli import app

runner = CliRunner()


@pytest.fixture
def files(tmp_path, synthetic_recording, synthetic_cbss_config):
    """The synthetic recording, its ground truth and a CBSSConfig, written as the CLI reads them."""
    paths = {
        "emg": tmp_path / "emg.npz",
        "gt": tmp_path / "spikes.npz",
        "cbss_config": tmp_path / "cbss_config.yaml",
    }
    np.savez(paths["emg"], emg=synthetic_recording.emg)
    np.savez(paths["gt"], spikes=synthetic_recording.spikes)
    synthetic_cbss_config.to_yaml(paths["cbss_config"])
    return paths


def invoke(*args):
    """Run adapt-decomp with args, failing the test with its output if it fails."""
    result = runner.invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result


def test_decompose_then_process_data_from_the_calibration_end(tmp_path, files, synthetic_recording):
    n_cal = synthetic_recording.n_cal
    invoke(
        "decompose", files["emg"], "--cbss_config", files["cbss_config"], "--stop", n_cal,
        "--gt", files["gt"], "--out_dir", tmp_path / "calib",
    )  # fmt: skip
    calibration = CBSSResult.load(tmp_path / "calib" / "calibration.pkl")
    assert calibration.gt_matched_indices is not None
    assert calibration.sources.shape[0] == n_cal

    invoke(
        "process_data", files["emg"],
        "--calibration", tmp_path / "calib" / "calibration.pkl",
        "--calibration_config", tmp_path / "calib" / "calibration_config.yaml",
        "--preset", "muniverse", "--start", n_cal, "--source_fifo_from_calib",
        "--out", tmp_path / "adapted.pkl",
    )  # fmt: skip
    adapted = AdaptationResult.load(tmp_path / "adapted.pkl")
    assert adapted.spikes.shape == (
        synthetic_recording.emg.shape[0] - n_cal,
        calibration.spikes.shape[1],
    )


def test_calibrate_and_process_covers_the_recording_and_saves_no_ground_truth(
    tmp_path, files, synthetic_recording
):
    invoke(
        "calibrate_and_process", files["emg"], "--cbss_config", files["cbss_config"],
        "--calib_stop", synthetic_recording.n_cal, "--gt", files["gt"], "--preset", "muniverse",
        "--out_dir", tmp_path / "out",
    )  # fmt: skip
    adapted = AdaptationResult.load(tmp_path / "out" / "adapted.pkl")
    assert adapted.spikes.shape[0] == synthetic_recording.emg.shape[0]
    assert CBSSResult.load(tmp_path / "out" / "calibration.pkl").gt_matched_indices is not None
    # The saved config is the one passed in: the ground truth stays out of the YAML
    assert CBSSConfig.from_yaml(tmp_path / "out" / "calibration_config.yaml").selection is None


def test_preset_and_adapt_config_cannot_both_be_given(tmp_path, files):
    AdaptConfig().to_yaml(tmp_path / "adapt_config.yaml")
    result = runner.invoke(
        app,
        [
            "calibrate_and_process", str(files["emg"]), "--calib_stop", "100",
            "--preset", "fixed", "--adapt_config", str(tmp_path / "adapt_config.yaml"),
            "--out_dir", str(tmp_path / "out"),
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "not both" in result.output


@pytest.fixture
def pool_config(tmp_path, files, synthetic_recording):
    """A one-recording pool YAML, calibrated by decompose, adapted from the calibration end."""
    n_cal = synthetic_recording.n_cal
    invoke(
        "decompose", files["emg"], "--cbss_config", files["cbss_config"], "--stop", n_cal,
        "--out_dir", tmp_path / "calib",
    )  # fmt: skip
    data_config = tmp_path / "pool.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "root": str(tmp_path),
                "datasets": [
                    {
                        "name": "synthetic",
                        "path_emg": "emg.npz",
                        "path_calib": "calib/calibration.pkl",
                        "path_calib_config": "calib/calibration_config.yaml",
                        "path_gt": "spikes.npz",
                        "start": n_cal,
                    }
                ],
            }
        )
    )
    return data_config


@pytest.mark.slow
def test_optimize_adapt_decomp_writes_the_best_config(tmp_path, pool_config):
    result = invoke(
        "optimize_adapt_decomp", "--data_config", pool_config, "--preset", "muniverse",
        "--source_fifo_from_calib", "--n_trials", 2, "--objectives", "wh_loss",
        "--objectives", "sv_loss", "--best_result_path", tmp_path / "search",
    )  # fmt: skip
    best = AdaptConfig.from_yaml(tmp_path / "search" / "best_config.yaml")
    assert best.source_fifo_from_calib
    assert (tmp_path / "search" / "study.pkl").exists()
    assert "Best setting" in result.output


class FakeWandb:
    """Stands in for the wandb module: records what is logged, runs sweeps locally."""

    def __init__(self, sweep_params):
        self.sweep_params = sweep_params  # what the sweep "chooses" for every run
        self.config, self.summary, self.logged, self.runs = {}, {}, [], 0

    def init(self, project=None, config=None, **kwargs):
        self.runs += 1

    def log(self, values):
        self.logged.append(values)

    def finish(self):
        pass

    def sweep(self, settings, project=None):
        assert "sweep_counts" not in settings
        return "sweep-id"

    def agent(self, sweep_id, function, count):
        for _ in range(count):
            self.config = dict(self.sweep_params)
            function()


@pytest.mark.slow
def test_optimize_adapt_decomp_logs_every_trial_to_wandb(tmp_path, pool_config, monkeypatch):
    fake = FakeWandb({})
    monkeypatch.setitem(sys.modules, "wandb", fake)
    invoke(
        "optimize_adapt_decomp", "--data_config", pool_config, "--preset", "muniverse",
        "--source_fifo_from_calib", "--n_trials", 2, "--best_result_path", tmp_path / "search",
        "--wandb_project", "test",
    )  # fmt: skip
    trials = [log for log in fake.logged if "optuna/trial_number" in log]
    assert len(trials) == 2 and "optuna/roa_mean" in trials[0]
    assert "best_config" in fake.summary and "sv_loss" in fake.summary


def test_wandb_sweep_adapts_the_pool_with_the_chosen_parameters(tmp_path, pool_config, monkeypatch):
    fake = FakeWandb({"wh_learning_rate": 0.01, "not_a_field": 1})
    monkeypatch.setitem(sys.modules, "wandb", fake)
    sweep_config = tmp_path / "sweep.yaml"
    sweep_config.write_text("sweep_counts: 2\nmethod: random\n")
    invoke(
        "wandb_sweep", "--data_config", pool_config, "--sweep_config", sweep_config,
        "--preset", "muniverse", "--source_fifo_from_calib", "--wandb_project", "test",
    )  # fmt: skip
    assert fake.runs == 2
    assert {"synthetic/wh_loss", "synthetic/roa", "sv_loss"} <= set(fake.summary)


def test_wandb_options_say_how_to_install_wandb(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb raises ImportError
    data_config = tmp_path / "pool.yaml"
    data_config.write_text("datasets: []\n")
    result = runner.invoke(
        app,
        [
            "optimize_adapt_decomp", "--data_config", str(data_config),
            "--best_result_path", str(tmp_path / "search"), "--wandb_project", "test",
        ],
    )  # fmt: skip
    assert result.exit_code == 1
    assert 'pip install "adapt-decomp[wandb]"' in result.output


def test_data_lists_the_archives():
    assert "neuromotion-data" in invoke("data", "list").output
