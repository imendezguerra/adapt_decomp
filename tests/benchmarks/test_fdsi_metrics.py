"""Tests for benchmarks/fdsi/fdsi.py: paths, loaders, triangular phases and per-unit metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.fdsi import fdsi


class _Outputs:
    """A minimal stand-in for AdaptationResult: spikes (samples, units) and sil (units,)."""

    def __init__(self, spikes: np.ndarray, sil=None):
        self.spikes = torch.from_numpy(spikes.astype(np.int32))
        self.sil = sil


class _Calibration:
    """A minimal stand-in for CBSSResult: spikes, its ground-truth match and quality metrics."""

    def __init__(self, n_units: int, gt_matched_indices=None, roa=None, sil=None, cov_isi=None):
        self.spikes = np.zeros((10, n_units), dtype=np.int32)
        self.gt_matched_indices = gt_matched_indices
        self.roa = roa
        self.sil = sil
        self.cov_isi = cov_isi


def _spike_train(n_samples: int, n_units: int, period: int = 40) -> np.ndarray:
    spikes = np.zeros((n_samples, n_units), dtype=np.float32)
    for u in range(n_units):
        spikes[5 + u :: period, u] = 1.0
    return spikes


def test_recording_stub_and_paths_follow_the_fdsi_layout():
    root = Path("/data")
    assert fdsi.recording_stub("sub-05", "staircase", 15) == "sub-05_FDSI_staircase_snr15dB"
    assert fdsi.emg_path(root, "sub-01", "staircase", 30) == (
        root / "sub-01" / "noisy" / "sub-01_FDSI_staircase_snr30dB_emg.npz"
    )
    assert fdsi.gt_spikes_path(root, "sub-01", "staircase") == (
        root / "sub-01" / "clean" / "sub-01_FDSI_staircase_spikes.npz"
    )


def test_v10_paths_match_the_v1_0_cache_layout():
    root = Path("/out")
    assert fdsi.v10_calibration_path(root, "sub-01", "staircase", 30) == (
        root / "calibration" / "sub-01" / "sub-01_FDSI_staircase_snr30dB_cbss.pkl"
    )
    assert fdsi.v10_result_path(root, "sub-01", "staircase", 30, None) == (
        root / "adaptation" / "sub-01" / "fixed" / "sub-01_FDSI_staircase_snr30dB_adapt_fixed.pkl"
    )
    assert fdsi.v10_result_path(root, "sub-01", "staircase", 30, "tpe_pareto") == (
        root
        / "adaptation"
        / "sub-01"
        / "lr_fixed"
        / "tpe_pareto"
        / "sub-01_FDSI_staircase_snr30dB_adapt_lr_fixed_tpe_pareto.pkl"
    )


def test_load_raw_emg_reads_the_noisy_npz(tmp_path):
    emg = np.random.randn(30, 4).astype(np.float32)
    path = fdsi.emg_path(tmp_path, "sub-01", "triangular-ramp40s", 30)
    path.parent.mkdir(parents=True)
    np.savez(path, emg=emg)
    np.testing.assert_array_equal(
        fdsi.load_raw_emg(tmp_path, "sub-01", "triangular-ramp40s", 30), emg
    )


def test_raw_data_loaders_read_the_fdsi_clean_and_noisy_files(tmp_path):
    clean = tmp_path / "sub-01" / "clean"
    noisy = tmp_path / "sub-01" / "noisy"
    clean.mkdir(parents=True)
    noisy.mkdir(parents=True)
    stem = "sub-01_FDSI_staircase"
    emg = np.random.randn(20, 3).astype(np.float32)
    np.savez(clean / f"{stem}_emg.npz", emg=emg)
    np.savez(clean / f"{stem}_angle.npz", angle=np.linspace(0, 40, 20))
    np.savez(clean / f"{stem}_effort.npz", effort=np.full(20, 0.15))
    (clean / f"{stem}_metadata.json").write_text('{"n_channels": 3, "fs": 2048}')
    (noisy / f"{stem}_snr30dB_noise_metadata.json").write_text('{"snr_db_target": 30}')

    np.testing.assert_array_equal(fdsi.load_clean_emg(tmp_path, "sub-01", "staircase"), emg)
    angle = fdsi.load_angle_profile(tmp_path, "sub-01", "staircase")
    assert angle.dtype == np.float32 and angle[-1] == 40
    assert fdsi.load_effort_profile(tmp_path, "sub-01", "staircase")[0] == pytest.approx(0.15)
    assert fdsi.load_recording_metadata(tmp_path, "sub-01", "staircase") == {
        "n_channels": 3,
        "fs": 2048,
    }
    assert fdsi.load_noise_metadata(tmp_path, "sub-01", "staircase", 30) == {"snr_db_target": 30}


def test_load_gt_full_bin_selects_and_orders_the_matched_units(tmp_path):
    path = fdsi.gt_spikes_path(tmp_path, "sub-01", "triangular-ramp40s")
    path.parent.mkdir(parents=True)
    spikes = np.array([np.array([2, 5]), np.array([7]), np.array([1, 9])], dtype=object)
    np.savez(path, spikes=spikes)

    class _Calibration:
        gt_matched_indices = np.array([2, 0])

    out = fdsi.load_gt_full_bin(tmp_path, "sub-01", "triangular-ramp40s", _Calibration(), 20)

    assert out.shape == (20, 2)
    assert out[1, 0] == 1 and out[9, 0] == 1  # ground-truth unit 2, now column 0
    assert out[2, 1] == 1 and out[5, 1] == 1  # ground-truth unit 0, now column 1


def test_load_gt_full_bin_is_none_without_a_supervised_match(tmp_path):
    class _Calibration:
        gt_matched_indices = None

    assert fdsi.load_gt_full_bin(tmp_path, "sub-01", "staircase", _Calibration(), 10) is None


def test_triangular_phases_cover_the_recording_without_gaps():
    phases = fdsi.get_triangular_phases(2048 * 50, 2048 * 5, 2048 * 5)
    assert phases["first_iso"] == slice(0, 2048 * 10)
    assert phases["first_iso"].stop == phases["ramp"].start
    assert phases["ramp"].stop == phases["last_iso"].start
    assert phases["last_iso"].stop == 2048 * 50


def test_compute_roa_subset_is_nan_for_an_empty_slice_and_one_for_identical_trains():
    spikes = _spike_train(400, 2)
    assert np.all(np.isnan(fdsi.compute_roa_subset(spikes, spikes, slice(50, 50), 2048, 1.0)))
    np.testing.assert_allclose(
        fdsi.compute_roa_subset(spikes, spikes, slice(0, 400), 2048, 1.0), [1.0, 1.0]
    )


def test_unit_metrics_of_a_perfect_triangular_result():
    spikes = _spike_train(2000, 2)
    metrics = fdsi.unit_metrics(
        _Outputs(spikes, sil=np.array([0.9, 0.8])),
        spikes,
        _Calibration(2, gt_matched_indices=np.array([7, 3]), roa=np.array([0.95, 1.0])),
        "triangular-ramp10s",
        cal_end=400,
        iso_dur=400,
        fs=2048,
        tol_spike_ms=1.0,
    )

    assert list(metrics["unit"]) == [0, 1]
    assert list(metrics["gt_unit"]) == [7, 3]  # the simulation motor units they track
    np.testing.assert_allclose(metrics["roa_calib"], [0.95, 1.0])
    for column in ("roa_full", "roa_after_cal", "roa_first_iso", "roa_ramp", "roa_last_iso"):
        np.testing.assert_allclose(metrics[column], [1.0, 1.0])
    np.testing.assert_allclose(metrics["sil"], [0.9, 0.8])
    assert list(metrics["n_spikes"]) == list(spikes.sum(axis=0).astype(int))
    assert list(metrics["n_spikes"]) == list(metrics["n_spikes_gt"])


def test_unit_metrics_without_phases_or_ground_truth():
    spikes = _spike_train(2000, 3)
    calibration = _Calibration(3)  # no supervised match
    staircase = fdsi.unit_metrics(
        _Outputs(spikes), spikes, calibration, "staircase", 400, 400, 2048, 1.0
    )
    no_gt = fdsi.unit_metrics(_Outputs(spikes), None, calibration, "staircase", 400, 400, 2048, 1.0)

    assert staircase["roa_ramp"].isna().all()  # phases only for triangular contractions
    assert staircase["roa_after_cal"].notna().all()
    assert no_gt["roa_full"].isna().all()
    assert (no_gt["n_spikes_gt"] == -1).all()
    assert no_gt["sil"].isna().all()
    assert (no_gt["gt_unit"] == -1).all() and no_gt["roa_calib"].isna().all()


def test_calibration_unit_metrics_list_each_unit_with_its_ground_truth_match():
    calibration = _Calibration(
        3,
        gt_matched_indices=np.array([12, 0, 5]),
        roa=np.array([0.97, 0.91, 1.0]),
        sil=np.array([0.95, 0.85, 0.9]),
        cov_isi=np.array([0.1, 0.2, 0.15]),
    )
    units = fdsi.calibration_unit_metrics(calibration)

    assert list(units.columns) == ["unit", "gt_unit", "roa_calib", "sil_calib", "cov_isi_calib"]
    assert list(units["unit"]) == [0, 1, 2]
    assert list(units["gt_unit"]) == [12, 0, 5]
    np.testing.assert_allclose(units["roa_calib"], [0.97, 0.91, 1.0])
    assert int((units["sil_calib"] >= 0.9).sum()) == 2


def test_with_recording_labels_prepends_the_recording_and_branch():
    units = fdsi.calibration_unit_metrics(_Calibration(2))
    labelled = fdsi.with_recording_labels(units, "sub-01", "staircase", 30, branch="sv_mean")
    unlabelled = fdsi.with_recording_labels(units, "sub-01", "staircase", 30)

    assert list(labelled.columns[:5]) == ["branch", "recording", "sub", "condition", "snr"]
    assert set(labelled["recording"]) == {"sub-01_FDSI_staircase_snr30dB"}
    assert list(unlabelled.columns[:4]) == ["recording", "sub", "condition", "snr"]
    assert len(labelled) == len(units)


def test_spikes_digest_depends_only_on_the_spikes_and_their_shape():
    spikes = _spike_train(100, 2)
    digest = fdsi.spikes_digest(spikes)

    assert fdsi.spikes_digest(spikes.astype(np.int32)) == digest
    assert fdsi.spikes_digest(spikes.astype(bool)) == digest
    moved = spikes.copy()
    moved[5, 0], moved[6, 0] = 0, 1
    assert fdsi.spikes_digest(moved) != digest
    assert fdsi.spikes_digest(spikes.reshape(50, 4)) != digest
