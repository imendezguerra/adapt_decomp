"""Tests for notebooks/fdsi_benchmark/fdsi_common.py's pure, IO-free helpers.

Mirrors tests/test_fdsi_common.py's coverage of the original
notebooks/muniverse_simulations/fdsi_common.py, minus the functions this trimmed copy
dropped in favour of adapt_decomp.utils.load_emg/load_gt (build_gt_cal_binary,
build_gt_full_binary_paired) and adapt_decomp.utils.plots (build_plot1/build_plot2) --
see fdsi_common.py's own module docstring. Those two loaders have their own tests in
tests/utils/test_loaders.py; this file only covers what's still local to fdsi_common.py.

fdsi_common.py lives outside src/adapt_decomp/ (FDSI-benchmark-specific notebook glue, not
part of the installed package), so it's imported here via a path-scoped sys.path insertion
local to this module, rather than a package import.
"""

import importlib.util
from pathlib import Path

import numpy as np

# Loaded by explicit file path under a unique module name (not "fdsi_common" via sys.path)
# because notebooks/muniverse_simulations/ has its own same-named module -- both get imported
# in the same pytest session (see tests/test_fdsi_common.py), and a bare `import fdsi_common`
# in both files would collide on sys.modules['fdsi_common'], silently testing whichever one
# happened to import first.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "fdsi_benchmark" / "fdsi_common.py"
_spec = importlib.util.spec_from_file_location("fdsi_benchmark_common", _MODULE_PATH)
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)


def test_recording_stub_formats_sub_condition_snr():
    """recording_stub() builds the canonical '<sub>_FDSI_<cond>_snr<N>dB' string."""
    assert fc.recording_stub('sub-01', 'triangular-ramp40s', 30) == 'sub-01_FDSI_triangular-ramp40s_snr30dB'
    assert fc.recording_stub('sub-05', 'staircase', 15) == 'sub-05_FDSI_staircase_snr15dB'


def test_gt_spikes_path_matches_load_spikes_obj_convention():
    """gt_spikes_path() must match the original fdsi_common.py's own load_spikes_obj
    path convention exactly -- it reads the same on-disk cache."""
    path = fc.gt_spikes_path(Path('/data'), 'sub-01', 'triangular-ramp40s')
    assert path == Path('/data') / 'sub-01' / 'clean' / 'sub-01_FDSI_triangular-ramp40s_spikes.npz'


def test_get_triangular_phases_boundaries():
    """Phase slices split [0, n_full) into first_iso/ramp/last_iso in order, no gaps/overlap."""
    fs, cal_end, iso_dur, n_full = 2048, 2048 * 5, 2048 * 5, 2048 * 50
    phases = fc.get_triangular_phases(n_full, cal_end, iso_dur)

    assert phases['first_iso'] == slice(0, cal_end + iso_dur)
    assert phases['ramp'] == slice(cal_end + iso_dur, n_full - iso_dur)
    assert phases['last_iso'] == slice(n_full - iso_dur, n_full)
    # No gaps/overlap: consecutive slices' stop/start line up exactly.
    assert phases['first_iso'].stop == phases['ramp'].start
    assert phases['ramp'].stop == phases['last_iso'].start


def test_get_triangular_phases_empty_ramp_when_recording_too_short():
    """A short recording collapses 'ramp' to an empty (but valid) slice, not an error."""
    cal_end, iso_dur = 2048 * 5, 2048 * 5
    n_full = cal_end + 2 * iso_dur  # exactly enough for both isometric bookends, no ramp left
    phases = fc.get_triangular_phases(n_full, cal_end, iso_dur)
    assert phases['ramp'].start == phases['ramp'].stop


def test_compute_roa_subset_returns_nan_for_empty_slice():
    """An empty temporal slice short-circuits to a NaN-filled array, no RoA computation."""
    gt_full_bin = np.zeros((100, 3), dtype=np.float32)
    pred = np.zeros((100, 3), dtype=np.float32)
    roa = fc.compute_roa_subset(gt_full_bin, pred, slice(50, 50), fs=2048, tol_spike_ms=1.0)
    assert roa.shape == (3,)
    assert np.all(np.isnan(roa))


def test_compute_roa_subset_perfect_agreement_on_matching_spikes():
    """Identical ref/pred spike trains within a non-empty slice give RoA == 1."""
    n = 200
    gt_full_bin = np.zeros((n, 1), dtype=np.float32)
    gt_full_bin[10:n:20, 0] = 1.0
    pred = gt_full_bin.copy()
    roa = fc.compute_roa_subset(gt_full_bin, pred, slice(0, n), fs=2048, tol_spike_ms=1.0)
    assert roa.shape == (1,)
    np.testing.assert_allclose(roa, [1.0], atol=1e-6)


def test_load_raw_emg_reads_the_fdsi_noisy_npz_layout(tmp_path):
    """load_raw_emg() delegates to adapt_decomp.utils.load_emg on the FDSI 'noisy' path
    convention -- same file, same 'emg' key, just via the shared loader now."""
    emg = np.random.randn(30, 4).astype(np.float32)
    sub_dir = tmp_path / 'sub-01' / 'noisy'
    sub_dir.mkdir(parents=True)
    np.savez(sub_dir / 'sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz', emg=emg)

    out = fc.load_raw_emg(tmp_path, 'sub-01', 'triangular-ramp40s', 30)

    np.testing.assert_array_equal(out, emg)


def test_load_gt_full_bin_returns_none_without_supervised_match(tmp_path):
    """load_gt_full_bin() returns None when the calibration has no gt_matched_indices,
    without touching disk at all."""
    class _FakeCBSSResult:
        gt_matched_indices = None

    out = fc.load_gt_full_bin(tmp_path, 'sub-01', 'triangular-ramp40s', _FakeCBSSResult(), n_samples=10)
    assert out is None


def test_load_gt_full_bin_slices_to_matched_units(tmp_path):
    """load_gt_full_bin() densifies GT then selects/orders columns by gt_matched_indices."""
    n_samples = 20
    spikes_obj = np.array([np.array([2, 5]), np.array([7]), np.array([1, 9])], dtype=object)
    sub_dir = tmp_path / 'sub-01' / 'clean'
    sub_dir.mkdir(parents=True)
    np.savez(sub_dir / 'sub-01_FDSI_triangular-ramp40s_spikes.npz', spikes=spikes_obj)

    class _FakeCBSSResult:
        gt_matched_indices = np.array([2, 0])  # reorders and subsets to GT units 2 then 0

    out = fc.load_gt_full_bin(tmp_path, 'sub-01', 'triangular-ramp40s', _FakeCBSSResult(),
                               n_samples=n_samples)

    assert out.shape == (n_samples, 2)
    assert out[1, 0] == 1 and out[9, 0] == 1  # GT unit 2's spikes, now in column 0
    assert out[2, 1] == 1 and out[5, 1] == 1  # GT unit 0's spikes, now in column 1


def test_compute_roa_for_result_none_without_gt():
    """compute_roa_for_result() returns None (no computation) when gt_full_bin is None."""
    class _FakeOutputs:
        pass  # never touched -- the None short-circuit must happen before outputs.spikes is read

    out = fc.compute_roa_for_result(_FakeOutputs(), gt_full_bin=None, fs=2048, tol_spike_ms=1.0)
    assert out is None


def test_compute_roa_for_result_perfect_agreement(tmp_path):
    """compute_roa_for_result() matches rate_of_agreement_paired on identical spike trains."""
    import torch

    n = 200
    spikes = np.zeros((n, 1), dtype=np.float32)
    spikes[10:n:20, 0] = 1.0

    class _FakeOutputs:
        pass

    outputs = _FakeOutputs()
    outputs.spikes = torch.from_numpy(spikes.astype(np.int32))

    roa = fc.compute_roa_for_result(outputs, gt_full_bin=spikes, fs=2048, tol_spike_ms=1.0)

    assert roa.shape == (1,)
    np.testing.assert_allclose(roa, [1.0], atol=1e-6)


def test_calibration_ground_truth_densification_matches_manual_expectation(tmp_path):
    """The calibration gt_bin (built via adapt_decomp.utils.load_gt against gt_spikes_path())
    still one-hot-places each unit's in-range spike indices and drops out-of-range ones -- the
    same contract the original build_gt_cal_binary() had (see tests/test_fdsi_common.py)."""
    from adapt_decomp.utils import load_gt

    n_cal = 10
    spikes_obj = np.array([np.array([2, 5, n_cal + 3]), np.array([-1, 7])], dtype=object)
    sub_dir = tmp_path / 'sub-01' / 'clean'
    sub_dir.mkdir(parents=True)
    gt_path = sub_dir / 'sub-01_FDSI_triangular-ramp40s_spikes.npz'
    np.savez(gt_path, spikes=spikes_obj)

    gt_bin = load_gt(fc.gt_spikes_path(tmp_path, 'sub-01', 'triangular-ramp40s'), n_samples=n_cal)

    assert gt_bin.shape == (n_cal, 2)
    assert gt_bin[:, 0].sum() == 2  # index n_cal+3 dropped (out of range)
    assert gt_bin[2, 0] == 1 and gt_bin[5, 0] == 1
    assert gt_bin[:, 1].sum() == 1  # index -1 dropped
    assert gt_bin[7, 1] == 1
