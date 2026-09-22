"""Tests for notebooks/muniverse_simulations/fdsi_common.py's pure, IO-free helpers.

fdsi_common.py lives outside src/adapt_decomp/ (FDSI-benchmark-specific notebook glue, not
part of the installed package), so it's imported here via a path-scoped sys.path insertion
local to this module, rather than a package import.
"""

import importlib.util
from pathlib import Path

import numpy as np
import optuna

# Loaded by explicit file path under a unique module name (not "fdsi_common" via sys.path)
# because notebooks/fdsi_benchmark/ has its own same-named module (see
# tests/test_fdsi_benchmark_common.py) -- both get imported in the same pytest session, and a
# bare `import fdsi_common` in both files would collide on sys.modules['fdsi_common'], silently
# testing whichever one happened to import first.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "muniverse_simulations" / "fdsi_common.py"
_spec = importlib.util.spec_from_file_location("muniverse_simulations_fdsi_common", _MODULE_PATH)
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)


def test_recording_stub_formats_sub_condition_snr():
    """recording_stub() builds the canonical '<sub>_FDSI_<cond>_snr<N>dB' string."""
    assert fc.recording_stub('sub-01', 'triangular-ramp40s', 30) == 'sub-01_FDSI_triangular-ramp40s_snr30dB'
    assert fc.recording_stub('sub-05', 'staircase', 15) == 'sub-05_FDSI_staircase_snr15dB'


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
    fs, cal_end, iso_dur = 2048, 2048 * 5, 2048 * 5
    n_full = cal_end + 2 * iso_dur  # exactly enough for both isometric bookends, no ramp left
    phases = fc.get_triangular_phases(n_full, cal_end, iso_dur)
    assert phases['ramp'].start == phases['ramp'].stop


def test_build_gt_cal_binary_places_spikes_and_drops_out_of_range():
    """build_gt_cal_binary() one-hot-places each unit's in-range spike indices, drops the rest."""
    n_cal = 10
    spikes_obj = np.array([np.array([2, 5, n_cal + 3]), np.array([-1, 7])], dtype=object)
    gt_bin = fc.build_gt_cal_binary(spikes_obj, n_gt=2, n_cal=n_cal)

    assert gt_bin.shape == (n_cal, 2)
    assert gt_bin[:, 0].sum() == 2  # index n_cal+3 dropped (out of range)
    assert gt_bin[2, 0] == 1 and gt_bin[5, 0] == 1
    assert gt_bin[:, 1].sum() == 1  # index -1 dropped
    assert gt_bin[7, 1] == 1


def test_build_gt_full_binary_paired_reorders_by_matched_unit():
    """Columns follow gt_matched_indices' order, not the raw spikes_obj order."""
    n_full = 10
    spikes_obj = np.array([np.array([1]), np.array([4]), np.array([7])], dtype=object)
    gt_bin = fc.build_gt_full_binary_paired(spikes_obj, gt_matched_indices=np.array([2, 0]), n_full=n_full)

    assert gt_bin.shape == (n_full, 2)
    assert gt_bin[7, 0] == 1  # column 0 <- gt unit 2
    assert gt_bin[1, 1] == 1  # column 1 <- gt unit 0


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


def _make_single_objective_study(seed: int) -> optuna.Study:
    study = optuna.create_study(direction='minimize')

    def objective(trial):
        wh = trial.suggest_float('wh_learning_rate', 1e-4, 1e-1, log=True)
        sv = trial.suggest_float('sv_learning_rate', 1e-4, 1e-1, log=True)
        trial.set_user_attr('wh_loss', wh)
        trial.set_user_attr('sv_loss', sv)
        trial.set_user_attr('total_loss', wh + sv)
        trial.set_user_attr('roa_mean_pooled', 100.0 * (1.0 - wh - sv))
        return wh + sv

    study.optimize(objective, n_trials=5)
    return study


def test_build_plot1_shows_each_marker_legend_entry_exactly_once():
    """Composing 2 rows (lr_fixed/lr_relerror) must not duplicate the 'max RoA'/'selected'
    legend entries across rows -- each should appear in the legend exactly once, even though
    every row's own trial-scatter series still gets its own entry."""
    studies = {'lr_fixed': _make_single_objective_study(1), 'lr_relerror': _make_single_objective_study(2)}
    best_trials = {t: s.best_trial for t, s in studies.items()}

    fig = fc.build_plot1(studies, {t: t for t in studies}, best_trials)

    legend_names = [tr.name for tr in fig.data if tr.showlegend]
    assert legend_names.count('max RoA') == 1
    assert legend_names.count('selected') == 1
    # Each row's own trial series still gets its own legend entry.
    assert 'lr_fixed' in legend_names and 'lr_relerror' in legend_names
