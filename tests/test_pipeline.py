"""End to end on synthetic data: calibration, adaptation and the hyperparameter search.

synthetic_recording (conftest.py) mixes known motor unit firings through MUAPs that
drift after the calibration window, so each stage is checked against ground truth:
calibration must recover every unit, the fixed decomposition must lose them as the
MUAPs drift (or the recording is too easy to test adaptation with), and adaptation must
keep tracking them. Each test runs in about a second.
"""

from dataclasses import replace

import numpy as np
import pytest

from adapt_decomp.adaptation import AdaptConfig, AdaptDecomp
from adapt_decomp.adaptation.optimize import DEFAULT_UNIT_SELECTION_KWARGS, optimize_adapt_decomp
from adapt_decomp.cbss import CBSS
from adapt_decomp.spikes import rate_of_agreement_paired
from adapt_decomp.utils.loaders import PooledDatasetMemory

TOL_SPIKE_MS = 1


def _roa(rec, calibration, spikes, start=0):
    """Per-unit RoA of spikes[start:] against the ground truth of calibration's units."""
    gt = rec.spikes[start:, calibration.gt_matched_indices]
    roa, _, _ = rate_of_agreement_paired(
        gt, np.asarray(spikes)[start:], fs=rec.fs, tol_spike_ms=TOL_SPIKE_MS
    )
    return roa


def _preset(name, cbss_config):
    config = AdaptConfig.from_preset(name)
    config.device = "cpu"
    config.ext_fact = cbss_config.ext_fact
    config.__post_init__()
    return config


def test_calibration_recovers_every_ground_truth_unit(
    synthetic_recording, synthetic_cbss_config, synthetic_calibration
):
    rec, cal = synthetic_recording, synthetic_calibration
    n_units = rec.spikes.shape[1]
    dim = rec.emg.shape[1] * synthetic_cbss_config.ext_fact

    assert sorted(cal.gt_matched_indices) == list(range(n_units))
    assert cal.roa.min() >= 0.95
    assert cal.sep_vectors.shape == (dim, n_units)
    assert cal.whitening.shape == (dim, dim)
    assert (cal.spikes_centr > cal.base_centr).all()
    # Every true unit passes the unsupervised quality criteria too
    assert cal.unsupervised_mask(sil_th=0.9, **DEFAULT_UNIT_SELECTION_KWARGS).all()
    assert ((cal.dr > 8) & (cal.dr < 17)).all()


def test_apply_reproduces_the_calibration(
    synthetic_recording, synthetic_cbss_config, synthetic_calibration
):
    rec, cal = synthetic_recording, synthetic_calibration
    applied = CBSS(synthetic_cbss_config).apply(rec.emg[: rec.n_cal], cal)

    np.testing.assert_allclose(applied.sources, cal.sources, rtol=1e-3, atol=1e-3)
    roa, _, _ = rate_of_agreement_paired(cal.spikes, applied.spikes, fs=rec.fs, tol_spike_ms=1)
    assert roa.min() >= 0.99


def test_adaptation_tracks_the_units_the_fixed_decomposition_loses(
    synthetic_recording, synthetic_cbss_config
):
    """calibrate_and_process end to end, offline and online (streaming) modes."""
    rec = synthetic_recording
    cbss_config = replace(
        synthetic_cbss_config,
        selection="supervised",
        selection_kwargs={"gt_spikes": rec.spikes[: rec.n_cal], "tol_spike_ms": TOL_SPIKE_MS},
    )

    def run(preset, processing_mode="offline"):
        return AdaptDecomp.calibrate_and_process(
            rec.emg,
            timestamps=np.arange(len(rec.emg)) / rec.fs,
            calib_indices=slice(0, rec.n_cal),
            cbss_config=cbss_config,
            adapt_config=_preset(preset, cbss_config),
            processing_mode=processing_mode,
        )

    fixed, cal = run("fixed")
    adapted, _ = run("neuromotion")
    streamed, _ = run("neuromotion", processing_mode="online")

    assert adapted.spikes.shape == (len(rec.emg), len(cal.gt_matched_indices))
    last_third = rec.n_cal + 2 * (len(rec.emg) - rec.n_cal) // 3
    assert _roa(rec, cal, fixed.spikes, last_third).mean() < 0.6
    assert _roa(rec, cal, adapted.spikes, rec.n_cal).min() >= 0.95
    assert _roa(rec, cal, adapted.spikes, last_third).min() >= 0.95
    assert np.isfinite(np.asarray(adapted.wh_loss)).all()
    assert np.isfinite(np.asarray(adapted.sv_loss)).all()

    # Streaming the raw EMG batch by batch matches preprocessing it all at once
    roa, _, _ = rate_of_agreement_paired(
        np.asarray(adapted.spikes), np.asarray(streamed.spikes), fs=rec.fs, tol_spike_ms=1
    )
    assert roa.min() >= 0.99


def test_search_picks_the_parameters_that_track_the_drift(
    synthetic_recording, synthetic_cbss_config, synthetic_calibration
):
    """The unsupervised sv_loss objective ranks tuned parameters above none at all (the
    fixed decomposition), as their RoA against the ground truth does."""
    rec, cal = synthetic_recording, synthetic_calibration
    pool = {
        "synthetic": PooledDatasetMemory(
            emg=rec.emg[rec.n_cal :],
            calibration=cal,
            cbss_config=synthetic_cbss_config,
            preprocess=True,
            gt_paired_bin=rec.spikes[rec.n_cal :, cal.gt_matched_indices],
        )
    }
    param_space = {
        "wh_learning_rate": ("float", 0.0, 1e-2),
        "sv_learning_rate": ("float", 0.0, 1e-2),
        "centroid_momentum": ("float", 0.0, 1.0),
    }
    fixed = {"wh_learning_rate": 0.0, "sv_learning_rate": 0.0, "centroid_momentum": 1.0}
    tuned = {"wh_learning_rate": 7e-3, "sv_learning_rate": 3e-3, "centroid_momentum": 0.8}
    logs = []

    result = optimize_adapt_decomp(
        pool=pool,
        objectives="sv_loss",
        param_space=param_space,
        base_config=_preset("neuromotion", synthetic_cbss_config),
        compute_roa=True,
        roa_kwargs={"tol_spike_ms": TOL_SPIKE_MS},
        n_trials=2,
        initial_params=[fixed, tuned],
        on_trial=logs.append,
    )

    assert result.study.best_trial.params == pytest.approx(tuned)
    assert result.best_config.sv_learning_rate == pytest.approx(tuned["sv_learning_rate"])
    assert logs[0]["roa_mean"] < 60 and logs[1]["roa_mean"] >= 95  # in %
