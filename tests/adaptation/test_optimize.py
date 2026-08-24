"""Tests for adaptation/optimize.py: optimize_adapt_decomp / _pooled.

All marked slow -- each spins up a real (if tiny) Optuna study end to end.
"""

import numpy as np
import pytest

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.optimize import (
    optimize_adapt_decomp,
    optimize_adapt_decomp_pooled,
    PooledCondition,
    DEFAULT_PARAM_SPACE,
)
from adapt_decomp.adaptation.data_structures import AdaptationResult

pytestmark = pytest.mark.slow


def test_optimize_adapt_decomp_on_trial_fires_once_per_trial_with_log_vars(make_optimize_kwargs):
    """on_trial is invoked exactly once per completed trial, from inside the
    objective itself, with the canonical {trial_number, loss, params} log dict
    -- not Optuna's own (study, trial) callback shape."""
    common, _ = make_optimize_kwargs()
    seen = []

    def _on_trial(log_vars):
        seen.append(log_vars)

    optimize_adapt_decomp(
        param_space=DEFAULT_PARAM_SPACE, n_trials=3, on_trial=_on_trial, **common,
    )
    assert [v["trial_number"] for v in seen] == [0, 1, 2]
    for v in seen:
        assert isinstance(v["loss"], float)
        assert set(v["params"]) == set(DEFAULT_PARAM_SPACE)
        assert "roa_mean" not in v  # compute_roa=False here


def test_optimize_adapt_decomp_best_result_path_optional(tmp_path, make_optimize_kwargs):
    """best_result_path stays a fully optional opt-in: omitted -> (best_config,
    study); set -> (best_outputs, best_config, study) with the winning trial's
    full AdaptationResult and its files written to disk."""
    common, _ = make_optimize_kwargs()

    result_no_path = optimize_adapt_decomp(param_space=DEFAULT_PARAM_SPACE, n_trials=2, **common)
    assert len(result_no_path) == 2
    best_config, study = result_no_path
    assert isinstance(best_config, AdaptConfig)
    assert isinstance(study.best_value, float)

    best_dir = tmp_path / "best_trial"
    result_with_path = optimize_adapt_decomp(
        param_space=DEFAULT_PARAM_SPACE, n_trials=2, best_result_path=str(best_dir), **common,
    )
    assert len(result_with_path) == 3
    outputs, best_config2, study2 = result_with_path
    assert isinstance(outputs, AdaptationResult)
    assert outputs.wh_loss is not None
    for fname in ("result.pkl", "config.yaml", "study.pkl"):
        assert (best_dir / fname).exists()


def test_optimize_adapt_decomp_compute_roa(tmp_path, make_optimize_kwargs):
    """compute_roa=True logs roa_mean/roa_per_unit as user_attrs on every trial
    AND on on_trial's log_vars, and (with best_result_path set) writes RoA onto
    the winning trial's AdaptationResult.roa."""
    common, M = make_optimize_kwargs()
    n_samples = common["emg"].shape[0]
    gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
    gt_full_bin[::30] = 1
    # Default roa_kwargs tol_spike_ms=2 rounds to 0 samples at this fixture's low
    # fs=200 (round(2 * 200 / 1000) == 0), which rate_of_agreement_paired can't
    # convolve with -- widen it for this fs, same as a real low-fs caller would.
    roa_kwargs = {"tol_spike_ms": 25}
    seen = []

    best_config, study = optimize_adapt_decomp(
        param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        compute_roa=True, gt_full_bin=gt_full_bin, roa_kwargs=roa_kwargs,
        on_trial=seen.append, **common,
    )
    for trial in study.trials:
        assert "roa_mean" in trial.user_attrs
        assert "roa_per_unit" in trial.user_attrs
    assert len(seen) == 2
    for v in seen:
        assert "roa_mean" in v and "roa_per_unit" in v
        assert len(v["roa_per_unit"]) == M

    best_dir = tmp_path / "best_trial_roa"
    outputs, _, _ = optimize_adapt_decomp(
        param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        compute_roa=True, gt_full_bin=gt_full_bin, roa_kwargs=roa_kwargs,
        best_result_path=str(best_dir), **common,
    )
    assert outputs.roa is not None
    assert outputs.roa.shape == (M,)


def test_optimize_adapt_decomp_pooled_on_trial_log_vars(make_optimize_kwargs):
    """optimize_adapt_decomp_pooled's on_trial log dict carries the pooled SUM
    as "loss" plus a "per_condition" entry (with its own loss/RoA) for every
    condition in the pool -- the schema that fixed the original bug (an
    external wandb callback assuming the single-condition user_attrs keys,
    which pooled never sets)."""
    common_a, M = make_optimize_kwargs()
    common_b, _ = make_optimize_kwargs()
    # preprocess lives per-condition on PooledCondition (not as a top-level
    # optimize_adapt_decomp_pooled param) -- carried over from each
    # make_optimize_kwargs() dict, same False value the single-condition
    # tests use (this fixture's low fs=200 is below AdaptConfig's default
    # highcut=500, so leaving preprocess at PooledCondition's own True default
    # would raise -- see AdaptConfig.preprocess).
    _pooled_keys = {"emg", "whitening", "sep_vectors", "base_centr", "spikes_centr",
                    "emg_calib", "ipts_calib", "spikes_calib", "preprocess"}
    pool = {
        "condA": PooledCondition(**{k: v for k, v in common_a.items() if k in _pooled_keys}),
        "condB": PooledCondition(**{k: v for k, v in common_b.items() if k in _pooled_keys}),
    }
    n_samples = common_a["emg"].shape[0]
    gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
    gt_full_bin[::30] = 1
    pool["condA"].gt_full_bin = gt_full_bin
    pool["condB"].gt_full_bin = gt_full_bin
    seen = []

    optimize_adapt_decomp_pooled(
        pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
        base_config=common_a["base_config"],
        compute_roa=True, roa_kwargs={"tol_spike_ms": 25}, on_trial=seen.append,
    )

    assert len(seen) == 2
    for v in seen:
        assert set(v["per_condition"]) == {"condA", "condB"}
        assert v["loss"] == pytest.approx(
            sum(c["loss"] for c in v["per_condition"].values())
        )
        for cond_vars in v["per_condition"].values():
            assert "roa_mean" in cond_vars and "roa_per_unit" in cond_vars
        assert "roa_mean" in v  # pooled mean, present since compute_roa=True


def test_optimize_adapt_decomp_compute_roa_requires_gt_full_bin(make_optimize_kwargs):
    """compute_roa=True without gt_full_bin raises ValueError up front, before
    any trial runs (matches the docstring's Raises: entry)."""
    common, _ = make_optimize_kwargs()
    with pytest.raises(ValueError):
        optimize_adapt_decomp(param_space=DEFAULT_PARAM_SPACE, n_trials=1, compute_roa=True, **common)
