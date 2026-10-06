"""Tests for adaptation/optimize/'s single entry point, optimize_adapt_decomp:
dispatch over pool kind (memory/disk) and objective count (single/Pareto),
unit selection, front selection rules and the default sampler. Worker
processes are tested in test_optimize_resources.py. End-to-end searches are
marked slow; the rest is pure logic.
"""

import numpy as np
import optuna
import pytest

from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    OptimisationResult,
    optimize_adapt_decomp,
    optimize_adapt_decomp_pooled_memory,
)
from adapt_decomp.adaptation.optimize.pareto import _select_knee
from adapt_decomp.adaptation.optimize.scoring import suggest_overrides
from adapt_decomp.adaptation.optimize.search import _make_study
from adapt_decomp.adaptation.optimize.units import select_pool_units
from adapt_decomp.utils.loaders import PooledDatasetMemory
from tests.adaptation.test_optimize import (
    _make_pooled_disk_base_config,
    _make_pooled_disk_dataset,
)


def _memory_pool(make_optimize_kwargs, names=("dataset_a",), cov_isi=None, gt=False):
    """In-memory pool of identical synthetic datasets; optionally override the
    calibration's per-unit CoV-ISI and attach paired ground truth."""
    pool = {}
    for name in names:
        common, M = make_optimize_kwargs()
        calibration = common["calibration"]
        if cov_isi is not None:
            calibration.cov_isi = np.asarray(cov_isi, dtype=np.float32)
        gt_paired_bin = None
        if gt:
            gt_paired_bin = np.zeros((common["emg"].shape[0], M), dtype=np.float32)
            gt_paired_bin[::30] = 1
        pool[name] = PooledDatasetMemory(
            emg=common["emg"],
            calibration=calibration,
            cbss_config=common["cbss_config"],
            preprocess=common["preprocess"],
            gt_paired_bin=gt_paired_bin,
        )
    return pool, common["base_config"]


# A parameter set on DEFAULT_PARAM_SPACE's grid
V10_WINNER = {"wh_learning_rate": 0.036, "sv_learning_rate": 0.0053, "centroid_momentum": 0.9}


def _front(points):
    """FrozenTrials with the given (wh_loss, sv_loss) values, sv_loss logged as a user_attr."""
    return [
        optuna.trial.create_trial(
            values=list(p), params={}, distributions={}, user_attrs={"sv_loss": p[-1]}
        )
        for p in points
    ]


# ------------------------------------------------------------------
# Front selection, sampler, unit selection -- no search run
# ------------------------------------------------------------------


def test_select_knee_picks_the_point_farthest_from_the_extremes_chord():
    front = _front([(0.0, 1.0), (0.1, 0.4), (0.3, 0.3), (1.0, 0.0)])
    assert _select_knee(front).values == [0.1, 0.4]


def test_select_knee_falls_back_to_min_sv_loss_on_tiny_fronts_and_rejects_three_objectives():
    assert _select_knee(_front([(0.0, 1.0), (1.0, 0.0)])).values == [1.0, 0.0]
    with pytest.raises(ValueError, match="exactly 2"):
        _select_knee(_front([(0.0, 1.0, 2.0), (1.0, 0.0, 2.0), (0.5, 0.5, 2.0)]))


def test_a_stepped_range_draws_only_its_grid():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    momenta = [
        suggest_overrides(study.ask(), DEFAULT_PARAM_SPACE)["centroid_momentum"] for _ in range(50)
    ]
    grid = np.round(np.arange(0.1, 1.0, 0.1), 10)
    assert set(np.round(momenta, 10)) <= set(grid)
    assert len(set(np.round(momenta, 10))) > 5  # and spreads over it
    int_space = {"batch_ms": ("int", 50, 200, 50)}
    assert {suggest_overrides(study.ask(), int_space)["batch_ms"] for _ in range(30)} <= {
        50,
        100,
        150,
        200,
    }


def test_default_sampler_is_multivariate_tpe_with_constant_liar_only_when_concurrent():
    sequential = _make_study(("wh_loss", "sv_loss"), None, 0, n_jobs=1).sampler
    concurrent = _make_study(("sv_loss",), None, 0, n_jobs=2).sampler
    assert isinstance(sequential, optuna.samplers.TPESampler)
    assert sequential._multivariate and not sequential._constant_liar
    assert concurrent._multivariate and concurrent._constant_liar


def test_unsupervised_unit_selection_subsets_calibration_and_gt_and_drops_empty_datasets(
    make_optimize_kwargs,
):
    """Units over cov_th are dropped from the calibration (and their ground
    truth columns with them); a dataset left with no unit leaves the pool."""
    pool, _ = _memory_pool(make_optimize_kwargs, ("keep",), cov_isi=[0.1, 0.5], gt=True)
    empty, _ = _memory_pool(make_optimize_kwargs, ("empty",), cov_isi=[0.6, 0.5], gt=True)
    gt_before = pool["keep"].gt_paired_bin

    selected = select_pool_units({**pool, **empty}, "unsupervised", {"cov_th": 0.3})

    assert set(selected) == {"keep"}
    assert selected["keep"].calibration.sources.shape[1] == 1
    np.testing.assert_array_equal(selected["keep"].gt_paired_bin, gt_before[:, [0]])
    with pytest.raises(ValueError, match="keeps any unit"):
        select_pool_units(empty, "unsupervised", {"cov_th": 0.3})
    assert select_pool_units(empty, None, {}) is empty  # ablation: every unit kept


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(unit_selection="supervised"), ValueError, "Ground truth"),
        (dict(objectives="roa"), ValueError, "Ground truth"),
        (dict(selection="knee"), ValueError, "exactly 2"),
        (dict(selection="bogus", objectives=("wh_loss", "sv_loss")), ValueError, "selection"),
        (dict(unit_selection="bogus"), ValueError, "unit_selection"),
        (dict(objectives=("sv_loss", "sv_loss")), ValueError, "duplicates"),
        (dict(initial_params=[{"wh_learning_rate": 1e-3}]), ValueError, "exactly the param_space"),
        (dict(initial_params=[{**V10_WINNER, "centroid_momentum": 0.95}]), ValueError, "outside"),
        (dict(initial_params=[{**V10_WINNER, "centroid_momentum": 0.25}]), ValueError, "outside"),
    ],
)
def test_invalid_search_inputs_raise_before_any_trial(make_optimize_kwargs, kwargs, error, match):
    pool, base_config = _memory_pool(make_optimize_kwargs)
    with pytest.raises(error, match=match):
        optimize_adapt_decomp(pool=pool, base_config=base_config, n_trials=1, **kwargs)


def test_mixed_memory_and_disk_pool_raises(make_optimize_kwargs, tmp_path):
    pool, base_config = _memory_pool(make_optimize_kwargs)
    pool["on_disk"] = _make_pooled_disk_dataset(tmp_path, "on_disk")
    with pytest.raises(TypeError, match="all PooledDatasetMemory or all PooledDatasetDisk"):
        optimize_adapt_decomp(pool=pool, base_config=base_config, n_trials=1)


# ------------------------------------------------------------------
# End-to-end searches
# ------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["memory", "disk"])
@pytest.mark.parametrize("objectives", ["sv_loss", ("wh_loss", "sv_loss")])
def test_dispatches_on_pool_kind_and_objective_count(
    kind, objectives, make_optimize_kwargs, tmp_path
):
    """One function covers all four former entry points: a Pareto front only
    for several objectives, reloaded outputs only for an in-memory pool."""
    if kind == "memory":
        pool, base_config = _memory_pool(make_optimize_kwargs)
    else:
        pool = {"dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a")}
        base_config = _make_pooled_disk_base_config()
    best_dir = tmp_path / "best"

    result = optimize_adapt_decomp(
        pool=pool,
        objectives=objectives,
        base_config=base_config,
        n_trials=3,
        best_result_path=str(best_dir),
    )

    assert isinstance(result, OptimisationResult)
    single = isinstance(objectives, str)
    assert (result.pareto_front is None) == single
    assert (result.outputs is None) == (kind == "disk")
    if kind == "memory":
        assert isinstance(result.outputs["dataset_a"], AdaptationResult)
    member = best_dir if single else best_dir / f"trial_{result.study.best_trials[0].number}"
    assert (member / "dataset_a.pkl").exists() and (best_dir / "study.pkl").exists()
    assert not best_dir.with_name("best_temp").exists()
    assert set(result.study.trials[0].params) == {
        "wh_learning_rate",
        "sv_learning_rate",
        "centroid_momentum",
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs, n_units",
    [({}, 2), (dict(unit_selection="unsupervised"), 1)],
    ids=["default_keeps_every_unit", "unsupervised"],
)
def test_unit_selection_runs_the_search_on_the_selected_units(
    make_optimize_kwargs, tmp_path, kwargs, n_units
):
    pool, base_config = _memory_pool(make_optimize_kwargs, cov_isi=[0.1, 0.5])
    result = optimize_adapt_decomp(
        pool=pool,
        base_config=base_config,
        n_trials=1,
        best_result_path=str(tmp_path / "best"),
        **kwargs,
    )
    assert result.outputs["dataset_a"].spikes.shape[1] == n_units


@pytest.mark.slow
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_initial_params_run_first_and_the_rest_is_sampled(make_optimize_kwargs, n_jobs):
    pool, base_config = _memory_pool(make_optimize_kwargs)
    second = {**V10_WINNER, "centroid_momentum": 0.5}
    result = optimize_adapt_decomp(
        pool=pool,
        base_config=base_config,
        n_trials=3,
        n_jobs=n_jobs,
        initial_params=[V10_WINNER, second],
    )
    trials = result.study.trials
    assert trials[0].params == pytest.approx(V10_WINNER)
    assert trials[1].params == pytest.approx(second)
    assert trials[2].params != trials[0].params


@pytest.mark.slow
def test_knee_selection_builds_best_config_from_the_knee_member(make_optimize_kwargs):
    pool, base_config = _memory_pool(make_optimize_kwargs)
    result = optimize_adapt_decomp(
        pool=pool,
        objectives=("wh_loss", "sv_loss"),
        selection="knee",
        base_config=base_config,
        n_trials=6,
    )
    knee = _select_knee(result.pareto_front)
    assert result.best_config.wh_learning_rate == pytest.approx(knee.params["wh_learning_rate"])


@pytest.mark.slow
def test_deprecated_entry_points_warn_and_keep_their_return_shapes(make_optimize_kwargs):
    pool, base_config = _memory_pool(make_optimize_kwargs)
    with pytest.warns(FutureWarning, match="optimize_adapt_decomp"):
        best_config, study = optimize_adapt_decomp_pooled_memory(
            pool=pool,
            param_space={"sv_learning_rate": ("log_float", 1e-4, 1e-1)},
            n_trials=1,
            base_config=base_config,
        )
    assert best_config.sv_learning_rate == pytest.approx(study.best_params["sv_learning_rate"])
