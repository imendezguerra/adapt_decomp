"""Tests for adaptation/optimize/'s search, optimize_adapt_decomp: Pareto-front logic,
front selection, the sampler, unit selection, input validation and the deprecated entry
points, then a few end-to-end searches (marked slow) that each cover one pool kind and
objective count. Resource planning and worker processes are in test_optimize_resources.py;
a search on a drifting synthetic recording is in tests/test_pipeline.py.
"""

from unittest.mock import patch

import numpy as np
import optuna
import pytest

from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    OptimisationResult,
    deprecated,
    optimize_adapt_decomp,
)
from adapt_decomp.adaptation.optimize.pareto import (
    _dominates,
    _select_knee,
    front_mask,
    update_front,
)
from adapt_decomp.adaptation.optimize.persistence import save_study_snapshot
from adapt_decomp.adaptation.optimize.scoring import suggest_overrides
from adapt_decomp.adaptation.optimize.search import _make_study
from adapt_decomp.adaptation.optimize.units import select_pool_units

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
# Pareto front: dominance, the running front and its selection rules
# ------------------------------------------------------------------


def test_dominates_needs_no_worse_everywhere_and_better_somewhere():
    assert _dominates((1, 1), (2, 2)) and not _dominates((2, 2), (1, 1))
    assert not _dominates((1, 1), (1, 1))  # a tie dominates neither
    assert not _dominates((1, 5), (5, 1)) and not _dominates((5, 1), (1, 5))


@pytest.mark.parametrize(
    "front, point, joined, evicted, after",
    [
        ({0: (5.0, 5.0)}, (3.0, 3.0), True, [0], {1: (3.0, 3.0)}),
        ({0: (1.0, 1.0)}, (2.0, 2.0), False, [], {0: (1.0, 1.0)}),
        ({0: (1.0, 1.0)}, (1.0, 1.0), True, [], {0: (1.0, 1.0), 1: (1.0, 1.0)}),
        ({0: (1.0, 5.0)}, (5.0, 1.0), True, [], {0: (1.0, 5.0), 1: (5.0, 1.0)}),
        (
            {0: (5.0, 5.0), 2: (5.0, 1.0), 3: (1.0, 5.0)},
            (1.0, 1.0),
            True,
            [0, 2, 3],
            {1: (1.0, 1.0)},
        ),
    ],
    ids=[
        "evicts_dominated",
        "dominated_stays_out",
        "tie_joins",
        "non_dominated_joins",
        "evicts_all",
    ],
)
def test_update_front(front, point, joined, evicted, after):
    got_joined, got_evicted = update_front(front, 1, point)
    assert (got_joined, sorted(got_evicted)) == (joined, evicted)
    assert front == after


def test_front_mask_matches_the_running_front_and_drops_non_finite_rows():
    rng = np.random.default_rng(0)
    values = rng.random((40, 2))
    values[7] = values[3]  # a tie stays on the front with its twin, as in update_front
    front = {}
    for number, row in enumerate(values):
        update_front(front, number, tuple(row))
    assert set(np.flatnonzero(front_mask(values))) == set(front)

    values = np.array([[1.0, 1.0], [np.nan, 0.0], [0.5, np.inf], [2.0, 2.0]])
    assert list(front_mask(values)) == [True, False, False, False]


def test_select_knee_picks_the_point_farthest_from_the_extremes_chord():
    front = _front([(0.0, 1.0), (0.1, 0.4), (0.3, 0.3), (1.0, 0.0)])
    assert _select_knee(front).values == [0.1, 0.4]
    # Too few points for a knee: the min-sv_loss end; and two objectives only
    assert _select_knee(_front([(0.0, 1.0), (1.0, 0.0)])).values == [1.0, 0.0]
    with pytest.raises(ValueError, match="exactly 2"):
        _select_knee(_front([(0.0, 1.0, 2.0), (1.0, 0.0, 2.0), (0.5, 0.5, 2.0)]))


# ------------------------------------------------------------------
# Sampler, parameter space and unit selection -- no search run
# ------------------------------------------------------------------


def test_a_stepped_range_draws_only_its_grid():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    momenta = {
        round(suggest_overrides(study.ask(), DEFAULT_PARAM_SPACE)["centroid_momentum"], 10)
        for _ in range(50)
    }
    assert momenta <= set(np.round(np.arange(0.1, 1.0, 0.1), 10)) and len(momenta) > 5
    int_space = {"batch_ms": ("int", 50, 200, 50)}
    drawn = {suggest_overrides(study.ask(), int_space)["batch_ms"] for _ in range(30)}
    assert drawn <= {50, 100, 150, 200}


def test_default_sampler_is_multivariate_tpe_with_constant_liar_only_when_concurrent():
    sequential = _make_study(("wh_loss", "sv_loss"), None, 0, n_jobs=1).sampler
    concurrent = _make_study(("sv_loss",), None, 0, n_jobs=2).sampler
    assert isinstance(sequential, optuna.samplers.TPESampler)
    assert sequential._multivariate and not sequential._constant_liar
    assert concurrent._multivariate and concurrent._constant_liar


def test_unsupervised_unit_selection_subsets_calibration_and_gt_and_drops_empty_datasets(
    make_memory_pool,
):
    """Units over cov_th are dropped from the calibration (and their ground
    truth columns with them); a dataset left with no unit leaves the pool."""
    pool, _ = make_memory_pool(("keep",), cov_isi=[0.1, 0.5], gt=True)
    empty, _ = make_memory_pool(("empty",), cov_isi=[0.6, 0.5], gt=True)
    gt_before = pool["keep"].gt_paired_bin

    selected = select_pool_units({**pool, **empty}, "unsupervised", {"cov_th": 0.3})

    assert set(selected) == {"keep"}
    assert selected["keep"].calibration.sources.shape[1] == 1
    np.testing.assert_array_equal(selected["keep"].gt_paired_bin, gt_before[:, [0]])
    with pytest.raises(ValueError, match="keeps any unit"):
        select_pool_units(empty, "unsupervised", {"cov_th": 0.3})
    assert select_pool_units(empty, None, {}) is empty  # ablation: every unit kept


# ------------------------------------------------------------------
# Input validation: every bad input raises before any trial runs
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(objectives="bogus"), ValueError, "objective"),
        (dict(objectives=("wh_loss", "bogus")), ValueError, "objective"),
        (dict(objectives=("sv_loss", "sv_loss")), ValueError, "duplicates"),
        (dict(objectives="roa"), ValueError, "Ground truth"),
        (dict(compute_roa=True), ValueError, "Ground truth"),
        (dict(unit_selection="supervised"), ValueError, "Ground truth"),
        (dict(unit_selection="bogus"), ValueError, "unit_selection"),
        (dict(selection="knee"), ValueError, "exactly 2"),
        (dict(selection="bogus", objectives=("wh_loss", "sv_loss")), ValueError, "selection"),
        (dict(initial_params=[{"wh_learning_rate": 1e-3}]), ValueError, "exactly the param_space"),
        (dict(initial_params=[{**V10_WINNER, "centroid_momentum": 0.95}]), ValueError, "outside"),
        (dict(initial_params=[{**V10_WINNER, "centroid_momentum": 0.25}]), ValueError, "outside"),
    ],
)
def test_invalid_search_inputs_raise_before_any_trial(make_memory_pool, kwargs, error, match):
    pool, base_config = make_memory_pool()
    with pytest.raises(error, match=match):
        optimize_adapt_decomp(pool=pool, base_config=base_config, n_trials=1, **kwargs)


def test_empty_or_mixed_memory_and_disk_pools_raise(make_memory_pool, make_disk_dataset):
    pool, base_config = make_memory_pool()
    with pytest.raises(ValueError, match="empty"):
        optimize_adapt_decomp(pool={}, base_config=base_config, n_trials=1)
    pool["on_disk"] = make_disk_dataset("on_disk")
    with pytest.raises(TypeError, match="all PooledDatasetMemory or all PooledDatasetDisk"):
        optimize_adapt_decomp(pool=pool, base_config=base_config, n_trials=1)


# ------------------------------------------------------------------
# Deprecated entry points: thin wrappers over optimize_adapt_decomp
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kwargs, forwarded, returns",
    [
        ("optimize_adapt_decomp_pooled_memory", {}, {"objectives": "sv_loss"}, ("cfg", "study")),
        (
            "optimize_adapt_decomp_pooled_memory",
            {"best_result_path": "best"},
            {"objectives": "sv_loss", "best_result_path": "best"},
            ("outputs", "cfg", "study"),
        ),
        (
            "optimize_adapt_decomp_pooled_disk",
            {"objective": "wh_loss", "best_result_path": "best"},
            {"objectives": "wh_loss", "best_result_path": "best"},
            ("cfg", "study"),
        ),
        (
            "optimize_adapt_decomp_pooled_memory_pareto",
            {},
            {"objectives": ("wh_loss", "sv_loss"), "selection": "min_sv_loss"},
            ("cfg", "front", "study"),
        ),
        (
            "optimize_adapt_decomp_pooled_memory_pareto",
            {"best_result_path": "best", "selection_rule": max},
            {"best_result_path": "best", "selection": max},
            ("outputs", "cfg", "front", "study"),
        ),
        (
            "optimize_adapt_decomp_pooled_disk_pareto",
            {"objectives": ("wh_loss", "roa")},
            {"objectives": ("wh_loss", "roa")},
            ("cfg", "front", "study"),
        ),
    ],
)
def test_deprecated_entry_points_warn_forward_and_keep_their_return_shapes(
    monkeypatch, name, kwargs, forwarded, returns
):
    calls = []

    def fake_search(**kw):
        calls.append(kw)
        return OptimisationResult("cfg", "study", pareto_front="front", outputs="outputs")

    monkeypatch.setattr(deprecated, "optimize_adapt_decomp", fake_search)
    with pytest.warns(FutureWarning, match="optimize_adapt_decomp"):
        result = getattr(deprecated, name)(pool="pool", param_space="space", **kwargs)

    assert result == returns
    assert calls[0]["unit_selection"] is None  # the old entry points never selected units
    assert calls[0]["pool"] == "pool" and calls[0]["param_space"] == "space"
    assert {k: calls[0][k] for k in forwarded} == forwarded


def test_deprecated_pareto_entry_points_need_two_objectives(monkeypatch):
    monkeypatch.setattr(deprecated, "optimize_adapt_decomp", lambda **kw: pytest.fail("ran"))
    for name in (
        "optimize_adapt_decomp_pooled_memory_pareto",
        "optimize_adapt_decomp_pooled_disk_pareto",
    ):
        with pytest.raises(ValueError, match="at least 2"):
            getattr(deprecated, name)(pool={}, param_space={}, objectives=("wh_loss",))


# ------------------------------------------------------------------
# End-to-end searches
# ------------------------------------------------------------------


@pytest.mark.slow
def test_single_objective_search_over_a_memory_pool(make_memory_pool, tmp_path):
    """Two datasets with ground truth, two trials at a time, the first one enqueued, and
    unsupervised unit selection (which narrows the ground truth too): every trial's log
    pools its datasets, and the best trial's results are promoted to best_result_path."""
    pool, base_config = make_memory_pool(("dataset_a", "dataset_b"), cov_isi=[0.1, 0.5], gt=True)
    best_dir = tmp_path / "best"
    logs = []

    result = optimize_adapt_decomp(
        pool=pool,
        objectives="total_loss",
        base_config=base_config,
        compute_roa=True,
        roa_kwargs={"tol_spike_ms": 25},  # the default 2 ms rounds to 0 samples at fs=200
        unit_selection="unsupervised",
        n_trials=4,
        n_jobs=2,
        initial_params=[V10_WINNER],
        best_result_path=str(best_dir),
        on_trial=logs.append,
    )

    study = result.study
    assert len(study.trials) == 4 and result.pareto_front is None
    assert study.trials[0].params == pytest.approx(V10_WINNER)
    assert set(study.trials[0].params) == set(DEFAULT_PARAM_SPACE)
    assert result.best_config.sv_learning_rate == pytest.approx(
        study.best_params["sv_learning_rate"]
    )

    assert len(logs) == 4
    for log in logs:
        per_dataset = log["per_dataset"]
        assert set(per_dataset) == set(pool) and isinstance(log["on_front"], bool)
        assert log["objective"] == "total_loss" and log["loss"] == pytest.approx(log["total_loss"])
        for key in ("loss", "sv_loss", "wh_loss", "total_loss", "roa"):
            assert log[key] == pytest.approx(sum(d[key] for d in per_dataset.values()))
        assert log["roa_mean"] == pytest.approx(
            np.mean([d["roa_mean"] for d in per_dataset.values()])
        )
        assert all(len(d["roa_per_unit"]) == 1 for d in per_dataset.values())

    # The best trial's results, reloaded for an in-memory pool, on the selected unit only
    assert set(result.outputs) == set(pool)
    for name, output in result.outputs.items():
        assert isinstance(output, AdaptationResult)
        assert output.spikes.shape[1] == 1 and output.roa.shape == (1,)
        assert (best_dir / f"{name}.pkl").exists()
    assert (best_dir / "config.yaml").exists() and (best_dir / "study.pkl").exists()
    assert not best_dir.with_name("best_temp").exists()


@pytest.mark.slow
def test_pareto_search_over_a_disk_pool(make_disk_dataset, make_optimize_kwargs, tmp_path):
    """Two on-disk datasets, two trials at a time: the front kept on disk is exactly the
    study's, with one complete trial_<n>/ per member; the study is saved after every
    trial; the knee member builds best_config; on-disk results are not reloaded."""
    base_config = make_optimize_kwargs()[0]["base_config"]
    pool = {name: make_disk_dataset(name, seed=i) for i, name in enumerate(("a", "b"))}
    best_dir = tmp_path / "front"
    logs = []

    with patch(
        "adapt_decomp.adaptation.optimize.search.save_study_snapshot",
        side_effect=save_study_snapshot,
    ) as snapshot:
        result = optimize_adapt_decomp(
            pool=pool,
            objectives=("wh_loss", "sv_loss"),
            selection="knee",
            base_config=base_config,
            n_trials=6,
            n_jobs=2,
            best_result_path=str(best_dir),
            on_trial=logs.append,
        )

    study = result.study
    assert len(study.trials) == 6 and snapshot.call_count == 6
    with pytest.raises(RuntimeError):
        study.best_value  # genuinely multi-objective
    on_disk = {int(p.name.removeprefix("trial_")) for p in best_dir.glob("trial_*")}
    assert (
        on_disk == {t.number for t in study.best_trials} == {t.number for t in result.pareto_front}
    )
    for n in on_disk:
        assert {p.name for p in (best_dir / f"trial_{n}").iterdir()} == {
            "a.pkl",
            "b.pkl",
            "config.yaml",
        }
    assert AdaptationResult.load(best_dir / f"trial_{min(on_disk)}" / "a.pkl").wh_loss is not None
    assert (best_dir / "study.pkl").exists() and not best_dir.with_name("front_temp").exists()
    assert result.outputs is None

    knee = _select_knee(result.pareto_front)
    assert result.best_config.wh_learning_rate == pytest.approx(knee.params["wh_learning_rate"])
    for log in logs:
        assert "loss" not in log and log["objectives"] == ("wh_loss", "sv_loss")
        assert len(log["values"]) == 2 and isinstance(log["on_front"], bool)
