"""Tests for adaptation/optimize.py's Pareto/multi-objective search:
optimize_adapt_decomp_pooled_memory_pareto / _pooled_disk_pareto, and their
private dominance/front-maintenance helpers.

Kept in its own file, separate from test_optimize.py, so the whole Pareto
feature (this file plus its counterpart in optimize.py) stays deletable as
a unit without touching the single-objective search's own tests.

Unit tests for the pure dominance/front logic are fast and unmarked; the
end-to-end Optuna-study tests are marked slow, matching this module's
existing convention (see pyproject.toml's marker registration).
"""

from unittest.mock import patch

import numpy as np
import optuna
import pytest

from adapt_decomp.adaptation.data_structures import AdaptationResult
from adapt_decomp.adaptation.optimize import (
    DEFAULT_PARAM_SPACE,
    optimize_adapt_decomp_pooled_memory_pareto,
    optimize_adapt_decomp_pooled_disk_pareto,
    _dominates,
    _update_front,
    _save_study_snapshot,
)
from adapt_decomp.utils.loaders import PooledDatasetMemory

from tests.adaptation.test_optimize import (
    _make_pooled_disk_dataset,
    _make_pooled_disk_base_config,
)


# ------------------------------------------------------------------
# Unit layer: _dominates / _update_front -- pure logic, no Optuna/torch/IO
# ------------------------------------------------------------------

def test_dominates_strict():
    assert _dominates((1, 1), (2, 2)) is True
    assert _dominates((2, 2), (1, 1)) is False


def test_dominates_tie_dominates_neither():
    assert _dominates((1, 1), (1, 1)) is False


def test_dominates_mixed_dimensions_neither_dominates():
    """Neither point is <= the other in every dimension -- neither dominates."""
    assert _dominates((1, 5), (5, 1)) is False
    assert _dominates((5, 1), (1, 5)) is False


def test_update_front_join_evicts_dominated_member():
    front = {0: (5.0, 5.0)}
    joined, evicted = _update_front(front, 1, (3.0, 3.0))
    assert joined is True
    assert evicted == [0]
    assert front == {1: (3.0, 3.0)}


def test_update_front_no_join_when_dominated():
    front = {0: (1.0, 1.0)}
    joined, evicted = _update_front(front, 1, (2.0, 2.0))
    assert joined is False
    assert evicted == []
    assert front == {0: (1.0, 1.0)}  # unchanged


def test_update_front_tie_joins_without_evicting():
    front = {0: (1.0, 1.0)}
    joined, evicted = _update_front(front, 1, (1.0, 1.0))
    assert joined is True
    assert evicted == []
    assert front == {0: (1.0, 1.0), 1: (1.0, 1.0)}


def test_update_front_dominating_point_evicts_every_dominated_resident():
    """A point that's <= every resident member in both dimensions (and
    strictly < in at least one, for each) dominates and evicts all of
    them -- not just the ones it "beats" in a single dimension."""
    front = {0: (5.0, 5.0), 1: (5.0, 1.0), 2: (1.0, 5.0)}
    joined, evicted = _update_front(front, 3, (1.0, 1.0))
    assert joined is True
    assert sorted(evicted) == [0, 1, 2]
    assert front == {3: (1.0, 1.0)}


def test_update_front_mutually_non_dominated_members_both_survive():
    """Two points that are each better in a different dimension neither
    dominate one another -- both stay resident."""
    front = {0: (1.0, 5.0)}
    joined, evicted = _update_front(front, 1, (5.0, 1.0))
    assert joined is True
    assert evicted == []
    assert front == {0: (1.0, 5.0), 1: (5.0, 1.0)}


# ------------------------------------------------------------------
# Integration layer: optimize_adapt_decomp_pooled_memory_pareto
# ------------------------------------------------------------------

@pytest.mark.slow
class TestOptimizeAdaptDecompPooledMemoryPareto:
    def test_best_result_path_optional(self, tmp_path, make_optimize_kwargs):
        """Without best_result_path: 3-tuple (best_config, pareto_front,
        study), and study.best_value raises RuntimeError -- proof the study
        is genuinely multi-objective, not silently single-objective. With
        best_result_path: 4-tuple, best_outputs is a real per-dataset
        AdaptationResult dict, and at least one trial_<n>/ subdirectory plus
        study.pkl exist on disk."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}

        result_no_path = optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=3,
            base_config=common["base_config"],
        )
        assert len(result_no_path) == 3
        best_config, pareto_front, study = result_no_path
        assert len(pareto_front) >= 1
        with pytest.raises(RuntimeError):
            study.best_value

        best_dir = tmp_path / "pareto_front"
        result_with_path = optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=3,
            base_config=common["base_config"], best_result_path=str(best_dir),
        )
        assert len(result_with_path) == 4
        best_outputs, best_config2, pareto_front2, study2 = result_with_path
        assert set(best_outputs.keys()) == {"dataset_a"}
        assert isinstance(best_outputs["dataset_a"], AdaptationResult)
        assert (best_dir / "study.pkl").exists()
        trial_dirs = sorted(p.name for p in best_dir.iterdir() if p.name.startswith("trial_"))
        assert len(trial_dirs) >= 1

    def test_front_matches_study_best_trials(self, tmp_path, make_optimize_kwargs):
        """The single strongest correctness check: once the search
        completes, the set of resident trial_<n>/ subdirectories under
        best_dir equals {t.number for t in study.best_trials} exactly --
        broken eviction would leave stale extra directories (a superset),
        broken joining would leave missing ones (a subset)."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}
        best_dir = tmp_path / "pareto_front"

        _, _, pareto_front, study = optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=8,
            base_config=common["base_config"], best_result_path=str(best_dir),
        )

        on_disk = {
            int(p.name.removeprefix("trial_"))
            for p in best_dir.iterdir() if p.name.startswith("trial_")
        }
        assert on_disk == {t.number for t in study.best_trials}
        assert on_disk == {t.number for t in pareto_front}
        # Every resident trial's directory actually holds a reloadable result.
        for n in on_disk:
            reloaded = AdaptationResult.load(best_dir / f"trial_{n}" / "dataset_a.pkl")
            assert reloaded.wh_loss is not None

    def test_study_saved_after_every_trial(self, tmp_path, make_optimize_kwargs):
        """study.pkl is overwritten via an Optuna callback after every
        trial, not only once study.optimize() returns -- proven both by
        call count and by each snapshot genuinely reflecting that many
        completed trials (so a crash mid-search wouldn't just leave an
        empty/stale file)."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}
        best_dir = tmp_path / "pareto_front"
        trial_counts_seen = []

        def _wrapped(best_dir_arg, study_arg, lock_arg):
            trial_counts_seen.append(len(study_arg.trials))
            return _save_study_snapshot(best_dir_arg, study_arg, lock_arg)

        with patch(
            "adapt_decomp.adaptation.optimize._save_study_snapshot", side_effect=_wrapped,
        ) as mock_save:
            optimize_adapt_decomp_pooled_memory_pareto(
                pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=3,
                base_config=common["base_config"], best_result_path=str(best_dir),
            )

        assert mock_save.call_count == 3
        assert trial_counts_seen == [1, 2, 3]  # strictly growing, not 3 calls all at the end

    def test_on_trial_log_vars(self, make_optimize_kwargs):
        """Log dict carries objectives/values (the Pareto-shaped
        replacement for the single-objective search's "loss") plus the
        usual pooled sv_loss/wh_loss/total_loss/params/per_dataset -- and
        no "loss" key, since there is no single scalar here."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}
        seen = []

        optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            base_config=common["base_config"], on_trial=seen.append,
        )

        assert len(seen) == 2
        for v in seen:
            assert "loss" not in v
            assert v["objectives"] == ("wh_loss", "sv_loss")
            assert len(v["values"]) == 2
            assert set(v) >= {"trial_number", "objectives", "values", "sv_loss", "wh_loss",
                               "total_loss", "params", "per_dataset"}
            assert "on_front" not in v  # best_result_path not set here
            assert "loss" not in v["per_dataset"]["dataset_a"]

    def test_on_trial_log_vars_carry_on_front_when_best_result_path_set(self, tmp_path, make_optimize_kwargs):
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}
        seen = []

        optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            base_config=common["base_config"], on_trial=seen.append,
            best_result_path=str(tmp_path / "pareto_front"),
        )

        assert len(seen) == 2
        for v in seen:
            assert isinstance(v["on_front"], bool)

    @pytest.mark.parametrize("objectives", [
        ("wh_loss",),                    # too few
        ("wh_loss", "wh_loss"),          # duplicate
        ("wh_loss", "bogus"),            # unknown
    ])
    def test_invalid_objectives_raises(self, objectives, make_optimize_kwargs):
        """Bad objectives raises ValueError up front, before any trial runs."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}
        with pytest.raises(ValueError):
            optimize_adapt_decomp_pooled_memory_pareto(
                pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1,
                base_config=common["base_config"], objectives=objectives,
            )

    def test_selection_rule_pluggable(self, make_optimize_kwargs):
        """A custom selection_rule is actually used to build best_config,
        not silently overridden by the default _select_min_sv_loss."""
        common, _ = make_optimize_kwargs()
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
        )}

        def _select_highest_trial_number(pareto_front):
            return max(pareto_front, key=lambda t: t.number)

        best_config, pareto_front, study = optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=5,
            base_config=common["base_config"], selection_rule=_select_highest_trial_number,
        )

        chosen = _select_highest_trial_number(pareto_front)
        assert best_config.wh_learning_rate == pytest.approx(chosen.params["wh_learning_rate"])
        assert best_config.sv_learning_rate == pytest.approx(chosen.params["sv_learning_rate"])

    def test_compute_roa_via_objectives(self, make_optimize_kwargs):
        """"roa" in objectives implies compute_roa=True, mirroring
        objective="roa" for the single-objective search, and roa_mean_pooled
        travels through on_trial the same way."""
        common, M = make_optimize_kwargs()
        n_samples = common["emg"].shape[0]
        gt_full_bin = np.zeros((n_samples, M), dtype=np.float32)
        gt_full_bin[::30] = 1
        pool = {"dataset_a": PooledDatasetMemory(
            emg=common["emg"], calibration=common["calibration"],
            cbss_config=common["cbss_config"], preprocess=common["preprocess"],
            gt_paired_bin=gt_full_bin,
        )}
        seen = []

        optimize_adapt_decomp_pooled_memory_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=2,
            objectives=("wh_loss", "roa"), roa_kwargs={"tol_spike_ms": 25},
            base_config=common["base_config"], on_trial=seen.append,
        )

        assert len(seen) == 2
        for v in seen:
            assert "roa_mean" in v and "roa" in v


# ------------------------------------------------------------------
# Integration layer: optimize_adapt_decomp_pooled_disk_pareto
# ------------------------------------------------------------------

@pytest.mark.slow
class TestOptimizeAdaptDecompPooledDiskPareto:
    def test_best_result_path_optional_and_writes_reloadable_results(self, tmp_path):
        """As the memory variant's equivalent test, but never returns
        AdaptationResults in memory -- reload from
        best_dir/trial_<n>/<dataset>.pkl instead."""
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }

        result_no_path = optimize_adapt_decomp_pooled_disk_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=3, base_config=base_config,
        )
        assert len(result_no_path) == 3
        best_config, pareto_front, study = result_no_path
        assert len(pareto_front) >= 1
        with pytest.raises(RuntimeError):
            study.best_value

        best_dir = tmp_path / "pareto_front"
        result_with_path = optimize_adapt_decomp_pooled_disk_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=3,
            base_config=base_config, best_result_path=str(best_dir),
        )
        assert len(result_with_path) == 3  # never a 4-tuple, unlike the memory variant

        assert (best_dir / "study.pkl").exists()
        assert not best_dir.with_name(best_dir.name + "_temp").exists()
        trial_dirs = [p for p in best_dir.iterdir() if p.name.startswith("trial_")]
        assert len(trial_dirs) >= 1
        for trial_dir in trial_dirs:
            for name in pool:
                reloaded = AdaptationResult.load(trial_dir / f"{name}.pkl")
                assert reloaded.wh_loss is not None
            assert (trial_dir / "config.yaml").exists()

    def test_front_matches_study_best_trials(self, tmp_path):
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }
        best_dir = tmp_path / "pareto_front"

        _, pareto_front, study = optimize_adapt_decomp_pooled_disk_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=8,
            base_config=base_config, best_result_path=str(best_dir),
        )

        on_disk = {
            int(p.name.removeprefix("trial_"))
            for p in best_dir.iterdir() if p.name.startswith("trial_")
        }
        assert on_disk == {t.number for t in study.best_trials}
        assert on_disk == {t.number for t in pareto_front}

    @pytest.mark.parametrize("objectives", [
        ("wh_loss",),
        ("wh_loss", "wh_loss"),
        ("wh_loss", "bogus"),
    ])
    def test_invalid_objectives_raises(self, objectives, tmp_path):
        base_config = _make_pooled_disk_base_config()
        pool = {"dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a")}
        with pytest.raises(ValueError):
            optimize_adapt_decomp_pooled_disk_pareto(
                pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=1,
                base_config=base_config, objectives=objectives,
            )

    def test_n_jobs_promotes_correct_files_without_scratch_collisions(self, tmp_path):
        """Under n_jobs>1, every resident trial_<n>/ directory ends up with
        exactly one .pkl per dataset plus config.yaml (no partial writes
        from a race between the front-join check and this trial's own
        scratch cleanup), and the scratch "_temp" directory is fully
        cleaned up once the search finishes."""
        base_config = _make_pooled_disk_base_config()
        pool = {
            "dataset_a": _make_pooled_disk_dataset(tmp_path, "dataset_a", seed=0),
            "dataset_b": _make_pooled_disk_dataset(tmp_path, "dataset_b", seed=1),
        }
        best_dir = tmp_path / "pareto_front"

        _, pareto_front, study = optimize_adapt_decomp_pooled_disk_pareto(
            pool=pool, param_space=DEFAULT_PARAM_SPACE, n_trials=8, n_jobs=4,
            base_config=base_config, best_result_path=str(best_dir),
        )

        assert len(study.trials) == 8
        trial_dirs = [p for p in best_dir.iterdir() if p.name.startswith("trial_")]
        assert len(trial_dirs) >= 1
        for trial_dir in trial_dirs:
            files = {p.name for p in trial_dir.iterdir()}
            assert files == {"dataset_a.pkl", "dataset_b.pkl", "config.yaml"}
        assert not best_dir.with_name(best_dir.name + "_temp").exists()
