"""Tests for spikes/comparison.py: agreement metrics, lag search, Hungarian pairing,
greedy duplicate removal, and the deprecated 1.0.0 names."""

from typing import Sequence, Tuple

import numpy as np
import pytest

from adapt_decomp.spikes import rate_of_agreement, rate_of_agreement_paired
from adapt_decomp.spikes.comparison import (
    _agreement_matrix,
    compare_paired_spikes,
    compare_spikes,
    find_duplicates,
)

FS = 2048


def _train(seed: int, n: int = 100, isi: Tuple[int, int] = (60, 140), start: int = 200):
    """Irregular spike train: n spikes with uniform random inter-spike intervals (samples)."""
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.integers(*isi, n))


def _binary(n_samples: int, *trains: np.ndarray) -> np.ndarray:
    """Binary (n_samples, n_units) spike matrix of the given spike trains."""
    out = np.zeros((n_samples, len(trains)), dtype=np.int32)
    for unit, train in enumerate(trains):
        out[train, unit] = 1
    return out


def _binary_of(trains: Sequence[np.ndarray]) -> np.ndarray:
    """Binary spike matrix just long enough for the given trains."""
    return _binary(max(int(t.max()) for t in trains) + 1, *trains)


# --- metrics and lag -----------------------------------------------------------------


def test_metrics_count_matched_missed_and_extra_spikes():
    """Test train = reference + 5 samples, one reference spike missed, one extra spike."""
    ref = _binary(1000, np.array([100, 300, 520, 700, 910]))
    test = _binary(1000, np.array([105, 305, 525, 640, 915]))

    agreement = compare_paired_spikes(ref, test, fs=FS, tol_spike_ms=1)

    np.testing.assert_array_equal(agreement.pairs, [[0, 0]])
    assert (agreement.lag[0], agreement.tp[0], agreement.fn[0], agreement.fp[0]) == (-5, 4, 1, 1)
    np.testing.assert_allclose(agreement.roa, 4 / 6)
    np.testing.assert_allclose(agreement.f1, 0.8)
    np.testing.assert_allclose(agreement.precision, 0.8)
    np.testing.assert_allclose(agreement.sensitivity, 0.8)
    np.testing.assert_allclose(agreement.f1, 2 * agreement.roa / (1 + agreement.roa))


def test_lag_search_stays_within_max_lag():
    """A shift beyond max_lag is not aligned, between sets as well as paired."""
    t = _train(0)
    ref = _binary(t.max() + 300, t)
    near, far = _binary(ref.shape[0], t + 5), _binary(ref.shape[0], t + 120)  # 120 > 82 samples

    assert compare_paired_spikes(ref, near, fs=FS).lag[0] == -5
    assert compare_spikes(ref, near, fs=FS).roa[0] == 1.0
    for agreement in (
        compare_paired_spikes(ref, far, fs=FS),
        compare_spikes(ref, far, fs=FS),
    ):
        assert abs(agreement.lag[0]) <= round(40 / 1000 * FS)
        assert agreement.roa[0] < 0.3


@pytest.mark.parametrize(
    "spikes, kwargs, expected",
    [
        (_binary(1000, np.array([100, 500]), np.array([], int)), {}, [1.0, 0.0]),
        (_binary(50, np.array([5, 30])), {}, [1.0]),
        (_binary(1000, np.array([100, 500])), {"tol_spike_ms": 0.2}, [1.0]),
    ],
    ids=["unit-silent-in-both", "recording-shorter-than-max-lag", "tolerance-rounds-to-0"],
)
def test_paired_edge_cases_score_without_crashing(spikes, kwargs, expected):
    agreement = compare_paired_spikes(spikes, spikes.copy(), fs=FS, **kwargs)
    np.testing.assert_allclose(agreement.roa, expected)


def test_hungarian_pairs_exact_copy_over_busy_unit():
    """A unit holding all of a's spikes plus many extras must not take a's partner."""
    a, b = _train(0), _train(1)
    busy = np.union1d(a, _train(2, n=300, isi=(10, 30)))
    n_samples = max(a.max(), b.max(), busy.max()) + 1

    agreement = compare_spikes(_binary(n_samples, a, b), _binary(n_samples, b, busy, a), fs=FS)

    np.testing.assert_array_equal(agreement.pairs, [[1, 0], [0, 2]])  # ordered by test unit
    np.testing.assert_allclose(agreement.roa, [1.0, 1.0])


def test_symmetric_agreement_matrix_mirrors_the_full_one():
    times = [_train(0), _train(0) + 3, _train(1), np.sort(_train(0)[::2] + 6)]
    lag_sym, tp_sym = _agreement_matrix(times, times, max_lag=82, tol=1, symmetric=True)
    lag_full, tp_full = _agreement_matrix(times, times, max_lag=82, tol=1)
    off_diagonal = ~np.eye(len(times), dtype=bool)
    np.testing.assert_array_equal(tp_sym[off_diagonal], tp_full[off_diagonal])
    np.testing.assert_array_equal(lag_sym[off_diagonal], lag_full[off_diagonal])


@pytest.mark.parametrize("min_score, expected_pairs", [(0.0, [[1, 0], [0, 1]]), (0.5, [[1, 0]])])
def test_compare_spikes_min_score_drops_poor_pairs(min_score, expected_pairs):
    """Ground truth 1 copies unit 0; ground truth 0 shares a third of unit 1's spikes."""
    a, b = _train(0), _train(1)
    part = np.sort(np.random.default_rng(1).choice(b, 30, replace=False))
    n_samples = max(a.max(), b.max()) + 1

    agreement = compare_spikes(
        _binary(n_samples, part, a), _binary(n_samples, a, b), fs=FS, min_score=min_score
    )

    np.testing.assert_array_equal(agreement.pairs, expected_pairs)


def test_compare_spikes_without_test_units_returns_no_pairs():
    agreement = compare_spikes(np.zeros((50, 2), int), np.zeros((50, 0), int), fs=FS)
    assert agreement.pairs.shape == (0, 2)
    assert not agreement.tp.any()


def test_deprecated_names_warn_and_forward():
    t = _train(0)
    ref = _binary(t.max() + 300, t)
    test = _binary(ref.shape[0], t + 5)
    with pytest.warns(FutureWarning, match="compare_paired_spikes"):
        paired = rate_of_agreement_paired(ref, test, fs=FS, tol_train_ms=40)
    with pytest.warns(FutureWarning, match="compare_spikes"):
        matched = rate_of_agreement(ref, test, fs=FS)
    expected = compare_paired_spikes(ref, test, fs=FS)
    for agreement in (paired, matched):
        np.testing.assert_array_equal(agreement.lag, expected.lag)
        np.testing.assert_array_equal(agreement.tp, expected.tp)


# --- find_duplicates -----------------------------------------------------------------


@pytest.mark.parametrize(
    "quality, kept_copy", [([-0.2, -0.1, -0.3, -0.1], 1), ([0.95, 0.9, 0.97, 0.9], 2)]
)
def test_find_duplicates_collapses_every_copy_to_the_best_unit(quality, kept_copy):
    clean, other = _train(0), _train(1)
    rng = np.random.default_rng(3)
    copies = [clean, clean + 3, np.sort(rng.choice(clean, 70, replace=False)) + 6]

    keep = find_duplicates(_binary_of([*copies, other]), np.array(quality), fs=FS)

    np.testing.assert_array_equal(keep, [kept_copy, 3])


def test_find_duplicates_ranks_nan_quality_last_and_ties_by_unit_order():
    a = _train(0)
    np.testing.assert_array_equal(find_duplicates(_binary_of([a, a]), [np.nan, 0.5], fs=FS), [1])
    np.testing.assert_array_equal(find_duplicates(_binary_of([a, a]), [0.5, 0.5], fs=FS), [0])


def test_find_duplicates_bridge_unit_does_not_merge_two_units():
    """A unit mixing two motor units is a duplicate of both, which stay separate."""
    a, c = _train(0), _train(1, start=237)
    bridge = np.sort(np.concatenate([a[::2], c[1::2]]))
    keep = find_duplicates(_binary_of([bridge, a, c]), np.array([-0.5, -0.1, -0.1]), fs=FS)
    np.testing.assert_array_equal(keep, [1, 2])


def test_find_duplicates_keeps_independent_fast_units():
    """Two unrelated ~30 Hz units: distinct at 0.5 ms, false duplicates at 10 ms."""
    a, b = _train(0, n=1800, isi=(55, 80)), _train(1, n=1800, isi=(55, 80))
    spikes, quality = _binary_of([a, b]), np.zeros(2)
    assert len(find_duplicates(spikes, quality, fs=FS)) == 2
    assert len(find_duplicates(spikes, quality, fs=FS, tol_spike_ms=10)) == 1


@pytest.mark.parametrize("metric, n_kept", [("roa", 2), ("match", 2), ("f1", 1)])
def test_find_duplicates_metric(metric, n_kept):
    """A unit with 40 % of another's spikes: RoA = match = 0.4, F1 = 0.57."""
    a = _train(0)
    part = np.sort(np.random.default_rng(1).choice(a, 40, replace=False))
    keep = find_duplicates(_binary_of([a, part]), np.zeros(2), fs=FS, threshold=0.5, metric=metric)
    assert len(keep) == n_kept
