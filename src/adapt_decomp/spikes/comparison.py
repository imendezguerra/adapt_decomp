"""Spike train comparison: agreement of paired units, one-to-one matching and duplicates."""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import _check_mu_format

AgreementMetric = Literal["roa", "f1", "precision", "sensitivity", "match"]
SymmetricMetric = Literal["roa", "f1", "match"]  # same score either way round


def _agreement_score(tp, n_ref, n_test, metric: AgreementMetric) -> np.ndarray:
    """Agreement score from common-spike counts, 0 where undefined.

    Args:
        tp (array-like): Common spikes (true positives).
        n_ref (array-like): Spikes in the reference trains.
        n_test (array-like): Spikes in the test trains.
        metric (AgreementMetric): "roa" TP/(TP+FN+FP), "f1" 2TP/(2TP+FN+FP),
            "precision" TP/(TP+FP), "sensitivity" TP/(TP+FN), or "match"
            TP/max(n_ref, n_test) (muniverse's duplicate score).

    Returns:
        np.ndarray: Scores in [0, 1], broadcast shape of the inputs.
    """
    tp, n_ref, n_test = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (tp, n_ref, n_test))
    )
    den = {
        "roa": n_ref + n_test - tp,
        "f1": (n_ref + n_test) / 2,
        "precision": n_test,
        "sensitivity": n_ref,
        "match": np.maximum(n_ref, n_test),
    }[metric]
    return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)


@dataclass
class SpikeAgreement:
    """Spike-by-spike agreement of paired reference and test units.

    Attributes:
        pairs (np.ndarray): (reference, test) unit index of each pair, shape (n_pairs, 2).
        lag (np.ndarray): Samples added to the test train to align it, shape (n_pairs,).
        tp (np.ndarray): Spikes in both trains, shape (n_pairs,).
        fn (np.ndarray): Reference spikes missing from the test train, shape (n_pairs,).
        fp (np.ndarray): Test spikes missing from the reference train, shape (n_pairs,).
    """

    pairs: np.ndarray
    lag: np.ndarray
    tp: np.ndarray
    fn: np.ndarray
    fp: np.ndarray

    def score(self, metric: AgreementMetric) -> np.ndarray:
        """Per-pair score.

        Args:
            metric (AgreementMetric): "roa", "f1", "precision", "sensitivity" or
                "match" (TP / max(n_ref, n_test)).

        Returns:
            np.ndarray: Score of each pair, shape (n_pairs,).
        """
        return _agreement_score(self.tp, self.tp + self.fn, self.tp + self.fp, metric)

    @property
    def roa(self) -> np.ndarray:
        """Rate of agreement, TP / (TP + FN + FP)."""
        return self.score("roa")

    @property
    def f1(self) -> np.ndarray:
        """F1 score, 2TP / (2TP + FN + FP) = 2 RoA / (1 + RoA)."""
        return self.score("f1")

    @property
    def precision(self) -> np.ndarray:
        """Precision, TP / (TP + FP): share of test spikes that are real."""
        return self.score("precision")

    @property
    def sensitivity(self) -> np.ndarray:
        """Sensitivity, TP / (TP + FN): share of reference spikes found."""
        return self.score("sensitivity")


def _to_samples(ms: float, fs: float) -> int:
    """Milliseconds to whole samples."""
    return round(ms / 1000 * fs)


def _spike_times(spike_trains: np.ndarray) -> List[np.ndarray]:
    """Sorted spike samples of each unit of a binary (samples, units) array."""
    return [np.flatnonzero(spike_trains[:, unit]) for unit in range(spike_trains.shape[1])]


def _best_lag(t_ref: np.ndarray, t_test: np.ndarray, max_lag: int, tol: int) -> int:
    """Lag in [-max_lag, max_lag] that best aligns t_test + lag with t_ref.

    Histograms the differences of all spike pairs closer than max_lag + tol and
    smooths them with a triangle of half-width tol + 1. Ties go to the smallest |lag|.

    Args:
        t_ref (np.ndarray): Sorted reference spike samples.
        t_test (np.ndarray): Sorted test spike samples.
        max_lag (int): Largest lag searched, in samples.
        tol (int): Spike coincidence tolerance, in samples.

    Returns:
        int: The lag, in samples.
    """
    w = max_lag + tol
    lo = np.searchsorted(t_test, t_ref - w, side="left")
    hi = np.searchsorted(t_test, t_ref + w, side="right")
    n_close = hi - lo
    # Ragged arange: range(lo[i], hi[i]) for every reference spike, concatenated
    offsets = np.cumsum(n_close) - n_close
    j = np.repeat(lo - offsets, n_close) + np.arange(n_close.sum())
    diffs = np.repeat(t_ref, n_close) - t_test[j]
    hist = np.bincount(diffs + w, minlength=2 * w + 1)
    weight = np.convolve(hist, np.bartlett(2 * tol + 3)[1:-1], mode="valid")
    lags = np.flatnonzero(weight == weight.max()) - max_lag
    return int(lags[np.argmin(np.abs(lags))])


def _count_common(t_ref: np.ndarray, t_test: np.ndarray, tol: int) -> int:
    """Spikes matched one-to-one within ±tol, by a two-pointer sweep of sorted trains."""
    t_ref, t_test = t_ref.tolist(), t_test.tolist()
    i = j = common = 0
    while i < len(t_ref) and j < len(t_test):
        diff = t_ref[i] - t_test[j]
        if abs(diff) <= tol:
            common, i, j = common + 1, i + 1, j + 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return common


def _compare(t_ref: np.ndarray, t_test: np.ndarray, max_lag: int, tol: int) -> Tuple[int, int]:
    """Lag and common spikes of one pair of trains."""
    lag = _best_lag(t_ref, t_test, max_lag, tol)
    return lag, _count_common(t_ref, t_test + lag, tol)


def _agreement_matrix(
    times_ref: List[np.ndarray],
    times_test: List[np.ndarray],
    max_lag: int,
    tol: int,
    symmetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lag and common spikes of every (reference, test) unit pair.

    Args:
        times_ref (List[np.ndarray]): Sorted spike samples of each reference unit.
        times_test (List[np.ndarray]): Sorted spike samples of each test unit.
        max_lag (int): Largest lag searched, in samples.
        tol (int): Spike coincidence tolerance, in samples.
        symmetric (bool): Both lists are the same set: compute the upper
            triangle only and mirror it (diagonal left at 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: lag and tp, each with shape (n_ref, n_test).
    """
    lag = np.zeros((len(times_ref), len(times_test)), dtype=int)
    tp = np.zeros_like(lag)
    for r, t_ref in enumerate(times_ref):
        for t in range(r + 1 if symmetric else 0, len(times_test)):
            lag[r, t], tp[r, t] = _compare(t_ref, times_test[t], max_lag, tol)
    if symmetric:
        lag, tp = lag - lag.T, tp + tp.T
    return lag, tp


def _agreement(
    pairs: np.ndarray,
    lag: np.ndarray,
    tp: np.ndarray,
    times_ref: List[np.ndarray],
    times_test: List[np.ndarray],
) -> SpikeAgreement:
    """SpikeAgreement of the given (reference, test) pairs."""
    n_ref = np.array([len(times_ref[r]) for r in pairs[:, 0]], dtype=int)
    n_test = np.array([len(times_test[t]) for t in pairs[:, 1]], dtype=int)
    return SpikeAgreement(pairs=pairs, lag=lag, tp=tp, fn=n_ref - tp, fp=n_test - tp)


def _prepare(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: float,
    tol_spike_ms: float,
    max_lag_ms: float,
    same_units: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Validate two binary arrays; return their spike times, max lag and tolerance in samples.

    Args:
        spike_trains_ref (np.ndarray): Binary reference trains, shape (samples, n_ref) or (samples,).
        spike_trains_test (np.ndarray): Binary test trains, shape (samples, n_test) or (samples,).
        fs (float): Sampling frequency in Hz.
        tol_spike_ms (float): Spike coincidence tolerance in ms.
        max_lag_ms (float): Largest lag searched between two trains in ms.
        same_units (bool): Whether both arrays must also have the same number of units.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], int, int]: Reference and test spike
        times, max lag and spike tolerance in samples.

    Raises:
        ValueError: If the shapes do not match.
    """
    ref, test = (_check_mu_format(np.asarray(x)) for x in (spike_trains_ref, spike_trains_test))
    if ref.shape[0] != test.shape[0] or (same_units and ref.shape != test.shape):
        raise ValueError(f"Shape mismatch between ref {ref.shape} and test {test.shape}.")
    return (
        _spike_times(ref),
        _spike_times(test),
        _to_samples(max_lag_ms, fs),
        _to_samples(tol_spike_ms, fs),
    )


def compare_paired_spikes(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    *,
    fs: float,
    tol_spike_ms: float = 1.0,
    max_lag_ms: float = 40.0,
) -> SpikeAgreement:
    """Agreement of column-paired spike trains: unit i of ref against unit i of test.

    Args:
        spike_trains_ref (np.ndarray): Binary reference trains, shape (samples, n)
            or (samples,).
        spike_trains_test (np.ndarray): Binary test trains, same shape.
        fs (float): Sampling frequency in Hz.
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 1.
        max_lag_ms (float): Largest lag searched between two trains in ms. Defaults to 40.

    Returns:
        SpikeAgreement: One pair per unit, in unit order.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    times_ref, times_test, max_lag, tol = _prepare(
        spike_trains_ref, spike_trains_test, fs, tol_spike_ms, max_lag_ms, same_units=True
    )
    lag_tp = np.array(
        [_compare(r, t, max_lag, tol) for r, t in zip(times_ref, times_test)], dtype=int
    ).reshape(-1, 2)
    units = np.arange(len(times_ref))
    return _agreement(
        np.column_stack([units, units]), lag_tp[:, 0], lag_tp[:, 1], times_ref, times_test
    )


def compare_spikes(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    *,
    fs: float,
    metric: AgreementMetric = "roa",
    min_score: float = 0.0,
    tol_spike_ms: float = 1.0,
    max_lag_ms: float = 40.0,
) -> SpikeAgreement:
    """Pair the units of two sets one-to-one and compute their agreement.

    Every (reference, test) pair is aligned and scored. The Hungarian algorithm
    then picks the one-to-one pairing with the highest total score, as in
    muniverse's evaluation.

    Args:
        spike_trains_ref (np.ndarray): Binary reference trains, shape (samples, n_ref).
        spike_trains_test (np.ndarray): Binary test trains, shape (samples, n_test).
        fs (float): Sampling frequency in Hz.
        metric (AgreementMetric): Score the pairing maximises. Defaults to "roa".
        min_score (float): Pairs scoring below it on metric are dropped. Defaults to 0
            (keep all).
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 1.
        max_lag_ms (float): Largest lag searched between two trains in ms. Defaults to 40.

    Returns:
        SpikeAgreement: At most min(n_ref, n_test) pairs, ordered by test unit.

    Raises:
        ValueError: If the two arrays have different numbers of samples.
    """
    times_ref, times_test, max_lag, tol = _prepare(
        spike_trains_ref, spike_trains_test, fs, tol_spike_ms, max_lag_ms, same_units=False
    )
    lag, tp = _agreement_matrix(times_ref, times_test, max_lag, tol)
    n_ref, n_test = (np.array([len(t) for t in times]) for times in (times_ref, times_test))
    score = _agreement_score(tp, n_ref[:, None], n_test[None, :], metric)
    rows, cols = linear_sum_assignment(score, maximize=True)
    keep = score[rows, cols] >= min_score
    order = np.argsort(cols[keep])
    rows, cols = rows[keep][order], cols[keep][order]
    return _agreement(
        np.column_stack([rows, cols]), lag[rows, cols], tp[rows, cols], times_ref, times_test
    )


def find_duplicates(
    spike_trains: np.ndarray,
    quality: np.ndarray,
    *,
    fs: float,
    metric: SymmetricMetric = "roa",
    threshold: float = 0.3,
    tol_spike_ms: float = 0.5,
    max_lag_ms: float = 40.0,
) -> np.ndarray:
    """Units kept after greedy duplicate removal, as in muniverse.

    The best unit is kept and every unit scoring at least threshold with it is
    dropped; then the next best survivor, and so on. Kept units are pairwise below
    threshold, so any number of copies of one motor unit collapse to one.

    Args:
        spike_trains (np.ndarray): Binary spike trains, shape (samples, n_units).
        quality (np.ndarray): Per-unit quality, higher is better (e.g. SIL, or
            -CoV-ISI), shape (n_units,). NaN ranks last; ties keep unit order.
        fs (float): Sampling frequency in Hz.
        metric (SymmetricMetric): Score compared with threshold. Defaults to "roa".
        threshold (float): Score at or above which two units are duplicates.
            Defaults to 0.3.
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 0.5.
        max_lag_ms (float): Largest lag searched between two units in ms. Defaults to 40.

    Returns:
        np.ndarray: Sorted indices of the kept units.
    """
    times, _, max_lag, tol = _prepare(
        spike_trains, spike_trains, fs, tol_spike_ms, max_lag_ms, same_units=True
    )
    _, tp = _agreement_matrix(times, times, max_lag, tol, symmetric=True)
    n_spikes = np.array([len(t) for t in times])
    duplicate = _agreement_score(tp, n_spikes[:, None], n_spikes[None, :], metric) >= threshold
    np.fill_diagonal(duplicate, False)

    quality = np.asarray(quality, dtype=float)
    alive = np.ones(len(times), dtype=bool)
    for unit in np.argsort(np.where(np.isnan(quality), np.inf, -quality), kind="stable"):
        if alive[unit]:
            alive[duplicate[unit] & alive] = False
    return np.flatnonzero(alive)
