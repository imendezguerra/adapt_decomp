# Evaluate against ground truth

With simulated recordings, `gt_matched_indices` pairs each unit with the simulated motor unit
it tracks, so the rate of agreement (RoA) can be computed unit by unit.

## Rate of agreement

```python
--8<-- "workflow.py:evaluate"
```

The adapted results start where the calibration window ends, so the ground truth is sliced the
same way: row `i` of both is sample `CAL_END + i` of the recording.

## Over any window

```python
--8<-- "workflow.py:evaluate-window"
```

## RoA, F1, precision and sensitivity

`compare_paired_spikes()` compares unit `i` of the ground truth with unit `i` of the
decomposition and returns a `SpikeAgreement`: per unit, the common spikes (TP),
the ground-truth spikes missed (FN), the extra spikes (FP) and the lag applied, from which every
metric follows:

```python
--8<-- "workflow.py:metrics"
```

| Metric | Formula | Reads as |
|---|---|---|
| `roa` | TP / (TP + FN + FP) | Rate of agreement (Jaccard index) |
| `f1` | 2 TP / (2 TP + FN + FP) | F1 score (Dice); F1 = 2 RoA / (1 + RoA), so both rank units alike |
| `precision` | TP / (TP + FP) | Share of the detected spikes that are real |
| `sensitivity` | TP / (TP + FN) | Share of the real spikes that are detected |

`agreement.score("match")` gives TP / max(n_ref, n_test), MUniverse's duplicate score.

## How two spike trains are compared

1. **Align:** the test train may lag the reference by up to `max_lag_ms` (40 ms). The lag is
   found from the differences between every pair of reference and test spikes closer than
   that, never from the samples in between: their histogram, smoothed with a triangle as wide
   as the spike tolerance, peaks at the lag where most spikes coincide.
2. **Match:** after shifting the test train by that lag, spikes are matched one to one within
   `tol_spike_ms`, in time order. Matched spikes are TP, unmatched reference spikes FN and
   unmatched test spikes FP.

Between two sets of units (`compare_spikes()`, used by `select_supervised()`), every pair
is compared this way and the Hungarian algorithm picks the one-to-one pairing with the highest
total RoA (or `metric`); `min_score` drops the pairs below it. `find_duplicates()` compares the
units of one set the same way and keeps the best of each group of duplicates, as CBSS does.

## Without ground truth: silhouette

The silhouette (SIL) of each unit's source measures how well its spikes stand out from the
background; values of 0.9 and above usually mean a reliable unit:

```python
--8<-- "workflow.py:sil"
```
