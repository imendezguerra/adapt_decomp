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

## Without ground truth: silhouette

The silhouette (SIL) of each unit's source measures how well its spikes stand out from the
background; values of 0.9 and above usually mean a reliable unit:

```python
--8<-- "workflow.py:sil"
```
