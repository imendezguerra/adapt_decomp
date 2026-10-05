# Calibrate a recording

Run CBSS on a short calibration window (here the first 5 s) to find the motor units, then
keep the reliable ones.

## Load the recording

```python
--8<-- "workflow.py:load"
```

## Decompose the calibration window

```python
--8<-- "workflow.py:calibrate"
```

`random_seed` makes the search for units reproducible on the CPU. Every `CBSSConfig` field
is listed in the [API reference](../reference/cbss.md#adapt_decomp.cbss.CBSSConfig).

## Keep the reliable units

Without ground truth (real recordings), filter on quality metrics:

```python
--8<-- "workflow.py:select-unsupervised"
```

With ground truth (simulations), keep the units that match a simulated motor unit. Their
`gt_matched_indices` record which one each unit tracks:

```python
--8<-- "workflow.py:select-supervised"
```

Both return a new `CBSSResult` and leave the original as it is. To have `decompose()` apply
either filter itself, set `CBSSConfig.selection` (see
[Calibration](../calibration.md#unit-selection)).

## Save it

```python
--8<-- "workflow.py:save"
```

Save the config next to the result: rebuilding the model for adaptation needs both.
