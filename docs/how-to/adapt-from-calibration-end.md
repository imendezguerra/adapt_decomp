# Adapt from the calibration end

Build the online model from a calibration and track its units over the rest of the
recording. `process_from_calib_end` keeps CBSS's own output over the calibration window and
adapts from its last sample, so the adapted part is never scored on the window it was
calibrated on.

```python
--8<-- "workflow.py:adapt"
```

`default_muniverse.yaml` holds tuned hyperparameters; see
[Tune hyperparameters](tune-hyperparameters.md) to choose your own. When the calibration
window isn't at the start of the recording, pass `backward=True` to also adapt backwards from
its first sample.

## Compare with no adaptation

The baseline is the same calibration with every adaptation switched off
(`configs/adapt_configs/default_fixed.yaml`):

```python
--8<-- "workflow.py:baseline"
```

[Evaluate against ground truth](evaluate-against-ground-truth.md) compares the two.
