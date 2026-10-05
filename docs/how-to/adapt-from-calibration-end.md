# Adapt from the calibration end

Build the online model from a calibration and track its units over the rest of the
recording. `process_data` adapts over exactly the EMG it is given, so pass the samples after the
calibration window; `source_fifo_from_calib` starts the source FIFO from the calibration's last
sources, which come right before them. The adapted part is then never scored on the window the
model was calibrated on.

```python
--8<-- "workflow.py:adapt"
```

The `muniverse` preset holds hyperparameters tuned on this dataset (see the
[presets](../guide/adaptation.md#presets)); see
[Tune hyperparameters](tune-hyperparameters.md) to choose your own.

To score the whole recording, prepend the calibration's own output over its window (here the
first 5 s; for a window starting later, `CBSS(cbss_config).apply(emg[:b], calibration)` gives
CBSS's output over every sample before its end `b`):

```python
--8<-- "workflow.py:prepend"
```

`AdaptDecomp.calibrate_and_process(emg, timestamps, calib_indices, ...)` does all of this in one
call when it runs the calibration too: it adapts from the end of the window and fills the samples
up to it with CBSS's output (see [Where to start adapting](../guide/adaptation.md#where-to-start-adapting)).

## Compare with no adaptation

The baseline is the same calibration with every adaptation switched off
(the `fixed` preset):

```python
--8<-- "workflow.py:baseline"
```

[Evaluate against ground truth](evaluate-against-ground-truth.md) compares the two.
