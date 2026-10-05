# Quickstart

This page runs the whole pipeline on one recording, in a few minutes on a CPU: calibrate on
the first 5 s, keep the units that match the ground truth, adapt over the rest of the
recording, and compare with no adaptation.

The recording is synthetic, from the [FDSI benchmark](../benchmarks/fdsi.md): 90 s of HD-EMG
(100 channels, 2048 Hz) while the wrist angle ramps up and down, with the spike trains of its
64 simulated motor units. Download it (70 MB), then run the code from the directory that holds
`data/`:

```sh
adapt-decomp-data get fdsi_example-data
```

The code on this page is
[`docs/snippets/workflow.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/docs/snippets/workflow.py),
which is tested, so it runs as shown. The outputs below come from running it.

## 1. Load the recording

The EMG is an array of shape (samples, channels), and the ground truth one binary spike train
per simulated motor unit.

```python
--8<-- "workflow.py:load"
```

```text
(184320, 100) (184320, 64)
```

## 2. Calibrate

[CBSS](../guide/calibration.md) decomposes the first 5 s into motor units. `random_seed` makes
the result reproducible.

```python
--8<-- "workflow.py:calibrate"
```

```text
48 units found
```

## 3. Keep the reliable units

On a simulation, keep the units that match a simulated motor unit, with a rate of agreement
(RoA) of at least 90 % over the calibration window. On a real recording, filter on quality
metrics instead, with `select_unsupervised` (see
[Calibrate a recording](../how-to/calibrate-a-recording.md)).

```python
--8<-- "workflow.py:select-supervised"
```

```text
[28 32 38 37 35 24 34 36 33 29 30 31  7 22]
[1.  1.  1.  1.  1.  0.986  0.982  1.  1.  0.984  1.  0.983  0.927  0.947]
```

## 4. Save the calibration

Save the config next to the result: building the adaptation needs both.

```python
--8<-- "workflow.py:save"
```

## 5. Adapt

`from_calibration` builds the [adaptation](../guide/adaptation.md) from the calibration, with
the `muniverse` preset, tuned on this dataset. `process_data` adapts over the samples after the
calibration window, 100 ms at a time; `source_fifo_from_calib` hands it the calibration's last
sources, which come right before them.

```python
--8<-- "workflow.py:adapt"
```

```text
torch.Size([174080, 14]) 17.1 ms per 100 ms batch
```

## 6. Compare with no adaptation

The `fixed` preset switches every adaptation off: the calibration is applied as it is.

```python
--8<-- "workflow.py:baseline"
```

## 7. Score

Each unit's spikes are compared with those of the simulated motor unit it tracks, over the
adapted samples after the calibration window.

```python
--8<-- "workflow.py:evaluate"
```

```text
no adaptation: mean RoA after calibration 33.8 %
adapted: mean RoA after calibration 52.0 %
```

As the wrist moves, the units of the fixed calibration drift away from the motor units they
were matched to; the adaptation tracks them more closely. The [benchmark results](../benchmarks/fdsi/report.ipynb) show the same comparison over 100
recordings, and with tuned hyperparameters.

## Next

- The [tutorial](../notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) goes
  through the adaptation in depth: losses, timings, online processing and how the model changes.
- The [user guide](../guide/overview.md) explains each stage and its parameters.
- The [how-to guides](../how-to/index.md) cover tuning the hyperparameters, evaluating,
  plotting and processing online.
