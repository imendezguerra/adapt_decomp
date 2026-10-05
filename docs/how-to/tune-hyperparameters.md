# Tune hyperparameters

`optimize_adapt_decomp` searches `wh_learning_rate`, `sv_learning_rate` and
`centroid_momentum` (by default) over a pool of calibrated recordings: every trial adapts every
recording with one suggested setting. The [optimisation guide](../guide/optimisation.md) explains
each option in depth.

## Build a pool

```python
--8<-- "tune.py:pool"
```

`start` (and `stop`) select the samples every trial adapts and scores, the EMG and the ground
truth alike: here from the end of the 5 s calibration window, so the search scores the same part
of the recording as [adapting from the calibration end](adapt-from-calibration-end.md).
`source_fifo_from_calib` starts each trial's source FIFO from the calibration's last sources.

For pools too large to keep in memory, `load_pooled_cbss_disk` takes the same `datasets` list
and loads each recording per trial instead.

## One objective

`sv_loss` (how far the units' contrast drifts from calibration) needs no ground truth:

```python
--8<-- "tune.py:single"
```

## Two objectives: a Pareto front

Score `wh_loss` and `sv_loss` jointly and pick one member of the front with `selection`:

```python
--8<-- "tune.py:pareto"
```

## Without ground truth

On real recordings, adapt and score only the regularly firing units during the search; the
chosen config is then applied to every unit:

```python
--8<-- "tune.py:no-ground-truth"
```

When the calibrations already keep only ground-truth-matched units, leave `unit_selection` at
its default (`None`).

## Keep the result

```python
--8<-- "tune.py:save"
```

`sv_loss_reduction` on the base config sets whether `sv_loss` is averaged (`"mean"`, the
default) or summed over units. `n_jobs` and `n_cores` set how many trials run together and on
how many cores; see [Speed and resources](../guide/optimisation.md#speed-and-resources).
The [FDSI benchmark results](../benchmarks/fdsi/report.ipynb) compare these choices.
