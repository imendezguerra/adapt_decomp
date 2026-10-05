# Adaptation

`AdaptDecomp` runs a [calibration](calibration.md) over a recording in batches, as it would in
real time. In each batch it:

1. whitens the batch with the current whitening matrix;
2. estimates each unit's source with its separation vector;
3. detects the unit's spikes in its source;
4. updates the model so it keeps matching the calibration.

It never searches for new units: it tracks the ones it was given.

```python
from adapt_decomp import AdaptDecomp
from adapt_decomp.adaptation import AdaptConfig

adapt_config = AdaptConfig.from_preset("muniverse")
adapt_config.source_fifo_from_calib = True  # the EMG starts where calibration ends

adapter = AdaptDecomp.from_calibration(
    calibration=calibration, cbss_config=cbss_config, adapt_config=adapt_config
)
result = adapter.process_data(emg[5 * 2048 :])  # after a 5 s calibration window
```

## How it adapts

Three parts of the model adapt, each switched on or off by its own flag:

| Part | Flag | Update |
|---|---|---|
| Whitening | `adapt_wh` | After whitening, the batch's covariance should be the identity, as it was in calibration. A step of size `wh_learning_rate` reduces the difference. |
| Separation vectors | `adapt_sv` | Each unit's source should keep, at its spikes, the contrast it had in calibration. A step of size `sv_learning_rate` reduces the difference. |
| Spike detection | `adapt_sd` | Spikes are told from the baseline by two centroids, the typical amplitude of each. They move towards the batch's values, with momentum `centroid_momentum`. |

The two differences are the adaptation's losses: `wh_loss`, a KL divergence, and `sv_loss`,
per unit. Both are normalised by their spread during calibration, so values are comparable
across units and recordings. They measure how far the model drifted from the calibration, and
are what the [hyperparameter search](optimisation.md) minimises. The adaptation itself doesn't
need them: `compute_loss=False` skips them, for speed.

## Where to start adapting

`process_data` adapts over exactly the EMG it is given, from its first sample. Where it starts
is chosen by slicing that EMG. For a recording calibrated on its samples `[a, b)`:

| Call | Output over `[0, b)` | Adapts |
|---|---|---|
| `process_data(emg)` | Adapted, like every other sample | From the first sample of `emg` |
| `process_data(emg[b:])`, with `source_fifo_from_calib=True` | Not returned: prepend the calibration's output to score the whole recording | From `b` |
| `calibrate_and_process(emg, ..., calib_indices)` | CBSS's: the calibration's own output, or `CBSS.apply` over `emg[:b]` when `a > 0` | From `b` (`adapt_from="calib_end"`, the default) |
| `calibrate_and_process(..., adapt_from="emg_start")` | Adapted | From the first sample of `emg` |

Starting at `b` is the usual choice: the model starts from the samples it was just calibrated
on, and the adapted output is never scored on the calibration window itself.

## Presets

`AdaptConfig.from_preset(name)` loads a config shipped with the package:

| Preset | Tuned on | `wh_learning_rate` | `sv_learning_rate` | `centroid_momentum` |
|---|---|---|---|---|
| `muniverse` | The [FDSI benchmark](../benchmarks/fdsi.md)'s search pool (simulated, 100 channels) | 4.7e-4 | 1.0e-3 | 0.95 |
| `neuromotion` | The NeuroMotion simulation of the [tutorial](../notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) (320 channels) | 7e-3 | 3e-3 | 0.8 |
| `wrist` | The paper's experimental recordings, electrodes on the wrist | 1e-3 | 5e-4 | 0.8 |
| `forearm` | The paper's experimental recordings, electrodes on the forearm | 2e-3 | 5e-4 | 0.8 |
| `fixed` | No adaptation, the baseline | 0 | 0 | 1 |

For other data, start from the closest preset and [tune](optimisation.md) the learning rates
and the momentum.

## Key parameters

| Field | Default | Controls | Change it |
|---|---|---|---|
| `wh_learning_rate` | 5e-3 | Step size of the whitening update | Tune it |
| `sv_learning_rate` | 1e-3 | Step size of the separation-vector update | Tune it |
| `centroid_momentum` | 0.95 | How slowly the spike detection follows each batch, from 0 to 1 | Tune it |
| `adapt_wh` | True | Whether the whitening adapts | False to freeze it; all three flags False is no adaptation (the `fixed` preset) |
| `adapt_sv` | True | Whether the separation vectors adapt | False to freeze them |
| `adapt_sd` | True | Whether the spike detection centroids adapt | False to freeze them |
| `batch_ms` | 100 | Batch duration, in ms | Shorter for lower latency; the tuned learning rates assume their batch duration |
| `source_fifo_from_calib` | False | Whether the first batch sees the calibration's last sources, to detect spikes at its start | True when the EMG starts where the calibration window ends |
| `compute_loss` | True | Whether to compute the losses | False in real time |
| `device` | None | Compute device; None picks CUDA, then MPS, then the CPU | |
| `save_params` | False | Whether to write the model of every batch to an HDF5 file (`save_path`) | To study how the model changes |
| `lr_mode` | "fixed" | Update rule: a plain step of the learning rate, or `"rel_error"`, a step scaled by the normalised error | Keep the default: the presets were tuned with it |
| `wh_mode` | "kl_to_identity" | Whitening error: the divergence of the whitened covariance from the identity, or `"kl_to_cal"`, from the calibration's | Keep the default: the presets were tuned with it |
| `contrast_scope` | "spike_based" | Samples the separation-vector update uses: the detected spikes, or `"batch_based"`, the whole batch | Keep the default: the presets were tuned with it |

The alternatives of the last three exist for ablation studies, and need their own tuning. The
[API reference](../reference/adaptation.md#adapt_decomp.adaptation.AdaptConfig) lists every
field.

**Settings shared with calibration.** The preprocessing, channel and extension settings, and
`spike_det_exp`, must be the calibration's. `from_calibration` copies them from `cbss_config`
and warns about any it changed. An `ext_fact` that disagrees with the calibration's raises an
error, since it means the wrong calibration was passed.

## Online processing

`process_data(emg, processing_mode="online")` processes the recording as raw batches,
filtering, centring and extending each one as it arrives, which checks the real-time path. For a
live feed, call `process_batch` yourself, one batch of `adapt_config.batch_size` raw samples at
a time:

```python
adapter = AdaptDecomp.from_calibration(calibration, cbss_config, adapt_config)
for batch_idx, batch in enumerate(live_feed):  # torch.Tensor, (batch_size, channels)
    spikes, sources = adapter.process_batch(batch, batch_idx)
```

[Process online](../how-to/process-online.md) runs this loop on a recording, and checks it
gives the same output as `processing_mode="online"`.

## What you get back

An `AdaptationResult`, with `M` units and one entry per batch for the per-batch fields:

| Field | Shape | Holds |
|---|---|---|
| `spikes` | (samples, M) | Binary spike trains |
| `sources` | (samples, M) | Each unit's source |
| `total_time_ms` | (batches,) | Time per batch, in ms |
| `preprocess_time_ms` | (batches,) | Preprocessing time per batch, in ms; zero offline, where the recording is preprocessed upfront |
| `wh_time_ms` | (batches,) | Whitening step time per batch, in ms |
| `sv_time_ms` | (batches,) | Separation-vector step time per batch, in ms |
| `sd_time_ms` | (batches,) | Spike detection step time per batch, in ms |
| `wh_loss` | (batches,) | Whitening loss per batch, if `compute_loss` |
| `sv_loss` | (batches, M) | Separation-vector loss per batch and unit, if `compute_loss` |
| `wh_loss_total` | scalar | The whitening loss over the whole run, as the search scores it |
| `sv_loss_total` | scalar | The separation-vector loss over the whole run, as the search scores it |
| `total_loss` | scalar | `wh_loss_total + sv_loss_total` |
| `gt_matched_indices` | (M,) | The ground-truth unit each unit tracks, carried over from the calibration |

Save it with `result.save(path)` and reload it with `AdaptationResult.load(path)`.

## Reproducibility

The adaptation has no randomness: with the same calibration, EMG and config it gives the same
output on the CPU. The batch duration is part of that: each update starts from the state the
previous batch left, so a different `batch_ms` is a different sequence of updates. A live loop
reproduces only if it is fed the same chunks. On a GPU, small arithmetic differences can
accumulate over a long recording.

## Recipes

- [Adapt from the calibration end](../how-to/adapt-from-calibration-end.md)
- [Process online](../how-to/process-online.md)
- [Evaluate against ground truth](../how-to/evaluate-against-ground-truth.md)
- [Plot results](../how-to/plot-results.md)
