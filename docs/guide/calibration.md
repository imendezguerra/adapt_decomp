# Calibration

Calibration finds the motor units in a short window of the recording, typically a few seconds
of steady contraction. It runs once, offline, and its result is the starting point of the
[adaptation](adaptation.md).

```python
from adapt_decomp import CBSS, CBSSConfig

cbss_config = CBSSConfig(fs=2048, ext_fact=10)
calibration = CBSS(cbss_config).decompose(emg_window, timestamps)  # emg_window: (samples, channels)
```

## How it works

```mermaid
flowchart LR
    A["Preprocess\n(band-pass, notch)"] --> B["Extend\n(ext_fact delayed copies)"]
    B --> P["PCA\n(optional, n_components)"]
    P --> C["Whiten"]
    C --> D["ICA search\n(search_iter starts,\nica_iter iterations each,\nrefinement)"]
    D --> E["Keep reliable units\n(sil_th, min_spikes)"]
    E --> F["Remove duplicates\n(roa_th)"]
    F --> G["Unit properties\n(SIL, CoV-ISI, PNR, DR, MUAPs)"]
    G --> H["Optional selection"]
    H --> I["CBSSResult"]
```

1. **Preprocess:** the EMG is band-pass filtered and the powerline frequency notched out.
2. **Extend:** each channel is stacked with `ext_fact` delayed copies of itself, which turns
   the convolutive mixture of motor unit action potentials into an instantaneous one.
3. **Reduce (optional):** with `n_components` set, PCA keeps only that many components of the
   extended channels, which speeds up the following steps. The adaptation reuses the fitted
   PCA as it is.
4. **Whiten:** the extended channels are decorrelated and scaled to unit variance.
5. **Search:** from `search_iter` starting points, fixed-point ICA looks for a separation
   vector whose source shows the spikes of one unit, for up to `ica_iter` iterations each (it
   stops earlier once the vector converges). A refinement loop then re-estimates the vector
   from the unit's own spikes.
6. **Keep reliable units:** a unit is kept if its silhouette (SIL, how well its spikes stand
   out from the baseline) reaches `sil_th` and it has at least `min_spikes` spikes. Two units
   whose spike trains agree above `roa_th` are duplicates, and one is dropped.
7. **Describe the units:** the silhouette, the coefficient of variation of the inter-spike
   intervals (CoV-ISI), the pulse-to-noise ratio (PNR), the discharge rate and the MUAPs.

## Key parameters

| Field | Default | Controls | Change it |
|---|---|---|---|
| `fs` | 2048 | Sampling frequency, in Hz | Always, to your recording's |
| `lowcut` | 20 | High-pass cutoff, in Hz | For other EMG types or sampling rates |
| `highcut` | 500 | Low-pass cutoff, in Hz | For other EMG types or sampling rates; keep it below half of `fs` |
| `powerline_freq` | 50 | Notch frequency, in Hz | To 60 where the mains is 60 Hz |
| `ch_mask` | None | Boolean mask of the channels to keep; None keeps them all | When some channels are noisy |
| `replace_bad_channels` | False | False drops the masked-out channels; True interpolates them from their neighbours on the grid | To keep the grid complete |
| `ch_map` | None | Electrode grid layout, holding raw channel indices | To interpolate channels and to compute MUAPs |
| `ext_fact` | 10 | Delayed copies per channel | Larger to capture longer MUAPs, at a higher cost |
| `n_components` | None | PCA components kept before whitening | To speed up very large extensions |
| `search_iter` | 100 | Starting points of the ICA search, each yielding at most one unit | Larger to find more units, at a higher cost |
| `ica_iter` | 100 | Maximum fixed-point iterations per starting point; the search stops earlier once it converges | Larger if units fail to converge, smaller for speed |
| `sil_th` | 0.9 | Minimum silhouette of a unit | Lower to keep more, less reliable units |
| `roa_th` | 0.3 | Agreement above which two units are duplicates | |
| `selection` | None | Unit selection at the end of `decompose()`: `"unsupervised"`, `"supervised"`, or None for none | See [below](#keeping-the-reliable-units) |
| `selection_kwargs` | None | The thresholds, or the ground truth, the selection uses | See [below](#keeping-the-reliable-units) |
| `random_seed` | 1909 | ICA starting order | None for a different decomposition on every run |
| `device` | "cpu" | Compute device | "cuda" or "mps" for speed |

The preprocessing, channel and extension settings are reused by the adaptation, which takes
them from this config. The [API reference](../reference/cbss.md#adapt_decomp.cbss.CBSSConfig)
lists every field.

## Keeping the reliable units

`decompose()` returns every unit that passed `sil_th`, `min_spikes` and duplicate removal.
Two filters keep a stricter subset, each returning a new `CBSSResult`:

```python
# On real recordings: keep the units passing quality thresholds
calibration = calibration.select_unsupervised(sil_th=0.9, cov_th=0.3, pnr_th=30)

# On simulations: keep the units that match a ground-truth unit
calibration = calibration.select_supervised(gt_spikes, roa_th=0.9, fs=2048)
```

`select_unsupervised` takes any of `sil_th`, `pnr_th` (dB), `cov_th`, `dr_min` and `dr_max`
(Hz). `select_supervised` records the ground-truth unit each kept unit tracks in
`gt_matched_indices`, and its rate of agreement in `roa`. To apply either inside
`decompose()`, set `CBSSConfig.selection` to `"unsupervised"` or `"supervised"` and pass the
arguments in `selection_kwargs`.

## What you get back

A `CBSSResult`, with `n_mu` units:

| Field | Shape | Holds |
|---|---|---|
| `sources` | (samples, n_mu) | Each unit's source |
| `spikes` | (samples, n_mu) | Binary spike trains |
| `sil` | (n_mu,) | Silhouette of each unit |
| `cov_isi` | (n_mu,) | Coefficient of variation of each unit's inter-spike intervals |
| `pnr` | (n_mu,) | Pulse-to-noise ratio, in dB, if `compute_properties` |
| `dr` | (n_mu,) | Mean discharge rate, in Hz, if `compute_properties` |
| `muaps` | (n_mu, rows, cols, window) | Spike-triggered average MUAPs, if `compute_properties` |
| `sep_vectors` | (dim, n_mu) | Separation vectors, part of the model the adaptation starts from |
| `whitening` | (dim, dim) | Whitening matrix, part of the model the adaptation starts from |
| `gt_matched_indices` | (n_mu,) | The ground-truth unit each unit tracks, after `select_supervised` |
| `roa` | (n_mu,) | Each unit's rate of agreement with that unit over the calibration window, after `select_supervised` |

Save it with `calibration.save(path)` and reload it with `CBSSResult.load(path)`. Save the
config next to it (`cbss_config.to_yaml(path)`): building the adaptation needs both.

## Reusing a calibration

- `CBSS(cbss_config).apply(new_emg, calibration, timestamps)` applies a calibration's model to
  other EMG without searching again, for example to score it on another recording.
- A calibration from another tool can be used by building a `CBSSResult` from its fields
  (`sources`, `spikes`, `sep_vectors`, `whitening`, `extension_mean`, the centroids, `sil`,
  `cov_isi`, `ext_fact`, and the calibration `emg` and `timestamps`), or by building the
  adaptation directly (see [Building the adaptation](overview.md#building-the-adaptation)).

## Reproducibility

With the same `random_seed`, EMG and config, `decompose()` gives the same units on the CPU.
On a GPU the arithmetic is not bit-for-bit deterministic, which can change which borderline
units are kept. Calibrate on the CPU when the calibration must be reproduced exactly.

## Recipes

- [Calibrate a recording](../how-to/calibrate-a-recording.md)
- [Evaluate against ground truth](../how-to/evaluate-against-ground-truth.md)
