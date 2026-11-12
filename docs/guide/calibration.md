# Calibration

Calibration finds the motor units in a short window of the recording, typically a few seconds
of steady contraction. It runs once, offline, and its result is the starting point of the
[adaptation](adaptation.md).

```python
from adapt_decomp import CBSS, CBSSConfig

cbss_config = CBSSConfig(fs=2048, ext_fact=10)
calibration = CBSS(cbss_config).decompose(emg_window, timestamps)  # emg_window: (samples, channels)
```

## Methods

Two methods find the units, both on the same pipeline (`Decomposition`): preprocessing,
extension, optional PCA and whitening are shared, and so are duplicate removal, unit
properties, selection and `apply()`. Only the search for units differs.

| Method | Config | Search | Needs |
|---|---|---|---|
| `CBSS` | `CBSSConfig` | Fixed-point ICA with deflation and a refinement loop | Nothing extra |
| `SCD` | `SCDConfig` | Swarm-Contrastive Decomposition: swarm-optimised ICA with peel-off ([scd](https://github.com/AgneGris/swarm-contrastive-decomposition)) | The `scd` extra: `pip install "adapt_decomp[scd]"` |

The config's type picks the method: `Decomposition.from_config(config)` builds `CBSS` or
`SCD`, and `AdaptDecomp.calibrate_and_process(..., calib_config=config)` and the command line
(`--calib_config`, whose YAML records the `method`) do the same. Both return a
`DecompositionResult` (`CBSSResult` is its former name), so the adaptation takes either.
The steps below describe CBSS; [SCD](#scd) lists what it does differently.

## How it works

```mermaid
flowchart LR
    A["Preprocess\n(band-pass, notch)"] --> B["Extend\n(ext_fact delayed copies)"]
    B --> P["PCA\n(optional, n_components)"]
    P --> C["Whiten"]
    C --> D["ICA search\n(search_iter starts,\nica_iter iterations each,\nrefinement)"]
    D --> E["Keep reliable units\n(sil_th, min_spikes)"]
    E --> F["Remove duplicates\n(roa_th, duplicate_*)"]
    F --> G["Unit properties\n(SIL, CoV-ISI, PNR, DR, MUAPs)"]
    G --> H["Optional selection"]
    H --> I["DecompositionResult"]
```

1. **Preprocess:** the EMG is band-pass filtered and the powerline frequency notched out.
2. **Extend:** each channel is stacked with `ext_fact` delayed copies of itself, which turns
   the convolutive mixture of motor unit action potentials into an instantaneous one.
3. **Reduce (optional):** with `n_components` set, PCA keeps only that many components of the
   extended channels, which speeds up the following steps. The adaptation reuses the fitted
   PCA as it is.
4. **Whiten:** the extended channels are decorrelated and scaled to unit variance. Their
   covariance is badly conditioned, so it is eigendecomposed in float64; the rest of CBSS
   runs in float32.
5. **Search:** from `search_iter` starting points, fixed-point ICA looks for a separation
   vector whose source shows the spikes of one unit, for up to `ica_iter` iterations each (it
   stops earlier once the vector converges). A refinement loop then re-estimates the vector
   from the unit's own spikes.
6. **Keep reliable units:** a unit is kept if its silhouette (SIL, how well its spikes stand
   out from the baseline) reaches `sil_th` and it has at least `min_spikes` spikes. Two units
   whose spike trains agree at or above `roa_th` (on `duplicate_metric`) are duplicates.
   Removal is greedy: the best unit (lowest CoV-ISI or highest SIL, `duplicate_keep_by`) is
   kept and all its duplicates dropped, then the next best remaining unit, and so on. Kept
   units are pairwise below `roa_th`, so any number of copies of one motor unit reduce to one.
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
| `roa_th` | 0.3 | Agreement (on `duplicate_metric`) at or above which two units are duplicates | |
| `duplicate_metric` | "roa" | Agreement score for duplicates: `"roa"`, `"f1"`, or `"match"` (common spikes over the larger spike count, as in MUniverse) | To match an evaluation's duplicate criterion |
| `duplicate_keep_by` | "cov_isi" | Unit kept from each group of duplicates: lowest CoV-ISI or highest SIL (`"sil"`) | |
| `duplicate_tol_spike_ms` | 0.5 | Spike coincidence tolerance for duplicates, in ms | |
| `duplicate_tol_train_ms` | 40 | Largest lag searched between two units, in ms | |
| `selection` | None | Unit selection at the end of `decompose()`: `"unsupervised"`, `"supervised"`, or None for none | See [below](#keeping-the-reliable-units) |
| `selection_kwargs` | None | The thresholds, or the ground truth, the selection uses | See [below](#keeping-the-reliable-units) |
| `random_seed` | 1909 | ICA starting order | None for a different decomposition on every run |
| `device` | "cpu" | Compute device | "cuda" or "mps" for speed |

The preprocessing, channel and extension settings are reused by the adaptation, which takes
them from this config. The [API reference](../reference/cbss.md#adapt_decomp.cbss.CBSSConfig)
lists every field.

## Keeping the reliable units

The decomposition is blind: CBSS separates sources without knowing which motor units are
there, so not every source it returns is a reliable unit, and not every motor unit is found.
Some sources mix two units, follow noise or fire irregularly. `decompose()` returns every unit
that passed `sil_th`, `min_spikes` and duplicate removal, and the
[adaptation](adaptation.md) then tracks every unit it is given: it never drops one. Select the
reliable units after calibrating, with one of two filters, each returning a new
`DecompositionResult`:

- **Unsupervised**, on real recordings: keep the units whose quality metrics pass thresholds
  (silhouette, pulse-to-noise ratio, CoV-ISI, discharge rate).
- **Supervised**, on simulations or recordings with a reference decomposition: keep the units
  that match a ground-truth unit.

```python
# On real recordings: keep the units passing quality thresholds
calibration = calibration.select_unsupervised(sil_th=0.9, cov_th=0.3, pnr_th=30)

# On simulations: keep the units that match a ground-truth unit
calibration = calibration.select_supervised(gt_spikes, roa_th=0.9, fs=2048)
```

`select_unsupervised` takes any of `sil_th`, `pnr_th` (dB), `cov_th`, `dr_min`, `dr_max`
(Hz) and, for SCD, `native_roa_th` (CBSS units, which have no `native_roa`, pass it).
`select_supervised` records the ground-truth unit each kept unit tracks in
`gt_matched_indices`, and its rate of agreement in `roa`. To apply either inside
`decompose()`, set the config's `selection` to `"unsupervised"` or `"supervised"` and pass the
arguments in `selection_kwargs` (`AdaptDecomp.calibrate_and_process` uses it the same way).

Reliable units matter most in a [hyperparameter search](optimisation.md#unit-selection): its
losses are averaged over the units, so noisy units can steer it towards a poor setting.

## What you get back

A `DecompositionResult`, with `n_mu` units:

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
| `sep_vectors_native`, `sources_native`, `spikes_dict_native`, `native_roa` | (dim, n_mu), (samples, n_mu), dict, (n_mu,) | SCD's own filters, sources and spikes, and the agreement of its spikes with `spikes`; None for CBSS |
| `contrast_exp` | (n_mu,) | Exponent e of each unit's contrast G ∝ \|u\|^e: optimised per unit by SCD, the config's for CBSS (3 for `"square"`, 4 for `"cube"`); None for `"logcosh"` |
| `config` | | The config (`CBSSConfig`, `SCDConfig`) that produced it |

Save it with `calibration.save(path)`, an HDF5 file with the config embedded, and reload it with
`DecompositionResult.load(path)`: building the adaptation needs nothing else. To share or log the config,
write it as YAML too (`calibration.config.to_yaml(path)`). The ground truth in
`selection_kwargs` is data, so it stays out of both.

## Reusing a calibration

- `Decomposition.from_config(calibration.config).apply(new_emg, calibration, timestamps)`
  applies a calibration's model to other EMG without searching again, for example to score it
  on another recording.
- A calibration from another tool can be used by building a `DecompositionResult` from its fields
  (`sources`, `spikes`, `sep_vectors`, `whitening`, `extension_mean`, the centroids, `sil`,
  `cov_isi`, `ext_fact`, and the calibration `emg` and `timestamps`), or by building the
  adaptation directly (see [Building the adaptation](overview.md#building-the-adaptation)).

## Reproducibility

With the same `random_seed`, EMG and config, `decompose()` gives the same units on the CPU with
the same number of threads. Another thread count, or a GPU, changes the order of floating-point
sums, which can change which borderline units are kept. Calibrate on the CPU, with the threads
pinned (see [Run on a cluster](../how-to/run-on-a-cluster.md#reproducibility)), when the
calibration must be reproduced exactly. The same holds for SCD.

On a CUDA GPU, a seeded `decompose()` or `apply()` uses PyTorch's deterministic kernels, and
`import adapt_decomp` sets `CUBLAS_WORKSPACE_CONFIG` (unless you set it) for deterministic
matrix products, so the run repeats on the same GPU and software. It is somewhat slower, still
differs from a CPU run and from other GPUs, and an operation with no deterministic kernel warns.
Import `adapt_decomp` before running any CUDA code, or set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
yourself, since PyTorch reads it on its first CUDA matrix product.

## SCD

`SCD` runs the search of the [scd](https://github.com/AgneGris/swarm-contrastive-decomposition)
package unchanged on the whitened EMG of the shared pipeline: its own filtering, extension and
whitening are not used, so the adaptation sees exactly the preprocessing the calibration did.

```python
--8<-- "calibrate_scd.py:calibrate"
```

What differs from CBSS:

- **Search options** are scd's own: start from one of its parameter sets with `scd_preset`
  (`"default"`, `"surface"` or `"intramuscular"`) and override any `scd.Config` search option
  in `scd_kwargs` (`max_iterations`, `acceptance_silhouette`, `peel_off`, `swarm_*`, ...).
  The options the shared config sets (sampling frequency, filters, extension, whitening,
  bad channels, device, verbosity) are rejected there. `sil_th` and the ICA fields of
  `CBSSConfig` do not exist: SCD accepts units on `acceptance_silhouette`.
- **Separation vectors from SCD's spikes.** SCD finds units one at a time and peels each off
  the data, so its filters are fitted to a residual that does not exist online, and they are
  not orthogonal. Each unit's separation vector is instead the whitened, extended EMG averaged
  at the unit's SCD spikes. The adaptation orthonormalises the separation vectors on every
  update, so these vectors are orthonormalised once at calibration, with the
  order-independent symmetric transform: the first adaptation batch then starts from the
  calibrated units instead of rotating them. SCD's filters are kept in `sep_vectors_native`.
- **Reconstructed sources.** SCD's own sources come from the peeled-off residual. `sources`,
  `spikes`, the centroids and `sil` are therefore recomputed from the orthonormal vectors on
  the full whitened EMG, as the adaptation computes them. SCD's own output is kept in
  `sources_native` and `spikes_dict_native`, and `native_roa` gives each unit's agreement
  between the two. A low `native_roa` flags a unit that relied on the peel-off: drop such
  units with `select_unsupervised(native_roa_th=0.9)`.
- **Contrast exponents.** SCD's swarm optimises the contrast function's exponent per unit;
  `contrast_exp` keeps each unit's best one. CBSS stores its fixed exponent there on the same
  scale, so the two are comparable.
- **Fixed setting:** `spike_det_exp` must be 2 (SCD detects spikes on squared sources). PCA
  (`n_components`) runs in the shared pipeline, before SCD.
- **Cost:** SCD searches far longer than CBSS. The adaptation's cost grows with the whitened
  dimension (channels × `ext_fact`), so set `ext_fact` explicitly rather than following SCD's
  own rule of about 1000 extended channels.

## Recipes

- [Calibrate a recording](../how-to/calibrate-a-recording.md)
- [Evaluate against ground truth](../how-to/evaluate-against-ground-truth.md)
