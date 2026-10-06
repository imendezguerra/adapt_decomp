# FDSI benchmark

What to expect from `adapt_decomp`, measured on 100 synthetic HD-EMG recordings with ground
truth: the FDSI dataset, 5 subjects × 5 contractions × 4 SNR levels, simulated with
NeuroMotion via MUniverse (a tour: [Dataset](fdsi/dataset.ipynb)). Every recording changes the
wrist angle during the contraction, so a fixed decomposition loses units that the adaptation
should keep. The [Results](fdsi/report.ipynb) page compares adaptation with no adaptation,
by contraction, by noise level and by hyperparameter search.

## Design

The benchmark runs in three stages:

1. **Calibrate** every recording with CBSS on its first 5 s, a steady hold, and keep the units
   that match a simulated motor unit (`select_supervised(roa_th=0.9)`).
2. **Search** the adaptation's hyperparameters (`wh_learning_rate`, `sv_learning_rate`,
   `centroid_momentum`) on a pool of three recordings: subject 1, 30 dB, triangular ramps of
   40, 10 and 5 s. Five searches, 100 trials each, one at a time after 15 random start-up trials:
   - `sv_loss`, averaged or summed over units;
   - a Pareto front of `wh_loss` and `sv_loss`, averaged or summed, choosing the member with
     the lowest `sv_loss`;
   - `roa`, the rate of agreement with the ground truth: the best a search could do.
3. **Apply** each search's best config, and the no-adaptation baseline, to all 100
   recordings, adapting from the end of the calibration window. Every unit is scored by its
   rate of agreement (RoA) with the motor unit it tracks and by its silhouette (SIL).

[`benchmark.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/benchmark.yaml)
declares the whole experiment.

## Reproduce it

From a clone of the repository, with the `adapt_decomp` environment active:

```sh
adapt-decomp-data get fdsi_benchmark-data          # inputs, about 10 GB

python -m benchmarks.fdsi calibrate --all --n-workers 8
python -m benchmarks.fdsi search --all             # each search uses 12 cores (see the spec)
python -m benchmarks.fdsi apply --all --n-workers 8
python -m benchmarks.fdsi collect                  # the tables the results read
```

To check the pipeline in about 20 minutes first, add
`--spec benchmarks/fdsi/benchmark_smoke.yaml` (1 subject, 4 recordings, 4 trials per search).
Outputs are cached, so an interrupted run resumes where it stopped, and on the pinned
environment they reproduce bit for bit.
[Running the benchmark](fdsi/running.md) covers the commands, caching, outputs and running on a
PBS cluster.

## Relation to v1.0

v1.0's benchmark (notebooks at the `v1.0.0` tag) adapted each recording from its first sample,
searched only the two learning rates with `centroid_momentum` fixed at 0.95, and ran its 50
trials one at a time. Given those settings, this version's code reproduces v1.0's spike trains
(bit for bit on two of the three pool recordings, 99.99 % of samples on the third).
