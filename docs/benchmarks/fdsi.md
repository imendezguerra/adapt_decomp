# FDSI benchmark

What to expect from `adapt_decomp`, measured on 100 synthetic HD-EMG recordings with ground
truth: the FDSI dataset, 5 subjects × 5 contractions × 4 SNR levels, simulated with
NeuroMotion via MUniverse (a tour: [Dataset](fdsi/0_dataset.ipynb)). Every recording changes the
wrist angle during the contraction, so a fixed decomposition loses units that the adaptation
should keep.

The benchmark contains five notebooks, each a stage shown on one recording and then run on all of
them, with its results:

0. [Dataset](fdsi/0_dataset.ipynb): Description and visualisation of the dataset.
1. [Calibration](fdsi/1_calibrate.ipynb): CBSS on the first 5 s of every recording, a steady
   hold, keeping the units that match a simulated motor unit (`select_supervised(roa_th=0.9)`).
2. [Search](fdsi/2_search.ipynb): the adaptation's hyperparameters (`wh_learning_rate`,
   `sv_learning_rate`, `centroid_momentum`) tuned on a pool of three recordings (subject 1,
   30 dB, triangular ramps of 40, 10 and 5 s), by five searches of 100 trials:
    - `sv_loss`, averaged or summed over units;
    - a Pareto front of `wh_loss` and `sv_loss`, averaged or summed, choosing the member with
      the lowest `sv_loss`;
    - `roa`, the rate of agreement with the ground truth: the best a search could do.
3. [Apply](fdsi/3_apply.ipynb): no adaptation and each search's winner on all 100 recordings,
   from the end of the calibration window. Every unit is scored by its rate of agreement (RoA)
   with the motor unit it tracks and by its silhouette (SIL).
4. [Results](fdsi/4_results.ipynb): the gains over no adaptation, by contraction, noise level,
   pool and held-out recordings and phase, and against the previous version.

[`config.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/config.yaml)
declares the whole experiment.

## Reproduce it

Each step reproduces more, at more cost (times on 8 cores):

| To | You need | Takes |
|---|---|---|
| Read and re-plot every result, and compare versions | A clone of the repository: the results are in [`benchmarks/fdsi/results/`](https://github.com/imendezguerra/adapt_decomp/tree/main/benchmarks/fdsi/results) | Seconds ([Results](fdsi/4_results.ipynb) runs without data) |
| Re-score the published spikes and sources | `fdsi_benchmark-data` and `fdsi_benchmark-outputs-v1.1.0`, then `python -m benchmarks.fdsi collect` | Minutes |
| Recompute every calibration and application with the published tuned configs | `fdsi_benchmark-data`, then `calibrate`, `apply` and `collect` | About 5 h, under 1 h for one subject or config |
| Re-run the searches too | The same, plus `search` | About 15 h on 12 cores, a cluster or a day on a workstation |

From a clone of the repository, with the `adapt_decomp` environment active:

```sh
adapt-decomp data get fdsi_benchmark-data          # the recordings, about 10 GB

python -m benchmarks.fdsi calibrate --workers 8
python -m benchmarks.fdsi search                   # optional: else the published configs apply
python -m benchmarks.fdsi apply --workers 8
python -m benchmarks.fdsi collect                  # writes benchmarks/fdsi/results/<version>/
```

or the `RUN` cells of the notebooks, which call the same code. To check the pipeline in about
20 minutes first, add `--quick` (1 subject, 4 recordings, 4 trials per search; its results go
to `<version>-quick`). Every stage skips the tasks whose outputs exist, so an interrupted run
resumes where it stopped, and on the pinned environment the outputs reproduce bit for bit.
[Running the benchmark](fdsi/running.md) covers the commands, the outputs and running on a PBS
cluster.

## Versions

Every version's results stay in `benchmarks/fdsi/results/<version>/`: per-unit scores
(`units.csv`), the calibrations' units, every search trial, the tuned configs, and `run.yaml`
with the adapt_decomp version, commit and config that produced them. Its spikes and sources
are published as `fdsi_benchmark-outputs-<version>` on Zenodo. v1.0.0's outputs, in that
version's layout, are at [10.5281/zenodo.22882323](https://doi.org/10.5281/zenodo.22882323)
(notebooks at the `v1.0.0` tag); its per-unit scores are in `results/v1.0.0/`.

v1.0.0 adapted each recording from its first sample, searched only the two learning rates with
`centroid_momentum` fixed at 0.95, and ran its 50 trials one at a time. Given those settings,
this version's code reproduces v1.0's spike trains (bit for bit on two of the three pool
recordings, 99.99 % of samples on the third).
