# How-to guides

Short, task-focused recipes. Their code comes from two scripts,
[`docs/snippets/workflow.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/docs/snippets/workflow.py)
(one recording: calibrate, adapt, evaluate, plot, record) and
[`docs/snippets/tune.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/docs/snippets/tune.py)
(hyperparameter searches). Both run as written on one FDSI recording and are tested, so the
code on these pages is known to work. To run them yourself, from the repository root:

```sh
python scripts/download_data.py get fdsi_benchmark-data
python docs/snippets/workflow.py
python docs/snippets/tune.py
```

| Guide | When you want to |
|---|---|
| [Calibrate a recording](calibrate-a-recording.md) | find motor units in a calibration window and keep the reliable ones |
| [Adapt from the calibration end](adapt-from-calibration-end.md) | track those units over the rest of the recording, and compare with no adaptation |
| [Tune hyperparameters](tune-hyperparameters.md) | choose the learning rates and centroid momentum for your data |
| [Evaluate against ground truth](evaluate-against-ground-truth.md) | score tracking with the rate of agreement and the silhouette |
| [Plot results](plot-results.md) | look at sources, spikes and searches |
| [Run on a cluster](run-on-a-cluster.md) | run many recordings or searches on PBS or SLURM, reproducibly |
| [Record provenance](record-provenance.md) | store with each result how, where and from which code it was produced |
| [Process online](process-online.md) | process raw batches as they would arrive in real time |
