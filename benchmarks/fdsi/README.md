# FDSI benchmark

A reproducible, cached run of `adapt_decomp` on the FDSI dataset: calibration, hyperparameter
searches and application to 100 recordings, locally or as PBS Pro array jobs. The full
documentation (commands, settings, caching, outputs, run metadata, release checklist) is on the
[documentation site](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/), with the
[results](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/report/) and a
[tour of the dataset](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/dataset/).

From the repository root, with the `adapt_decomp` environment active:

```sh
python scripts/download_data.py get fdsi_benchmark-data

python -m benchmarks.fdsi calibrate --all --n-workers 8
python -m benchmarks.fdsi search --all
python -m benchmarks.fdsi apply --all --n-workers 8
python -m benchmarks.fdsi collect

bash benchmarks/fdsi/pbs/submit.sh   # or all of it on PBS Pro
```

| File | Holds |
|---|---|
| `benchmark.yaml`, `benchmark_smoke.yaml` | The experiment, and a small version to check the pipeline |
| `cli.py`, `__main__.py` | `python -m benchmarks.fdsi`: one command per stage |
| `spec.py` | Spec validation, tasks, cache keys and status |
| `stages.py` | The stages, each calling the library directly |
| `fdsi.py` | FDSI paths, loaders and metrics |
| `report.py`, `report.ipynb` | The results, read from the collected tables |
| `dataset.ipynb` | A tour of the raw data |
| `pbs/` | PBS Pro array-job template and submission script |
