# FDSI benchmark

`adapt_decomp` on the FDSI dataset's 100 recordings with ground truth: calibration,
hyperparameter searches and application, as four notebooks that each show a stage on one
recording and then run it on all of them, or from the command line and as PBS array jobs. The
[documentation site](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/) renders
the notebooks with their results.

From the repository root, with the `adapt_decomp` environment active:

```sh
adapt-decomp data get fdsi_benchmark-data   # the recordings, about 10 GB

python -m benchmarks.fdsi calibrate --workers 8
python -m benchmarks.fdsi search            # optional: else the published configs apply
python -m benchmarks.fdsi apply --workers 8
python -m benchmarks.fdsi collect           # writes results/<version>/

bash benchmarks/fdsi/pbs/submit.sh          # or all of it on PBS Pro
```

Add `--quick` to any of them for a 20-minute check of the pipeline.

| File | Holds |
|---|---|
| `config.yaml` | The experiment, and its `quick` version |
| `0_dataset.ipynb` | Description and plots of the dataset |
| `1_calibrate.ipynb` … `4_results.ipynb` | The stages, then the results |
| `pipeline.py` | The stages, one function per task, and `run` |
| `fdsi.py` | FDSI paths, loaders and metrics |
| `report.py` | The tables and comparisons of the notebooks |
| `__main__.py`, `pbs/` | `python -m benchmarks.fdsi STAGE`, and PBS Pro array jobs |
| `results/<version>/` | Each version's scores, search trials, tuned configs and `run.yaml` |
