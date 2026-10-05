# Getting started

## Install

Clone the repository and create the conda environment, which pins every dependency to an
exact version and installs the package in editable mode:

```sh
git clone https://github.com/imendezguerra/adapt_decomp.git
cd adapt_decomp
conda env create -f environment.yaml
conda activate adapt_decomp
```

Alternatively, with pip only (dependency versions are then not pinned):
`pip install -e ".[dev]"`, adding `docs` to build this site: `pip install -e ".[dev,docs]"`.

The environment installs the CPU build of PyTorch. For GPU acceleration, install a CUDA build
afterwards ([instructions](https://pytorch.org/get-started/locally/)).

## Get the data

The examples use datasets published on Zenodo, downloaded into `data/`:

```sh
python scripts/download_data.py list                     # archives, sizes and DOIs
python scripts/download_data.py get neuromotion-data     # the tutorial's recording (1.6 GB)
python scripts/download_data.py get fdsi_benchmark-data  # the FDSI benchmark (10 GB)
```

| Archive | What it is |
|---|---|
| `neuromotion-data` | The [tutorial](notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb)'s simulated recording and its calibration |
| `fdsi_benchmark-data` | The [FDSI benchmark](benchmarks/fdsi.md)'s 100 recordings, used by the how-to guides |
| `fdsi_benchmark-outputs` | The v1.0 benchmark's cached results |

## The whole pipeline on one recording

Calibrate on the first 5 s, keep the units that match the ground truth, adapt over the rest of
the recording, and score it. This is
[`docs/snippets/workflow.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/docs/snippets/workflow.py),
which runs as is from the repository root.

```python
--8<-- "workflow.py:load"

--8<-- "workflow.py:calibrate"

--8<-- "workflow.py:select-supervised"

--8<-- "workflow.py:save"

--8<-- "workflow.py:adapt"

--8<-- "workflow.py:baseline"

--8<-- "workflow.py:evaluate"
```

## Next

- The [how-to guides](how-to/index.md) take each step further: unsupervised unit selection,
  tuning, plotting, clusters, provenance and online processing.
- The [user guide](architecture.md) explains how the pieces fit together.
- The [tutorial](notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) compares online and offline processing, and
  adaptation against none, on a NeuroMotion simulation.
