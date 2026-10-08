# Installation

--8<-- "README.md:install"

For GPU acceleration, install the PyTorch build for your CUDA version
([instructions](https://pytorch.org/get-started/locally/)) and set `device="cuda"` in the
configs. Results on the GPU are not much faster than on CPU and are not bit-for-bit reproducible; see the reproducibility notes in
the [API concepts](../guide/calibration.md#reproducibility).

To log adaptations and hyperparameter searches to [wandb](https://wandb.ai) from the
[command line](../how-to/use-the-command-line.md), install the `wandb` extra:

```sh
pip install "adapt-decomp[wandb]"
```

## Get the data

The examples use datasets published on Zenodo. `adapt-decomp data`, part of the command installed
with the package, downloads them into `data/` under the current directory, checks their checksums and
unpacks them:

```sh
adapt-decomp data list                     # archives, sizes and DOIs
adapt-decomp data get fdsi_example-data    # the quickstart's recording
```

| Archive | Size | What it is | Used by | DOI |
|---|---|---|---|---|
| `fdsi_example-data` | 70 MB | One FDSI recording with its ground truth | [Quickstart](quickstart.md), [user guide](../how-to/index.md) | Not published yet |
| `neuromotion-data` | 1.6 GB | The NeuroMotion simulation of the paper and its calibration | [Paper example](../notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) | [10.5281/zenodo.22880910](https://doi.org/10.5281/zenodo.22880910) |
| `fdsi_benchmark-data` | 10.3 GB | The 100 FDSI recordings with their ground truth | [FDSI benchmark](../benchmarks/fdsi.md) | [10.5281/zenodo.22882346](https://doi.org/10.5281/zenodo.22882346) |
| `fdsi_benchmark-outputs-v1.1.0` | 3.5 GB | The v1.1.0 benchmark's spikes and sources, and its searches | Re-scoring the [FDSI benchmark](../benchmarks/fdsi.md) without recomputing it | Not published yet |

The DOIs are those of the archive versions `adapt-decomp data` downloads.

Each archive unpacks to `data/<dataset>/` and includes a `README.md` describing its files. Run
the examples from the directory that holds `data/`. From Python, use
`adapt_decomp.utils.download_data(["fdsi_example-data"])`.
