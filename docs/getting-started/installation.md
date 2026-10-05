# Installation

--8<-- "README.md:install"

For GPU acceleration, install the PyTorch build for your CUDA version
([instructions](https://pytorch.org/get-started/locally/)) and set `device="cuda"` in the
configs. Results on the GPU are not bit-for-bit reproducible; see the reproducibility notes in
the [user guide](../guide/calibration.md#reproducibility).

## Get the data

The examples use datasets published on Zenodo. `adapt-decomp-data`, installed with the
package, downloads them into `data/` under the current directory, checks their checksums and
unpacks them:

```sh
adapt-decomp-data list                     # archives, sizes and DOIs
adapt-decomp-data get fdsi_example-data    # the quickstart's recording
```

| Archive | Size | What it is | Used by | DOI |
|---|---|---|---|---|
| `fdsi_example-data` | 70 MB | One FDSI recording with its ground truth | [Quickstart](quickstart.md), [how-to guides](../how-to/index.md) | Not published yet |
| `neuromotion-data` | 1.6 GB | The NeuroMotion simulation of the paper and its calibration | [Tutorial](../notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) | [10.5281/zenodo.22880910](https://doi.org/10.5281/zenodo.22880910) |
| `fdsi_benchmark-data` | 10.3 GB | The 100 FDSI recordings with their ground truth | [FDSI benchmark](../benchmarks/fdsi.md) | [10.5281/zenodo.22882346](https://doi.org/10.5281/zenodo.22882346) |
| `fdsi_benchmark-outputs` | 10.8 GB | The v1.0 benchmark's results | The benchmark's v1.0 comparison | [10.5281/zenodo.22882323](https://doi.org/10.5281/zenodo.22882323) |

The DOIs are those of the archive versions `adapt-decomp-data` downloads.

Each archive unpacks to `data/<dataset>/` and includes a `README.md` describing its files. Run
the examples from the directory that holds `data/`. From Python, use
`adapt_decomp.utils.download_data(["fdsi_example-data"])`.
