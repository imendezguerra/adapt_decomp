# Adaptive EMG decomposition in dynamic conditions based on online learning metrics with tunable hyperparameters

This repository contains functions to adaptively decompose electromyography (EMG) into motor unit firings during dynamic conditions in real-time (~22 ms per 100 ms batch, CPU only with loss calculation) based on online learning metrics with tunable hyperparameters as described in [Mendez Guerra et al, JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf). The code is implemented in Python using PyTorch.

The full pipeline is as follows including calibration, hyperparameter optimisation, and adaptation:

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart LR
    EMG["Raw EMG\n(samples, channels)"]

    subgraph CAL["Calibration — cbss/"]
        CBSSC["CBSS(config)\n.decompose()"]
        RES["CBSSResult"]
        CBSSC --> RES
    end

    subgraph ADAPT["Online adaptation — adaptation/"]
        AD["AdaptDecomp"]
        OUT["AdaptationResult\n(spikes, sources, losses)"]
        AD -->|".process_data(emg, ...)"| OUT
    end

    subgraph OPT["Optimisation — adaptation/optimize.py"]
        O1["optimize_adapt_decomp_pooled_memory\n(preloaded CBSSResults)"]
        O2["optimize_adapt_decomp_pooled_disk\n(on-disk CBSSResults, loaded per trial)"]
        BEST["best AdaptConfig"]
        O1 --> BEST
        O2 --> BEST

        P1["optimize_adapt_decomp_pooled_memory_pareto\n(preloaded CBSSResults)"]
        P2["optimize_adapt_decomp_pooled_disk_pareto\n(on-disk CBSSResults, loaded per trial)"]
        FRONT["Pareto front\n(study.best_trials)"]
        P1 --> FRONT
        P2 --> FRONT
        FRONT -->|"selection_rule(front)"| BEST
    end

    EMG --> CBSSC
    EMG -->|".calibrate_and_process()"| OUT
    RES -->|".from_calibration()"| AD
    RES -->|".to_adapt_tensors()"| O1
    RES -->|".to_adapt_tensors()"| O2
    RES -->|".to_adapt_tensors()"| P1
    RES -->|".to_adapt_tensors()"| P2
```

## Table of Contents
- [Installation](#installation)
- [Dcomumentation](#Dcomumentation)
- [Tutorial](#tutorial)
- [FDSI Benchmark](#fdsi-benchmark)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/adapt_decomp.git
    ```
2. Navigate to the project directory:
    ```sh
    cd adapt_decomp
    ```
3. Create the conda environment from the `environment.lock.yaml` file:
    ```sh
    conda env create -f environment.lock.yaml
    ```
4. Activate the environment:
    ```sh
    conda activate adapt_decomp
    ```
5. Install the package:
    ```sh
    pip install -e .
    ```

Please note that `environment.lock.yaml` only installs the `cpu` version of `pytorch`. To enable GPU acceleration, `cuda` will need to be installed manually (check command [here](https://pytorch.org/get-started/locally/)).

The code has been tested on macOS, Windows, and Linux.

## Documentation

A task-oriented manual for each entry point, with diagrams of how the
pieces connect.

| Page | Covers |
|------|--------|
| [docs/architecture.md](docs/architecture.md) | Repo layout, subpackage dependencies, how objects hand off between stages |
| [docs/calibration.md](docs/calibration.md) | `cbss/`: running CBSS on raw EMG, loading/reusing an existing calibration |
| [docs/adaptation.md](docs/adaptation.md) | `adaptation/`: the three ways to build an `AdaptDecomp` and run it |
| [docs/optimisation.md](docs/optimisation.md) | `adaptation/optimize.py`: single-contraction and pooled hyperparameter search — single-objective (`objective`) and Pareto/multi-objective (`objectives`) |

### Where to start

- **Have raw EMG, need motor units:** [docs/calibration.md](docs/calibration.md).
- **Have a calibration, need an online decomposition:**
  [docs/adaptation.md](docs/adaptation.md).
- **Have one or more calibrations, need tuned hyperparameters:**
  [docs/optimisation.md](docs/optimisation.md).
- **New to the codebase, want the map first:**
  [docs/architecture.md](docs/architecture.md).

## Tutorials
To learn how to use the adaptive decomposition go to [adaptive_emg_decomp_dyn_example](https://github.com/imendezguerra/adapt_decomp/blob/feature_structure/notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb) for a step by step tutorial. The notebook loads synethetic data and a precomputed decomposition model and runs the adaptation pipeline on a simulated wrist dynamic contraction ([NeuroMotion](https://github.com/shihan-ma/NeuroMotion): a 15% MVC index flexion recorded while the wrist ramps from 0° to -40° in a staircase pattern, precalibrated on the first 30 s plateau).

For more examples go to [fdsi_benchmark](https://github.com/imendezguerra/adapt_decomp/blob/main/notebooks/fdsi_benchmark), where there is a collection of notebooks covering decomposition claibration using cbss, hyperparameter optimization (3 methods), online decomposition with and without adaptation (best configs), and model comparison. The dataset used by these notebooks comprises 100 synthetic HD-EMG recordings (100 chs) simulated with NeuroMotion using [MUniverse](https://github.com/dfarinagroup/muniverse) (5 subjects x 5 wrist-kinematic conditions x 4 SNR levels, each with a matching ground-truth spike train), configured via [configs/data_configs/fdsi_benchmark_grid.yaml](configs/data_configs/fdsi_benchmark_grid.yaml). For more information on the datset start by [00_dataset.ipynb](notebooks/fdsi_benchmark/00_dataset.ipynb) and follow the notebooks in order.

## Downloading the data

The notebooks read data that is too large to keep in the repository. It is published as three
independent archives, so you only fetch what you need:

| Archive | Size | What it is | DOI |
|---|---|---|---|
| `neuromotion-data` | 1.64 GB | The tutorial's simulated recording and its calibration. **Required by the tutorial.** | [10.5281/zenodo.22880910](https://doi.org/10.5281/zenodo.22880910) |
| `fdsi_benchmark-data` | 10.35 GB | The 100-recording benchmark itself. **Required by the benchmark notebooks.** | [10.5281/zenodo.22882346](https://doi.org/10.5281/zenodo.22882346) |
| `fdsi_benchmark-outputs` | 10.75 GB | Every cached pipeline stage. Optional, but recomputing the full grid takes hours. | [10.5281/zenodo.22882323](https://doi.org/10.5281/zenodo.22882323) |

A `data` archive is the dataset; the matching `outputs` archive is what this repository
produced from it. The benchmark notebooks are load-only by default, so
`fdsi_benchmark-outputs` is what lets you reproduce every figure in seconds instead of hours.
The tutorial has no outputs archive — it regenerates its own in minutes.

From the repository root, with the `adapt_decomp` environment active:

```sh
python scripts/download_data.py list                      # archives, sizes and DOIs
python scripts/download_data.py get neuromotion-data      # just the tutorial's input
python scripts/download_data.py get fdsi_benchmark        # both benchmark archives (prefix match)
python scripts/download_data.py get                       # everything (22 GB)
```

Files land in `data/<dataset>/{data,outputs}/`, which is exactly where the configs and
notebooks expect them — no manual placement needed. Each download is checked against the
checksum Zenodo publishes, and anything already unpacked is skipped, so re-running the command
is cheap.

Prefer to do it by hand? Download the zips from the DOIs list and unpack them all into `data/`. Each archive
carries its own full path, so any subset lands correctly:

```sh
unzip '*.zip' -d data/
```

Each dataset folder then contains its own `README.md` describing how the data was generated
and what every field means.

## Contributing
We welcome contributions! Here's how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This repository is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```
@article{Mendez Guerra_2024,
   author={Mendez Guerra, Irene and Barsakcioglu, Deren Y. and Farina, Dario},
   title={Adaptive EMG decomposition in dynamic conditions based on online learning metrics with tunable hyperparameters},
   journal={Journal of Neural Engineering},
   publisher={IOP Publishing},
   volume={21},
   number={4},
   ISSN={1741-2552},
   DOI={10.1088/1741-2552/ad5ebf},
   url={https://dx.doi.org/10.1088/1741-2552/ad5ebf}
   }
```

## Contact

For any questions or inquiries, please contact us at:
```
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
