# adapt_decomp: adaptive EMG decomposition

[![PyPI](https://img.shields.io/pypi/v/adapt-decomp)](https://pypi.org/project/adapt-decomp/)
[![Python versions](https://img.shields.io/pypi/pyversions/adapt-decomp)](https://pypi.org/project/adapt-decomp/)
[![CI](https://github.com/imendezguerra/adapt_decomp/actions/workflows/ci.yml/badge.svg)](https://github.com/imendezguerra/adapt_decomp/actions/workflows/ci.yml)
[![Docs](https://github.com/imendezguerra/adapt_decomp/actions/workflows/docs.yml/badge.svg)](https://imendezguerra.github.io/adapt_decomp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22902494.svg)](https://doi.org/10.5281/zenodo.22902494)

## Overview
<!-- --8<-- [start:overview] -->
adapt_decomp decomposes high-density electromyography (HD-EMG) into motor unit firings during
dynamic contractions, in real time (about 40 ms per 100 ms batch on 8 CPU cores). It calibrates a
decomposition on a short window, then adapts it batch by batch as the contraction changes, as
described in [Mendez Guerra et al., J. Neural Eng., 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf).
It is written in Python with PyTorch.

The package covers the three steps of the pipeline:

- **Calibration** (`adapt_decomp.cbss`): convolutive blind source separation (CBSS) finds the
  motor units in a calibration window.
- **Adaptation** (`adapt_decomp.adaptation`): `AdaptDecomp` tracks those units over the rest of
  the recording, updating the whitening, the separation vectors and the spike detection in every
  batch.
- **Hyperparameter optimisation** (`adapt_decomp.adaptation.optimize`): an Optuna search
  chooses the adaptation's learning rates for your data, with or without ground truth.
<!-- --8<-- [end:overview] -->

## Table of Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Data](#data)
- [FDSI benchmark](#fdsi-benchmark)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation
<!-- --8<-- [start:install] -->
Install the latest release from PyPI (Python 3.10 to 3.12, Linux, macOS or Windows):

```sh
pip install adapt-decomp
```

The package is imported as `adapt_decomp`. On Linux, pip installs PyTorch's CUDA build (about
2 GB); for the CPU-only build, install PyTorch first:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install adapt-decomp
```

To reproduce published results exactly, use the conda environment from a clone of the
repository instead. It pins every dependency to the versions the results were produced with,
and installs the CPU build of PyTorch:

```sh
git clone https://github.com/imendezguerra/adapt_decomp.git
cd adapt_decomp
conda env create -f environment.yaml
conda activate adapt_decomp
```
<!-- --8<-- [end:install] -->

## Quick start

The [quickstart](https://imendezguerra.github.io/adapt_decomp/getting-started/quickstart/)
calibrates, adapts and scores a synthetic recording in a few minutes on a CPU. Download it
first (70 MB):

```sh
adapt-decomp data get fdsi_example-data
```

## Documentation

The documentation, **<https://imendezguerra.github.io/adapt_decomp/>**, has the quickstart, the
API concepts (calibration, adaptation and hyperparameter optimisation, with their parameters), a
user guide of recipes, including the `adapt-decomp` command line, an example on the paper's
simulated contraction, the FDSI benchmark and the API reference.

## Data

The example, paper example and benchmark datasets are published on Zenodo. `adapt-decomp data list`
shows them, and `adapt-decomp data get <archive>` downloads one into `data/`. See
[Get the data](https://imendezguerra.github.io/adapt_decomp/getting-started/installation/#get-the-data).

## FDSI benchmark

[`benchmarks/fdsi/`](https://github.com/imendezguerra/adapt_decomp/tree/main/benchmarks/fdsi)
runs `adapt_decomp` on 100 synthetic HD-EMG recordings with ground truth (5 subjects × 5
contractions × 4 SNR levels): four notebooks that show each stage on one recording and run it on
all of them, also from the command line or on a PBS cluster. Each version's results are in
`benchmarks/fdsi/results/`. See its
[description](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/) and
[results](https://imendezguerra.github.io/adapt_decomp/benchmarks/fdsi/4_results/).

## Contributing

Contributions are welcome!
[CONTRIBUTING.md](https://github.com/imendezguerra/adapt_decomp/blob/main/CONTRIBUTING.md)
explains how to set up the development environment, run the tests, preview the docs, and what
the automated checks do.

## License

This project is licensed under the
[MIT License](https://github.com/imendezguerra/adapt_decomp/blob/main/LICENSE).

## Citation
<!-- --8<-- [start:citation] -->
If you use adapt_decomp in your research, please cite the paper:

```bibtex
@article{MendezGuerra2024,
  author    = {Mendez Guerra, Irene and Barsakcioglu, Deren Y. and Farina, Dario},
  title     = {Adaptive EMG decomposition in dynamic conditions based on online learning
               metrics with tunable hyperparameters},
  journal   = {Journal of Neural Engineering},
  publisher = {IOP Publishing},
  volume    = {21},
  number    = {4},
  year      = {2024},
  issn      = {1741-2552},
  doi       = {10.1088/1741-2552/ad5ebf},
  url       = {https://dx.doi.org/10.1088/1741-2552/ad5ebf}
}
```

To cite the version of the software you used, use its Zenodo DOI
([10.5281/zenodo.22902494](https://doi.org/10.5281/zenodo.22902494)) or the **Cite this
repository** button on [GitHub](https://github.com/imendezguerra/adapt_decomp), which reads
[`CITATION.cff`](https://github.com/imendezguerra/adapt_decomp/blob/main/CITATION.cff).
<!-- --8<-- [end:citation] -->

## Contact

For any questions or inquiries, please contact:
```
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
