# Adaptive EMG decomposition in dynamic conditions based on online learning metrics with tunable hyperparameters

## Overview 
This repository contains functions to adaptively decompose electromyography (EMG) into motor unit firings during dynamic conditions in real-time (~22 ms per 100 ms batch, CPU only with loss calculation) based online learning metrics with tunable hyperparameters as described in [Mendez Guerra et al, JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf). The code is implemented in `python` using `pytorch`.

## Table of Contents
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Package structure](#packagestructure)
- [Command line inteface](#commandlineinterface)
- [Data loaders](#dataloaders)
- [Optimization](#optimization)
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
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate adapt_decomp
    ```
5. Install adaptation package:
    ```sh
    pip install -e .
    ```

Please note that the `environment.yml` only installs the `cpu` version of `pytorch`. To enable gpu acceleration, `cuda` will need to be installed manually (check command [here](https://pytorch.org/get-started/locally/))

The code has been tested in MacOs, Windows, and Linux. 

## Tutorial
To learn how to use the dynamic decomposition go to [adaptive_emg_decomp_dyn_example](https://github.com/imendezguerra/adapt_decomp/blob/main/tutorials/adaptive_emg_decomp_dyn_example.ipynb) for a step by step tutorial. Please note that the model requires a precalibrated decomposition model including extension factor, whitening, separation vectors, spike and baseline centroids, emg during calibration as well as the resulting IPTs and spikes. The model uses the last three variables to compute the whitening and separation vector losses based on the median squared error between the Kullback–Leibler divergence of the whitened covariance and the kurtosis of the sources between the adaptive and calibration conditions. To execute the code, download this [example contraction](https://imperiallondon-my.sharepoint.com/:f:/g/personal/im4417_ic_ac_uk/EkJvoEffPmdEnkoHeRItVt8BWyQd6kztbrszu6njnfHM0Q?e=wbbuZF) and save it in the repository directory under `data\example` folders.

## Package structure 
The package is composed of the following modules:
- `loaders.py`: Functions to load EMG and decomposition files.
- `data_structures.py`: Classes for the input EMG and initial decomposition parameters.
- `adaptation.py`: Main class with the adaptive decomposition.
- `config.py`: Dataclass with the parameters for the decomposition adaptation.
- `preprocesing.py`: Functions for EMG preprocessing such as filtering.
- `plots.py`: Functions to plot the results.
- `utils.py`: Functions to extract motor unit properties and compute rate of agreement.
- `io.py`: Functions to save and load the adaptive decomposition outputs.

## Command line interface
The code is integrated with [Weights and Biases](https://wandb.ai) to track and visualise the results. To enable this, run the code via the command line using:

```sh
python scripts/run.py --data_config configs/data_configs/data_example.yml --model_config configs/model_configs/default_neuromotion.yml --wandb_project_name adapt_decomp
```

This command executes the decomposition adaptation for a given input described in data_config with parameters specified in model_config. Take a look at the [config](https://github.com/imendezguerra/adapt_decomp/blob/main/configs) to see examples of both structures. The data_config is simply a wrapper `.yml` file with the paths to the calibrated decomposition model (`path_decomp`), the input EMG for the decomposition adaptation (`path_emg`), the ground truth data if available (`path_gt`), a flag to activate/deactivate data preprocessing (`preprocess`), and the corresponding data loader (`loader`).

## Data loaders
To use the command line interface with your own data, please implement the corresponding data loader in [loaders.py](https://github.com/imendezguerra/adapt_decomp/blob/main/src/adapt_decomp/loaders.py), and add it to the `load_data` wrapper function. Also, create the corresponding `.yml` file and store it under `data_configs`.

## Optimization 
The default hyperparameters are the optimal for the dataset presented in [Mendez Guerra et al, JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf) for simulations (neuromotion) and experimental wrist and forearm data. However, if those were not appropriate for a given dataset, the code is integrated with [Weights and Biases](https://wandb.ai) which enables hyperparameter sweeps based on random sampling, grid search, and bayesian optimisation (default here). To access this functionality, use the command line to execute the code as:
```sh
python scripts/run.py --data_config configs/data_configs/data_example.yml --model_config configs/model_configs/default_neuromotion.yml --sweep_config configs/model_configs/sweep_loss.yml --sweep_counts 30 --wandb_project_name adapt_decomp
```
This will perform a hyperparameter sweep with 30 iterations (can be changed) for the hyperparameters, range, criteria, and method specified in [sweep_loss.yml](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/model_configs/sweep_loss.yml).

## Contributing
We welcome contributions! Here’s how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This repository is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```sh
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
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```