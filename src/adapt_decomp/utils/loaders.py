"""Data loaders"""

import h5py
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from scipy.io import loadmat
from typing import Any, Dict, Literal, Optional, Tuple, TYPE_CHECKING, Union

from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.preprocessing.extension import extend_data
from adapt_decomp.spikes import firings_to_spikes, pair_ground_truth

if TYPE_CHECKING:
    from adapt_decomp.cbss.config import CBSSConfig

# ------------------------------------------------------------------
# Loaders for dataset 1 (neuromotion)
# ------------------------------------------------------------------

def _load_neuromotion(path_file: str) -> Dict:
    """Load neuromotion HDF5 data from the specified file.

    Args:
        path_file (str): Path to the neuromotion-format HDF5 file.

    Returns:
        Dict: Parsed fields -- emg, rms, ch_map, ch_cols, bad_ch, fs,
        timestamps, angle_profile, force_profile, staircase_phases, spikes,
        muaps, muap_muscle_labels, muap_angle_labels, paired_units,
        roa_0deg, lags_0deg.
    """
    data = dict.fromkeys([
        'emg', 'rms', 'ch_map', 'ch_cols', 'bad_ch',
        'fs', 'timestamps', 'angle_profile', 'force_profile', 'staircase_phases',
        'spikes', 'muaps', 'muap_muscle_labels', 'muap_angle_labels',
        'paired_units', 'roa_0deg', 'lags_0deg'
    ])

    with h5py.File(path_file, 'r') as h5:

        for key in data.keys():
            if key in ['staircase_phases', 'paired_units']:
                data[key] = dict.fromkeys( h5[key].keys() )
                for subkey in h5[key].keys():
                    data[key][subkey] = h5[key][subkey][()]
            else:
                data[key] = h5[key][()]

    return data


def _cbss_result_from_mat_decomp(path_decomp: str, ext_mode: str = "block") -> CBSSResult:
    """Build a CBSSResult from example MATLAB decomposition.

    Args:
        path_decomp (str): Path to the legacy .mat decomposition file.
        ext_mode (str, optional): Extension layout used to rebuild
            extension_mean -- the .mat file doesn't record which convention
            produced BRT/WH, so this is an assumption, not a verified fact.
            Defaults to "block".

    Returns:
        CBSSResult: Calibration result with sources, spikes, spikes_dict,
        sep_vectors, whitening, extension_mean, spikes_centr, base_centr,
        sil, cov_isi, ext_fact, dr, pnr, emg, and timestamps populated.
        pca_components/pca_mean, muaps (ragged MATLAB cell array, doesn't
        match CBSSResult's dense convention), and gt_matched_indices/roa are
        left None.
    """
    decomp = loadmat(path_decomp, simplify_cells=True)

    ext_fact = int(decomp["EXT_FACT"])
    emg_calib = np.asarray(decomp["EMG"].T, dtype=np.float32)  # [T, C]

    spikes = firings_to_spikes(decomp["firings"], decomp["IPTs"], matlab_index=True)  # [n_mu, T]
    spikes_dict = {
        i: (np.asarray(firing) - 1).astype(np.int64)
        for i, firing in enumerate(decomp["firings"])
    }

    extension_mean = (
        extend_data(torch.from_numpy(emg_calib), ext_fact, ext_mode=ext_mode)
        .mean(dim=0, keepdim=True)
        .numpy()
        .astype(np.float32)
    )

    return CBSSResult(
        sources=np.asarray(decomp["IPTs"].T, dtype=np.float32),
        spikes=np.asarray(spikes.T, dtype=np.int32),
        spikes_dict=spikes_dict,
        sep_vectors=np.asarray(decomp["BRT"], dtype=np.float32),
        whitening=np.asarray(decomp["WH"], dtype=np.float32),
        extension_mean=extension_mean,
        spikes_centr=np.asarray(decomp["SIG_CENT"], dtype=np.float32),
        base_centr=np.asarray(decomp["N_CENT"], dtype=np.float32),
        sil=np.asarray(decomp["SIL"], dtype=np.float32),
        cov_isi=np.asarray(decomp["CoV"], dtype=np.float32),
        ext_fact=ext_fact,
        dr=np.asarray(decomp["DR"], dtype=np.float32),
        pnr=np.asarray(decomp["PNR"], dtype=np.float32),
        emg=emg_calib,
        timestamps=np.asarray(decomp["timestamps"], dtype=np.float64),
    )


def load_example(
    path_emg: str,
    path_decomp: str,
    preprocess: bool,
    path_gt: Optional[str] = None,
) -> Dict:
    """Load the legacy v1 (JNE 2024) simulated dataset via the CBSSResult contract.

    Args:
        path_emg (str): Path to the neuromotion-format HDF5 recording (also
            the source of ground truth, unless path_gt points elsewhere).
        path_decomp (str): Path to the legacy .mat decomposition file.
        preprocess (bool): Whether the online path should preprocess emg
            before extension (forwarded unchanged).
        path_gt (Optional[str], optional): Path to an alternative
            neuromotion-format HDF5 file to source ground truth from.
            Defaults to None, which reuses path_emg's own 'spikes' key (no
            second file read).

    Returns:
        Dict: "cbss_result" (CBSSResult), "cbss_config" (CBSSConfig, a
        best-effort reconstruction), "emg" (torch.Tensor, shape
        (samples, channels)), "gt_full_bin" (Optional[np.ndarray], paired to
        cbss_result's units), "roa_calib" (Optional[np.ndarray], per-unit
        rate of agreement at the calibration window), "preprocess" (bool),
        "fs" (int).
    """
    from adapt_decomp.cbss.config import CBSSConfig

    sim_data = _load_neuromotion(path_emg)
    cbss_result = _cbss_result_from_mat_decomp(path_decomp)
    cbss_config = CBSSConfig(
        preprocess_emg = preprocess,
        ext_fact = cbss_result.ext_fact,
        fs = int(sim_data["fs"]),
        ext_mode = "block",
        save_emg = False,
    )

    gt_source = sim_data if (path_gt is None or path_gt == path_emg) else _load_neuromotion(path_gt)
    spikes_gt = np.asarray(gt_source["spikes"])
    gt_full_bin, roa_calib = pair_ground_truth(
        spikes_gt, cbss_result.spikes, fs=int(sim_data["fs"]),
    )

    return {
        "cbss_result": cbss_result,
        "cbss_config": cbss_config,
        "emg": torch.from_numpy(np.asarray(sim_data["emg"], dtype=np.float32)),
        "gt_full_bin": gt_full_bin,
        "roa_calib": roa_calib,
        "preprocess": preprocess,
        "fs": int(sim_data["fs"]),
    }


# ------------------------------------------------------------------
# Loaders for dataset 2 (muniverse)
# ------------------------------------------------------------------

def _load_bin_spikes_muniverse(path: Union[str, Path], n_samples: int) -> np.ndarray:
    """Load ground-truth firings as binary array of spikes.

    Args:
        path (Union[str, Path]): Path to the .npz file (key "spikes").
        n_samples (int): Number of samples in the recording this ground
            truth must align to -- the densified array's row count.

    Returns:
        np.ndarray: Dense binary spike matrix with shape (n_samples, n_gt).
    """
    spikes_obj = np.load(path, allow_pickle=True)["spikes"]
    if spikes_obj.dtype != object:
        return np.asarray(spikes_obj)

    n_gt = len(spikes_obj)
    dense = np.zeros((n_samples, n_gt), dtype=np.float32)
    for gt_idx in range(n_gt):
        idx = np.asarray(spikes_obj[gt_idx]).astype(int)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        dense[idx, gt_idx] = 1.0
    return dense


# ------------------------------------------------------------------
# CBSSResult-based loader dispatch (calib_loader/emg_loader/gt_loader)
# ------------------------------------------------------------------

CalibLoaderName = Literal["class"]
EmgLoaderName = Literal["npz", "neuromotion"]
GtLoaderName = Literal["npz", "neuromotion"]


def load_calib(
    path_calib: Union[str, Path],
    path_calib_config: Optional[Union[str, Path]] = None,
    calib_loader: CalibLoaderName = "class",
) -> Tuple[CBSSResult, Optional["CBSSConfig"]]:
    """Load a calibration result and its config by calib_loader's mechanism.

    Args:
        path_calib (Union[str, Path]): Path to the calibration result.
        path_calib_config (Optional[Union[str, Path]], optional): Path to
            the calibration config, the sibling of path_calib from the same
            calibration run. Defaults to None, which skips loading a config
            (e.g. callers with no use for it, like load_pooled_cbss_memory).
        calib_loader (CalibLoaderName, optional): Which mechanism loads
            path_calib/path_calib_config -- "class" delegates to
            CBSSResult.load()/CBSSConfig.from_yaml(). Defaults to "class".

    Returns:
        Tuple[CBSSResult, Optional[CBSSConfig]]: The loaded calibration
        result, and its config (None if path_calib_config was None).

    Raises:
        ValueError: If calib_loader is not "class".
    """
    from adapt_decomp.cbss.config import CBSSConfig

    if calib_loader == "class":
        cbss_result = CBSSResult.load(path_calib)
        cbss_config = CBSSConfig.from_yaml(path_calib_config) if path_calib_config is not None else None
        return cbss_result, cbss_config
    raise ValueError(f"Unknown calib_loader: {calib_loader!r}. Expected 'class'.")


def load_emg(path_emg: Union[str, Path], emg_loader: EmgLoaderName = "npz") -> np.ndarray:
    """Load a full EMG recording by emg_loader's format.

    Args:
        path_emg (Union[str, Path]): Path to the EMG recording.
        emg_loader (EmgLoaderName, optional): On-disk format of path_emg.
            Defaults to "npz".

    Returns:
        np.ndarray: EMG data with shape (samples, channels), dtype float32.

    Raises:
        ValueError: If emg_loader is not "npz" or "neuromotion".
    """
    if emg_loader == "npz":
        return np.asarray(np.load(path_emg)["emg"], dtype=np.float32)
    if emg_loader == "neuromotion":
        return np.asarray(_load_neuromotion(path_emg)["emg"], dtype=np.float32)
    raise ValueError(f"Unknown emg_loader: {emg_loader!r}. Expected 'npz' or 'neuromotion'.")


def load_gt(
    path_gt: Union[str, Path], n_samples: int, gt_loader: GtLoaderName = "npz",
) -> np.ndarray:
    """Load a full ground-truth spike train by gt_loader's format.

    Args:
        path_gt (Union[str, Path]): Path to the ground-truth spikes.
        n_samples (int): Number of samples in the recording this ground
            truth must align to.
        gt_loader (GtLoaderName, optional): On-disk format of path_gt.
            Defaults to "npz".

    Returns:
        np.ndarray: Dense binary spike matrix with shape (n_samples, n_gt).

    Raises:
        ValueError: If gt_loader is not "npz" or "neuromotion".
    """
    if gt_loader == "npz":
        return _load_bin_spikes_muniverse(path_gt, n_samples)
    if gt_loader == "neuromotion":
        # n_samples unused -- neuromotion's spikes are already dense.
        return np.asarray(_load_neuromotion(path_gt)["spikes"], dtype=np.float32)
    raise ValueError(f"Unknown gt_loader: {gt_loader!r}. Expected 'npz' or 'neuromotion'.")


def _resolve_pool_loaders(
    dataset: Dict, defaults: Dict[str, str],
) -> Tuple[CalibLoaderName, EmgLoaderName, GtLoaderName]:
    """Resolve one dataset's calib_loader/emg_loader/gt_loader against pool defaults.

    Args:
        dataset (Dict): One dataset's data_config entry.
        defaults (Dict[str, str]): Pool-level "calib_loader"/"emg_loader"/
            "gt_loader" defaults.

    Returns:
        Tuple[CalibLoaderName, EmgLoaderName, GtLoaderName]: This
        dataset's resolved (calib_loader, emg_loader, gt_loader).
    """
    return (
        dataset.get("calib_loader", defaults["calib_loader"]),
        dataset.get("emg_loader", defaults["emg_loader"]),
        dataset.get("gt_loader", defaults["gt_loader"]),
    )


# ------------------------------------------------------------------
# Pooled dataset dataclasses
# ------------------------------------------------------------------

@dataclass
class PooledDatasetMemory:
    """One dataset's calibration and optional ground truth for a hyperparameter search --
    whether the pool has one dataset or many (optimize_adapt_decomp_pooled_memory).

    Attributes:
        emg (torch.Tensor): Online EMG to decompose, with shape
            (samples, channels).
        calibration (CBSSResult): This dataset's calibration result, with
            calibration.emg set (see AdaptDecomp.from_calibration). Already
            narrowed to GT-matched units when gt_paired_bin is set.
        cbss_config (CBSSConfig): The CBSSConfig that produced calibration.
        preprocess (bool, optional): Whether to preprocess emg before
            extension, for every dataset's AdaptDecomp. Defaults to True.
        gt_paired_bin (Optional[np.ndarray]): Ground-truth binary spike
            train for the full recording, matched and reordered to
            calibration's units (see CBSSResult.select_supervised), with
            shape (samples, M). Required when
            optimize_adapt_decomp_pooled_memory(compute_roa=True) is used;
            ignored otherwise. Defaults to None.
    """

    emg: torch.Tensor
    calibration: CBSSResult
    cbss_config: "CBSSConfig"
    preprocess: bool = True
    gt_paired_bin: Optional[np.ndarray] = None

    def resolve(
        self,
    ) -> Tuple[torch.Tensor, CBSSResult, "CBSSConfig", bool, Optional[np.ndarray]]:
        """Return this dataset's already-loaded inputs, unchanged.

        Trivial passthrough -- matches PooledDatasetDisk.resolve()'s return
        shape so both can be called identically from a pooled optimizer's
        trial loop.

        Returns:
            Tuple[torch.Tensor, CBSSResult, CBSSConfig, bool, Optional[np.ndarray]]:
            (emg, calibration, cbss_config, preprocess, gt_paired_bin).
        """
        return self.emg, self.calibration, self.cbss_config, self.preprocess, self.gt_paired_bin


@dataclass
class PooledDatasetDisk:
    """One dataset's on-disk calibration paths -- the lazy counterpart to
    PooledDatasetMemory, for optimize_adapt_decomp_pooled_disk().

    Attributes:
        path_calib (Path): Calibration result (e.g. a CBSSResult pickle
            written by CBSSResult.save()), with .emg set.
        path_calib_config (Path): Calibration config (e.g. a CBSSConfig
            YAML written by CBSSConfig.to_yaml()) -- the sibling of
            path_calib, from the same calibration run.
        path_emg (Path): .npz file, key "emg", with shape
            (samples, channels) -- the full online recording for this
            dataset.
        calib_loader (CalibLoaderName, optional): Mechanism loading
            path_calib/path_calib_config, resolved via load_calib() each
            trial. Defaults to "class".
        emg_loader (EmgLoaderName, optional): On-disk format of path_emg,
            resolved via load_emg() each trial. Defaults to "npz".
        gt_loader (GtLoaderName, optional): On-disk format of path_gt,
            resolved via load_gt() each trial. Defaults to "npz".
        preprocess (bool, optional): Whether to preprocess emg before
            extension. Defaults to True.
        path_gt (Optional[Path]): Ground-truth spikes for the full
            recording, matched to calibration's units fresh every trial --
            see optimize_adapt_decomp_pooled_disk. Required when
            optimize_adapt_decomp_pooled_disk(compute_roa=True) is used;
            ignored otherwise. Defaults to None.
        fs (Optional[int]): Sampling frequency override for matching
            path_gt. Defaults to None, which uses the loaded calibration's
            own fs each trial.
    """

    path_calib: Path
    path_calib_config: Path
    path_emg: Path
    calib_loader: CalibLoaderName = "class"
    emg_loader: EmgLoaderName = "npz"
    gt_loader: GtLoaderName = "npz"
    preprocess: bool = True
    path_gt: Optional[Path] = None
    fs: Optional[int] = None

    def resolve(
        self,
    ) -> Tuple[np.ndarray, CBSSResult, "CBSSConfig", bool, Optional[np.ndarray]]:
        """Load this dataset's calibration/EMG/ground truth fresh from disk.

        Matches ground truth to calibration units via
        CBSSResult.select_supervised() when path_gt is set, narrowing
        calibration to GT-matched units in the process. Nothing is cached --
        call this once per trial.

        Returns:
            Tuple[np.ndarray, CBSSResult, CBSSConfig, bool, Optional[np.ndarray]]:
            (emg, calibration, cbss_config, preprocess, gt_paired_bin), matching
            PooledDatasetMemory.resolve()'s shape.
        """
        calibration, cbss_config = load_calib(self.path_calib, self.path_calib_config, self.calib_loader)
        emg = load_emg(self.path_emg, self.emg_loader)

        gt_paired_bin = None
        if self.path_gt is not None:
            spikes_gt_full = load_gt(self.path_gt, emg.shape[0], self.gt_loader)
            fs = self.fs if self.fs is not None else int(calibration.fs)
            calibration = calibration.select_supervised(
                spikes_gt_full[: calibration.sources.shape[0]], fs=fs,
            )
            gt_paired_bin = spikes_gt_full[:, calibration.gt_matched_indices]

        return emg, calibration, cbss_config, self.preprocess, gt_paired_bin


def load_pooled_cbss_memory(data_config: Dict) -> Dict[str, Any]:
    """Load a pooled data_config into optimize_adapt_decomp_pooled_memory()'s pool= input.

    data_config format::

        name: <label>
        root: <base dir; every path below is relative to it>
        loader: load_pooled_cbss_memory
        preprocess: true            # default for every dataset, overridable per-dataset
        calib_loader: class         # default for every dataset, overridable per-dataset
        emg_loader: npz             # default for every dataset, overridable per-dataset
        gt_loader: npz              # default for every dataset, overridable per-dataset
        datasets:
          - name: <pool key, e.g. a dataset/recording label>
            path_emg: <emg_loader format, e.g. .npz key "emg", shape (samples, channels)>
            path_calib: <calib_loader format, e.g. CBSSResult pickle written by CBSSResult.save()>
            path_calib_config: <CBSSConfig YAML written by CBSSConfig.to_yaml()
                                 -- the sibling of path_calib, same calibration run>
            path_gt: <optional -- gt_loader format, e.g. .npz key "spikes">
            preprocess: <optional per-dataset override>
            calib_loader: <optional per-dataset override>
            emg_loader: <optional per-dataset override>
            gt_loader: <optional per-dataset override>
            fs: <optional -- defaults to CBSSResult.fs (needs timestamps set)>

    Ground truth (if given) is matched to each dataset's calibration via
    CBSSResult.select_supervised() -- the calibration itself (and therefore
    the resulting PooledDatasetMemory) is narrowed to only GT-matched
    units.

    Args:
        data_config (Dict): Parsed pooled data_config YAML, as above.

    Returns:
        Dict[str, Any]: Dataset name -> PooledDatasetMemory, ready to
        pass directly as optimize_adapt_decomp_pooled_memory(pool=...).
    """
    root = Path(data_config.get("root", "."))
    default_preprocess = data_config.get("preprocess", True)
    defaults = {
        "calib_loader": data_config.get("calib_loader", "class"),
        "emg_loader": data_config.get("emg_loader", "npz"),
        "gt_loader": data_config.get("gt_loader", "npz"),
    }

    pool: Dict[str, PooledDatasetMemory] = {}
    for dataset in data_config["datasets"]:
        calib_loader, emg_loader, gt_loader = _resolve_pool_loaders(dataset, defaults)
        cbss_result, cbss_config = load_calib(
            root / dataset["path_calib"], root / dataset["path_calib_config"], calib_loader,
        )
        emg_full = load_emg(root / dataset["path_emg"], emg_loader)

        gt_paired_bin = None
        if dataset.get("path_gt") is not None:
            spikes_gt_full = load_gt(root / dataset["path_gt"], emg_full.shape[0], gt_loader)
            fs = int(dataset["fs"]) if "fs" in dataset else int(cbss_result.fs)
            cbss_result = cbss_result.select_supervised(
                spikes_gt_full[: cbss_result.sources.shape[0]], fs=fs,
            )
            gt_paired_bin = spikes_gt_full[:, cbss_result.gt_matched_indices]

        pool[dataset["name"]] = PooledDatasetMemory(
            emg=torch.from_numpy(emg_full),
            calibration=cbss_result,
            cbss_config=cbss_config,
            preprocess=dataset.get("preprocess", default_preprocess),
            gt_paired_bin=gt_paired_bin,
        )
    return pool


def load_pooled_cbss_disk(data_config: Dict) -> Dict[str, Any]:
    """Load a pooled data_config into optimize_adapt_decomp_pooled_disk()'s pool= input.

    The lazy counterpart to load_pooled_cbss_memory: builds a
    Dict[str, PooledDatasetDisk] of on-disk paths and loader names,
    touching no files at all -- every path is resolved and loaded fresh per
    trial by optimize_adapt_decomp_pooled_disk().

    data_config format:: (same shape as load_pooled_cbss_memory's)

        name: <label>
        root: <base dir; every path below is relative to it>
        loader: load_pooled_cbss_disk
        preprocess: true            # default for every dataset, overridable per-dataset
        calib_loader: class         # default for every dataset, overridable per-dataset
        emg_loader: npz             # default for every dataset, overridable per-dataset
        gt_loader: npz              # default for every dataset, overridable per-dataset
        datasets:
          - name: <pool key, e.g. a dataset/recording label>
            path_emg: <emg_loader format, e.g. .npz key "emg", shape (samples, channels)>
            path_calib: <calib_loader format, e.g. CBSSResult pickle written by CBSSResult.save()>
            path_calib_config: <CBSSConfig YAML written by CBSSConfig.to_yaml()
                                 -- the sibling of path_calib, same calibration run>
            path_gt: <optional -- gt_loader format, e.g. .npz key "spikes">
            preprocess: <optional per-dataset override>
            calib_loader: <optional per-dataset override>
            emg_loader: <optional per-dataset override>
            gt_loader: <optional per-dataset override>
            fs: <optional -- defaults to the loaded calibration's own fs each trial>

    Args:
        data_config (Dict): Parsed pooled data_config YAML, as above.

    Returns:
        Dict[str, Any]: Dataset name -> PooledDatasetDisk, ready to
        pass directly as optimize_adapt_decomp_pooled_disk(pool=...).
    """
    root = Path(data_config.get("root", "."))
    default_preprocess = data_config.get("preprocess", True)
    defaults = {
        "calib_loader": data_config.get("calib_loader", "class"),
        "emg_loader": data_config.get("emg_loader", "npz"),
        "gt_loader": data_config.get("gt_loader", "npz"),
    }

    pool: Dict[str, PooledDatasetDisk] = {}
    for dataset in data_config["datasets"]:
        calib_loader, emg_loader, gt_loader = _resolve_pool_loaders(dataset, defaults)
        path_gt = dataset.get("path_gt")
        pool[dataset["name"]] = PooledDatasetDisk(
            path_calib=root / dataset["path_calib"],
            path_calib_config=root / dataset["path_calib_config"],
            path_emg=root / dataset["path_emg"],
            calib_loader=calib_loader,
            emg_loader=emg_loader,
            gt_loader=gt_loader,
            preprocess=dataset.get("preprocess", default_preprocess),
            path_gt=root / path_gt if path_gt is not None else None,
            fs=int(dataset["fs"]) if "fs" in dataset else None,
        )
    return pool


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_data(data_config: Dict) -> Dict:
    """Dispatch a data_config to its loader, by data_config["loader"].

    Args:
        data_config (Dict): Parsed data_config YAML. Must contain a
            "loader" key naming one of the loaders below. "root" (optional,
            default ".") is prepended to every path field.

    Returns:
        Dict: load_example()'s return shape for loader == "load_example";
        load_pooled_cbss_memory()'s return shape (Dict[str,
        PooledDatasetMemory]) for loader == "load_pooled_cbss_memory" --
        one dataset or many alike; load_pooled_cbss_disk()'s return shape
        (Dict[str, PooledDatasetDisk]) for loader == "load_pooled_cbss_disk".

    Raises:
        ValueError: If data_config["loader"] is not a known loader name.
    """
    loader = data_config["loader"]
    if loader == "load_example":
        root = Path(data_config.get("root", "."))
        path_gt = data_config.get("path_gt")
        data = load_example(
            str(root / data_config["path_emg"]),
            str(root / data_config["path_decomp"]),
            data_config["preprocess"],
            path_gt=str(root / path_gt) if path_gt is not None else None,
        )
    elif loader == "load_pooled_cbss_memory":
        data = load_pooled_cbss_memory(data_config)
    elif loader == "load_pooled_cbss_disk":
        data = load_pooled_cbss_disk(data_config)
    else:
        raise ValueError(f"Unknown data loader: {loader}")
    return data
