"""Decomposition: template base class shared by every calibration method (CBSS, SCD, ...).

decompose() runs the shared pipeline (preprocess, extend and centre, PCA, whiten), calls
the method's _extract() to find the units in the whitened EMG, then finalises the result
(spike trains, duplicate removal, properties, selection). apply() is fully shared.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import copy
from functools import wraps
from typing import Callable, ClassVar, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch
from loguru import logger

from adapt_decomp.decomposition.config import DecompositionConfig
from adapt_decomp.decomposition.result import DecompositionResult
from adapt_decomp.preprocessing import (
    extend_data,
    filter_kwargs,
    pca_reduction,
    preprocess_emg,
    select_channels,
    whiten,
)
from adapt_decomp.spikes import detect_spikes, find_duplicates, spikes_dict_to_binary
from adapt_decomp.spikes.metrics import (
    emg_to_ch_array,
    get_coefficient_of_variation,
    get_discharge_rate,
    get_muaps,
    get_pulse_to_noise_ratio,
)
from adapt_decomp.utils.system import deterministic_algorithms, set_cublas_workspace

# method name -> Decomposition subclass, filled by Decomposition.__init_subclass__
_DECOMPOSITIONS: Dict[str, Type["Decomposition"]] = {}

set_cublas_workspace()  # on import, before any cuBLAS call

_Method = TypeVar("_Method", bound=Callable)


def _deterministic_when_seeded_on_cuda(method: _Method) -> _Method:
    """Run a Decomposition method with deterministic CUDA kernels when it is seeded on CUDA.

    CPU runs are already deterministic and stay untouched.

    Args:
        method (_Method): A Decomposition method.

    Returns:
        _Method: The wrapped method.
    """

    @wraps(method)
    def wrapper(self: "Decomposition", *args, **kwargs):
        seeded_cuda = self.config.random_seed is not None and self._device.type == "cuda"
        with deterministic_algorithms(seeded_cuda):
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class Decomposition(ABC):
    """Calibration template: shared pipeline around a method-specific unit search.

    Subclasses set config_class (a registered DecompositionConfig subclass) and
    implement _extract().
    """

    config_class: ClassVar[Type[DecompositionConfig]] = DecompositionConfig

    def __init_subclass__(cls, **kwargs) -> None:
        """Register a subclass under its config's method, for from_config()."""
        super().__init_subclass__(**kwargs)
        if cls.config_class.method:
            _DECOMPOSITIONS[cls.config_class.method] = cls

    @staticmethod
    def from_config(config: DecompositionConfig) -> "Decomposition":
        """Build the decomposition that config's method names.

        Args:
            config (DecompositionConfig): A registered config subclass (e.g.
                CBSSConfig, SCDConfig).

        Returns:
            Decomposition: e.g. CBSS(config) for a CBSSConfig.

        Raises:
            ValueError: If no decomposition is registered for config's method.
        """
        if config.method not in _DECOMPOSITIONS:
            raise ValueError(
                f"No decomposition registered for method {config.method!r}; "
                f"known: {sorted(_DECOMPOSITIONS)}."
            )
        return _DECOMPOSITIONS[config.method](config)

    def __init__(self, config: DecompositionConfig) -> None:
        self.config = config
        self._device = torch.device(config.device or "cpu")
        self._dtype = config.dtype
        self._rng = torch.Generator(device="cpu")
        if config.random_seed is not None:
            self._rng.manual_seed(config.random_seed)
            os.environ["PYTHONHASHSEED"] = str(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.random_seed)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_emg(self, emg: np.ndarray, fs: float) -> torch.Tensor:
        """Bandpass + optional notch + zero-mean + optional bad-channel handling."""
        emg_f = preprocess_emg(emg, fs, **filter_kwargs(self.config))
        emg_f = emg_f - emg_f.mean(axis=0, keepdims=True)
        emg_f = select_channels(
            emg_f, self.config.ch_mask, self.config.ch_map, self.config.replace_bad_channels
        )
        return torch.from_numpy(emg_f.astype(np.float32)).to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_timestamps(self, timestamps, n_samples: int) -> torch.Tensor:
        if timestamps is None:
            return torch.arange(n_samples, dtype=self._dtype, device=self._device) / self.config.fs
        ts = torch.as_tensor(timestamps, dtype=self._dtype, device=self._device)
        if ts.ndim != 1 or ts.shape[0] != n_samples:
            raise ValueError("timestamps must be a 1-D array matching the number of EMG samples.")
        return ts

    def _cov_isi_from_idx(self, idx: torch.Tensor, timestamps: torch.Tensor) -> float:
        n = timestamps.shape[0]
        bin_src = torch.zeros(n, 1, dtype=torch.int32, device=self._device)
        valid = idx[(idx >= 0) & (idx < n)]
        bin_src[valid, 0] = 1
        return get_coefficient_of_variation(bin_src, timestamps, None)[0].item()

    def _compute_muaps(self, spike_trains: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        """Compute MUAPs from spike trains and EMG.

        Args:
            spike_trains (torch.Tensor): Binary spike train with shape
                (samples, n_mu).
            emg (torch.Tensor): Preprocessed EMG with shape (samples,
                raw_channels) or (samples, good_channels) if channels were
                dropped.

        Returns:
            torch.Tensor: MUAPs with shape (n_mu, rows, cols, win).

        Note:
            ch_map defaults to a line array if not provided. Bad channels are
            replaced with zeros in the returned MUAPs.
        """
        ch_mask = self.config.ch_mask
        n_raw = ch_mask.shape[0] if ch_mask is not None else emg.shape[1]

        if emg.shape[1] != n_raw:
            ch_mask_t = torch.as_tensor(ch_mask, device=emg.device)
            emg_full = torch.zeros(emg.shape[0], n_raw, dtype=emg.dtype, device=emg.device)
            emg_full[:, ch_mask_t] = emg
            emg = emg_full

        ch_map = self.config.ch_map
        if ch_map is None:
            ch_map = np.arange(n_raw).reshape(1, n_raw)

        half_win = round(25 / 2 / 1000 * self.config.fs)
        emg_ch = emg_to_ch_array(emg.cpu(), ch_map)
        return get_muaps(spike_trains.cpu(), emg_ch, half_win)

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(dim: int) -> Dict:
        """Per-unit fields of a decomposition that found no units.

        Args:
            dim (int): Whitened dimension.

        Returns:
            Dict: Empty per-unit fields, sep_vectors with shape (dim, 0).
        """
        return {
            "sources": torch.zeros((0, 0)),
            "spikes_dict": {},
            "spikes": torch.zeros((0, 0), dtype=torch.int32),
            "sil": torch.empty(0, dtype=torch.float32),
            "cov_isi": torch.empty(0, dtype=torch.float32),
            "sep_vectors": torch.zeros((dim, 0)),
            "spikes_centr": torch.empty(0, dtype=torch.float32),
            "base_centr": torch.empty(0, dtype=torch.float32),
        }

    @staticmethod
    def _keep_units(result: Dict, keep: np.ndarray) -> Dict:
        """Restrict a result dict to the given units, renumbered in their order.

        Args:
            result (Dict): Results from _extract(), with "spikes".
            keep (np.ndarray): Sorted indices of the units to keep.

        Returns:
            Dict: result with every per-unit field sliced to keep.
        """
        idx = torch.as_tensor(keep, dtype=torch.long)
        spikes_dict = result["spikes_dict"]
        result["spikes_dict"] = {new: spikes_dict[int(old)] for new, old in enumerate(keep)}
        for key in ("sources", "spikes", "sep_vectors"):
            result[key] = result[key][:, idx]
        for key in ("sil", "cov_isi", "spikes_centr", "base_centr"):
            result[key] = result[key][idx]
        # Optional native fields (set by methods such as SCD)
        for key in ("sep_vectors_native", "sources_native"):
            if result.get(key) is not None:
                result[key] = result[key][:, idx]
        for key in ("native_roa", "contrast_exp"):
            if result.get(key) is not None:
                result[key] = result[key][idx]
        native = result.get("spikes_dict_native")
        if native is not None:
            result["spikes_dict_native"] = {new: native[int(old)] for new, old in enumerate(keep)}
        return result

    @staticmethod
    def _to_numpy(result: Dict) -> Dict:
        for key in (
            "sources",
            "spikes",
            "sil",
            "cov_isi",
            "sep_vectors",
            "spikes_centr",
            "base_centr",
            "whitening",
            "extension_mean",
            "pnr",
            "dr",
            "muaps",
            "emg",
            "timestamps",
            "sep_vectors_native",
            "sources_native",
            "native_roa",
            "contrast_exp",
        ):
            value = result.get(key)
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
        return result

    # ------------------------------------------------------------------
    # Method-specific unit search
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract(self, emg_wh: torch.Tensor, timestamps: torch.Tensor) -> Dict:
        """Find the units in the whitened EMG.

        Args:
            emg_wh (torch.Tensor): Whitened, extended EMG with shape (samples, dim).
            timestamps (torch.Tensor): Sample times, in s, with shape (samples,).

        Returns:
            Dict: Per-unit "sources" (samples, n_mu), "spikes_dict", "sil",
            "cov_isi", "sep_vectors" (dim, n_mu), "spikes_centr" and
            "base_centr", as torch tensors on the CPU, optionally with the
            native fields "sep_vectors_native", "sources_native", "spikes_dict_native" and
            "native_roa", and "contrast_exp"; _empty_result(dim) when no unit is found.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_deterministic_when_seeded_on_cuda
    def decompose(
        self,
        emg: Union[np.ndarray, torch.Tensor],
        timestamps: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> DecompositionResult:
        """Decompose HD-EMG with this method.

        Args:
            emg:        [T, C] EMG (float32 recommended).
            timestamps: Optional [T] sample times in seconds.

        Raises:
            ValueError: If self.config.selection is not one of
                "unsupervised"/"supervised"/None, or if the selected filter
                keeps no units (raised by DecompositionResult.select_unsupervised()/
                select_supervised()).

        Returns:
            DecompositionResult with source signals, spike trains, quality metrics,
            and fitted model parameters (whitening, sep_vectors, centroids).
        """
        emg_np = emg.cpu().numpy() if isinstance(emg, torch.Tensor) else np.asarray(emg)
        emg_t = (
            self._preprocess_emg(emg_np, self.config.fs)
            if self.config.preprocess_emg
            else torch.from_numpy(emg_np.astype(np.float32)).to(
                device=self._device, dtype=self._dtype
            )
        )
        timestamps_t = self._make_timestamps(timestamps, emg_t.shape[0])

        # 1. Extend
        emg_ext = extend_data(emg_t, self.config.ext_fact, ext_mode=self.config.ext_mode)
        extension_mean = emg_ext.mean(dim=0, keepdim=True)
        emg_ext = emg_ext - extension_mean
        emg_ext[: self.config.ext_fact, :] = 0
        emg_ext[-self.config.ext_fact :, :] = 0

        # 2. PCA
        emg_for_wh, pca_model = pca_reduction(emg_ext, self.config.n_components)

        # 3. Whitening
        emg_wh, W = whiten(
            emg_for_wh, self.config.whitening_method, self.config.regularization, self.config.eps
        )

        dict_results: Dict = {
            "whitening": W.cpu(),
            "extension_mean": extension_mean.cpu(),
            "pca_components": pca_model.components_.astype(np.float32) if pca_model else None,
            "pca_mean": pca_model.mean_.astype(np.float32) if pca_model else None,
        }

        # 4. Method-specific unit search
        dict_results.update(self._extract(emg_wh, timestamps_t))

        # 5. Binary spike matrix
        sources = dict_results.get("sources")
        spikes_dict = dict_results.get("spikes_dict", {})
        if sources is not None and spikes_dict:
            dict_results["spikes"] = spikes_dict_to_binary(
                spikes_dict, sources.shape[0], device=self._device
            ).cpu()

        # 6. Duplicate removal
        if self.config.run_duplicate_removal and len(spikes_dict) > 1:
            keep_by = self.config.duplicate_keep_by
            quality = np.asarray(dict_results[keep_by], dtype=float)
            keep = find_duplicates(
                dict_results["spikes"].numpy(),
                -quality if keep_by == "cov_isi" else quality,
                fs=self.config.fs,
                metric=self.config.duplicate_metric,
                threshold=self.config.roa_th,
                tol_spike_ms=self.config.duplicate_tol_spike_ms,
                max_lag_ms=self.config.duplicate_tol_train_ms,
            )
            if self.config.verbose and len(keep) < len(spikes_dict):
                logger.debug(f"Removed {len(spikes_dict) - len(keep)} duplicate units")
            dict_results = self._keep_units(dict_results, keep)

        # 7. Properties
        if self.config.compute_properties:
            sources = dict_results.get("sources")
            if sources is not None and dict_results.get("spikes_dict"):
                sources_d = torch.as_tensor(sources, dtype=self._dtype, device=self._device)
                spike_trains = dict_results["spikes"].to(self._device)
                dict_results["pnr"] = get_pulse_to_noise_ratio(
                    spike_trains, sources_d, self.config.ext_fact, self.config.spike_min_dist
                ).cpu()
                dict_results["dr"] = get_discharge_rate(spike_trains, timestamps_t).cpu()
                dict_results["muaps"] = self._compute_muaps(spike_trains, emg_t).cpu()

        dict_results["emg"] = emg_t.cpu() if self.config.save_emg else None
        dict_results["timestamps"] = timestamps_t.cpu() if self.config.save_emg else None
        d = self._to_numpy(dict_results)

        result = DecompositionResult(
            sources=d["sources"],
            spikes=d["spikes"],
            spikes_dict=d["spikes_dict"],
            sil=d["sil"],
            cov_isi=d["cov_isi"],
            sep_vectors=d["sep_vectors"],
            whitening=d["whitening"],
            extension_mean=d["extension_mean"],
            spikes_centr=d["spikes_centr"],
            base_centr=d["base_centr"],
            ext_fact=self.config.ext_fact,
            pca_components=d["pca_components"],
            pca_mean=d["pca_mean"],
            pnr=d.get("pnr"),
            dr=d.get("dr"),
            muaps=d.get("muaps"),
            emg=d.get("emg"),
            timestamps=d.get("timestamps"),
            sep_vectors_native=d.get("sep_vectors_native"),
            sources_native=d.get("sources_native"),
            spikes_dict_native=d.get("spikes_dict_native"),
            native_roa=d.get("native_roa"),
            contrast_exp=d.get("contrast_exp"),
            config=copy(self.config),
        )

        # 8. Optional post-hoc unit selection (config-driven, see DecompositionConfig.selection).
        # fs is taken from self.config.fs rather than result.fs, so supervised
        # selection's fs auto-inference doesn't depend on save_emg/timestamps.
        if self.config.selection == "unsupervised":
            result = result.select_unsupervised(**(self.config.selection_kwargs or {}))
        elif self.config.selection == "supervised":
            kwargs = dict(self.config.selection_kwargs or {})
            kwargs.setdefault("fs", self.config.fs)
            result = result.select_supervised(**kwargs)
        elif self.config.selection is not None:
            raise ValueError(
                f"Unknown {type(self.config).__name__}.selection: {self.config.selection!r}. "
                "Expected 'unsupervised', 'supervised', or None."
            )
        return result

    @_deterministic_when_seeded_on_cuda
    def apply(
        self,
        emg: Union[np.ndarray, torch.Tensor],
        result: DecompositionResult,
        timestamps: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> DecompositionResult:
        """Apply stored decomposition parameters to a new EMG signal.

        Reuses extension mean, optional PCA, whitening matrix, and separation vectors
        from result. Spike detection is re-run on the projected sources.
        """
        emg_np = emg.cpu().numpy() if isinstance(emg, torch.Tensor) else np.asarray(emg)
        emg_t = (
            self._preprocess_emg(emg_np, self.config.fs)
            if self.config.preprocess_emg
            else torch.from_numpy(emg_np.astype(np.float32)).to(
                device=self._device, dtype=self._dtype
            )
        )
        emg_original = emg_t
        timestamps_t = self._make_timestamps(timestamps, emg_t.shape[0])

        emg_ext = extend_data(emg_t, self.config.ext_fact, ext_mode=self.config.ext_mode)
        ext_mean = torch.from_numpy(result.extension_mean).to(
            device=self._device, dtype=self._dtype
        )
        emg_ext = emg_ext - ext_mean
        emg_ext[: self.config.ext_fact, :] = 0
        emg_ext[-self.config.ext_fact :, :] = 0

        if result.pca_components is not None:
            pca_mean = torch.from_numpy(result.pca_mean).to(device=self._device, dtype=self._dtype)
            pca_comps = torch.from_numpy(result.pca_components).to(
                device=self._device, dtype=self._dtype
            )
            emg_pca = (emg_ext - pca_mean) @ pca_comps.T
        else:
            emg_pca = emg_ext

        W = torch.from_numpy(result.whitening).to(device=self._device, dtype=self._dtype)
        emg_wh = emg_pca @ W.T
        sep_vectors = torch.from_numpy(result.sep_vectors).to(
            device=self._device, dtype=self._dtype
        )
        n_mu = sep_vectors.shape[1]

        if n_mu == 0:
            return DecompositionResult(
                sources=np.zeros((emg_wh.shape[0], 0), dtype=np.float32),
                spikes=np.zeros((emg_wh.shape[0], 0), dtype=np.int32),
                spikes_dict={},
                sil=np.empty(0, dtype=np.float32),
                cov_isi=np.empty(0, dtype=np.float32),
                sep_vectors=result.sep_vectors,
                whitening=result.whitening,
                extension_mean=result.extension_mean,
                spikes_centr=np.empty(0, dtype=np.float32),
                base_centr=np.empty(0, dtype=np.float32),
                ext_fact=result.ext_fact,
                pca_components=result.pca_components,
                pca_mean=result.pca_mean,
                emg=emg_original.cpu().numpy() if self.config.save_emg else None,
                timestamps=timestamps_t.cpu().numpy() if self.config.save_emg else None,
                config=copy(self.config),
            )

        sources_t = emg_wh @ sep_vectors
        spikes_dict: Dict[int, np.ndarray] = {}
        sil_vals: List[float] = []
        cov_vals: List[float] = []

        for i in range(n_mu):
            sc = torch.tensor(float(result.spikes_centr[i]), dtype=self._dtype, device=self._device)
            bc = torch.tensor(float(result.base_centr[i]), dtype=self._dtype, device=self._device)
            spike_idx, _, _, sil = detect_spikes(
                sources_t[:, i],
                self.config.spike_min_dist,
                spike_centroid=sc,
                base_centroid=bc,
                peak_power=self.config.spike_det_exp,
                compute_sil=True,
            )
            cov_isi = self._cov_isi_from_idx(spike_idx, timestamps_t)
            spikes_dict[i] = spike_idx.cpu().numpy()
            sil_vals.append(sil)
            cov_vals.append(cov_isi)

        sources_np = sources_t.detach().cpu().numpy()
        spike_trains_dev = spikes_dict_to_binary(spikes_dict, emg_wh.shape[0], device=self._device)

        applied = DecompositionResult(
            sources=sources_np,
            spikes=spike_trains_dev.cpu().numpy(),
            spikes_dict=spikes_dict,
            sil=np.array(sil_vals, dtype=np.float32),
            cov_isi=np.array(cov_vals, dtype=np.float32),
            sep_vectors=result.sep_vectors,
            whitening=result.whitening,
            extension_mean=result.extension_mean,
            spikes_centr=result.spikes_centr,
            base_centr=result.base_centr,
            ext_fact=result.ext_fact,
            pca_components=result.pca_components,
            pca_mean=result.pca_mean,
            emg=emg_original.cpu().numpy() if self.config.save_emg else None,
            timestamps=timestamps_t.cpu().numpy() if self.config.save_emg else None,
            config=copy(self.config),
        )

        if self.config.compute_properties:
            sources_d = torch.from_numpy(sources_np).to(device=self._device, dtype=self._dtype)
            applied.pnr = (
                get_pulse_to_noise_ratio(
                    spike_trains_dev, sources_d, self.config.ext_fact, self.config.spike_min_dist
                )
                .cpu()
                .numpy()
            )
            applied.dr = get_discharge_rate(spike_trains_dev, timestamps_t).cpu().numpy()
            applied.muaps = self._compute_muaps(spike_trains_dev, emg_original).cpu().numpy()

        return applied
