"""DecompositionResult dataclass."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import h5py
import numpy as np
import torch
import yaml

from adapt_decomp import __version__
from adapt_decomp.utils.utils import read_h5, write_h5

if TYPE_CHECKING:
    from adapt_decomp.decomposition.config import DecompositionConfig

_FORMAT = "DecompositionResult"
_LEGACY_FORMAT = "CBSSResult"  # files saved before the rename


@dataclass
class DecompositionResult:
    """Output of Decomposition.decompose() and apply(). All array fields are CPU numpy arrays.

    Below, n_mu is the number of units and dim the whitened dimension (channels times
    ext_fact, or n_components with PCA).

    Attributes:
        sources (np.ndarray): Source of each unit with shape (samples, n_mu).
        spikes (np.ndarray): Binary spike trains with shape (samples, n_mu), int32.
        spikes_dict (Dict[int, np.ndarray]): Spike sample indices of each unit.
        sep_vectors (np.ndarray): Separation vectors with shape (dim, n_mu).
        whitening (np.ndarray): Whitening matrix with shape (dim, dim).
        extension_mean (np.ndarray): Mean of the extended EMG with shape
            (1, channels * ext_fact), subtracted before whitening.
        spikes_centr (np.ndarray): Spike centroid of each unit's source with shape
            (n_mu,).
        base_centr (np.ndarray): Baseline centroid of each unit's source with shape
            (n_mu,).
        sil (np.ndarray): Silhouette of each unit with shape (n_mu,).
        cov_isi (np.ndarray): Coefficient of variation of each unit's inter-spike
            intervals with shape (n_mu,), as a fraction.
        ext_fact (int): Extension factor used to build sep_vectors and whitening.
        pca_components (Optional[np.ndarray]): PCA components with shape
            (n_components, channels * ext_fact). None without PCA.
        pca_mean (Optional[np.ndarray]): PCA mean with shape (channels * ext_fact,).
            None without PCA.
        pnr (Optional[np.ndarray]): Pulse-to-noise ratio of each unit, in dB, with
            shape (n_mu,). Set when DecompositionConfig.compute_properties is True.
        dr (Optional[np.ndarray]): Mean discharge rate of each unit, in Hz, with shape
            (n_mu,). Set when DecompositionConfig.compute_properties is True.
        muaps (Optional[np.ndarray]): Spike-triggered average MUAPs with shape
            (n_mu, rows, cols, window). Set when DecompositionConfig.compute_properties is True.
        emg (Optional[np.ndarray]): Calibration EMG after preprocessing, before
            extension, with shape (samples, channels). Set when DecompositionConfig.save_emg is
            True; required by AdaptDecomp.from_calibration().
        timestamps (Optional[np.ndarray]): Time of each calibration sample, in s, with
            shape (samples,). Set when DecompositionConfig.save_emg is True.
        gt_matched_indices (Optional[np.ndarray]): Ground-truth unit tracked by each
            unit with shape (n_mu,). Set by select_supervised().
        roa (Optional[np.ndarray]): Rate of agreement of each unit with its
            ground-truth unit with shape (n_mu,), as a fraction. Set by
            select_supervised().
        sep_vectors_native (Optional[np.ndarray]): Separation vectors as the method
            itself output them, with shape (dim, n_mu), e.g. SCD's filters, applied to
            its peeled-off residual. Diagnostic only: adaptation uses sep_vectors. None
            for CBSS.
        sources_native (Optional[np.ndarray]): Sources as the method itself output
            them, with shape (samples, n_mu), e.g. SCD's from the peeled-off residual.
            Diagnostic only: adaptation uses sources. None for CBSS.
        spikes_dict_native (Optional[Dict[int, np.ndarray]]): Spike sample indices of
            each unit as the method itself detected them. None for CBSS.
        native_roa (Optional[np.ndarray]): Rate of agreement of each unit's native
            spikes with its spikes, with shape (n_mu,), as a fraction: how little the
            unit depends on the method's own processing (e.g. SCD's peel-off). None
            for CBSS.
        contrast_exp (Optional[np.ndarray]): Exponent e of each unit's contrast G ∝ |u|^e,
            with shape (n_mu,): optimised per unit by SCD, the config's fixed one for CBSS
            (3 for "square", 4 for "cube", contrast_exp for "smooth_abs"). None for
            contrasts that are not a power (CBSS "logcosh").
        config (Optional[DecompositionConfig]): The config that produced this result. Set by
            Decomposition.decompose() and apply(); None for legacy files.
    """

    sources: np.ndarray
    spikes: np.ndarray
    spikes_dict: Dict[int, np.ndarray]
    sep_vectors: np.ndarray
    whitening: np.ndarray
    extension_mean: np.ndarray
    spikes_centr: np.ndarray
    base_centr: np.ndarray
    sil: np.ndarray
    cov_isi: np.ndarray
    ext_fact: int
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    pnr: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    muaps: Optional[np.ndarray] = None
    emg: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    gt_matched_indices: Optional[np.ndarray] = None
    roa: Optional[np.ndarray] = None
    sep_vectors_native: Optional[np.ndarray] = None
    sources_native: Optional[np.ndarray] = None
    spikes_dict_native: Optional[Dict[int, np.ndarray]] = None
    native_roa: Optional[np.ndarray] = None
    contrast_exp: Optional[np.ndarray] = None
    config: Optional[DecompositionConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict, omitting fields that are still None.

        Mirrors AdaptationResult.to_dict()'s convention -- __getitem__/
        __contains__/get() below delegate to this, so a field only shows up in
        the dict-style view once it has actually been set (e.g. muaps only
        after MUAP extraction, roa/gt_matched_indices only after
        select_supervised).

        Returns:
            Dict[str, Any]: Mapping of field name to value, for every field
            that is not None.
        """
        out: Dict[str, Any] = {
            "sources": self.sources,
            "spikes": self.spikes,
            "spikes_dict": self.spikes_dict,
            "sil": self.sil,
            "cov_isi": self.cov_isi,
            "sep_vectors": self.sep_vectors,
            "whitening": self.whitening,
            "extension_mean": self.extension_mean,
            "spikes_centr": self.spikes_centr,
            "base_centr": self.base_centr,
            "ext_fact": self.ext_fact,
        }
        for key in (
            "pca_components",
            "pca_mean",
            "pnr",
            "dr",
            "muaps",
            "emg",
            "timestamps",
            "gt_matched_indices",
            "roa",
            "sep_vectors_native",
            "sources_native",
            "spikes_dict_native",
            "native_roa",
            "contrast_exp",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # Conversion for online adaptation
    # ------------------------------------------------------------------

    def to_adapt_tensors(self) -> Dict[str, Optional[torch.Tensor]]:
        """Unpack this result into the raw calibration tensors AdaptDecomp/optimize need.

        Returns:
            Dict[str, Optional[torch.Tensor]]: "whitening", "sep_vectors"
            ([n_mu, dim], transposed from this result's stored [dim, n_mu]),
            "base_centr", "spikes_centr", "emg_calib", "sources_calib",
            "spikes_calib", and "pca_components"/"pca_mean" (None if this
            calibration did not use PCA reduction).

        Raises:
            ValueError: If self.emg is None -- calibration EMG is required to
                build emg_calib (the same requirement
                AdaptDecomp.from_calibration() enforces up front, with a more
                specific message, before ever reaching this method).
        """
        if self.emg is None:
            raise ValueError(
                "DecompositionResult.emg is None; to_adapt_tensors() needs the calibration EMG "
                "to build emg_calib. Build/load a DecompositionResult with emg set (e.g. "
                "a config with save_emg=True, or DecompositionResult.load())."
            )

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(arr), dtype=torch.float32)

        return {
            "whitening": _t(self.whitening),
            "sep_vectors": _t(self.sep_vectors).T.contiguous(),
            "base_centr": _t(self.base_centr),
            "spikes_centr": _t(self.spikes_centr),
            "emg_calib": _t(self.emg),
            "sources_calib": _t(self.sources),
            "spikes_calib": _t(self.spikes),
            "pca_components": _t(self.pca_components) if self.pca_components is not None else None,
            "pca_mean": _t(self.pca_mean) if self.pca_mean is not None else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save this result to HDF5, with its config embedded as YAML.

        Args:
            path (Union[str, Path]): Destination file path.

        Returns:
            None
        """
        attrs = {"format": _FORMAT, "format_version": 1, "package_version": __version__}
        if self.config is not None:
            attrs["config"] = yaml.safe_dump(self.config.to_dict(), sort_keys=True)
        write_h5(path, self.to_dict(), attrs)

    @classmethod
    def load(cls, path: Union[str, Path]) -> DecompositionResult:
        """Load a DecompositionResult previously written by save().

        Fields missing from the file take their defaults, and unknown ones are
        dropped. Legacy pickle files are still read, with a DeprecationWarning.

        Args:
            path (Union[str, Path]): Path to a file written by save().

        Returns:
            DecompositionResult: The loaded result.

        Raises:
            ValueError: If the file does not contain a DecompositionResult.
        """
        if not h5py.is_hdf5(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            warnings.warn(
                f"{path} is a legacy pickle; reading pickled results is deprecated. "
                "Re-save it with DecompositionResult.load(path).save(new_path).",
                DeprecationWarning,
                stacklevel=2,
            )
            if not isinstance(obj, cls):
                raise ValueError(
                    f"{path} does not contain a DecompositionResult (got {type(obj).__name__})."
                )
            return obj

        from adapt_decomp.decomposition.config import DecompositionConfig

        data, attrs = read_h5(path)
        if attrs.get("format") not in (_FORMAT, _LEGACY_FORMAT):
            raise ValueError(
                f"{path} does not contain a DecompositionResult (got {attrs.get('format')})."
            )
        data = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        for key in ("spikes_dict", "spikes_dict_native"):
            if key in data:
                data[key] = {int(k): v for k, v in data[key].items()}
        data["ext_fact"] = int(data["ext_fact"])
        if "config" in attrs:
            data["config"] = DecompositionConfig.from_dict(yaml.safe_load(attrs["config"]))
        return cls(**data)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz, inferred from the median spacing of timestamps.

        Raises:
            ValueError: If timestamps is not set (or has fewer than 2 samples).
        """
        if self.timestamps is not None and len(self.timestamps) > 1:
            diffs = np.diff(self.timestamps)
            return float(1.0 / np.median(diffs))
        raise ValueError("Cannot determine sampling frequency: result.timestamps is not set.")

    # ------------------------------------------------------------------
    # Unit subsetting / selection
    # ------------------------------------------------------------------

    def subset(self, mask_or_idx: np.ndarray) -> DecompositionResult:
        """Return a copy keeping only the units selected by mask_or_idx.

        Args:
            mask_or_idx: Boolean per-unit keep mask, or an integer index array.

        Returns:
            New DecompositionResult with every per-unit field (sources, spikes,
            spikes_dict, sep_vectors, sil, cov_isi, pnr, dr, muaps,
            spikes_centr, base_centr, gt_matched_indices, roa, and the native
            fields) sliced to the kept units.
            Whole-model fields (whitening, extension_mean, ext_fact,
            pca_components, pca_mean, emg, timestamps) are carried over
            unchanged.
        """
        mask_or_idx = np.asarray(mask_or_idx)
        idx = np.where(mask_or_idx)[0] if mask_or_idx.dtype == bool else mask_or_idx

        def _sel1d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if arr is None else np.asarray(arr)[idx]

        def _sel2d_axis1(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if arr is None else np.asarray(arr)[:, idx]

        def _sel4d_axis0(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if arr is None else np.asarray(arr)[idx]

        def _sel_dict(d: Optional[Dict[int, np.ndarray]]) -> Optional[Dict[int, np.ndarray]]:
            if d is None:
                return None
            return {new_i: d[int(old_i)] for new_i, old_i in enumerate(idx) if int(old_i) in d}

        return type(self)(
            sources=_sel2d_axis1(self.sources),
            spikes=_sel2d_axis1(self.spikes),
            spikes_dict=_sel_dict(self.spikes_dict),
            sep_vectors=_sel2d_axis1(self.sep_vectors),
            whitening=self.whitening,
            extension_mean=self.extension_mean,
            spikes_centr=_sel1d(self.spikes_centr),
            base_centr=_sel1d(self.base_centr),
            sil=_sel1d(self.sil),
            cov_isi=_sel1d(self.cov_isi),
            ext_fact=self.ext_fact,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            pnr=_sel1d(self.pnr),
            dr=_sel1d(self.dr),
            muaps=_sel4d_axis0(self.muaps),
            emg=self.emg,
            timestamps=self.timestamps,
            gt_matched_indices=_sel1d(self.gt_matched_indices),
            roa=_sel1d(self.roa),
            sep_vectors_native=_sel2d_axis1(self.sep_vectors_native),
            sources_native=_sel2d_axis1(self.sources_native),
            spikes_dict_native=_sel_dict(self.spikes_dict_native),
            native_roa=_sel1d(self.native_roa),
            contrast_exp=_sel1d(self.contrast_exp),
            config=self.config,
        )

    def unsupervised_mask(
        self,
        *,
        sil_th: Optional[float] = None,
        pnr_th: Optional[float] = None,
        dr_min: Optional[float] = None,
        dr_max: Optional[float] = None,
        cov_th: Optional[float] = None,
        native_roa_th: Optional[float] = None,
    ) -> np.ndarray:
        """Flag the units that pass ALL provided quality thresholds (None = skip criterion).

        Args:
            sil_th:  Minimum silhouette score (units with sil >= sil_th are kept).
            pnr_th:  Minimum pulse-to-noise ratio.
            dr_min:  Minimum discharge rate (pps).
            dr_max:  Maximum discharge rate (pps).
            cov_th:  Maximum coefficient of variation of inter-spike intervals.
            native_roa_th: Minimum agreement of native and reconstructed spikes
                (native_roa; SCD). Every unit passes when native_roa is None.

        Returns:
            Boolean keep mask with shape (n_mu,), possibly all False.

        Raises:
            ValueError: If a threshold is given for a metric that is None.
        """
        n_mu = self.sources.shape[1]
        mask = np.ones(n_mu, dtype=bool)

        if sil_th is not None:
            if self.sil is None:
                raise ValueError("self.sil is None — cannot apply sil_th filter.")
            mask &= self.sil >= sil_th

        if pnr_th is not None:
            if self.pnr is None:
                raise ValueError("self.pnr is None — cannot apply pnr_th filter.")
            mask &= self.pnr >= pnr_th

        if dr_min is not None:
            if self.dr is None:
                raise ValueError("self.dr is None — cannot apply dr_min filter.")
            mask &= self.dr >= dr_min

        if dr_max is not None:
            if self.dr is None:
                raise ValueError("self.dr is None — cannot apply dr_max filter.")
            mask &= self.dr <= dr_max

        if cov_th is not None:
            if self.cov_isi is None:
                raise ValueError("self.cov_isi is None — cannot apply cov_th filter.")
            mask &= self.cov_isi <= cov_th

        # No native_roa (e.g. CBSS): the sources are already the reconstructed ones, so all pass
        if native_roa_th is not None and self.native_roa is not None:
            mask &= self.native_roa >= native_roa_th

        return mask

    def select_unsupervised(self, **thresholds: Optional[float]) -> DecompositionResult:
        """Keep units that pass ALL provided quality thresholds (None = skip criterion).

        Args:
            **thresholds: sil_th, pnr_th, dr_min, dr_max, cov_th and/or native_roa_th; see
                unsupervised_mask.

        Returns:
            New DecompositionResult with only the selected units.

        Raises:
            ValueError: If no units survive the filters, or see
                unsupervised_mask.
        """
        mask = self.unsupervised_mask(**thresholds)
        if not mask.any():
            raise ValueError(
                "No units survived unsupervised quality filtering. "
                "Loosen one or more thresholds or check DecompositionConfig."
            )
        return self.subset(mask)

    def select_supervised(
        self,
        gt_spikes: np.ndarray,
        *,
        roa_th: float = 0.9,
        tol_spike_ms: float = 2,
        fs: Optional[float] = None,
    ) -> DecompositionResult:
        """Match each decomposed unit to the best GT unit by RoA; keep matches above roa_th.

        Args:
            gt_spikes: [T_calib, M_gt] binary (int or bool) spike matrix aligned to the
                       calibration window. Must have the same number of samples as
                       self.sources.
            roa_th:    Minimum rate-of-agreement to keep a unit (default 0.9).
            tol_spike_ms: Tolerance window for coincident spikes in milliseconds (default 2).
            fs:        Sampling frequency in Hz. If None, inferred from self.fs
                       (which requires self.timestamps to be set).

        Returns:
            New DecompositionResult with:
            - only units whose best GT match has RoA >= roa_th
            - gt_matched_indices[i] set to the index of the matched GT unit

        Raises:
            ValueError: If no units match or fs cannot be determined.
        """
        T = self.sources.shape[0]
        if gt_spikes.shape[0] != T:
            raise ValueError(
                f"gt_spikes has {gt_spikes.shape[0]} samples but self.sources has {T}. "
                "gt_spikes must be aligned to the calibration window."
            )
        fs_val = float(fs) if fs is not None else self.fs

        from adapt_decomp.spikes import compare_spikes

        # pairs[i] = (gt_idx, dec_idx), sorted by dec_idx: 1:1 Hungarian assignment
        agreement = compare_spikes(
            gt_spikes, self.spikes, fs=fs_val, min_score=roa_th, tol_spike_ms=tol_spike_ms
        )
        if len(agreement.pairs) == 0:
            raise ValueError(
                f"No units had RoA >= {roa_th} with any ground-truth unit. "
                "Lower roa_th or check the calibration."
            )

        subset = self.subset(agreement.pairs[:, 1])
        subset.gt_matched_indices = agreement.pairs[:, 0].astype(np.int64)
        subset.roa = agreement.roa
        return subset

    # ------------------------------------------------------------------
    # Dict-style access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Dict-style subscript access, delegating to to_dict().

        Args:
            key (str): Field name.

        Returns:
            Any: The field's value.

        Raises:
            KeyError: If key is not a set field.
        """
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Check whether key is a set (non-None) field.

        Args:
            key (str): Field name.

        Returns:
            bool: True if key is present in to_dict().
        """
        return key in self.to_dict()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Dict-style .get(), delegating to to_dict().

        Args:
            key (str): Field name.
            default (Optional[Any], optional): Value to return if key is not
                set. Defaults to None.

        Returns:
            Any: The field's value, or default.
        """
        return self.to_dict().get(key, default)
