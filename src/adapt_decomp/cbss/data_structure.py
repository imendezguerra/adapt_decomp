"""CBSSResult dataclass."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


@dataclass
class CBSSResult:
    """Output of CBSS.decompose() and CBSS.apply(). All array fields are CPU numpy arrays."""

    sources: np.ndarray                    # [T, n_mu]
    spikes: np.ndarray                     # [T, n_mu]  int32
    spikes_dict: Dict[int, np.ndarray]
    sep_vectors: np.ndarray                # [dim, n_mu]
    whitening: np.ndarray                  # [dim, dim]
    extension_mean: np.ndarray             # [1, C*ext_fact]
    spikes_centr: np.ndarray               # [n_mu]
    base_centr: np.ndarray                 # [n_mu]
    sil: np.ndarray                        # [n_mu]
    cov_isi: np.ndarray                    # [n_mu] coefficient of variation of inter-spike intervals
    ext_fact: int                          # extension factor used to build sep_vectors/whitening
    pca_components: Optional[np.ndarray] = None   # [n_comp, C*ext_fact] or None
    pca_mean: Optional[np.ndarray] = None          # [C*ext_fact] or None
    pnr: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    muaps: Optional[np.ndarray] = None
    emg: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    gt_matched_indices: Optional[np.ndarray] = None  # [n_mu] index into GT units after supervised selection
    roa: Optional[np.ndarray] = None                 # [n_mu] RoA vs gt_matched_indices, set by select_supervised

    def to_dict(self) -> Dict:
        return {
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
            "pca_components": self.pca_components,
            "pca_mean": self.pca_mean,
            "pnr": self.pnr,
            "dr": self.dr,
            "muaps": self.muaps,
            "emg": self.emg,
            "timestamps": self.timestamps,
            "gt_matched_indices": self.gt_matched_indices,
            "roa": self.roa,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Pickle this result to disk.

        Args:
            path (Union[str, Path]): Destination file path.

        Returns:
            None
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> CBSSResult:
        """Load a CBSSResult previously written by save().

        Args:
            path (Union[str, Path]): Path to a pickle file written by save().

        Returns:
            CBSSResult: The unpickled result.

        Raises:
            ValueError: If the unpickled object is not a CBSSResult.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(
                f"{path} does not contain a CBSSResult (got {type(obj).__name__})."
            )
        return obj

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
        raise ValueError(
            "Cannot determine sampling frequency: result.timestamps is not set."
        )

    # ------------------------------------------------------------------
    # Unit subsetting / selection
    # ------------------------------------------------------------------

    def subset(self, mask_or_idx: np.ndarray) -> CBSSResult:
        """Return a copy keeping only the units selected by mask_or_idx.

        Args:
            mask_or_idx: Boolean per-unit keep mask, or an integer index array.

        Returns:
            New CBSSResult with every per-unit field (sources, spikes,
            spikes_dict, sep_vectors, sil, cov_isi, pnr, dr, muaps,
            spikes_centr, base_centr, gt_matched_indices, roa) sliced to the
            kept units.
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

        new_spikes_dict = {
            new_i: self.spikes_dict[int(old_i)]
            for new_i, old_i in enumerate(idx)
            if int(old_i) in self.spikes_dict
        }

        return CBSSResult(
            sources=_sel2d_axis1(self.sources),
            spikes=_sel2d_axis1(self.spikes),
            spikes_dict=new_spikes_dict,
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
        )

    def select_unsupervised(
        self,
        *,
        sil_th: Optional[float] = None,
        pnr_th: Optional[float] = None,
        dr_min: Optional[float] = None,
        dr_max: Optional[float] = None,
        cov_th: Optional[float] = None,
    ) -> CBSSResult:
        """Keep units that pass ALL provided quality thresholds (None = skip criterion).

        Args:
            sil_th:  Minimum silhouette score (units with sil >= sil_th are kept).
            pnr_th:  Minimum pulse-to-noise ratio.
            dr_min:  Minimum discharge rate (pps).
            dr_max:  Maximum discharge rate (pps).
            cov_th:  Maximum coefficient of variation of inter-spike intervals.

        Returns:
            New CBSSResult with only the selected units.

        Raises:
            ValueError: If no units survive the filters.
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

        n_kept = int(mask.sum())
        if n_kept == 0:
            raise ValueError(
                "No units survived unsupervised quality filtering. "
                "Loosen one or more thresholds or check CBSSConfig."
            )
        return self.subset(mask)

    def select_supervised(
        self,
        gt_spikes: np.ndarray,
        *,
        roa_th: float = 0.5,
        tol_spike_ms: float = 0.5,
        fs: Optional[float] = None,
    ) -> CBSSResult:
        """Match each decomposed unit to the best GT unit by RoA; keep matches above roa_th.

        Args:
            gt_spikes: [T_calib, M_gt] binary (int or bool) spike matrix aligned to the
                       calibration window. Must have the same number of samples as
                       self.sources.
            roa_th:    Minimum rate-of-agreement to keep a unit (default 0.5).
            tol_spike_ms: Tolerance window for coincident spikes in milliseconds (default 0.5).
            fs:        Sampling frequency in Hz. If None, inferred from self.fs
                       (which requires self.timestamps to be set).

        Returns:
            New CBSSResult with:
            - only units whose best GT match has RoA >= roa_th
            - gt_matched_indices[i] set to the index of the matched GT unit

        Raises:
            ValueError: If no units match or fs cannot be determined.
        """
        T = self.sources.shape[0]
        n_mu = self.sources.shape[1]

        if gt_spikes.shape[0] != T:
            raise ValueError(
                f"gt_spikes has {gt_spikes.shape[0]} samples but self.sources has {T}. "
                "gt_spikes must be aligned to the calibration window."
            )
        fs_val = float(fs) if fs is not None else self.fs

        from adapt_decomp.spikes import rate_of_agreement
        roa_vals, pairs, _ = rate_of_agreement(
            gt_spikes.astype(np.float32),
            self.spikes.astype(np.float32),
            fs=int(fs_val),
            tol_spike_ms=tol_spike_ms,
        )
        # pairs[i] = (gt_idx, dec_idx), sorted by dec_idx ascending — 1:1 global greedy assignment
        mask = np.zeros(n_mu, dtype=bool)
        gt_matched = np.zeros(n_mu, dtype=np.int64)
        roa_by_dec_idx = np.zeros(n_mu, dtype=np.float64)
        for (gt_idx, dec_idx), roa_val in zip(pairs, roa_vals):
            if roa_val >= roa_th:
                mask[dec_idx] = True
                gt_matched[dec_idx] = gt_idx
                roa_by_dec_idx[dec_idx] = roa_val

        n_kept = int(mask.sum())
        if n_kept == 0:
            raise ValueError(
                f"No units had RoA >= {roa_th} with any ground-truth unit. "
                "Lower roa_th or check the calibration."
            )

        subset = self.subset(mask)
        subset.gt_matched_indices = gt_matched[mask].astype(np.int64)
        subset.roa = roa_by_dec_idx[mask]
        return subset
