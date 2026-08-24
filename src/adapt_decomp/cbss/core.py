"""CBSS — Convolutive Blind Source Separation for HD-EMG decomposition."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

from loguru import logger
import numpy as np
import torch

from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.ica import _fast_fixed_point_ica, _gram_schmidt_deflate, _normalize
from adapt_decomp.cbss.data_structure import CBSSResult
from adapt_decomp.cbss.pca import pca_reduction
from adapt_decomp.cbss.whitening import whiten
from adapt_decomp.spikes import detect_spikes, remove_duplicates, spikes_dict_to_binary
from adapt_decomp.spikes.metrics import (
    emg_to_ch_array,
    get_coefficient_of_variation,
    get_discharge_rate,
    get_muaps,
    get_pulse_to_noise_ratio,
)
from adapt_decomp.preprocessing import extend_data, filter_kwargs, preprocess_emg, replace_bad_channels


class CBSS:
    """Convolutive Blind Source Separation for HD-EMG decomposition."""

    def __init__(self, config: CBSSConfig) -> None:
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
        """Bandpass + optional notch + zero-mean + optional bad-channel replacement.

        Shares preprocess_emg() with Data.preprocess_emg (online adaptation) so the
        whitening reference computed here at calibration and the online covariance see
        EMG at the same scale/spectral content.
        """
        emg_f = preprocess_emg(emg, fs, **filter_kwargs(self.config))
        emg_f = emg_f - emg_f.mean(axis=0, keepdims=True)
        if self.config.replace_bad_channels and self.config.bad_chs is not None and self.config.ch_map is not None:
            emg_f = replace_bad_channels(emg_f, self.config.bad_chs, self.config.ch_map, layout="samples_first")
        elif self.config.bad_chs is not None:
            if self.config.ch_map is not None:
                raise ValueError(
                    "Dropping bad channels (replace_bad_channels=False) while "
                    "ch_map is set is not supported: ch_map's indices would no "
                    "longer match the shrunk channel layout. Set "
                    "replace_bad_channels=True to keep channel count/indexing intact."
                )
            mask = np.ones(emg_f.shape[1], dtype=bool)
            mask[self.config.bad_chs] = False
            emg_f = emg_f[:, mask]
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

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _refinement_loop(
        self,
        emg_wh: torch.Tensor,
        timestamps: torch.Tensor,
        w: torch.Tensor,
        source: torch.Tensor,
        spike_idx: torch.Tensor,
        spike_centr: float,
        base_centr: float,
        sil: float,
        deflation_basis: Optional[torch.Tensor],
    ) -> dict:
        best = {
            "w": w, "source": source, "spike_idx": spike_idx,
            "spike_centr": spike_centr, "base_centr": base_centr,
            "sil": sil, "cov_isi": self._cov_isi_from_idx(spike_idx, timestamps),
        }
        for _ in range(self.config.refine_max_iter):
            w_ref = emg_wh[best["spike_idx"]].mean(0)
            w_ref = _normalize(w_ref, eps=self.config.eps)
            if deflation_basis is not None:
                w_ref = _gram_schmidt_deflate(w_ref, deflation_basis, eps=self.config.eps)
            if not torch.isfinite(w_ref).all() or w_ref.norm() <= self.config.eps:
                break
            source = emg_wh @ w_ref
            if not torch.isfinite(source).all():
                break
            spike_idx, spike_centr, base_centr, sil = detect_spikes(
                source, self.config.spike_min_dist, peak_power=self.config.spike_det_exp,
            )
            cov_isi = self._cov_isi_from_idx(spike_idx, timestamps)
            if self.config.refinement_mode == "sil":
                if not np.isfinite(sil) or sil <= best["sil"]:
                    break
            else:
                if not np.isfinite(cov_isi) or cov_isi >= best["cov_isi"]:
                    break
            best = {"w": w_ref, "source": source, "spike_idx": spike_idx,
                    "spike_centr": spike_centr, "base_centr": base_centr, "sil": sil, "cov_isi": cov_isi}
        return best

    def _empty_result(self, dict_results: Dict, dim: int) -> Dict:
        dict_results.update({
            "sources": torch.zeros((0, 0)),
            "spikes_dict": {},
            "spikes": torch.zeros((0, 0), dtype=torch.int32),
            "sil": torch.empty(0, dtype=torch.float32),
            "cov_isi": torch.empty(0, dtype=torch.float32),
            "sep_vectors": torch.zeros((dim, 0)),
            "spikes_centr": torch.empty(0, dtype=torch.float32),
            "base_centr": torch.empty(0, dtype=torch.float32),
        })
        return dict_results

    @staticmethod
    def _to_numpy(result: Dict) -> Dict:
        for key in ("sources", "spikes", "sil", "cov_isi", "sep_vectors", "spikes_centr",
                    "base_centr", "whitening", "extension_mean", "pnr", "dr", "muaps", "emg", "timestamps"):
            value = result.get(key)
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        emg: Union[np.ndarray, torch.Tensor],
        timestamps: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> CBSSResult:
        """Decompose HD-EMG using CBSS.

        If self.config.selection is set, the result is filtered via
        CBSSResult.select_unsupervised()/select_supervised() (using
        self.config.selection_kwargs) before being returned -- see
        CBSSConfig.selection. selection=None (default) returns every unit
        CBSS found, unfiltered.

        Args:
            emg:        [T, C] EMG (float32 recommended).
            timestamps: Optional [T] sample times in seconds.

        Raises:
            ValueError: If self.config.selection is not one of
                "unsupervised"/"supervised"/None, or if the selected filter
                keeps no units (raised by CBSSResult.select_unsupervised()/
                select_supervised()).

        Returns:
            CBSSResult with source signals, spike trains, quality metrics,
            and fitted model parameters (whitening, sep_vectors, centroids).
        """
        emg_np = emg.cpu().numpy() if isinstance(emg, torch.Tensor) else np.asarray(emg)
        emg_t = (
            self._preprocess_emg(emg_np, self.config.fs)
            if self.config.preprocess_emg
            else torch.from_numpy(emg_np.astype(np.float32)).to(device=self._device, dtype=self._dtype)
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

        # 4. ICA extraction — use a local version that properly captures sep_vectors
        dict_results = self._extraction_loop_full(emg_wh, timestamps_t, dict_results)

        # 5. Duplicate removal
        if self.config.run_duplicate_removal:
            dict_results = remove_duplicates(
                dict_results, fs=self.config.fs, roa_th=self.config.roa_th,
                tol_spike_ms=self.config.spike_min_dist_ms, dtype=self._dtype,
                device=str(self._device), verbose=self.config.verbose,
            )

        # 6. Binary spike matrix
        sources = dict_results.get("sources")
        spikes_dict = dict_results.get("spikes_dict", {})
        if sources is not None and spikes_dict:
            sources_t = torch.as_tensor(sources, dtype=self._dtype, device=self._device)
            dict_results["spikes"] = spikes_dict_to_binary(
                spikes_dict, sources_t.shape[0], device=self._device
            ).cpu()

        # 7. Properties
        if self.config.compute_properties:
            sources = dict_results.get("sources")
            spikes_dict = dict_results.get("spikes_dict", {})
            if sources is not None and spikes_dict:
                sources_d = torch.as_tensor(sources, dtype=self._dtype, device=self._device)
                spike_trains = spikes_dict_to_binary(spikes_dict, sources_d.shape[0], device=self._device)
                dict_results["pnr"] = get_pulse_to_noise_ratio(
                    spike_trains, sources_d, self.config.ext_fact, self.config.spike_min_dist
                ).cpu()
                dict_results["dr"] = get_discharge_rate(spike_trains, timestamps_t).cpu()
                if self.config.ch_map is not None:
                    half_win = round(25 / 2 / 1000 * self.config.fs)
                    emg_ch = emg_to_ch_array(emg_t.cpu(), self.config.ch_map)
                    dict_results["muaps"] = get_muaps(spike_trains.cpu(), emg_ch, half_win).cpu()

        dict_results["emg"] = emg_t.cpu() if self.config.save_emg else None
        dict_results["timestamps"] = timestamps_t.cpu() if self.config.save_emg else None
        d = self._to_numpy(dict_results)

        result = CBSSResult(
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
        )

        # 8. Optional post-hoc unit selection (config-driven, see CBSSConfig.selection).
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
                f"Unknown CBSSConfig.selection: {self.config.selection!r}. "
                "Expected 'unsupervised', 'supervised', or None."
            )
        return result

    def _extraction_loop_full(self, emg_wh: torch.Tensor, timestamps: torch.Tensor, dict_results: Dict) -> Dict:
        """Extraction loop that properly accumulates sep_vectors."""
        samples, dim = emg_wh.shape
        sources: List[torch.Tensor] = []
        spikes_dict: Dict[int, np.ndarray] = {}
        sil_vals: List[float] = []
        cov_vals: List[float] = []
        deflation_basis: List[torch.Tensor] = []
        accepted_filters: List[torch.Tensor] = []
        sc_list: List[float] = []
        bc_list: List[float] = []
        init_order = torch.empty(0, dtype=torch.long)

        for _ in range(self.config.search_iter):
            if init_order.numel() == 0:
                init_order = torch.randperm(samples, generator=self._rng)
            idx = int(init_order[0].item())
            init_order = init_order[1:]
            w = _normalize(emg_wh[idx], eps=self.config.eps)
            if deflation_basis:
                w = _gram_schmidt_deflate(w, torch.stack(deflation_basis, dim=1), self.config.eps)

            ica_result = _fast_fixed_point_ica(
                w=w, z=emg_wh.T,
                contrast_fun_type=self.config.contrast_fun,
                max_iter=self.config.ica_iter, tol=self.config.ica_tol, eps=self.config.eps,
                deflation_basis=torch.stack(deflation_basis, dim=1) if deflation_basis else None,
                contrast_exp=self.config.contrast_exp,
            )
            w = ica_result.w
            if ica_result.collapsed:
                continue

            source = emg_wh @ w
            spike_idx, spike_centr, base_centr, sil = detect_spikes(
                source, self.config.spike_min_dist, peak_power=self.config.spike_det_exp
            )

            if self.config.refinement_loop and spike_idx.shape[0] >= self.config.min_spikes:
                d_out = self._refinement_loop(
                    emg_wh, timestamps, w, source, spike_idx, spike_centr, base_centr, sil,
                    torch.stack(deflation_basis, dim=1) if deflation_basis else None,
                )
                w = d_out["w"]
                spike_idx, spike_centr, base_centr, sil = detect_spikes(
                    d_out["source"], self.config.spike_min_dist, peak_power=self.config.spike_det_exp
                )
                source = emg_wh @ w
                cov_isi = self._cov_isi_from_idx(spike_idx, timestamps)
            else:
                cov_isi = self._cov_isi_from_idx(spike_idx, timestamps)

            deflation_basis.append(w)

            if spike_idx.shape[0] < self.config.min_spikes or sil < self.config.sil_th:
                if self.config.verbose:
                    logger.debug(f"SIL={sil:.3f} < {self.config.sil_th}: rejected")
                continue

            unit_id = len(sources)
            sources.append(source)
            accepted_filters.append(w)
            spikes_dict[unit_id] = spike_idx.cpu().numpy()
            sil_vals.append(sil)
            cov_vals.append(cov_isi)
            sc_list.append(spike_centr)
            bc_list.append(base_centr)

            if self.config.verbose:
                logger.debug(f"unit {unit_id} accepted (SIL={sil:.3f}, CoV={cov_isi:.3f})")

        if not sources:
            logger.warning("CBSS found no motor units.")
            return self._empty_result(dict_results, dim)

        dict_results.update({
            "sources": torch.stack(sources, dim=1).cpu(),
            "spikes_dict": spikes_dict,
            "sil": torch.tensor(sil_vals),
            "cov_isi": torch.tensor(cov_vals),
            "sep_vectors": torch.stack(accepted_filters, dim=1).cpu(),
            "spikes_centr": torch.tensor(sc_list, dtype=self._dtype),
            "base_centr": torch.tensor(bc_list, dtype=self._dtype),
        })
        return dict_results

    def apply(
        self,
        emg: Union[np.ndarray, torch.Tensor],
        result: CBSSResult,
        timestamps: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> CBSSResult:
        """Apply stored decomposition parameters to a new EMG signal.

        Reuses extension mean, optional PCA, whitening matrix, and ICA filters
        from result. Spike detection is re-run on the projected sources.
        """
        emg_np = emg.cpu().numpy() if isinstance(emg, torch.Tensor) else np.asarray(emg)
        emg_t = (
            self._preprocess_emg(emg_np, self.config.fs)
            if self.config.preprocess_emg
            else torch.from_numpy(emg_np.astype(np.float32)).to(device=self._device, dtype=self._dtype)
        )
        emg_original = emg_t
        timestamps_t = self._make_timestamps(timestamps, emg_t.shape[0])

        emg_ext = extend_data(emg_t, self.config.ext_fact, ext_mode=self.config.ext_mode)
        ext_mean = torch.from_numpy(result.extension_mean).to(device=self._device, dtype=self._dtype)
        emg_ext = emg_ext - ext_mean
        emg_ext[: self.config.ext_fact, :] = 0
        emg_ext[-self.config.ext_fact :, :] = 0

        if result.pca_components is not None:
            pca_mean = torch.from_numpy(result.pca_mean).to(device=self._device, dtype=self._dtype)
            pca_comps = torch.from_numpy(result.pca_components).to(device=self._device, dtype=self._dtype)
            emg_pca = (emg_ext - pca_mean) @ pca_comps.T
        else:
            emg_pca = emg_ext

        W = torch.from_numpy(result.whitening).to(device=self._device, dtype=self._dtype)
        emg_wh = emg_pca @ W.T
        sep_vectors = torch.from_numpy(result.sep_vectors).to(device=self._device, dtype=self._dtype)
        n_mu = sep_vectors.shape[1]

        if n_mu == 0:
            return CBSSResult(
                sources=np.zeros((emg_wh.shape[0], 0), dtype=np.float32),
                spikes=np.zeros((emg_wh.shape[0], 0), dtype=np.int32),
                spikes_dict={}, sil=np.empty(0, dtype=np.float32),
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
            )

        sources_t = emg_wh @ sep_vectors
        spikes_dict: Dict[int, np.ndarray] = {}
        sil_vals: List[float] = []
        cov_vals: List[float] = []

        for i in range(n_mu):
            sc = torch.tensor(float(result.spikes_centr[i]), dtype=self._dtype, device=self._device)
            bc = torch.tensor(float(result.base_centr[i]), dtype=self._dtype, device=self._device)
            spike_idx, _, _, sil = detect_spikes(
                sources_t[:, i], self.config.spike_min_dist, spike_centroid=sc, base_centroid=bc,
                peak_power=self.config.spike_det_exp, compute_sil=True,
            )
            cov_isi = self._cov_isi_from_idx(spike_idx, timestamps_t)
            spikes_dict[i] = spike_idx.cpu().numpy()
            sil_vals.append(sil)
            cov_vals.append(cov_isi)

        sources_np = sources_t.detach().cpu().numpy()
        spike_trains_dev = spikes_dict_to_binary(spikes_dict, emg_wh.shape[0], device=self._device)

        applied = CBSSResult(
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
        )

        if self.config.compute_properties:
            sources_d = torch.from_numpy(sources_np).to(device=self._device, dtype=self._dtype)
            applied.pnr = get_pulse_to_noise_ratio(
                spike_trains_dev, sources_d, self.config.ext_fact, self.config.spike_min_dist
            ).cpu().numpy()
            applied.dr = get_discharge_rate(spike_trains_dev, timestamps_t).cpu().numpy()
            if self.config.ch_map is not None:
                half_win = round(25 / 2 / 1000 * self.config.fs)
                emg_ch = emg_to_ch_array(emg_original.cpu(), self.config.ch_map)
                applied.muaps = get_muaps(spike_trains_dev.cpu(), emg_ch, half_win).cpu().numpy()

        return applied
