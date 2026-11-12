"""CBSS — Convolutive Blind Source Separation for HD-EMG decomposition."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.ica import (
    _fast_fixed_point_ica,
    _gram_schmidt_deflate,
    _normalize,
    contrast_exponent,
)
from adapt_decomp.decomposition.base import Decomposition
from adapt_decomp.spikes import detect_spikes


class CBSS(Decomposition):
    """Convolutive Blind Source Separation: fixed-point ICA with deflation and refinement."""

    config_class = CBSSConfig
    config: CBSSConfig

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
            "w": w,
            "source": source,
            "spike_idx": spike_idx,
            "spike_centr": spike_centr,
            "base_centr": base_centr,
            "sil": sil,
            "cov_isi": self._cov_isi_from_idx(spike_idx, timestamps),
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
                source,
                self.config.spike_min_dist,
                peak_power=self.config.spike_det_exp,
            )
            cov_isi = self._cov_isi_from_idx(spike_idx, timestamps)
            if self.config.refinement_mode == "sil":
                if not np.isfinite(sil) or sil <= best["sil"]:
                    break
            else:
                if not np.isfinite(cov_isi) or cov_isi >= best["cov_isi"]:
                    break
            best = {
                "w": w_ref,
                "source": source,
                "spike_idx": spike_idx,
                "spike_centr": spike_centr,
                "base_centr": base_centr,
                "sil": sil,
                "cov_isi": cov_isi,
            }
        return best

    # ------------------------------------------------------------------
    # Unit search
    # ------------------------------------------------------------------

    def _extract(self, emg_wh: torch.Tensor, timestamps: torch.Tensor) -> Dict:
        """Find units by deflationary fixed-point ICA, each refined on its own spikes.

        Args:
            emg_wh (torch.Tensor): Whitened, extended EMG with shape (samples, dim).
            timestamps (torch.Tensor): Sample times, in s, with shape (samples,).

        Returns:
            Dict: Per-unit fields; see Decomposition._extract().
        """
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
                w=w,
                z=emg_wh.T,
                contrast_fun_type=self.config.contrast_fun,
                max_iter=self.config.ica_iter,
                tol=self.config.ica_tol,
                eps=self.config.eps,
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
                    emg_wh,
                    timestamps,
                    w,
                    source,
                    spike_idx,
                    spike_centr,
                    base_centr,
                    sil,
                    torch.stack(deflation_basis, dim=1) if deflation_basis else None,
                )
                w = d_out["w"]
                spike_idx, spike_centr, base_centr, sil = detect_spikes(
                    d_out["source"],
                    self.config.spike_min_dist,
                    peak_power=self.config.spike_det_exp,
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
            return self._empty_result(dim)

        exponent = contrast_exponent(self.config.contrast_fun, self.config.contrast_exp)
        return {
            "sources": torch.stack(sources, dim=1).cpu(),
            "spikes_dict": spikes_dict,
            "sil": torch.tensor(sil_vals),
            "cov_isi": torch.tensor(cov_vals),
            "sep_vectors": torch.stack(accepted_filters, dim=1).cpu(),
            "spikes_centr": torch.tensor(sc_list, dtype=self._dtype),
            "base_centr": torch.tensor(bc_list, dtype=self._dtype),
            "contrast_exp": (None if exponent is None else torch.full((len(sources),), exponent)),
        }
