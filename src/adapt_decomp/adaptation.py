"""Online adaptive EMG decomposition."""

import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from adapt_decomp.config import Config
from adapt_decomp.data_structures import Data, Decomposition
from adapt_decomp.io import H5ParamsBatchWriter
from adapt_decomp.ops import (
    clip_global_delta,
    find_peaks_multisource,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    update_B_spike_gated,
    gate_spikes_by_iqr,
)


class AdaptDecomp:
    """Online adaptive decomposition with natural-gradient whitening and spike-gated source updates.

    Public API (constructor signature, run(), run_optimisation()) is unchanged.
    Backend algorithm:
      1. Natural-gradient relative update of V using retained KL calibration error.
      2. Vectorised NMS peak detection on the whitened source matrix Y.
      3. Adaptive centroid-based spike classification.
      4. Spike-gated B update with retained contrast error and QR orthonormalisation.
    """

    def __init__(
        self,
        emg: torch.Tensor,
        whitening: torch.Tensor,
        sep_vectors: torch.Tensor,
        base_centr: torch.Tensor,
        spikes_centr: torch.Tensor,
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        preprocess: Optional[bool] = True,
        config: Optional[Config] = None,
        save_path: Optional[str] = None,
    ) -> None:
        if config is None:
            config = Config()
        self.config = config

        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "cuda"
            elif torch.backends.mps.is_available():
                self.config.device = "mps"
            else:
                self.config.device = "cpu"

        self.decomp = Decomposition(
            whitening, sep_vectors, base_centr, spikes_centr,
            emg_calib, ipts_calib, spikes_calib, self.config,
        )
        self.data = Data(emg, preprocess, config)
        self.save_path = save_path

        # Store originals for reset between optimisation trials
        self._V_orig = whitening.to(dtype=torch.float32).clone()
        self._B_orig = sep_vectors.to(dtype=torch.float32).clone()

    # ------------------------------------------------------------------
    # Reset / initialisation helpers
    # ------------------------------------------------------------------

    def _reset_params(self) -> None:
        """Reset adaptive state to calibration originals for a fresh optimisation trial."""
        # Sync config-derived fields cached on Decomposition as instance vars.
        # Required when config_overrides changes them between optimisation trials.
        self.decomp.shrinkage = self.config.shrinkage
        self.decomp.contrast_scope = self.config.contrast_scope
        self.decomp.wh_mode = self.config.wh_mode
        self.decomp.batch_size = self.config.batch_size

        self.decomp.V = self._V_orig.clone().to(device=self.config.device)
        self.decomp.B = self._B_orig.clone().to(device=self.config.device)
        self.decomp.init_sd_update()
        self.decomp.init_wh_update()   # recomputes K_cal, sigma_K_cal, fifo_cov
        self.decomp.init_sv_update()   # recomputes kappa_cal, sigma_kappa_cal

    def init_exe_time(self, batches: int) -> None:
        self.time_sv_ms = torch.zeros(batches, dtype=torch.float32)
        self.time_wh_ms = torch.zeros(batches, dtype=torch.float32)
        self.time_sd_ms = torch.zeros(batches, dtype=torch.float32)

    def init_outputs(self, samples: int, units: int) -> None:
        self.units = units
        self.samples = samples
        self.spikes = torch.zeros(samples, units, dtype=torch.int32, device=self.config.device)
        self.ipts = torch.zeros(samples, units, dtype=torch.float32, device=self.config.device)

    def init_losses(self, batches: int) -> None:
        if self.config.log_loss:
            self.wh_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.sv_loss = torch.zeros((batches, self.units), dtype=torch.float32, device=self.config.device)
            self.centroid_loss = torch.zeros((batches, self.units), dtype=torch.float32, device=self.config.device)
            self.wh_trace = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.total_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)

    def format_outputs(self) -> Dict:
        outputs = {
            "spikes": self.spikes.detach().cpu().clone(),
            "ipts": self.ipts.detach().cpu().clone(),
            "wh_time_ms": self.time_wh_ms,
            "sv_time_ms": self.time_sv_ms,
            "sd_time_ms": self.time_sd_ms,
            "total_time_ms": self.time_wh_ms + self.time_sv_ms + self.time_sd_ms,
        }
        if hasattr(self, "wh_loss"):
            outputs["wh_loss"] = self.wh_loss.detach().cpu().clone()
        if hasattr(self, "sv_loss"):
            outputs["sv_loss"] = self.sv_loss.detach().cpu().clone()
        if hasattr(self, "centroid_loss"):
            outputs["centroid_loss"] = self.centroid_loss.detach().cpu().clone()
        if hasattr(self, "wh_trace"):
            outputs["wh_trace"] = self.wh_trace.detach().cpu().clone()
        if hasattr(self, "total_loss"):
            outputs["total_loss"] = self.total_loss.detach().cpu().clone()
        if self.config.debug and hasattr(self, "diagnostics"):
            outputs["diagnostics"] = self.diagnostics
        return outputs

    # ------------------------------------------------------------------
    # Per-batch decomposition
    # ------------------------------------------------------------------

    def run_decomp(
        self, emg_batch: torch.Tensor, batch_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one EMG batch: whiten → source estimate → detect → adapt.

        Returns (spikes, ipts), both shape [N, M].
        ipts is Y from before the B update so outputs are consistent across batches.
        """
        N = emg_batch.shape[0]
        X = emg_batch - emg_batch.mean(0, keepdim=True)

        # --- Whitening update ---
        t0 = time.time()
        Z, coupling_matrix = self._update_V(X, batch_idx)
        self.time_wh_ms[batch_idx] = (time.time() - t0) * 1000

        # --- V→B coupling correction ---
        # Applies the first-order frame correction implied by the V step to B before
        # source estimates are formed, so spike detection and contrast update see a
        # B that is already aligned with the new whitening frame.
        if coupling_matrix is not None:
            delta_B_coupling = self.decomp.B @ coupling_matrix
            delta_B_coupling = clip_global_delta(
                delta_B_coupling, self.decomp.B, self.config.max_rel_delta_b, self.config.eps
            )
            self.decomp.B = self.decomp.B + delta_B_coupling

        # --- Source estimates ---
        Y = Z @ self.decomp.B.T  # [N, M]

        # --- Spike detection (edge-aware via source FIFO) ---
        t0 = time.time()
        spike_mask, peak_mask = self._detect_spikes(Y, N)
        self.time_sd_ms[batch_idx] = (time.time() - t0) * 1000

        # --- IQR spike gate: compute trusted_spike_mask for adaptation ---
        # Outlier spikes (amplitude above Tukey upper fence) must NOT update
        # centroids or B. They are still present in spike_mask for output.
        if self.config.adapt_iqr_gate and (self.config.adapt_sd or self.config.adapt_sv):
            trusted_spike_mask = gate_spikes_by_iqr(
                Y, spike_mask,
                self.decomp.Q75_cal, self.decomp.IQR_cal,
                gate_factor=self.config.iqr_gate_factor,
                peak_power=self.config.peak_power,
                use_abs_for_detection=self.config.use_abs_for_detection,
                eps=self.config.eps,
            )
        else:
            trusted_spike_mask = spike_mask

        # --- Centroid update (current-batch portion only) ---
        if self.config.adapt_sd:
            self.decomp.spike_centroid, self.decomp.base_centroid = (
                update_centroids_from_peaks(
                    Y, peak_mask, trusted_spike_mask,
                    self.decomp.spike_centroid, self.decomp.base_centroid,
                    peak_power=self.config.peak_power,
                    centroid_momentum=self.config.centroid_momentum,
                    min_spikes_for_centroid=self.config.min_spikes_for_centroid,
                    min_base_peaks_for_centroid=self.config.min_base_peaks_for_centroid,
                    use_abs_for_detection=self.config.use_abs_for_detection,
                    eps=self.config.eps,
                )
            )

        # --- Centroid loss (separation ratio deviation from calibration) ---
        # Measures whether spike/base centroids remain well-separated relative to
        # calibration, not their absolute position. Scale-invariant: amplitude shifts
        # that move both centroids together don't inflate the loss.
        if self.config.log_loss:
            sep_t   = self.decomp.spike_centroid - self.decomp.base_centroid
            sep_cal = self.decomp.spike_centroid_cal - self.decomp.base_centroid_cal
            self.centroid_loss[batch_idx] = (
                sep_t / sep_cal.clamp_min(self.config.eps) - 1.0
            ) ** 2

        # --- Source (B) update ---
        t0 = time.time()
        if self.config.adapt_sv:
            B_curr = self.decomp.B
            Y_curr = Y
            for _ in range(self.config.max_iter_b):
                B_new, sv_diag = update_B_spike_gated(
                    B=B_curr,
                    Z=Z,
                    Y=Y_curr,
                    kappa_cal=self.decomp.kappa_cal,
                    spike_mask=trusted_spike_mask,
                    max_rel_delta_b=self.config.max_rel_delta_b,
                    min_spikes_for_update=self.config.min_spikes_for_update,
                    orthonormalization=self.config.orthonormalization,
                    contrast_scope=self.config.contrast_scope,
                    eps=self.config.eps,
                    sigma_kappa_cal=getattr(self.decomp, "sigma_kappa_cal", None),
                )
                delta_rel = (
                    torch.linalg.norm(B_new - B_curr)
                    / (torch.linalg.norm(B_curr) + self.config.eps)
                )
                B_curr = B_new
                Y_curr = Z @ B_curr.T
                if delta_rel < self.config.tol_b:
                    break
            self.decomp.B = B_curr
        else:
            # Still compute contrast for loss tracking even when not adapting
            sv_diag = self._compute_sv_diag(Y, trusted_spike_mask)

        self.time_sv_ms[batch_idx] = (time.time() - t0) * 1000

        # --- Store losses ---
        if self.config.log_loss:
            sv_err = sv_diag["contrast_error"]
            self.sv_loss[batch_idx] = sv_err ** 2
            self.total_loss[batch_idx] += (sv_err ** 2).nanmean().item()

        # --- Debug diagnostics ---
        # Use setdefault+update so _update_V's whitening keys are not overwritten.
        if self.config.debug:
            idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
            d = self.diagnostics.setdefault(idx, {})
            d.update({
                **sv_diag,
                "kappa_cal": self.decomp.kappa_cal.clone(),
                "base_centroid": self.decomp.base_centroid.clone(),
                "spike_centroid": self.decomp.spike_centroid.clone(),
                "base_centroid_cal": self.decomp.base_centroid_cal.clone(),
                "spike_centroid_cal": self.decomp.spike_centroid_cal.clone(),
                "centroid_drift": (
                    self.decomp.spike_centroid - self.decomp.spike_centroid_cal
                ).abs().mean(),
                "peak_counts_before": peak_mask.sum(dim=0),
                "peak_counts_after": spike_mask.sum(dim=0),
                "outlier_spike_counts": (spike_mask & ~trusted_spike_mask).sum(dim=0),
            })

        return spike_mask.to(torch.int32), Y

    # ------------------------------------------------------------------
    # Whitening update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_V(self, X: torch.Tensor, batch_idx) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Natural-gradient whitening update.

        Two modes controlled by config.wh_mode:
          "kl_to_identity" — error = K − K_cal,  direction = (Rz − I) @ V
          "kl_to_cal"      — error = KL(Rz‖Rz_cal), direction = (Rz_cal⁻¹Rz − I) @ V
        Both use the same trust-region clip on ‖ΔV‖_F.
        """
        cfg = self.config
        decomp = self.decomp

        coupling_matrix = None
        if cfg.adapt_wh or cfg.log_loss:
            decomp._update_fifo_cov(X)
            Rz = decomp._compute_Rz_from_fifo()

            if cfg.log_loss:
                self.wh_trace[batch_idx] = Rz.trace()

            sign, logdet = torch.linalg.slogdet(Rz)
            if sign <= 0:
                if cfg.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    self.diagnostics.setdefault(idx, {})["wh_skip_invalid_slogdet"] = True
                return X @ decomp.V.T, None

            if cfg.wh_mode == "kl_to_identity":
                K_online = 0.5 * (Rz.trace() - logdet - decomp.D)
                K_ref    = decomp.K_cal
                e_v_raw  = K_online - K_ref
                direction = Rz - decomp.I
            else:  # kl_to_cal
                A         = decomp.Rz_cal_inv @ Rz
                logdet_A  = logdet - decomp.logdet_cal   # logdet(A) = logdet(Rz) - logdet(Rz_cal)
                K_online  = 0.5 * (A.trace() - logdet_A - decomp.D)   # KL(Rz ‖ Rz_cal)
                K_ref     = decomp.K_cal      # batch-wise mean KL at calibration (finite-sample bias)
                e_v_raw   = K_online - K_ref
                direction = A - decomp.I      # = Rz_cal⁻¹ Rz − I

            # Z-score the whitening error so eta_v is scale-free across contractions.
            sigma_K = getattr(decomp, "sigma_K_cal", None)
            e_v = e_v_raw / sigma_K.clamp_min(cfg.eps) if sigma_K is not None else e_v_raw

            if cfg.log_loss:
                self.wh_loss[batch_idx] = e_v.item() ** 2
                self.total_loss[batch_idx] += e_v.item() ** 2

            if cfg.adapt_wh:
                delta_V_raw = -e_v * (direction @ decomp.V)
                delta_V = clip_global_delta(delta_V_raw, decomp.V, cfg.max_rel_delta_v, cfg.eps)
                decomp.V = decomp.V + delta_V

                if cfg.wh_b_coupling:
                    raw_norm = torch.linalg.norm(delta_V_raw)
                    clip_scale = torch.linalg.norm(delta_V) / (raw_norm + cfg.eps)
                    coupling_matrix = (clip_scale * e_v * direction).detach()

                if cfg.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    d = self.diagnostics.setdefault(idx, {})
                    d.update({
                        "K":               K_online.item(),
                        "K_cal":           K_ref.item(),
                        "whitening_error": e_v.item(),
                        "delta_V_norm":    torch.linalg.norm(delta_V).item(),
                        "Rz_trace":        Rz.trace().item(),
                        "Rz_logdet":       logdet.item(),
                        "V_norm":          torch.linalg.norm(decomp.V).item(),
                    })

        return X @ decomp.V.T, coupling_matrix

    # ------------------------------------------------------------------
    # Spike detection (edge-aware via source FIFO)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_spikes(
        self, Y: torch.Tensor, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect spikes using vectorised NMS on source FIFO + current batch.

        Prepending the source FIFO ensures spikes at the left edge of the current
        batch are not missed because of insufficient left-context for NMS.

        Returns (spike_mask, peak_mask) for the current batch only, shape [N, M].
        """
        cfg = self.config
        decomp = self.decomp

        if decomp.source_fifo is not None:
            Y_full = torch.cat([decomp.source_fifo, Y], dim=0)
        else:
            Y_full = Y

        peak_mask_full, Y_det_full = find_peaks_multisource(
            Y_full,
            min_dist=cfg.spike_dist,
            peak_power=cfg.peak_power,
            strict=cfg.strict_peaks,
            use_abs=cfg.use_abs_for_detection,
        )
        spike_mask_full = classify_peaks_from_adaptive_centroids(
            Y_det_full, peak_mask_full,
            decomp.spike_centroid, decomp.base_centroid,
        )

        # Only the current-batch rows are used for outputs and adaptation
        spike_mask = spike_mask_full[-N:]
        peak_mask = peak_mask_full[-N:]

        # Update source FIFO (keep at most source_fifo_batches * N rows)
        max_fifo = cfg.source_fifo_batches * N
        if decomp.source_fifo is not None:
            decomp.source_fifo = torch.cat([decomp.source_fifo, Y], dim=0)[-max_fifo:]
        else:
            decomp.source_fifo = Y[-max_fifo:]

        return spike_mask, peak_mask

    # ------------------------------------------------------------------
    # Contrast diagnostic for non-adapt_sv path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_sv_diag(
        self, Y: torch.Tensor, spike_mask: torch.Tensor
    ) -> dict:
        """Compute contrast error for loss tracking when adapt_sv is False."""
        from adapt_decomp.ops import log_cosh
        cfg = self.config
        decomp = self.decomp

        if cfg.contrast_scope == "batch_based":
            kappa = log_cosh(Y).mean(dim=0)
        else:
            mask_f = spike_mask.to(Y.dtype)
            counts = mask_f.sum(dim=0)
            kappa = (log_cosh(Y) * mask_f).sum(dim=0) / counts.clamp_min(1.0)

        e_b_raw = kappa - decomp.kappa_cal
        sigma = getattr(decomp, "sigma_kappa_cal", None)
        e_b = e_b_raw / sigma.clamp_min(cfg.eps) if sigma is not None else e_b_raw
        spike_counts = spike_mask.to(Y.dtype).sum(dim=0)
        active = spike_counts >= cfg.min_spikes_for_update
        _nan = torch.tensor(float("nan"), device=Y.device, dtype=Y.dtype)
        return {
            "kappa":          torch.where(active, kappa,  _nan),
            "contrast_error": torch.where(active, e_b,    _nan),
            "spike_counts":   spike_counts,
            "active":         active,
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the full online decomposition over all batches."""
        dataset = DataLoader(
            self.data, batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )
        self.init_outputs(
            samples=len(self.data),
            units=self.decomp.B.shape[0],
        )
        self.init_losses(len(dataset))
        self.init_exe_time(len(dataset))

        if self.config.debug:
            self.diagnostics = {}

        if self.config.save_params and self.save_path is not None:
            self.saver = H5ParamsBatchWriter(
                path=self.save_path,
                wh_shape=self.decomp.V.shape,
                sv_shape=self.decomp.B.shape,
                sd_shape=self.decomp.spike_centroid.shape,
                batches=len(dataset),
                dtype="float32",
            )

        for i, (emg_batch, idx_labels) in enumerate(dataset):
            i_t = torch.tensor(i, device=self.config.device)
            emg_batch, idx_labels = self._check_batch(emg_batch, idx_labels)

            if self.config.save_params:
                self.saver._append({
                    "whitening": self.decomp.V.cpu().numpy(),
                    "sep_vectors": self.decomp.B.cpu().numpy(),
                    "base_centr": self.decomp.base_centroid.cpu().numpy(),
                    "spikes_centr": self.decomp.spike_centroid.cpu().numpy(),
                })

            spikes, ipts = self.run_decomp(emg_batch, i_t)
            self.spikes[idx_labels, :] = spikes
            self.ipts[idx_labels, :] = ipts

        outputs = self.format_outputs()
        if self.config.save_params:
            self.saver._save(outputs)
        return outputs

    def run_optimisation(
        self,
        wh_lr: Optional[float] = None,
        cov_alpha: Optional[float] = None,
        sv_lr: Optional[float] = None,
        config_overrides: Optional[dict] = None,
    ) -> float:
        """Run decomposition for hyperparameter optimisation.

        Resets adaptive state, runs the full dataset, and returns the combined
        normalised loss (whitening + contrast, both z-scored by calibration std).
        """
        self.config.log_loss = True

        if wh_lr is not None:
            self.config.max_rel_delta_v = wh_lr
        if sv_lr is not None:
            self.config.max_rel_delta_b = sv_lr
        for k, v in (config_overrides or {}).items():
            setattr(self.config, k, v)
        if config_overrides and "batch_ms" in config_overrides:
            self.config.batch_size = int(self.config.batch_ms * self.config.fs / 1000)

        self._reset_params()
        self.init_outputs(
            samples=self.data.emg_ext.shape[0],
            units=self.decomp.B.shape[0],
        )

        if self.config.debug:
            self.diagnostics = {}

        dataset = DataLoader(
            self.data, batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )
        self.init_losses(len(dataset))
        self.init_exe_time(len(dataset))

        for i, (emg_batch, idx_labels) in enumerate(dataset):
            emg_batch, idx_labels = self._check_batch(emg_batch, idx_labels)
            spikes, ipts = self.run_decomp(emg_batch, i)
            self.spikes[idx_labels, :] = spikes
            self.ipts[idx_labels, :] = ipts

        if self.config.trace_check:
            trace_ratios = self.wh_trace / self.decomp.trace_cal
            agg = trace_ratios.median() if self.config.trace_check_mode == "median" else trace_ratios[-1]
            if not (0.1 < agg.item() < 50.0):
                return 1e10 if self.config.optim_loss == "single_obj" else (1e10, 1e10, 1e10)

        wh = self._compute_total_wh_loss()
        if self.config.optim_loss == "single_obj":
            return wh + self._compute_total_sv_loss() + self._compute_total_centroid_loss()
        return (wh, self._compute_total_sv_loss(), self._compute_total_centroid_loss())

    # ------------------------------------------------------------------
    # Loss aggregation
    # ------------------------------------------------------------------

    def _compute_total_wh_loss(self) -> float:
        if torch.any(torch.isnan(self.wh_loss)):
            return 1e10
        return self.wh_loss.median().item()

    def _compute_total_sv_loss(self) -> float:
        return self.sv_loss.nanmedian().item()

    def _compute_total_centroid_loss(self) -> float:
        return self.centroid_loss.nanmedian().item()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _check_batch(
        self, emg_batch: torch.Tensor, idx_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discard extension-factor artefact samples from the first batch."""
        if torch.any(idx_labels < self.config.ext_fact):
            emg_batch = emg_batch[self.config.ext_fact:]
            idx_labels = idx_labels[self.config.ext_fact:]
        return emg_batch, idx_labels
