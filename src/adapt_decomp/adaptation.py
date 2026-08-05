"""Online adaptive EMG decomposition."""

import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_decomp.config import Config, validate_literals
from adapt_decomp.data_structures import Data, Decomposition
from adapt_decomp.io import H5ParamsBatchWriter
from adapt_decomp.ops import (
    clip_global_delta,
    find_peaks_multisource,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    update_sv_spike_gated,
    gate_spikes_by_iqr,
)
from adapt_decomp.cbss.result import CBSSResult


class AdaptDecomp:
    """Online adaptive decomposition with natural-gradient whitening and spike-gated separation vector updates.
    """

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        emg: Union[torch.Tensor, np.ndarray],
        calibration: CBSSResult,
        config: Optional[Config] = None,
        preprocess: bool = False,
        save_path: Optional[str] = None,
    ) -> "AdaptDecomp":
        """Build AdaptDecomp from a CBSSResult produced by calibrate_from_indices().

        calibration.emg must be set (guaranteed when using calibrate_from_indices()).
        calibration.gt_matched_indices (if present) is stored on the instance and
        propagated through format_outputs(). calibration.pca_components/pca_mean
        (set when CBSSConfig.n_components was used) are threaded through to
        Decomposition so the online path re-derives the same PCA-reduced space
        calibration used -- see Decomposition._apply_pca.
        """
        from adapt_decomp.cbss import CBSSResult  # local import to avoid circular

        if not isinstance(calibration, CBSSResult):
            raise TypeError(f"calibration must be a CBSSResult, got {type(calibration)}")
        if calibration.emg is None:
            raise ValueError(
                "calibration.emg is None. Use calibrate_from_indices() which sets save_emg=True."
            )

        def _t(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                return arr.float()
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))

        emg_t: torch.Tensor
        if isinstance(emg, np.ndarray):
            emg_t = torch.from_numpy(emg.astype(np.float32))
        else:
            emg_t = emg.float()

        sep_vectors_t = _t(calibration.sep_vectors)
        if sep_vectors_t is not None:
            sep_vectors_t = sep_vectors_t.T.contiguous()  # CBSSResult stores [dim, n_mu]; AdaptDecomp expects [n_mu, dim]

        instance = cls(
            emg=emg_t,
            whitening=_t(calibration.whitening),
            sep_vectors=sep_vectors_t,
            base_centr=_t(calibration.base_centr),
            spikes_centr=_t(calibration.spikes_centr),
            emg_calib=_t(calibration.emg),
            ipts_calib=_t(calibration.sources),
            spikes_calib=_t(calibration.spikes),
            preprocess=preprocess,
            config=config,
            save_path=save_path,
            pca_components=_t(calibration.pca_components),
            pca_mean=_t(calibration.pca_mean),
        )
        instance.gt_matched_indices = calibration.gt_matched_indices  # None if unsupervised
        return instance

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
        pca_components: Optional[torch.Tensor] = None,
        pca_mean: Optional[torch.Tensor] = None,
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
            pca_components=pca_components, pca_mean=pca_mean,
        )
        self.data = Data(emg, preprocess, config)
        self.save_path = save_path
        self.diagnostics: dict = {}

        # Store originals for reset between optimisation trials
        self._wh_orig = whitening.to(dtype=torch.float32).clone()
        self._sv_orig = sep_vectors.to(dtype=torch.float32).clone()

    # ------------------------------------------------------------------
    # Reset / initialisation helpers
    # ------------------------------------------------------------------

    def _reset_params(self) -> None:
        """Reset adaptive state to calibration originals for a fresh optimisation trial."""
        # Sync config-derived fields cached on Decomposition as instance vars.
        # Required when config_overrides changes them between optimisation trials.
        self.decomp.device = self.config.device
        self.decomp.ext_fact = self.config.ext_fact
        self.decomp.shrinkage = self.config.shrinkage
        self.decomp.contrast_scope = self.config.contrast_scope
        self.decomp.fifo_length_cfg = self.config.fifo_length
        self.decomp.source_fifo_batches = self.config.source_fifo_batches
        self.decomp.wh_mode = self.config.wh_mode
        self.decomp.batch_size = self.config.batch_size
        self.decomp.max_sigma_batches = self.config.max_sigma_batches
        self.decomp.peak_power = self.config.peak_power
        self.decomp.use_abs_for_detection = self.config.use_abs_for_detection
        self.decomp.eps = self.config.eps

        self.decomp.whitening = self._wh_orig.clone().to(device=self.config.device)
        self.decomp.sep_vectors = self._sv_orig.clone().to(device=self.config.device)
        self.decomp.init_sd_update()
        self.decomp.init_wh_update()   # recomputes kl_div_calib_mean, kl_div_calib_std, fifo_cov
        self.decomp.init_sv_update()   # recomputes contrast_calib_mean, contrast_calib_std

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
        if self.config.compute_loss:
            self.wh_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.sv_loss = torch.zeros((batches, self.units), dtype=torch.float32, device=self.config.device)
            self.centroid_loss = torch.zeros((batches, self.units), dtype=torch.float32, device=self.config.device)
            self.wh_trace = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.total_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)

    def format_outputs(self) -> Dict:
        """Collect per-batch results into a single output dict.

        Always present:
            spikes          [samples, M]    int32   — binary spike train
            ipts            [samples, M]    float32 — source signal before sv update
            wh_time_ms      [batches]       float32
            sv_time_ms      [batches]       float32
            sd_time_ms      [batches]       float32
            total_time_ms   [batches]       float32

        Present when config.compute_loss=True:
            wh_loss         [batches]       float32
            sv_loss         [batches, M]    float32
            centroid_loss   [batches, M]    float32
            wh_trace        [batches]       float32
            total_loss      [batches]       float32

        Present when config.debug=True:
            diagnostics     dict            per-batch diagnostic tensors
        """
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
        if hasattr(self, "gt_matched_indices"):
            outputs["gt_matched_indices"] = self.gt_matched_indices
        return outputs

    # ------------------------------------------------------------------
    # Per-batch decomposition
    # ------------------------------------------------------------------

    def run_decomp(
        self, emg_batch: torch.Tensor, batch_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one EMG batch: whiten → source estimate → detect → adapt.

        Returns (spikes, ipts), both shape [N, M].
        ipts is sources from before the sv update so outputs are consistent across batches.
        """
        N = emg_batch.shape[0]
        X = emg_batch - emg_batch.mean(0, keepdim=True)

        # --- Whitening update ---
        t0 = time.time()
        Z, coupling_matrix = self._update_wh(X, batch_idx)
        self.time_wh_ms[batch_idx] = (time.time() - t0) * 1000

        # --- wh→sv coupling correction ---
        # Applies the first-order frame correction implied by the wh step to sv before
        # source estimates are formed, so spike detection and contrast update see a
        # sv that is already aligned with the new whitening frame.
        if coupling_matrix is not None:
            delta_sv_coupling = self.decomp.sep_vectors @ coupling_matrix
            eff_max_rel_delta_sv = self.config.safety_clip_multiplier_sv * self.config.sv_learning_rate
            delta_sv_coupling = clip_global_delta(
                delta_sv_coupling, self.decomp.sep_vectors, eff_max_rel_delta_sv, self.config.eps
            )
            self.decomp.sep_vectors = self.decomp.sep_vectors + delta_sv_coupling

        # --- Source estimates ---
        sources = Z @ self.decomp.sep_vectors.T  # [N, M]

        # --- Spike detection (edge-aware via source FIFO) ---
        t0 = time.time()
        spike_mask, peak_mask = self._detect_spikes(sources, N)
        self.time_sd_ms[batch_idx] = (time.time() - t0) * 1000

        # --- IQR spike gate: compute trusted_spike_mask for adaptation ---
        # Outlier spikes (amplitude above Tukey upper fence) must NOT update
        # centroids or sv. They are still present in spike_mask for output.
        if self.config.adapt_iqr_gate and (self.config.adapt_sd or self.config.adapt_sv):
            trusted_spike_mask = gate_spikes_by_iqr(
                sources, spike_mask,
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
            self.decomp.spikes_centr, self.decomp.base_centr = (
                update_centroids_from_peaks(
                    sources, peak_mask, trusted_spike_mask,
                    self.decomp.spikes_centr, self.decomp.base_centr,
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
        if self.config.compute_loss:
            sep_t   = self.decomp.spikes_centr - self.decomp.base_centr
            sep_cal = self.decomp.spikes_centr_cal - self.decomp.base_centr_cal
            self.centroid_loss[batch_idx] = (
                sep_t / sep_cal.clamp_min(self.config.eps) - 1.0
            ) ** 2

        # --- Source (sv) update ---
        t0 = time.time()
        if self.config.adapt_sv:
            sv_curr = self.decomp.sep_vectors
            sources_curr = sources
            first_sv_diag = None
            eff_max_rel_delta_sv = self.config.safety_clip_multiplier_sv * self.config.sv_learning_rate
            ema_gradnorm_sv_batch = self.decomp.ema_gradnorm_sv   # carried over from last batch
            for it in range(self.config.sv_epochs):
                sv_new, sv_diag = update_sv_spike_gated(
                    sv=sv_curr,
                    Z=Z,
                    sources=sources_curr,
                    kappa_cal=self.decomp.contrast_calib_mean,
                    spike_mask=trusted_spike_mask,
                    max_rel_delta_sv=eff_max_rel_delta_sv,
                    min_spikes_for_update=self.config.min_spikes_for_update,
                    orthonormalization=self.config.orthonormalization,
                    contrast_scope=self.config.contrast_scope,
                    eps=self.config.eps,
                    sigma_kappa_cal=getattr(self.decomp, "contrast_calib_std", None),
                    contrast_error_silent=(
                        self.config.silence_penalty_zscore
                        if self.config.silence_penalty else None
                    ),
                    lr_sv=self.config.sv_learning_rate,
                    lr_alone=self.config.lr_alone,
                    ema_gradnorm_sv=ema_gradnorm_sv_batch,
                    # EMA blends only once per batch (iteration 0); later fixed-point
                    # sub-iterations reuse the just-updated EMA frozen (alpha=1.0 means
                    # new = 1.0*old + 0*new = old) so refinement isn't double-counted
                    # as extra "time steps" in the smoothing.
                    ema_alpha=self.config.ema_alpha if it == 0 else 1.0,
                )
                # First sub-iteration only: the natural-gradient step before any
                # fixed-point refinement, i.e. the step the safety clip actually
                # trust-region clips. Later sub-iterations are refinements of an
                # already-applied update and shrink toward sv_tol by construction,
                # which would dilute the clip/EMA signal.
                if first_sv_diag is None:
                    first_sv_diag = sv_diag
                    ema_gradnorm_sv_batch = sv_diag["ema_gradnorm_sv"]
                    self.decomp.ema_gradnorm_sv = ema_gradnorm_sv_batch
                delta_rel = (
                    torch.linalg.norm(sv_new - sv_curr)
                    / (torch.linalg.norm(sv_curr) + self.config.eps)
                )
                sv_curr = sv_new
                sources_curr = Z @ sv_curr.T
                if delta_rel < self.config.sv_tol:
                    break
            self.decomp.sep_vectors = sv_curr
        else:
            # Still compute contrast for loss tracking even when not adapting
            sv_diag = self._compute_sv_diag(sources, trusted_spike_mask)
            first_sv_diag = sv_diag

        self.time_sv_ms[batch_idx] = (time.time() - t0) * 1000

        # --- Store losses ---
        if self.config.compute_loss:
            sv_err = sv_diag["contrast_error"]
            self.sv_loss[batch_idx] = sv_err ** 2
            self.total_loss[batch_idx] += (sv_err ** 2).nanmean().item()

        # --- Debug diagnostics ---
        # Use setdefault+update so _update_wh's whitening keys are not overwritten.
        if self.config.debug:
            idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
            d = self.diagnostics.setdefault(idx, {})
            d.update({
                **sv_diag,
                "kappa_cal": self.decomp.contrast_calib_mean.clone(),
                "base_centroids": self.decomp.base_centr.clone(),
                "spike_centroids": self.decomp.spikes_centr.clone(),
                "base_centroids_cal": self.decomp.base_centr_cal.clone(),
                "spike_centroids_cal": self.decomp.spikes_centr_cal.clone(),
                "centroid_drift": (
                    self.decomp.spikes_centr - self.decomp.spikes_centr_cal
                ).abs().mean(),
                "peak_counts_before": peak_mask.sum(dim=0),
                "peak_counts_after": spike_mask.sum(dim=0),
                "outlier_spike_counts": (spike_mask & ~trusted_spike_mask).sum(dim=0),
            })
            if "delta_sv_raw_norm" in first_sv_diag:
                # Override the last-iteration values **sv_diag contributed above --
                # see the "First sub-iteration only" comment at the sv-update loop for why.
                d["delta_sv_norm"]     = first_sv_diag["delta_sv_norm"]
                d["delta_sv_raw_norm"] = first_sv_diag["delta_sv_raw_norm"]

        return spike_mask.to(torch.int32), sources

    # ------------------------------------------------------------------
    # Whitening update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_wh(self, X: torch.Tensor, batch_idx) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Natural-gradient whitening update.

        Two modes controlled by config.wh_mode:
          "kl_to_identity" — error = K − K_cal,  direction = (Rz − I) @ wh
          "kl_to_cal"      — error = KL(Rz‖Rz_cal), direction = (Rz_cal⁻¹Rz − I) @ wh
        Both direction-normalize (EMA of ‖direction@wh‖) before scaling by wh_learning_rate
        and the signed error, then apply the same (now rare-safety-net) trust-region clip
        on ‖Δwh‖_F. lr_alone=True skips the direction-normalization and error term entirely,
        using the raw step wh -= wh_learning_rate * direction @ wh instead (see
        Config.lr_alone docstring).
        """
        cfg = self.config
        decomp = self.decomp

        # Project through the fitted PCA transform (identity when calibration didn't
        # use one) -- must happen after centering (X is already centered by the
        # caller) and before any use of decomp.whitening, which is dimensioned for
        # the post-PCA space whenever decomp.pca_components is set.
        X = decomp._apply_pca(X)

        coupling_matrix = None
        if cfg.adapt_wh or cfg.compute_loss:
            decomp._update_fifo_cov(X)
            Rz = decomp._compute_Rz_from_fifo()

            if cfg.compute_loss:
                self.wh_trace[batch_idx] = Rz.trace()

            sign, logdet = torch.linalg.slogdet(Rz)
            if sign <= 0:
                if cfg.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    self.diagnostics.setdefault(idx, {})["wh_skip_invalid_slogdet"] = True
                return X @ decomp.whitening.T, None

            if cfg.wh_mode == "kl_to_identity":
                K_online = 0.5 * (Rz.trace() - logdet - decomp.n)
                K_ref    = decomp.kl_div_calib_mean
                e_wh_raw  = K_online - K_ref
                direction = Rz - decomp.I
            else:  # kl_to_cal
                A         = decomp.Rz_cal_inv @ Rz
                logdet_A  = logdet - decomp.logdet_cal   # logdet(A) = logdet(Rz) - logdet(Rz_cal)
                K_online  = 0.5 * (A.trace() - logdet_A - decomp.n)   # KL(Rz ‖ Rz_cal)
                K_ref     = decomp.kl_div_calib_mean      # batch-wise mean KL at calibration (finite-sample bias)
                e_wh_raw   = K_online - K_ref
                direction = A - decomp.I      # = Rz_cal⁻¹ Rz − I

            # Z-score the whitening error so eta_wh is scale-free across contractions.
            sigma_K = getattr(decomp, "kl_div_calib_std", None)
            e_wh = e_wh_raw / sigma_K.clamp_min(cfg.eps) if sigma_K is not None else e_wh_raw

            if cfg.compute_loss:
                self.wh_loss[batch_idx] = e_wh.item() ** 2
                self.total_loss[batch_idx] += e_wh.item() ** 2

            if cfg.adapt_wh:
                # Normalize the whitening natural-gradient direction to unit scale via an
                # EMA of its own norm (so one noisy/small batch can't skew the normalization),
                # then scale by wh_learning_rate and the full signed e_wh -- step size now
                # tracks how wrong the model actually is, instead of always being clipped to
                # a fixed size.
                # (lr_alone bypasses this normalization entirely -- see below; the EMA is
                # still tracked so state stays consistent if the flag changes mid-run.)
                M_wh = direction @ decomp.whitening
                M_wh_norm = torch.linalg.norm(M_wh)
                decomp.ema_dirnorm_wh = (
                    M_wh_norm.detach() if decomp.ema_dirnorm_wh is None
                    else (cfg.ema_alpha * decomp.ema_dirnorm_wh + (1 - cfg.ema_alpha) * M_wh_norm).detach()
                )
                wh_norm = torch.linalg.norm(decomp.whitening)
                # delta_wh_target = dir_coeff @ decomp.whitening; dir_coeff is factored out
                # (rather than computed from M_wh directly) so the same coefficient can be
                # reused below for the wh_b_coupling correction.
                if cfg.lr_alone:
                    # Main (v1)'s raw, un-normalized natural-gradient step: no EMA
                    # direction-normalization, no error term. Reproduces main's
                    # fixed-learning-rate whitening update -- the step shrinks on its own
                    # as direction -> 0 (Rz approaches its target), instead of being
                    # forced to a constant relative size every batch. wh_learning_rate
                    # means something different here than in main (v1) and must be
                    # tuned separately. The whitening step's sign is unaffected either way.
                    dir_coeff = -cfg.wh_learning_rate * direction
                else:
                    dir_coeff = (
                        -cfg.wh_learning_rate * wh_norm * e_wh
                        / (decomp.ema_dirnorm_wh + cfg.eps) * direction
                    )
                delta_wh_target = dir_coeff @ decomp.whitening
                eff_max_rel_delta_wh = cfg.safety_clip_multiplier_wh * cfg.wh_learning_rate
                delta_wh = clip_global_delta(delta_wh_target, decomp.whitening, eff_max_rel_delta_wh, cfg.eps)
                decomp.whitening = decomp.whitening + delta_wh

                if cfg.wh_b_coupling:
                    # -delta_wh @ wh^-1 (the first-order frame correction implied by the wh
                    # step), rescaled by the same clip factor applied to delta_wh itself --
                    # holds because clip_global_delta is a pure scalar rescale of
                    # delta_wh_target = dir_coeff @ decomp.whitening.
                    target_norm = torch.linalg.norm(delta_wh_target)
                    clip_scale = torch.linalg.norm(delta_wh) / (target_norm + cfg.eps)
                    coupling_matrix = (-clip_scale * dir_coeff).detach()

                if cfg.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    d = self.diagnostics.setdefault(idx, {})
                    d.update({
                        "K":                K_online.item(),
                        "K_cal":            K_ref.item(),
                        "whitening_error":  e_wh.item(),
                        "delta_wh_norm":     torch.linalg.norm(delta_wh).item(),
                        "delta_wh_raw_norm": torch.linalg.norm(delta_wh_target).item(),
                        "Rz_trace":         Rz.trace().item(),
                        "Rz_logdet":        logdet.item(),
                        "wh_norm":           torch.linalg.norm(decomp.whitening).item(),
                    })

        return X @ decomp.whitening.T, coupling_matrix

    # ------------------------------------------------------------------
    # Spike detection (edge-aware via source FIFO)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_spikes(
        self, sources: torch.Tensor, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect spikes using vectorised NMS on source FIFO + current batch.

        Prepending the source FIFO ensures spikes at the left edge of the current
        batch are not missed because of insufficient left-context for NMS.

        Returns (spike_mask, peak_mask) for the current batch only, shape [N, M].
        """
        cfg = self.config
        decomp = self.decomp

        if decomp.source_fifo is not None:
            sources_full = torch.cat([decomp.source_fifo, sources], dim=0)
        else:
            sources_full = sources

        peak_mask_full, sources_det_full = find_peaks_multisource(
            sources_full,
            min_dist=cfg.spike_dist,
            peak_power=cfg.peak_power,
            strict=cfg.strict_peaks,
            use_abs=cfg.use_abs_for_detection,
        )
        spike_mask_full = classify_peaks_from_adaptive_centroids(
            sources_det_full, peak_mask_full,
            decomp.spikes_centr, decomp.base_centr,
        )

        # Only the current-batch rows are used for outputs and adaptation
        spike_mask = spike_mask_full[-N:]
        peak_mask = peak_mask_full[-N:]

        # Update source FIFO (keep at most source_fifo_batches * N rows)
        max_fifo = cfg.source_fifo_batches * N
        if decomp.source_fifo is not None:
            decomp.source_fifo = torch.cat([decomp.source_fifo, sources], dim=0)[-max_fifo:]
        else:
            decomp.source_fifo = sources[-max_fifo:]

        return spike_mask, peak_mask

    # ------------------------------------------------------------------
    # Contrast diagnostic for non-adapt_sv path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_sv_diag(
        self, sources: torch.Tensor, spike_mask: torch.Tensor
    ) -> dict:
        """Compute contrast error for loss tracking when adapt_sv is False."""
        from adapt_decomp.ops import log_cosh
        cfg = self.config
        decomp = self.decomp

        if cfg.contrast_scope == "batch_based":
            kappa = log_cosh(sources).mean(dim=0)
        else:
            mask_f = spike_mask.to(sources.dtype)
            counts = mask_f.sum(dim=0)
            kappa = (log_cosh(sources) * mask_f).sum(dim=0) / counts.clamp_min(1.0)

        e_sv_raw = kappa - decomp.contrast_calib_mean
        sigma = getattr(decomp, "contrast_calib_std", None)
        e_sv = e_sv_raw / sigma.clamp_min(cfg.eps) if sigma is not None else e_sv_raw
        spike_counts = spike_mask.to(sources.dtype).sum(dim=0)
        if cfg.contrast_scope == "batch_based":
            # Mirrors ops.py::update_sv_spike_gated's batch_based branch, which never
            # gates by min_spikes_for_update -- only spike_based mode does.
            active = torch.ones_like(spike_counts, dtype=torch.bool)
        else:
            active = spike_counts >= cfg.min_spikes_for_update
        _nan = torch.tensor(float("nan"), device=sources.device, dtype=sources.dtype)
        _fallback = (
            torch.full_like(e_sv, cfg.silence_penalty_zscore)
            if cfg.silence_penalty else _nan
        )
        return {
            "kappa":          torch.where(active, kappa,  _nan),
            "contrast_error": torch.where(active, e_sv,   _fallback),
            "spike_counts":   spike_counts,
            "active":         active,
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the full online decomposition over all batches."""
        if self.config.save_params and self.save_path is None:
            raise ValueError("config.save_params=True requires save_path to be set.")
        dataset = DataLoader(
            self.data, batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )
        self.init_outputs(
            samples=len(self.data),
            units=self.decomp.sep_vectors.shape[0],
        )
        self.init_losses(len(dataset))
        self.init_exe_time(len(dataset))

        if self.config.debug:
            self.diagnostics.clear()

        if self.config.save_params and self.save_path is not None:
            self.saver = H5ParamsBatchWriter(
                path=self.save_path,
                wh_shape=self.decomp.whitening.shape,
                sv_shape=self.decomp.sep_vectors.shape,
                sd_shape=self.decomp.spikes_centr.shape,
                batches=len(dataset),
                dtype="float32",
            )

        for i, (emg_batch, idx_labels) in enumerate(dataset):
            i_t = torch.tensor(i, device=self.config.device)
            emg_batch, idx_labels = self._check_batch(emg_batch, idx_labels)

            if self.config.save_params:
                self.saver._append({
                    "whitening": self.decomp.whitening.cpu().numpy(),
                    "sep_vectors": self.decomp.sep_vectors.cpu().numpy(),
                    "base_centr": self.decomp.base_centr.cpu().numpy(),
                    "spikes_centr": self.decomp.spikes_centr.cpu().numpy(),
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
        sv_lr: Optional[float] = None,
        config_overrides: Optional[dict] = None,
    ) -> float:
        """Run decomposition for hyperparameter optimisation.

        Resets adaptive state, runs the full dataset, and returns the combined
        normalised loss (whitening + contrast, both z-scored by calibration std).
        """
        prev_log_loss = self.config.compute_loss
        self.config.compute_loss = True
        try:
            if wh_lr is not None:
                self.config.wh_learning_rate = wh_lr
            if sv_lr is not None:
                self.config.sv_learning_rate = sv_lr
            for k, v in (config_overrides or {}).items():
                setattr(self.config, k, v)
            if config_overrides and "batch_ms" in config_overrides:
                self.config.batch_size = int(self.config.batch_ms * self.config.fs / 1000)
            validate_literals(self.config)

            self._reset_params()
            self.init_outputs(
                samples=self.data.emg_ext.shape[0],
                units=self.decomp.sep_vectors.shape[0],
            )

            if self.config.debug:
                self.diagnostics.clear()

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
                return wh + self._compute_total_sv_loss()
            return (wh, self._compute_total_sv_loss(), self._compute_total_centroid_loss())
        finally:
            self.config.compute_loss = prev_log_loss

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
