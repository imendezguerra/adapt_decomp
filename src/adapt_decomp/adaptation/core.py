"""Online adaptive EMG decomposition.

AdaptDecomp's per-batch update (run_decomp) is split, per concern, into an
orchestrator method that decides whether to adapt (its own config flag) and
logs loss/diagnostics either way, paired with a narrow method that only
mutates state when adapting is actually on: _whiten/_update_whitening for
whitening, _update_spike_det for the spike/base centroids, and
_update_sep_vectors (the one exception -- see its own docstring for why it
stays a single if/else rather than an orchestrator/narrow pair) for the
separation vectors.

AdaptDecomp has no run_optimisation()/reset-in-place path: Optuna
hyperparameter search (adaptation/optimize.py) builds a fresh AdaptDecomp
per trial instead of resetting one reused instance in place, so there is no
need to snapshot/restore calibration originals here.
"""

import time
from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult, Data, Decomposition
from adapt_decomp.adaptation.io import H5ParamsBatchWriter
from adapt_decomp.adaptation.ops import (
    clip_global_delta,
    find_peaks_multisource,
    classify_peaks_from_adaptive_centroids,
    update_centroids_from_peaks,
    compute_contrast_error,
    update_sv_spike_gated,
    gate_spikes_by_iqr,
)
from adapt_decomp.cbss.config import CBSSConfig
from adapt_decomp.cbss.core import CBSS
from adapt_decomp.cbss.data_structure import CBSSResult


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
        adapt_config: Optional[AdaptConfig] = None,
        preprocess: bool = True,
        save_path: Optional[str] = None,
    ) -> "AdaptDecomp":
        """Build AdaptDecomp from a CBSSResult produced by calibrate_from_indices().

        This is the CBSS-specific entry point: it unpacks a CBSSResult's
        fields into AdaptDecomp.__init__'s raw tensor arguments. For a
        calibration produced by anything other than CBSS, build the instance
        directly via AdaptDecomp(...) instead -- see __init__ below.

        Args:
            emg (Union[torch.Tensor, np.ndarray]): Online EMG to decompose,
                with shape (samples, channels).
            calibration (CBSSResult): Calibration result from
                calibrate_from_indices(). calibration.emg must be set
                (guaranteed there, which forces save_emg=True).
            adapt_config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds
                AdaptConfig(ext_fact=calibration.ext_fact).
            preprocess (bool, optional): Whether to preprocess emg before
                extension. Defaults to True.
            save_path (Optional[str], optional): Path to save per-batch
                parameters during run(), used when adapt_config.save_params
                is True. Defaults to None.

        Raises:
            TypeError: If calibration is not a CBSSResult.
            ValueError: If calibration.emg is None, or if an explicitly-passed
                adapt_config.ext_fact conflicts with calibration.ext_fact.

        Returns:
            AdaptDecomp: Instance ready for run().

        Notes:
            calibration.gt_matched_indices (if present) is stored on the
            instance and propagated through format_outputs().
            calibration.pca_components/pca_mean (set when
            CBSSConfig.n_components was used) are threaded through to
            Decomposition so the online path re-derives the same PCA-reduced
            space calibration used -- see Decomposition._apply_pca.
        """
        if not isinstance(calibration, CBSSResult):
            raise TypeError(
                f"calibration must be a CBSSResult, got {type(calibration)}"
            )
        if calibration.emg is None:
            raise ValueError(
                "calibration.emg is None. Use calibrate_from_indices() which sets save_emg=True."
            )
        if adapt_config is None:
            adapt_config = AdaptConfig(ext_fact=calibration.ext_fact)
        elif adapt_config.ext_fact != calibration.ext_fact:
            raise ValueError(
                f"adapt_config.ext_fact={adapt_config.ext_fact} does not match "
                f"calibration.ext_fact={calibration.ext_fact}. Either omit adapt_config "
                "(a default AdaptConfig built from calibration.ext_fact will be used), "
                "or set adapt_config.ext_fact to match the extension factor used to "
                "produce this calibration before calling from_calibration()."
            )

        def _format(arr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                return arr.float()
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))

        emg_t = _format(emg)
        if emg_t is None:
            raise ValueError("emg is None. Must be a 2-D array of shape [samples, channels].")

        sep_vectors_t = _format(calibration.sep_vectors)
        if sep_vectors_t is not None:
            sep_vectors_t = sep_vectors_t.T.contiguous()  # CBSSResult stores [dim, n_mu]; AdaptDecomp expects [n_mu, dim]

        instance = cls(
            emg=emg_t,
            whitening=_format(calibration.whitening),
            sep_vectors=sep_vectors_t,
            base_centr=_format(calibration.base_centr),
            spikes_centr=_format(calibration.spikes_centr),
            emg_calib=_format(calibration.emg),
            ipts_calib=_format(calibration.sources),
            spikes_calib=_format(calibration.spikes),
            preprocess=preprocess,
            adapt_config=adapt_config,
            save_path=save_path,
            pca_components=_format(calibration.pca_components),
            pca_mean=_format(calibration.pca_mean),
        )
        instance.gt_matched_indices = calibration.gt_matched_indices  # None if unsupervised
        return instance

    @classmethod
    def calibrate_from_indices(
        cls,
        emg: Union[torch.Tensor, np.ndarray],
        timestamps: Union[torch.Tensor, np.ndarray],
        calib_indices: Union[slice, np.ndarray],
        cbss_config: Optional[CBSSConfig] = None,
        adapt_config: Optional[AdaptConfig] = None,
        preprocess: bool = True,
        save_path: Optional[str] = None,
    ) -> "AdaptDecomp":
        """Run CBSS on emg[calib_indices] and build an AdaptDecomp.

        Chains two steps that would otherwise be called by hand: running CBSS
        on the calibration window (optionally filtering units -- set
        cbss_config.selection/selection_kwargs, see CBSSConfig) and
        from_calibration(). For a calibration already computed elsewhere (e.g.
        loaded from disk), use from_calibration() directly instead of this method.

        Args:
            emg (Union[torch.Tensor, np.ndarray]): Full EMG recording, with
                shape (samples, channels). emg[calib_indices] is used for
                calibration; the full array is used for online adaptation.
            timestamps (Union[torch.Tensor, np.ndarray]): Sample times in
                seconds, with shape (samples,).
            calib_indices (Union[slice, np.ndarray]): Which samples to use
                for calibration -- a slice, an integer index array, or a
                boolean mask.
            cbss_config (Optional[CBSSConfig], optional): CBSS configuration.
                Defaults to CBSSConfig(). save_emg is forced to True
                regardless. Set selection/selection_kwargs on this to filter
                units (see CBSSConfig.selection).
            adapt_config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds
                AdaptConfig(ext_fact=calibration.ext_fact).
            preprocess (bool, optional): Whether to preprocess emg before
                extension. Defaults to True.
            save_path (Optional[str], optional): Path to save per-batch
                parameters during run(), used when adapt_config.save_params
                is True. Defaults to None.

        Raises:
            ValueError: If emg/timestamps are malformed, the calibration
                window is too short, CBSS finds no units, or
                cbss_config.selection is set and keeps no units (see
                CBSS.decompose()).

        Returns:
            AdaptDecomp: Instance ready for run().
        """

        # Format data
        emg_np = _to_numpy(emg)
        ts_np = _to_numpy(timestamps)

        if emg_np.ndim != 2:
            raise ValueError(f"emg must be 2-D [samples, channels], got shape {emg_np.shape}")
        if ts_np.ndim != 1 or ts_np.shape[0] != emg_np.shape[0]:
            raise ValueError(
                f"timestamps must be 1-D with length {emg_np.shape[0]}, got {ts_np.shape}"
            )

        # Get calibration window
        emg_calib = emg_np[calib_indices]
        ts_calib = ts_np[calib_indices]

        if emg_calib.shape[0] < 2:
            raise ValueError(
                f"Calibration window has only {emg_calib.shape[0]} samples — too short for CBSS."
            )

        # Set config
        cbss_config = copy(cbss_config) if cbss_config is not None else CBSSConfig()
        cbss_config.save_emg = True  # required for from_calibration()

        # Run CBSS decomposition on the calibration window
        result = CBSS(cbss_config).decompose(emg_calib, ts_calib)

        if result.sources.shape[1] == 0:
            raise ValueError(
                "CBSS found no motor units in the calibration window. "
                "Try relaxing sil_th or increasing search_iter and ica_iter in CBSSConfig."
            )

        # Format calibration result into an AdaptDecomp instance
        return cls.from_calibration(
            emg, result, adapt_config=adapt_config, preprocess=preprocess, save_path=save_path,
        )

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
        adapt_config: Optional[AdaptConfig] = None,
        save_path: Optional[str] = None,
        pca_components: Optional[torch.Tensor] = None,
        pca_mean: Optional[torch.Tensor] = None,
    ) -> None:
        # Get config
        if adapt_config is None:
            adapt_config = AdaptConfig()
        self.config = adapt_config

        # Set device
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "cuda"
            elif torch.backends.mps.is_available():
                self.config.device = "mps"
            else:
                self.config.device = "cpu"

        # Build decomposition object
        self.decomp = Decomposition(
            whitening, sep_vectors, base_centr, spikes_centr,
            emg_calib, ipts_calib, spikes_calib, self.config,
            pca_components=pca_components, pca_mean=pca_mean,
        )
        # Build data object, save path, and diagnostics dict
        self.data = Data(emg, preprocess, adapt_config)
        self.save_path = save_path
        self.diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_exe_time(self, batches: int) -> None:
        self.time_sv_ms = torch.zeros(batches, dtype=torch.float32)
        self.time_wh_ms = torch.zeros(batches, dtype=torch.float32)
        self.time_sd_ms = torch.zeros(batches, dtype=torch.float32)

    def _init_outputs(self, samples: int, units: int) -> None:
        self.units = units
        self.samples = samples
        self.spikes = torch.zeros(samples, units, dtype=torch.int32, device=self.config.device)
        self.ipts = torch.zeros(samples, units, dtype=torch.float32, device=self.config.device)

    def _init_losses(self, batches: int) -> None:
        if self.config.compute_loss:
            self.wh_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.sv_loss = torch.zeros((batches, self.units), dtype=torch.float32, device=self.config.device)
            self.wh_trace = torch.zeros(batches, dtype=torch.float32, device=self.config.device)
            self.total_loss = torch.zeros(batches, dtype=torch.float32, device=self.config.device)

    def _format_outputs(self) -> AdaptationResult:
        """Collect per-batch results into a typed AdaptationResult.

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

        Returns:
            AdaptationResult: Typed result; call .to_dict() for the equivalent
            plain dict, or subscript it directly (result["wh_loss"]) for
            backward-compatible dict-style access.
        """
        result = AdaptationResult(
            spikes=self.spikes.detach().cpu().clone(),
            ipts=self.ipts.detach().cpu().clone(),
            wh_time_ms=self.time_wh_ms,
            sv_time_ms=self.time_sv_ms,
            sd_time_ms=self.time_sd_ms,
            total_time_ms=self.time_wh_ms + self.time_sv_ms + self.time_sd_ms,
        )
        if hasattr(self, "wh_loss"):
            result.wh_loss = self.wh_loss.detach().cpu().clone()
        if hasattr(self, "sv_loss"):
            result.sv_loss = self.sv_loss.detach().cpu().clone()
        if hasattr(self, "centroid_loss"):
            result.centroid_loss = self.centroid_loss.detach().cpu().clone()
        if hasattr(self, "wh_trace"):
            result.wh_trace = self.wh_trace.detach().cpu().clone()
        if hasattr(self, "total_loss"):
            result.total_loss = self.total_loss.detach().cpu().clone()
        if self.config.debug and hasattr(self, "diagnostics"):
            result.diagnostics = self.diagnostics
        if hasattr(self, "gt_matched_indices"):
            result.gt_matched_indices = self.gt_matched_indices
        return result

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

        # --- Whitening ---
        t0 = time.time()
        # Whitening
        Z, coupling_matrix = self._whiten(X, batch_idx)
        self.time_wh_ms[batch_idx] = (time.time() - t0) * 1000

        # --- Source estimates and spike detection ---
        t0 = time.time()

        # Coupling correction (if adapting and requested)
        self._apply_wh_sv_coupling(coupling_matrix)

        # Estimate sources
        sources = Z @ self.decomp.sep_vectors.T  # [N, M]

        # Spike detection
        spike_mask, peak_mask = self._detect_spikes(sources, N)

        # Remove outlier spikes from separation vector and centroid updates
        if self.config.adapt_sd or self.config.adapt_sv:
            trusted_spike_mask = gate_spikes_by_iqr(
                sources, spike_mask,
                self.decomp.Q75_cal, self.decomp.IQR_cal,
                gate_factor=3.0,
                peak_power=self.config.spike_det_exp,
                use_abs_for_detection=True,
                eps=self.config.eps,
            )
        else:
            trusted_spike_mask = spike_mask

        # Update spike and base centroids
        self._update_spike_det(sources, spike_mask, peak_mask, trusted_spike_mask, batch_idx)
        self.time_sd_ms[batch_idx] = (time.time() - t0) * 1000

        # --- Update separation vectors ---
        t0 = time.time()
        self._update_sep_vectors(Z, sources, trusted_spike_mask, batch_idx)
        self.time_sv_ms[batch_idx] = (time.time() - t0) * 1000

        return spike_mask.to(torch.int32), sources

    # ------------------------------------------------------------------
    # Whitening — orchestrator (_whiten) + narrow update (_update_whitening)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _whiten(self, X: torch.Tensor, batch_idx) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-batch whitening step.

        Decides whether to adapt decomp.whitening (config.adapt_wh) and logs
        loss/diagnostics either way when asked (config.compute_loss/.debug) --
        see _update_whitening for the narrow update itself, called only when
        adapt_wh is True.

        Two modes controlled by config.wh_mode:
          "kl_to_identity" — error = K − K_cal,  direction = (Rz − I) @ wh
          "kl_to_cal"      — error = KL(Rz‖Rz_cal), direction = (Rz_cal⁻¹Rz − I) @ wh
        See _update_whitening for the direction-normalisation/clip/lr_mode
        details (AdaptConfig.lr_mode docstring).
        """

        # Apply the precomputed PCA transform if provided
        X = self.decomp._apply_pca(X)

        # Only run if adapt_wh or compute_loss is True
        coupling_matrix = None
        if self.config.adapt_wh or self.config.compute_loss:

            # Compute the whitened covariance Rz for this batch (from the FIFO)
            Rz = self._compute_wh_covariance(X)

            # Log the trace of the covariance for diagnostics and loss gating during optimisation
            if self.config.compute_loss:
                self.wh_trace[batch_idx] = Rz.trace()

            # Compute KL-divergence error and direction for the configured wh_mode
            sign, logdet = torch.linalg.slogdet(Rz)
            if sign <= 0:
                # If Rz is not positive definite, skip the whitening update and log a warning.
                if self.config.compute_loss:
                    if batch_idx > 0:
                        # If possible set loss to the previous batch's loss to avoid 0 init propagation
                        self.wh_loss[batch_idx] = self.wh_loss[batch_idx - 1]
                if self.config.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    self.diagnostics.setdefault(idx, {})["wh_skip_invalid_slogdet"] = True
                return X @ self.decomp.whitening.T, None
            K_online, e_wh, direction = self._compute_kl_error(Rz, logdet)

            # If Rz is positive definite, log the actual error
            if self.config.compute_loss:
                self.wh_loss[batch_idx] = e_wh.item() ** 2
                self.total_loss[batch_idx] += e_wh.item() ** 2

            # Update whitening if requested
            if self.config.adapt_wh:
                coupling_matrix, wh_diag = self._update_whitening(direction, e_wh)

                # Store whitening diagnosis
                if self.config.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    d = self.diagnostics.setdefault(idx, {})
                    d.update({
                        "K":                K_online.item(),
                        "K_cal":            self.decomp.kl_div_calib_mean.item(),
                        "whitening_error":  e_wh.item(),
                        "delta_wh_norm":     wh_diag["delta_wh_norm"],
                        "delta_wh_raw_norm": wh_diag["delta_wh_raw_norm"],
                        "Rz_trace":         Rz.trace().item(),
                        "Rz_logdet":        logdet.item(),
                        "wh_norm":           torch.linalg.norm(self.decomp.whitening).item(),
                    })

        # Return whitened data and wh-sv coupling update (init to None)
        return X @ self.decomp.whitening.T, coupling_matrix

    @torch.no_grad()
    def _compute_wh_covariance(self, X: torch.Tensor) -> torch.Tensor:
        """Push X into the extended-EMG FIFO and return the current regularised
        whitened covariance Rz (see Decomposition._update_fifo_cov/._compute_Rz_from_fifo).
        """
        self.decomp._update_fifo_cov(X)
        return self.decomp._compute_Rz_from_fifo()

    @torch.no_grad()
    def _compute_kl_error(
        self, Rz: torch.Tensor, logdet: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (K_online, e_wh, direction) for the configured wh_mode.

        The two modes share intermediates (A, logdet_A in kl_to_cal) so they're
        kept in one method rather than split further.

          "kl_to_identity" — error = K − K_cal,  direction = (Rz − I) @ wh
          "kl_to_cal"      — error = KL(Rz‖Rz_cal), direction = (Rz_cal⁻¹Rz − I) @ wh

        e_wh is z-scored by kl_div_calib_std so it's scale-free across contractions.
        """

        if self.config.wh_mode == "kl_to_identity":
            K_online = 0.5 * (Rz.trace() - logdet - self.decomp.n)
            K_ref    = self.decomp.kl_div_calib_mean
            e_wh_raw  = K_online - K_ref
            direction = Rz - self.decomp.I
        else:  # kl_to_cal
            A         = self.decomp.Rz_cal_inv @ Rz
            logdet_A  = logdet - self.decomp.logdet_cal
            K_online  = 0.5 * (A.trace() - logdet_A - self.decomp.n)
            K_ref     = self.decomp.kl_div_calib_mean
            e_wh_raw   = K_online - K_ref
            direction = A - self.decomp.I

        # Z-score the whitening error so eta_wh is scale-free across contractions.
        sigma_K = getattr(self.decomp, "kl_div_calib_std", None)
        e_wh = e_wh_raw / sigma_K.clamp_min(self.config.eps) if sigma_K is not None else e_wh_raw
        return K_online, e_wh, direction

    @torch.no_grad()
    def _update_whitening(
        self, direction: torch.Tensor, e_wh: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], dict]:
        """Apply the natural-gradient step to decomp.whitening.

        Narrow: only mutates decomp.whitening (and decomp.ema_dirnorm_wh);
        called only from _whiten, only when config.adapt_wh is True.

        Normalizes the direction to unit scale via an EMA of its own norm (so
        one noisy/small batch can't skew the normalization), then scales by
        wh_learning_rate and the full signed e_wh -- step size tracks how
        wrong the model actually is, instead of always being clipped to a
        fixed size. lr_mode="fixed" bypasses this normalization entirely,
        reproducing main (v1)'s raw fixed-learning-rate step (see
        AdaptConfig.lr_mode docstring); the EMA is still tracked so state
        stays consistent if the mode changes mid-run.

        Returns (coupling_matrix, diag): coupling_matrix is the wh→sv
        frame-correction matrix when config.wh_b_coupling, else None; diag
        carries delta_wh_norm/delta_wh_raw_norm for _whiten's debug logging.
        """

        # Compute the direction of the whitening update
        if self.config.lr_mode == "fixed": # Fixed learning rate
            dir_coeff = -self.config.wh_learning_rate * direction

        else: # Relative error

            # Compute the norm of the whitening and update (via exponential moving average) for scaling
            M_wh = direction @ self.decomp.whitening
            M_wh_norm = torch.linalg.norm(M_wh)
            self.decomp.ema_dirnorm_wh = (
                M_wh_norm.detach() if self.decomp.ema_dirnorm_wh is None
                else (self.config.ema_alpha * self.decomp.ema_dirnorm_wh + (1 - self.config.ema_alpha) * M_wh_norm).detach()
            )
            wh_norm = torch.linalg.norm(self.decomp.whitening)

            # Compute the direction of the whitening update using the relative error and the scaling
            dir_coeff = (
                -self.config.wh_learning_rate * wh_norm * e_wh
                / (self.decomp.ema_dirnorm_wh + self.config.eps) * direction
            )

        # Clip whitening update within tolerance region
        delta_wh_target = dir_coeff @ self.decomp.whitening
        eff_max_rel_delta_wh = self.config.safety_clip_multiplier_wh * self.config.wh_learning_rate
        delta_wh = clip_global_delta(delta_wh_target, self.decomp.whitening, eff_max_rel_delta_wh, self.config.eps)

        # Update whitening
        self.decomp.whitening = self.decomp.whitening + delta_wh

        coupling_matrix = None
        if self.config.wh_b_coupling:
            # Compute first order approximation of coupling matrix -delta_wh @ wh^-1
            # to align separation vectors to new whitening
            target_norm = torch.linalg.norm(delta_wh_target)
            clip_scale = torch.linalg.norm(delta_wh) / (target_norm + self.config.eps)
            coupling_matrix = (-clip_scale * dir_coeff).detach()

        # Diagnostics
        diag = {
            "delta_wh_norm": torch.linalg.norm(delta_wh).item(),
            "delta_wh_raw_norm": torch.linalg.norm(delta_wh_target).item(),
        }
        return coupling_matrix, diag

    # ------------------------------------------------------------------
    # wh→sv coupling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_wh_sv_coupling(self, coupling_matrix: Optional[torch.Tensor]) -> None:
        """Apply the first-order frame correction implied by the wh step to sv,
        before source estimates are formed, so spike detection and the contrast
        update see a sv that is already aligned with the new whitening frame.

        No-op when coupling_matrix is None (config.wh_b_coupling=False, or
        config.adapt_wh=False -- see _whiten/_update_whitening).
        """
        if coupling_matrix is None:
            return

        # Get coupling delta
        delta_sv_coupling = self.decomp.sep_vectors @ coupling_matrix

        # Clip coupling delta
        eff_max_rel_delta_sv = self.config.safety_clip_multiplier_sv * self.config.sv_learning_rate
        delta_sv_coupling = clip_global_delta(
            delta_sv_coupling, self.decomp.sep_vectors, eff_max_rel_delta_sv, self.config.eps
        )

        # Update separation vectors
        self.decomp.sep_vectors = self.decomp.sep_vectors + delta_sv_coupling

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

        # Append new sources to previous batch to avoid missing spikes at the edges
        if self.decomp.source_fifo is not None:
            sources_full = torch.cat([self.decomp.source_fifo, sources], dim=0)
        else:
            sources_full = sources

        # Find peaks for all sources
        peak_mask_full, sources_det_full = find_peaks_multisource(
            sources_full,
            min_dist=self.config.spike_min_dist,
            peak_power=self.config.spike_det_exp,
            use_abs=True,
        )

        # Get spikes form all soures
        spike_mask_full = classify_peaks_from_adaptive_centroids(
            sources_det_full, peak_mask_full,
            self.decomp.spikes_centr, self.decomp.base_centr,
        )

        # Keep only the current batch data
        spike_mask = spike_mask_full[-N:]
        peak_mask = peak_mask_full[-N:]

        # Update source FIFO (keep at most source_fifo_batches * N rows)
        max_fifo = self.config.source_fifo_batches * N
        if self.decomp.source_fifo is not None:
            self.decomp.source_fifo = torch.cat([self.decomp.source_fifo, sources], dim=0)[-max_fifo:]
        else:
            self.decomp.source_fifo = sources[-max_fifo:]

        return spike_mask, peak_mask

    @torch.no_grad()
    def _update_spike_det(
        self,
        sources: torch.Tensor,
        spike_mask: torch.Tensor,
        peak_mask: torch.Tensor,
        trusted_spike_mask: torch.Tensor,
        batch_idx,
    ) -> None:
        """Per-batch centroid step.

        Narrow, mirroring _update_whitening's shape: decides whether to adapt
        decomp.spikes_centr/.base_centr (config.adapt_sd) and logs
        centroid_loss/diagnostics either way when asked. Called after
        detection (self._detect_spikes) and IQR-gating (gate_spikes_by_iqr)
        have already produced trusted_spike_mask, and before the sv update --
        matching today's order exactly.
        """
        # Update centroids using the trusted spikes only
        if self.config.adapt_sd:
            self.decomp.spikes_centr, self.decomp.base_centr = (
                update_centroids_from_peaks(
                    sources, peak_mask, trusted_spike_mask,
                    self.decomp.spikes_centr, self.decomp.base_centr,
                    peak_power=self.config.spike_det_exp,
                    centroid_momentum=self.config.centroid_momentum,
                    min_spikes_for_centroid=1,
                    min_base_peaks_for_centroid=1,
                    use_abs_for_detection=True,
                    eps=self.config.eps,
                )
            )

        # Diagnostics
        if self.config.debug:
            idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
            d = self.diagnostics.setdefault(idx, {})
            sep_t   = self.decomp.spikes_centr - self.decomp.base_centr
            sep_cal = self.decomp.spikes_centr_cal - self.decomp.base_centr_cal
            d.update({
                "base_centroids": self.decomp.base_centr.clone(),
                "spike_centroids": self.decomp.spikes_centr.clone(),
                "base_centroids_cal": self.decomp.base_centr_cal.clone(),
                "spike_centroids_cal": self.decomp.spikes_centr_cal.clone(),
                "centroid_drift": (
                    self.decomp.spikes_centr - self.decomp.spikes_centr_cal
                ).abs().mean(),
                "centroid_separation": (
                    sep_t / sep_cal.clamp_min(self.config.eps) - 1.0
                ) ** 2,
                "peak_counts_before": peak_mask.sum(dim=0),
                "peak_counts_after": spike_mask.sum(dim=0),
                "outlier_spike_counts": (spike_mask & ~trusted_spike_mask).sum(dim=0),
            })

    # ------------------------------------------------------------------
    # Separation vectors
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_sep_vectors(
        self, Z: torch.Tensor, sources: torch.Tensor, spike_mask: torch.Tensor, batch_idx
    ) -> None:
        """Per-batch separation-vector step.

        Decides whether to run the fixed-point natural-gradient update
        (config.adapt_sv) and logs sv_loss/diagnostics either way when asked.
        Unlike whitening, there's no single "compute the error" call shared
        between the adapting and non-adapting paths: adapt_sv=True computes
        contrast error *and* the natural-gradient update together via
        update_sv_spike_gated inside the fixed-point loop, while adapt_sv=False
        uses the separate, already-existing self._compute_sv_diag, which only
        computes contrast error for logging -- so one method with an if/else
        is the right shape here, not an artificial orchestrator/narrow split.

        Mutates decomp.sep_vectors/.ema_gradnorm_sv in place; returns None.
        Note this does NOT change the sources returned by run_decomp -- that's
        deliberately the pre-adaptation sources computed before this call, per
        run_decomp's own docstring ("ipts is sources from before the sv update
        so outputs are consistent across batches"), matching main's
        source_sep convention.
        """

        if self.config.adapt_sv:
            sv_curr = self.decomp.sep_vectors
            sources_curr = sources
            first_sv_diag = None
            # Define clipping region and norm
            eff_max_rel_delta_sv = self.config.safety_clip_multiplier_sv * self.config.sv_learning_rate
            ema_gradnorm_sv_batch = self.decomp.ema_gradnorm_sv   # carried over from last batch

            # Apply update for requested epochs
            for it in range(self.config.sv_epochs):
                sv_new, sv_diag = update_sv_spike_gated(
                    sv=sv_curr,
                    Z=Z,
                    sources=sources_curr,
                    kappa_cal=self.decomp.contrast_calib_mean,
                    spike_mask=spike_mask,
                    max_rel_delta_sv=eff_max_rel_delta_sv,
                    contrast_scope=self.config.contrast_scope,
                    eps=self.config.eps,
                    sigma_kappa_cal=getattr(self.decomp, "contrast_calib_std", None),
                    lr_sv=self.config.sv_learning_rate,
                    lr_mode=self.config.lr_mode,
                    ema_gradnorm_sv=ema_gradnorm_sv_batch,
                    ema_alpha=self.config.ema_alpha if it == 0 else 1.0,
                )
                # Store the EMA norm of the sep vectors only once to avoid oversmoothing
                # when running multiple update epochs
                if first_sv_diag is None:
                    first_sv_diag = sv_diag
                    ema_gradnorm_sv_batch = sv_diag["ema_gradnorm_sv"]
                    self.decomp.ema_gradnorm_sv = ema_gradnorm_sv_batch

                # Update separation vectors and sources
                sv_curr = sv_new
                sources_curr = Z @ sv_curr.T

                # Check for convergence
                delta_rel = (
                    torch.linalg.norm(sv_new - sv_curr)
                    / (torch.linalg.norm(sv_curr) + self.config.eps)
                )
                if delta_rel < self.config.sv_tol:
                    break
            self.decomp.sep_vectors = sv_curr
        else:
            # Still compute contrast for loss tracking even when not adapting
            sv_diag = self._compute_sv_diag(sources, spike_mask)
            first_sv_diag = sv_diag

        # --- Store losses ---
        if self.config.compute_loss:
            sv_err = sv_diag["contrast_error"]
            self.sv_loss[batch_idx] = sv_err ** 2
            self.total_loss[batch_idx] += (sv_err ** 2).nanmean().item()

        # --- Debug diagnostics ---
        if self.config.debug:
            idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
            d = self.diagnostics.setdefault(idx, {}) 
            d.update({
                **sv_diag,
                "kappa_cal": self.decomp.contrast_calib_mean.clone(),
            })
            if "delta_sv_raw_norm" in first_sv_diag:
                # Override the last-iteration values **sv_diag contributed above --
                # see the "First sub-iteration only" comment above for why.
                d["delta_sv_norm"]     = first_sv_diag["delta_sv_norm"]
                d["delta_sv_raw_norm"] = first_sv_diag["delta_sv_raw_norm"]

    # ------------------------------------------------------------------
    # Contrast diagnostic for non-adapt_sv path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_sv_diag(
        self, sources: torch.Tensor, spike_mask: torch.Tensor
    ) -> dict:
        """Compute contrast error for loss tracking when adapt_sv is False."""
        kappa, e_sv, active, spike_counts = compute_contrast_error(
            sources, spike_mask, self.decomp.contrast_calib_mean,
            self.config.contrast_scope,
            getattr(self.decomp, "contrast_calib_std", None),
            self.config.eps,
        )
        _nan = torch.tensor(float("nan"), device=sources.device, dtype=sources.dtype)
        _fallback = torch.full_like(e_sv, -3.0)
        return {
            "kappa":          torch.where(active, kappa,  _nan),
            "contrast_error": torch.where(active, e_sv,   _fallback),
            "spike_counts":   spike_counts,
            "active":         active,
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self) -> AdaptationResult:
        """Run the full online decomposition over all batches."""
        if self.config.save_params and self.save_path is None:
            raise ValueError("config.save_params=True requires save_path to be set.")
        dataset = DataLoader(
            self.data, batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )
        self._init_outputs(
            samples=len(self.data),
            units=self.decomp.sep_vectors.shape[0],
        )
        self._init_losses(len(dataset))
        self._init_exe_time(len(dataset))

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

        outputs = self._format_outputs()
        if self.config.save_params:
            self.saver._save(outputs.to_dict())
        return outputs

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a CPU numpy array.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input tensor or array.

    Returns:
        np.ndarray: Equivalent CPU numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
