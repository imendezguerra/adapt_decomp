"""Online adaptive EMG decomposition."""

import time
import warnings
from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapt_decomp.adaptation.config import AdaptConfig
from adapt_decomp.adaptation.data_structures import AdaptationResult, Data, Decomposition, RawData
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
from adapt_decomp.preprocessing import (
    extend_data,
    filter_kwargs,
    preprocess_emg_stateful,
    select_channels,
    validate_channel_selection,
)


class AdaptDecomp:
    """Online adaptive decomposition with natural-gradient whitening and spike-gated separation vector updates."""

    # ------------------------------------------------------------------
    # Construction and data supply
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        calibration: CBSSResult,
        cbss_config: CBSSConfig,
        adapt_config: Optional[AdaptConfig] = None,
        save_path: Optional[str] = None,
    ) -> "AdaptDecomp":
        """Build AdaptDecomp from a CBSSResult. Call process_data(emg, ...) next.

        Args:
            calibration (CBSSResult): Calibration result from
                calibrate_and_process(). calibration.emg must be set
                (guaranteed there, which forces save_emg=True).
            cbss_config (CBSSConfig): The CBSSConfig that produced
                calibration. Treated as ground truth for every field it
                shares with AdaptConfig (ext_fact, ext_mode, spike_det_exp,
                and the preprocessing/filter fields); see Notes.
            adapt_config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds an AdaptConfig
                seeded entirely from cbss_config's shared fields.
            save_path (Optional[str], optional): Path to save per-batch
                parameters during process_data(), used when
                adapt_config.save_params is True. Defaults to None.

        Raises:
            TypeError: If calibration is not a CBSSResult.
            ValueError: If calibration.emg is None, or if
                cbss_config.ext_fact does not match calibration.ext_fact
                (they must come from the same calibration run).

        Returns:
            AdaptDecomp: Instance ready for process_data(emg, ...); this
            factory only builds the model, it never touches emg.

        Notes:
            calibration.gt_matched_indices (if present) is stored on the
            instance and propagated through _format_outputs().
            calibration.pca_components/pca_mean (set when
            CBSSConfig.n_components was used) are threaded through to
            Decomposition; see Decomposition._apply_pca.
            When adapt_config disagrees with cbss_config on any of
            _SHARED_CBSS_ADAPT_FIELDS, cbss_config wins and a single
            UserWarning lists what changed; see reconcile_with_calib_config.
        """
        if not isinstance(calibration, CBSSResult):
            raise TypeError(
                f"calibration must be a CBSSResult, got {type(calibration)}"
            )
        if calibration.emg is None:
            raise ValueError(
                "calibration.emg is None. Use calibrate_and_process() which sets save_emg=True."
            )
        if cbss_config.ext_fact != calibration.ext_fact:
            raise ValueError(
                f"cbss_config.ext_fact={cbss_config.ext_fact} does not match "
                f"calibration.ext_fact={calibration.ext_fact}; these must come from "
                "the same calibration run."
            )
        if adapt_config is None:
            adapt_config = AdaptConfig(
                **{field: getattr(cbss_config, field) for field in _SHARED_CBSS_ADAPT_FIELDS}
            )
        else:
            adapt_config = reconcile_with_calib_config(
                adapt_config, SharedCalibFields.from_cbss_config(cbss_config)
            )

        tensors = calibration.to_adapt_tensors()

        instance = cls(
            whitening=tensors["whitening"],
            sep_vectors=tensors["sep_vectors"],
            base_centr=tensors["base_centr"],
            spikes_centr=tensors["spikes_centr"],
            emg_calib=tensors["emg_calib"],
            ipts_calib=tensors["ipts_calib"],
            spikes_calib=tensors["spikes_calib"],
            adapt_config=adapt_config,
            save_path=save_path,
            pca_components=tensors["pca_components"],
            pca_mean=tensors["pca_mean"],
        )
        instance.gt_matched_indices = calibration.gt_matched_indices  # None if unsupervised
        return instance

    @classmethod
    def calibrate_and_process(
        cls,
        emg: Union[torch.Tensor, np.ndarray],
        timestamps: Union[torch.Tensor, np.ndarray],
        calib_indices: Union[slice, np.ndarray],
        cbss_config: Optional[CBSSConfig] = None,
        adapt_config: Optional[AdaptConfig] = None,
        preprocess: bool = True,
        processing_mode: Literal["offline", "online"] = "offline",
        save_path: Optional[str] = None,
    ) -> Tuple[AdaptationResult, CBSSResult]:
        """Run CBSS on emg[calib_indices] and run AdaptDecomp over the entire emg.

        Args:
            emg (Union[torch.Tensor, np.ndarray]): Full EMG recording, with
                shape (samples, channels). emg[calib_indices] is used for
                calibration; the full array is used for online adaptation.
            timestamps (Union[torch.Tensor, np.ndarray]): Sample times in
                seconds, with shape (samples,).
            calib_indices (Union[slice, np.ndarray]): Which samples to use
                for calibration: a slice, an integer index array, or a
                boolean mask.
            cbss_config (Optional[CBSSConfig], optional): CBSS configuration.
                Defaults to CBSSConfig(). save_emg is forced to True
                regardless. Set selection/selection_kwargs on this to filter
                units (see CBSSConfig.selection).
            adapt_config (Optional[AdaptConfig], optional): Online adaptation
                configuration. Defaults to None, which builds an AdaptConfig
                seeded entirely from cbss_config's shared fields; see
                from_calibration.
            preprocess (bool, optional): Forwarded to init_data(). Defaults
                to True.
            processing_mode (Literal["offline", "online"], optional):
                Forwarded to init_data(); see AdaptDecomp's class docstring
                mode table. Defaults to "offline".
            save_path (Optional[str], optional): Path to save per-batch
                parameters during process_data(), used when
                adapt_config.save_params is True. Defaults to None.

        Raises:
            ValueError: If emg/timestamps are malformed, the calibration
                window is too short, CBSS finds no units, or
                cbss_config.selection is set and keeps no units (see
                CBSS.decompose()).

        Returns:
            Tuple[AdaptationResult, CBSSResult]: outputs, this run's typed
            result; calibration, the CBSSResult CBSS produced on the
            calibration window.
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
        calibration = CBSS(cbss_config).decompose(emg_calib, ts_calib)

        if calibration.sources.shape[1] == 0:
            raise ValueError(
                "CBSS found no motor units in the calibration window. "
                "Try relaxing sil_th or increasing search_iter and ica_iter in CBSSConfig."
            )

        # Build the model, then run adaptation over the full recording.
        instance = cls.from_calibration(
            calibration, cbss_config=cbss_config, adapt_config=adapt_config, save_path=save_path,
        )
        outputs = instance.process_data(emg, preprocess=preprocess, processing_mode=processing_mode)
        return outputs, calibration

    def __init__(
        self,
        whitening: torch.Tensor,
        sep_vectors: torch.Tensor,
        base_centr: torch.Tensor,
        spikes_centr: torch.Tensor,
        emg_calib: torch.Tensor,
        ipts_calib: torch.Tensor,
        spikes_calib: torch.Tensor,
        adapt_config: Optional[AdaptConfig] = None,
        save_path: Optional[str] = None,
        pca_components: Optional[torch.Tensor] = None,
        pca_mean: Optional[torch.Tensor] = None,
        emg: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> None:
        """Build the decomposition model. Never touches emg, except a deprecated v1-compatible path (see emg).

        Args:
            whitening (torch.Tensor): Initial whitening matrix.
            sep_vectors (torch.Tensor): Initial separation vectors.
            base_centr (torch.Tensor): Calibration baseline centroids.
            spikes_centr (torch.Tensor): Calibration spike centroids.
            emg_calib (torch.Tensor): Raw, unextended calibration EMG.
            ipts_calib (torch.Tensor): Calibration source signal.
            spikes_calib (torch.Tensor): Calibration binary spike train.
            adapt_config (Optional[AdaptConfig], optional): Online
                adaptation configuration. Defaults to None, which builds
                AdaptConfig().
            save_path (Optional[str], optional): Path to save per-batch
                parameters during process_data(), used when
                adapt_config.save_params is True. Defaults to None.
            pca_components (Optional[torch.Tensor], optional): Fitted PCA
                components. Defaults to None (no PCA reduction).
            pca_mean (Optional[torch.Tensor], optional): Fitted PCA mean.
                Defaults to None.
            emg (Optional[Union[torch.Tensor, np.ndarray]], optional):
                Deprecated v1-compatible construction: pass emg here, then
                call .run(), equivalent to process_data(emg,
                processing_mode="offline"). Defaults to None, the
                recommended way to construct; process_data()/init_data()
                are the only places emg should otherwise enter. Raises
                FutureWarning when given.

        Returns:
            None
        """
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
        self.save_path = save_path
        self.diagnostics: dict = {}
        self.data_preprocessed = False
        self.processing_mode: Optional[Literal["offline", "online"]] = None

        # Seed the per-batch accumulators so process_batch() is callable
        # directly right after construction (full-online mode: no
        # process_data()/init_data() call at all); process_data() resets
        # these again at the start of its own run.
        self._init_outputs(units=self.decomp.sep_vectors.shape[0])
        self._init_losses()
        self._init_exe_time()

        # Deprecated v1-compatible construction: emg here is only ever
        # consumed by .run() (see its own docstring); process_data()/
        # init_data() never read self._emg_raw.
        self._emg_raw: Optional[Union[torch.Tensor, np.ndarray]] = None
        if emg is not None:
            self._emg_raw = emg
            warnings.warn(
                "Passing emg to AdaptDecomp.__init__ is deprecated and will "
                "be removed in a future version; use process_data(emg, ...) "
                "instead. emg passed here is only used if .run() is called "
                "afterward (equivalent to process_data(emg, "
                "processing_mode='offline')).",
                FutureWarning, stacklevel=2,
            )

    def init_data(
        self,
        emg: Union[torch.Tensor, np.ndarray],
        preprocess: bool = True,
        processing_mode: Literal["offline", "online"] = "offline",
    ) -> None:
        """Prepare emg for process_data(), building Data or RawData.

        Builds RawData (process_batch preprocesses each batch itself) when
        processing_mode is "online", else Data (preprocessed upfront). Sets
        self.data_preprocessed and self.processing_mode accordingly.

        Args:
            emg (Union[torch.Tensor, np.ndarray]): Online EMG to decompose,
                with shape (samples, channels).
            preprocess (bool, optional): Forwarded to Data when
                processing_mode is "offline"; ignored for "online".
                Defaults to True.
            processing_mode (Literal["offline", "online"], optional):
                "offline" preprocesses the whole recording upfront (the
                default); "online" defers preprocessing to process_batch,
                per batch. Defaults to "offline".

        Returns:
            None
        """
        self.processing_mode = processing_mode
        emg_t = torch.as_tensor(emg, dtype=torch.float32)
        if processing_mode == "online":
            self.data = RawData(emg_t, self.config)
            self.data_preprocessed = False
        else:
            self.data = Data(emg_t, preprocess, self.config)
            self.data_preprocessed = True

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def process_data(
        self,
        emg: Union[torch.Tensor, np.ndarray],
        preprocess: bool = True,
        processing_mode: Literal["offline", "online"] = "offline",
    ) -> AdaptationResult:
        """Run the full online decomposition over emg, batch by batch.

        Always calls init_data(emg, preprocess, processing_mode) first;
        this is the one place emg ever enters an AdaptDecomp (besides
        run()'s deprecated v1-compatible path).

        Args:
            emg (Union[torch.Tensor, np.ndarray]): Online EMG to decompose,
                with shape (samples, channels).
            preprocess (bool, optional): Forwarded to init_data(). Defaults
                to True.
            processing_mode (Literal["offline", "online"], optional):
                Forwarded to init_data(); see its docstring/the class
                docstring's mode table. Defaults to "offline".

        Raises:
            ValueError: If config.save_params is True with save_path unset.

        Returns:
            AdaptationResult: This run's typed output.
        """
        self.init_data(emg, preprocess, processing_mode)

        if self.config.save_params and self.save_path is None:
            raise ValueError("config.save_params=True requires save_path to be set.")

        dataset = DataLoader(
            self.data, batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )

        self._init_outputs(units=self.decomp.sep_vectors.shape[0])
        self._init_losses()
        self._init_exe_time()

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

        for batch_idx, (emg_batch, _) in enumerate(dataset):
            i_t = torch.tensor(batch_idx, device=self.config.device)

            if self.config.save_params:
                self.saver._append({
                    "whitening": self.decomp.whitening.cpu().numpy(),
                    "sep_vectors": self.decomp.sep_vectors.cpu().numpy(),
                    "base_centr": self.decomp.base_centr.cpu().numpy(),
                    "spikes_centr": self.decomp.spikes_centr.cpu().numpy(),
                })

            spikes, ipts = self.process_batch(emg_batch, i_t)
            self._spikes_accum.append(spikes)
            self._sources_accum.append(ipts)

        self._finalize_accumulators()
        if self.config.compute_loss:
            self.wh_loss_median, self.sv_loss_median, self.total_loss = self._compute_losses()

        outputs = self._format_outputs()
        if self.config.save_params:
            self.saver._save(outputs.to_dict())
        return outputs

    def run(self) -> AdaptationResult:
        """Deprecated v1-compatible alias: process_data(emg, processing_mode="offline")
        using the emg passed to __init__.

        Raises:
            ValueError: If no emg was passed to __init__.

        Returns:
            AdaptationResult: See process_data().
        """
        if self._emg_raw is None:
            raise ValueError(
                "AdaptDecomp.run() requires emg to have been passed to "
                "__init__ (the deprecated v1-compatible construction "
                "pattern); use process_data(emg, ...) instead."
            )
        warnings.warn(
            "AdaptDecomp.run() is deprecated and will be removed in a "
            "future version; use process_data(emg, ...) instead.",
            FutureWarning, stacklevel=2,
        )
        return self.process_data(self._emg_raw, processing_mode="offline")

    # ------------------------------------------------------------------
    # Per-batch pipeline
    # ------------------------------------------------------------------

    def process_batch(
        self, emg_batch: torch.Tensor, batch_idx: Optional[Union[int, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one EMG batch: preprocess → whiten → source estimate → detect → adapt.

        Preprocessing branches on self.data_preprocessed: when True,
        emg_batch is already extended and only the first batch's leading
        ext_fact rows are trimmed; when False, emg_batch is raw and is
        filtered/channel-selected/centred/extended by
        _preprocess_batch_raw. Either way, the returned (spikes, ipts)
        always have emg_batch.shape[0] rows (as called), zero-padded at
        the leading rows a batch's own trim discarded.

        Args:
            emg_batch (torch.Tensor): One batch of EMG, shape (N,
                channels) if data_preprocessed is False, else (N, D)
                already extended.
            batch_idx (Optional[Union[int, torch.Tensor]]): This batch's
                sequential index, starting at 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: spikes and ipts, both shape
            (N, M). ipts is sources from before the sv update so outputs
            are consistent across batches.
        """
        N_called = emg_batch.shape[0]

        # --- Preprocessing ---
        t0 = time.time()
        if self.data_preprocessed:
            pad_offset = self.config.ext_fact if batch_idx == 0 else 0
            if pad_offset:
                emg_batch = emg_batch[pad_offset:]
            preprocess_ms = 0.0
        else:
            emg_batch, pad_offset = self._preprocess_batch_raw(emg_batch)
            preprocess_ms = (time.time() - t0) * 1000
        self.time_preprocess_ms.append(preprocess_ms)

        N = emg_batch.shape[0]
        X = emg_batch - emg_batch.mean(0, keepdim=True)

        # --- Whitening ---
        t0 = time.time()
        # Whitening
        Z, coupling_matrix = self._whiten(X, batch_idx)
        self.time_wh_ms.append((time.time() - t0) * 1000)

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
        self.time_sd_ms.append((time.time() - t0) * 1000)

        # --- Update separation vectors ---
        t0 = time.time()
        self._update_sep_vectors(Z, sources, trusted_spike_mask, batch_idx)
        self.time_sv_ms.append((time.time() - t0) * 1000)

        # --- Pad back to N_called rows ---
        spikes_out = torch.zeros(
            N_called, self.decomp.sep_vectors.shape[0], dtype=torch.int32, device=self.config.device
        )
        sources_out = torch.zeros(
            N_called, self.decomp.sep_vectors.shape[0], dtype=torch.float32, device=self.config.device
        )
        spikes_out[pad_offset:] = spike_mask.to(torch.int32)
        sources_out[pad_offset:] = sources
        return spikes_out, sources_out

    # ------------------------------------------------------------------
    # Raw-batch preprocessing (streaming mode, data_preprocessed=False):
    # orchestrator (_preprocess_batch_raw) + narrow step (_center_and_extend_batch)
    # ------------------------------------------------------------------

    def _preprocess_batch_raw(self, emg_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Filter, select channels, and extend one raw batch.

        Args:
            emg_batch (torch.Tensor): Raw EMG chunk with shape (N, raw_channels).

        Returns:
            Tuple[torch.Tensor, int]: Extended batch and pad_offset, see
            _center_and_extend_batch.

        Raises:
            ValueError: If config.replace_bad_channels is True with
                config.ch_map unset, or config.ch_mask's length disagrees
                with emg_batch's raw channel count.
        """
        if self.decomp.zi is None:
            # Only checkable once the raw channel count is known, on the
            # first batch; later batches can't change it mid-run.
            validate_channel_selection(
                self.config.ch_mask, self.config.ch_map,
                self.config.replace_bad_channels, emg_batch.shape[1],
            )

        emg_np, zi_new = preprocess_emg_stateful(
            emg_batch.cpu().numpy(), self.config.fs, zi=self.decomp.zi, **filter_kwargs(self.config)
        )
        self.decomp.zi = zi_new

        emg_np = select_channels(
            emg_np, self.config.ch_mask, self.config.ch_map, self.config.replace_bad_channels
        )
        emg_batch = torch.from_numpy(emg_np).to(device=self.config.device, dtype=torch.float32)

        return self._center_and_extend_batch(emg_batch)

    def _center_and_extend_batch(self, emg_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Prepend the extension FIFO, centre, extend, and trim one batch.

        Updates decomp.ema_mean_online and decomp.ext_fifo in place.

        Args:
            emg_batch (torch.Tensor): Filtered, channel-selected batch
                with shape (N, channels).

        Returns:
            Tuple[torch.Tensor, int]: Extended batch and pad_offset, the
            number of leading rows process_batch must zero-pad the final
            output with.
        """
        ext_fact = self.config.ext_fact
        if self.decomp.ext_fifo is not None:
            window = torch.cat([self.decomp.ext_fifo, emg_batch], dim=0)
        else:
            window = emg_batch

        batch_mean = emg_batch.mean(0, keepdim=True)
        self.decomp.ema_mean_online = (
            batch_mean.detach() if self.decomp.ema_mean_online is None
            else (
                self.config.ema_alpha * self.decomp.ema_mean_online
                + (1 - self.config.ema_alpha) * batch_mean
            ).detach()
        )
        window = window - self.decomp.ema_mean_online

        window_ext = extend_data(window, ext_fact, ext_mode=self.config.ext_mode)

        if self.decomp.ext_fifo is None:
            # First call: no prior context, so the leading ext_fact rows
            # genuinely lack history and must be zero-padded by the caller.
            pad_offset = ext_fact
        else:
            # Later calls: the leading ext_fact rows duplicate the previous
            # call's own already-returned tail, so they're discarded here
            # (not zero-padded) to avoid emitting the same timestamps twice.
            pad_offset = 0
        window_ext = window_ext[ext_fact:]

        self.decomp.ext_fifo = emg_batch[-ext_fact:]
        return window_ext, pad_offset

    # ------------------------------------------------------------------
    # Whitening — orchestrator (_whiten) + narrow update (_update_whitening)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _whiten(
        self, X: torch.Tensor, batch_idx: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the whitening error and, if adapt_wh, update decomp.whitening.

        Logs loss/diagnostics either way when asked (config.compute_loss/
        .debug); see _update_whitening for the update itself.

        Args:
            X (torch.Tensor): Centred batch, with shape (N, D) or, if PCA
                is configured, (N, n) after projection.
            batch_idx (Union[int, torch.Tensor]): This batch's sequential index.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Whitened data Z,
            with shape (N, n); and the wh→sv coupling_matrix (see
            _update_whitening), or None when not applicable.

        Notes:
            config.wh_mode selects the error/direction formula:
            "kl_to_identity" — error = K − K_cal, direction = (Rz − I) @ wh.
            "kl_to_cal" — error = KL(Rz‖Rz_cal), direction = (Rz·Rz_cal⁻¹ − I) @ wh.
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
                self.wh_trace.append(Rz.trace())

            # Compute KL-divergence error and direction for the configured wh_mode
            sign, logdet = torch.linalg.slogdet(Rz)
            if sign <= 0:
                # If Rz is not positive definite, skip the whitening update and log a warning.
                if self.config.compute_loss:
                    # Use the previous batch's loss or 0.0 if this is the first batch.
                    self.wh_loss.append(self.wh_loss[-1] if self.wh_loss else 0.0)
                if self.config.debug:
                    idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
                    self.diagnostics.setdefault(idx, {})["wh_skip_invalid_slogdet"] = True
                return X @ self.decomp.whitening.T, None
            K_online, e_wh, direction = self._compute_kl_error(Rz, logdet)

            # If Rz is positive definite, log the actual error
            if self.config.compute_loss:
                self.wh_loss.append(e_wh.item() ** 2)

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
        """Push X into the extended-EMG FIFO and return the regularised whitened covariance.

        Args:
            X (torch.Tensor): Centred batch to push into the FIFO, with
                shape (N, n).

        Returns:
            torch.Tensor: Regularised whitened covariance Rz, with shape (n, n).
        """
        self.decomp._update_fifo_cov(X)
        return self.decomp._compute_Rz_from_fifo()

    @torch.no_grad()
    def _compute_kl_error(
        self, Rz: torch.Tensor, logdet: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute K_online, e_wh, and direction for the configured wh_mode.

        The two modes share intermediates (A, logdet_A in kl_to_cal), so
        they're kept in one method rather than split further.

        Args:
            Rz (torch.Tensor): Regularised whitened covariance, with shape (n, n).
            logdet (torch.Tensor): log|Rz|, scalar.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: K_online
            (scalar); e_wh, the z-scored whitening error (scalar); and
            direction, the natural-gradient direction, with shape (n, n).

        Notes:
            "kl_to_identity" — error = K − K_cal, direction = (Rz − I) @ wh.
            "kl_to_cal" — error = KL(Rz‖Rz_cal), direction = (Rz·Rz_cal⁻¹ − I) @ wh.
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
            direction = A.T - self.decomp.I   # exact steepest direction, not A - I; see Notes above

        # Z-score the whitening error so eta_wh is scale-free across contractions.
        sigma_K = getattr(self.decomp, "kl_div_calib_std", None)
        e_wh = e_wh_raw / sigma_K.clamp_min(self.config.eps) if sigma_K is not None else e_wh_raw
        return K_online, e_wh, direction

    @torch.no_grad()
    def _update_whitening(
        self, direction: torch.Tensor, e_wh: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """Apply the natural-gradient step to decomp.whitening.

        Narrow: only mutates decomp.whitening/.ema_dirnorm_wh; called only
        from _whiten, only when config.adapt_wh is True.

        Args:
            direction (torch.Tensor): Natural-gradient direction from
                _compute_kl_error, with shape (n, n).
            e_wh (torch.Tensor): Z-scored whitening error, scalar.

        Returns:
            Tuple[Optional[torch.Tensor], dict]: coupling_matrix, the
            wh→sv frame-correction matrix when config.wh_sv_coupling, else
            None; and diag, carrying delta_wh_norm/delta_wh_raw_norm for
            _whiten's debug logging.

        Notes:
            lr_mode="fixed" applies a raw fixed-learning-rate step (v1
            behaviour). lr_mode="rel_error" (default) first normalises
            direction to unit scale via an EMA of its own norm, then
            scales by wh_learning_rate and the signed e_wh.
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
        if self.config.wh_sv_coupling:
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
    # Whitening → separation coupling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_wh_sv_coupling(self, coupling_matrix: Optional[torch.Tensor]) -> None:
        """Apply the first-order frame correction implied by the wh step to sep_vectors.

        Runs before source estimates are formed, so spike detection and
        the contrast update see sep_vectors already aligned with the new
        whitening frame. No-op when coupling_matrix is None
        (config.wh_sv_coupling=False, or config.adapt_wh=False; see
        _whiten/_update_whitening).

        Args:
            coupling_matrix (Optional[torch.Tensor]): wh→sv frame-correction
                matrix from _update_whitening, with shape (n, n), or None.

        Returns:
            None
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
        """Detect spikes via vectorised NMS on the source FIFO plus the current batch.

        Prepending the source FIFO gives spikes at the left edge of the
        current batch enough left-context for NMS.

        Args:
            sources (torch.Tensor): Source estimates for this batch, with
                shape (N, M).
            N (int): Number of samples in this batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: spike_mask and peak_mask
            for this batch only, both with shape (N, M).
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
        batch_idx: Union[int, torch.Tensor],
    ) -> None:
        """Per-batch centroid step: update decomp.spikes_centr/.base_centr and log diagnostics.

        Narrow, mirroring _update_whitening's shape. Called after detection
        (_detect_spikes) and IQR-gating (gate_spikes_by_iqr) have already
        produced trusted_spike_mask, and before the sv update.

        Args:
            sources (torch.Tensor): Source estimates, with shape (N, M).
            spike_mask (torch.Tensor): Detected spikes, with shape (N, M).
            peak_mask (torch.Tensor): Detected peaks, before centroid
                classification, with shape (N, M).
            trusted_spike_mask (torch.Tensor): spike_mask with outliers
                gated out, with shape (N, M).
            batch_idx (Union[int, torch.Tensor]): This batch's sequential index.

        Returns:
            None
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
        self, Z: torch.Tensor, sources: torch.Tensor, spike_mask: torch.Tensor,
        batch_idx: Union[int, torch.Tensor],
    ) -> None:
        """Per-batch separation-vector step: update decomp.sep_vectors and log diagnostics.

        Runs the fixed-point natural-gradient update when config.adapt_sv,
        else only computes contrast error for logging (_compute_sv_diag).

        Args:
            Z (torch.Tensor): Whitened data, with shape (N, n).
            sources (torch.Tensor): Source estimates from before this
                update, with shape (N, M).
            spike_mask (torch.Tensor): Trusted spike mask, with shape (N, M).
            batch_idx (Union[int, torch.Tensor]): This batch's sequential index.

        Returns:
            None

        Notes:
            Kept as one method with an if/else, not an orchestrator/narrow
            pair like _whiten/_update_whitening: adapt_sv=True computes
            contrast error and the update together inside the fixed-point
            loop, so there's no separate "compute the error" call to share
            with the adapt_sv=False path.
            Does not change the sources process_batch returns: ipts is
            always the pre-update sources, so outputs stay consistent
            across batches.
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
            self.sv_loss.append(sv_err ** 2)

        # --- Debug diagnostics ---
        if self.config.debug:
            idx = batch_idx.item() if hasattr(batch_idx, "item") else batch_idx
            d = self.diagnostics.setdefault(idx, {})
            d.update({
                **sv_diag,
                "kappa_cal": self.decomp.contrast_calib_mean.clone(),
            })
            if "delta_sv_raw_norm" in first_sv_diag:
                # Override the last-iteration values **sv_diag contributed above;
                # see the "First sub-iteration only" comment above for why.
                d["delta_sv_norm"]     = first_sv_diag["delta_sv_norm"]
                d["delta_sv_raw_norm"] = first_sv_diag["delta_sv_raw_norm"]

    @torch.no_grad()
    def _compute_sv_diag(
        self, sources: torch.Tensor, spike_mask: torch.Tensor
    ) -> dict:
        """Compute contrast error for loss tracking when adapt_sv is False.

        Args:
            sources (torch.Tensor): Source estimates, with shape (N, M).
            spike_mask (torch.Tensor): Trusted spike mask, with shape (N, M).

        Returns:
            dict: kappa, contrast_error, spike_counts, active; all shape (M,).
        """
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
    # Per-run setup
    # ------------------------------------------------------------------

    def _init_exe_time(self) -> None:
        """Reset per-batch timing accumulators to empty lists.

        Returns:
            None
        """
        self.time_wh_ms: list = []
        self.time_sv_ms: list = []
        self.time_sd_ms: list = []
        self.time_preprocess_ms: list = []

    def _init_outputs(self, units: int) -> None:
        """Reset the per-batch spikes/sources accumulators to empty lists.

        Args:
            units (int): Number of motor units.

        Returns:
            None
        """
        self.units = units
        self._spikes_accum: list = []
        self._sources_accum: list = []

    def _init_losses(self) -> None:
        """Reset per-batch loss accumulators to empty lists, when config.compute_loss.

        Returns:
            None
        """
        if self.config.compute_loss:
            self.wh_loss: list = []
            self.sv_loss: list = []
            self.wh_trace: list = []
            # wh_loss_median/sv_loss_median/total_loss are set once, at the end of
            # process_data(), by _compute_losses(), not accumulated per batch.

    # ------------------------------------------------------------------
    # Per-run output finalisation
    # ------------------------------------------------------------------

    def _finalize_accumulators(self) -> None:
        """Convert this run's list accumulators into their final tensors.

        Returns:
            None
        """
        self.spikes = self._cat_list(self._spikes_accum, (0, self.units), dtype=torch.int32)
        self.ipts = self._cat_list(self._sources_accum, (0, self.units), dtype=torch.float32)
        if self.config.compute_loss:
            self.wh_loss = torch.tensor(self.wh_loss, dtype=torch.float32, device=self.config.device)
            self.sv_loss = self._stack_list(self.sv_loss, (0, self.units))
            self.wh_trace = self._stack_list(self.wh_trace, (0,))
        self.time_wh_ms = torch.tensor(self.time_wh_ms, dtype=torch.float32)
        self.time_sv_ms = torch.tensor(self.time_sv_ms, dtype=torch.float32)
        self.time_sd_ms = torch.tensor(self.time_sd_ms, dtype=torch.float32)
        self.time_preprocess_ms = torch.tensor(self.time_preprocess_ms, dtype=torch.float32)

    def _stack_list(self, values: list, empty_shape: Tuple[int, ...]) -> torch.Tensor:
        """Stack a list of same-shaped tensors, or an empty tensor if the list is empty.

        Args:
            values (list): Tensors to stack.
            empty_shape (Tuple[int, ...]): Shape to use when values is empty.

        Returns:
            torch.Tensor: torch.stack(values), or torch.zeros(empty_shape)
            when values is empty.
        """
        if not values:
            return torch.zeros(empty_shape, dtype=torch.float32, device=self.config.device)
        return torch.stack(values)

    def _cat_list(
        self, values: list, empty_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Concatenate a list of row tensors along dim 0, or an empty tensor if empty.

        Args:
            values (list): Row-shaped tensors to concatenate.
            empty_shape (Tuple[int, ...]): Shape to use when values is empty.
            dtype (torch.dtype): Dtype to use when values is empty.

        Returns:
            torch.Tensor: torch.cat(values, dim=0), or torch.zeros(empty_shape)
            when values is empty.
        """
        if not values:
            return torch.zeros(empty_shape, dtype=dtype, device=self.config.device)
        return torch.cat(values, dim=0)

    def _compute_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate wh_loss/sv_loss into guarded per-run scalars.

        The single canonical scores for a run; adaptation/optimize.py's
        Optuna objectives and scripts/run.py's wandb logging both read
        these directly.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: wh_loss_median
            (median(wh_loss)), sv_loss_median (nanmedian(sv_loss)), and
            total_loss (their sum); each 1e10 if wh_loss has any NaN or
            the wh_trace/trace_cal ratio indicates whitening diverged.
        """
        trace_ratio = (self.wh_trace / self.decomp.trace_cal).median()
        if torch.any(torch.isnan(self.wh_loss)) or not (0.1 < trace_ratio.item() < 50.0):
            invalid = torch.tensor(1e10, device=self.config.device)
            return invalid, invalid, invalid
        wh_loss_median = self.wh_loss.median()
        sv_loss_median = self.sv_loss.nanmedian()
        return wh_loss_median, sv_loss_median, wh_loss_median + sv_loss_median

    def _format_outputs(self) -> AdaptationResult:
        """Collect per-batch results into a typed AdaptationResult.

        Always present:
            spikes          [samples, M]    int32   — binary spike train
            ipts            [samples, M]    float32 — source signal before sv update
            wh_time_ms      [batches]       float32
            sv_time_ms      [batches]       float32
            sd_time_ms      [batches]       float32
            preprocess_time_ms [batches]    float32 — 0 when data_preprocessed=True
            total_time_ms   [batches]       float32

        Present when config.compute_loss=True:
            wh_loss         [batches]       float32
            sv_loss         [batches, M]    float32
            centroid_loss   [batches, M]    float32
            wh_trace        [batches]       float32
            wh_loss_median  scalar          float32 — see _compute_losses()
            sv_loss_median  scalar          float32 — see _compute_losses()
            total_loss      scalar          float32 — see _compute_losses()

        Present when config.debug=True:
            diagnostics     dict            per-batch diagnostic tensors

        Returns:
            AdaptationResult: Typed result; call .to_dict() for the equivalent
            plain dict, or subscript it directly (result["wh_loss"]) for
            dict-style access.
        """
        result = AdaptationResult(
            spikes=self.spikes.detach().cpu().clone(),
            ipts=self.ipts.detach().cpu().clone(),
            wh_time_ms=self.time_wh_ms,
            sv_time_ms=self.time_sv_ms,
            sd_time_ms=self.time_sd_ms,
            preprocess_time_ms=self.time_preprocess_ms,
            total_time_ms=(
                self.time_wh_ms + self.time_sv_ms + self.time_sd_ms + self.time_preprocess_ms
            ),
        )
        if hasattr(self, "wh_loss"):
            result.wh_loss = self.wh_loss.detach().cpu().clone()
        if hasattr(self, "sv_loss"):
            result.sv_loss = self.sv_loss.detach().cpu().clone()
        if hasattr(self, "centroid_loss"):
            result.centroid_loss = self.centroid_loss.detach().cpu().clone()
        if hasattr(self, "wh_trace"):
            result.wh_trace = self.wh_trace.detach().cpu().clone()
        if hasattr(self, "wh_loss_median"):
            result.wh_loss_median = self.wh_loss_median.detach().cpu().clone()
        if hasattr(self, "sv_loss_median"):
            result.sv_loss_median = self.sv_loss_median.detach().cpu().clone()
        if hasattr(self, "total_loss"):
            result.total_loss = self.total_loss.detach().cpu().clone()
        if self.config.debug and hasattr(self, "diagnostics"):
            result.diagnostics = self.diagnostics
        if hasattr(self, "gt_matched_indices"):
            result.gt_matched_indices = self.gt_matched_indices
        return result

# ---------------------------------------------------------------------------
# Config reconciliation
# ---------------------------------------------------------------------------

# Fields with identical names on both CBSSConfig and AdaptConfig that must
# agree for a calibration to be tracked correctly online; see
# from_calibration()/reconcile_with_calib_config() below.
_SHARED_CBSS_ADAPT_FIELDS = (
    "ext_fact", "ext_mode", "spike_det_exp",
    "fs", "lowcut", "highcut", "filter_order",
    "powerline", "powerline_freq",
    "notch_width_hz", "notch_n_harmonics", "notch_order",
    "ch_mask", "ch_map", "replace_bad_channels",
)


@dataclass
class SharedCalibFields:
    """The subset of calibration config that online adaptation must agree with.

    Same 15 field names/types as AdaptConfig's own; see
    _SHARED_CBSS_ADAPT_FIELDS. Lets reconcile_with_calib_config() work from
    any calibration source, not just a CBSSConfig.
    """
    ext_fact: int
    ext_mode: Literal["block", "toeplitz"]
    spike_det_exp: float
    fs: int
    lowcut: float
    highcut: float
    filter_order: int
    powerline: bool
    powerline_freq: float
    notch_width_hz: float
    notch_n_harmonics: int
    notch_order: int
    ch_mask: Optional[np.ndarray]
    ch_map: Optional[np.ndarray]
    replace_bad_channels: bool

    @classmethod
    def from_cbss_config(cls, cbss_config: CBSSConfig) -> "SharedCalibFields":
        """Extract this wrapper's fields from a real CBSSConfig.

        Args:
            cbss_config (CBSSConfig): Calibration config to read from.

        Returns:
            SharedCalibFields: This calibration's shared-field values.
        """
        return cls(**{field: getattr(cbss_config, field) for field in _SHARED_CBSS_ADAPT_FIELDS})


def _fields_differ(adapt_val: Any, cbss_val: Any) -> bool:
    """Compare one shared field's adapt_config/cbss_config values, array-safe.

    Plain != raises ValueError ("truth value of an array is ambiguous") when
    either side is an np.ndarray (ch_mask/ch_map); use np.array_equal for
    those instead, falling back to != for every scalar/bool/str field.

    Args:
        adapt_val (Any): This field's value on adapt_config.
        cbss_val (Any): This field's value on the calibration's own config.

    Returns:
        bool: True if the two values differ.
    """
    if isinstance(adapt_val, np.ndarray) or isinstance(cbss_val, np.ndarray):
        if adapt_val is None or cbss_val is None:
            return adapt_val is not cbss_val
        return not np.array_equal(adapt_val, cbss_val)
    return adapt_val != cbss_val


def reconcile_with_calib_config(adapt_config: AdaptConfig, shared: SharedCalibFields) -> AdaptConfig:
    """Copy adapt_config, overwriting any of _SHARED_CBSS_ADAPT_FIELDS that
    disagree with shared: the calibration's own values, treated as ground truth.

    Never mutates the caller's adapt_config in place. Warns once (not once per
    field) if anything was overwritten, listing every field actually changed.

    Args:
        adapt_config (AdaptConfig): Caller-supplied online adaptation
            configuration to reconcile.
        shared (SharedCalibFields): The calibration's own shared-field
            values, treated as ground truth.

    Returns:
        AdaptConfig: A copy of adapt_config with every disagreeing shared
        field set to shared's value.
    """
    adapt_config = copy(adapt_config)
    changed = []
    for field in _SHARED_CBSS_ADAPT_FIELDS:
        cbss_val, adapt_val = getattr(shared, field), getattr(adapt_config, field)
        if _fields_differ(adapt_val, cbss_val):
            changed.append((field, adapt_val, cbss_val))
            setattr(adapt_config, field, cbss_val)
    if changed:
        lines = "\n".join(f"  {f}: {old!r} -> {new!r}" for f, old, new in changed)
        warnings.warn(
            f"adapt_config disagreed with cbss_config on {len(changed)} shared field(s); "
            f"cbss_config is treated as ground truth and these were overwritten:\n{lines}\n"
            "Update how you construct AdaptConfig to match self.config to silence this.",
            UserWarning, stacklevel=3,
        )
    return adapt_config


# ---------------------------------------------------------------------------
# Module-level utilities
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
