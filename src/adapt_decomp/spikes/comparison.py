"""Rate of agreement functions for spike comparsion."""

import numpy as np
from typing import List, Optional, Tuple, Union, Dict
import scipy.signal as signal
import itertools
import torch
from loguru import logger
from .metrics import get_coefficient_of_variation

def rate_of_agreement_paired(
    spike_trains_ref: np.ndarray,
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the rate of agreement between two sets of paired spike trains.

    Args:
        spike_trains_ref (np.ndarray): Reference spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n),
            where m is the number of samples and n is the number of motor units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains,
              with shape (n).
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units, with shape (n).

    Note:
        - The function assumes that the spike trains between the sets are matched 
          and in the same order.
    """
    # Check spike trains shape
    if len(spike_trains_ref.shape) == 1:
        spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if len(spike_trains_test.shape) == 1:
        spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)

    if spike_trains_ref.shape != spike_trains_test.shape:
        raise ValueError(f'Dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms / 1000 * fs)
    tol_train = round(tol_train_ms / 1000 * fs)

    # Initialise test variables
    n_units = spike_trains_test.shape[1]

    #  If there are no spikes return empty RoA
    if (not np.any(spike_trains_ref)) | (not np.any(spike_trains_test)):
        pair_idx = [(unit, unit) for unit in range(n_units)]
        pair_lag = np.zeros((n_units))
        roa = np.zeros((n_units))
        return roa, pair_idx, pair_lag

    # Compute the RoA between the sets
    #  --------------------------------
    # Initialise correlation variables
    spikes_corr = np.zeros((n_units))
    roa = np.empty((n_units))
    pair_lag = np.zeros((n_units))
    pair_idx = [(unit, unit) for unit in range(n_units)]

    for unit in range(n_units):
        #  Align spike trains based on their correlation and spike tol
        # -----------------------------------------------------------
        #  Get trains
        train_ref = spike_trains_ref[:, unit]
        train_test = spike_trains_test[:, unit]
        # Apply spike tolerance
        train_ref = np.convolve(train_ref, np.ones(tol_spike), mode="same")
        train_test = np.convolve(train_test, np.ones(tol_spike), mode="same")
        # Compute correlation and lags
        curr_corr = signal.correlate(train_ref, train_test, mode="full")
        curr_lags = signal.correlation_lags(
            len(train_ref), len(train_test), mode="full"
        )
        # Apply train shift tolerance
        train_tol_idxs = np.nonzero(np.abs(curr_lags) == tol_train)[0]
        train_tol_mask = np.arange(train_tol_idxs[0], train_tol_idxs[-1] + 1).astype(
            int
        )
        curr_corr = curr_corr[train_tol_mask]
        curr_lags = curr_lags[train_tol_mask]
        # Identify optimal lag for alignment
        trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
        if not np.isscalar(trains_lag):
            # If there is more than one possible lag, choose the minimum
            trains_lag = np.amin(trains_lag)
        # Fill matrices
        spikes_corr[unit] = np.amax(curr_corr)
        pair_lag[unit] = trains_lag

        #  Compute rate of agreement between the aligned spike trains
        # ----------------------------------------------------------
        # Align spike trains
        firings_ref = np.nonzero(spike_trains_ref[:, unit])[0]
        firings_test = np.nonzero(spike_trains_test[:, unit])[0] + pair_lag[unit]
        # Initialise variables
        firings_common = 0
        firings_ref_only = 0
        firings_test_only = 0
        # Pair firings
        for firing in firings_ref:
            curr_firing_diff = np.abs(firings_test - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A common firing
                firings_common += 1
                firings_test = np.delete(firings_test, np.argmin(curr_firing_diff))
            else:
                # Only in reference firings
                firings_ref_only += 1
        firings_test_only = len(firings_test)
        # Compute rate of agreement
        roa[unit] = firings_common / (
            firings_common + firings_ref_only + firings_test_only
        )

    return roa, pair_idx, pair_lag


def rate_of_agreement(
    spike_trains_ref: Union[np.ndarray, None],
    spike_trains_test: np.ndarray,
    fs: Optional[int] = 2048,
    tol_spike_ms: Optional[int] = 1,
    tol_train_ms: Optional[int] = 40,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Compute the rate of agreement between two sets of spike trains.

    Args:
        spike_trains_ref (Union[np.ndarray, None]): Reference spike trains 
            with shape (m, n1) where m is the number of samples and n1 is the
            number of motor units in the reference set. If None are provided, 
            the function will compute the RoA within the test set.
        spike_trains_test (np.ndarray): Test spike trains with shape (m, n2),
            where m is the number of samples and n2 is the number of motor units
            in the test set.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (Optional[int], optional): Spike tolerance in milliseconds.
            Defaults to 1.
        tol_train_ms (Optional[int], optional): Train shift tolerance in milliseconds.
            Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: A tuple containing:
            - RoA (np.ndarray): Rate of agreement between the aligned spike trains.
            - pair_idx (List[Tuple[int, int]]): List of pairs of motor units that
              have the highest rate of agreement.
            - pair_lag (np.ndarray): Optimal lag for alignment between the pairs of
              motor units.

    Note:
        - The function does not assume that the spike trains between the sets are 
          matched nor in the same order.
        - The dimensions of the output arrays will depend on the number of matched
          pairs between the sets.
    """

    # Check spike trains shape
    if spike_trains_ref is not None:
        if len( spike_trains_ref.shape ) == 1:
            spike_trains_ref = np.expand_dims(spike_trains_ref, axis=-1)

    if spike_trains_test is not None:
        if len( spike_trains_test.shape ) == 1:
            spike_trains_test = np.expand_dims(spike_trains_test, axis=-1)
    
    if spike_trains_ref is not None and spike_trains_ref.shape[0] != spike_trains_test.shape[0]:
        raise ValueError(f'Time dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms/1000 * fs)
    tol_train = round(tol_train_ms/1000 * fs)

    # Initialise test variables
    n_units_test = spike_trains_test.shape[1]

    #  If no spike trains to test are provided, return empty RoA
    if not np.any(spike_trains_test):
        pair_idx = [(unit, unit) for unit in range(n_units_test)]
        pair_lag = np.zeros((n_units_test))
        roa = np.zeros((n_units_test))
        return roa, pair_idx, pair_lag

    if spike_trains_ref is None:
        # Only one set provided, compute the RoA within set
        # -------------------------------------------------
        # Initialise correlation variables
        spikes_corr = np.zeros((n_units_test, n_units_test))
        spikes_lag = np.zeros((n_units_test, n_units_test))

        #  Align spike trains based on their correlation and spike tol
        pairs = itertools.combinations(range(n_units_test), 2)
        for pair in pairs:
            #  Get trains
            train_0 = spike_trains_test[:, pair[0]]
            train_1 = spike_trains_test[:, pair[1]]
            # Apply spike tolerance
            train_0 = np.convolve(train_0, np.ones(tol_spike), mode="same")
            train_1 = np.convolve(train_1, np.ones(tol_spike), mode="same")
            # Compute correlation and lags
            curr_corr = signal.correlate(train_0, train_1, mode="full")
            curr_lags = signal.correlation_lags(len(train_0), len(train_1), mode="full")
            # Identify optimal lag for alignment
            trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
            if not np.isscalar(trains_lag):
                # If there is more than one possible lag, choose the minimum
                trains_lag = np.amin(trains_lag)
            # Ensure alignment is within tolerance
            if np.abs(trains_lag) > tol_train:
                trains_lag = 0
            # Fill matrices
            spikes_corr[pair] = np.amax(curr_corr)
            spikes_lag[pair] = int(trains_lag)

    else:
        # Compute the RoA between the sets
        #  --------------------------------
        # Initialise reference variables
        n_units_ref = spike_trains_ref.shape[-1]

        # Initialise correlation variables
        spikes_corr = np.zeros((n_units_ref, n_units_test))
        spikes_lag = np.zeros((n_units_ref, n_units_test))

        #  Align spike trains based on their correlation and spike tol
        for unit_ref in range(n_units_ref):
            for unit_test in range(n_units_test):
                #  Get trains
                train_0 = spike_trains_ref[:, unit_ref]
                train_1 = spike_trains_test[:, unit_test]
                # Apply spike tolerance
                train_0 = np.convolve(train_0, np.ones(tol_spike), mode="same")
                train_1 = np.convolve(train_1, np.ones(tol_spike), mode="same")
                # Compute correlation and lags
                curr_corr = signal.correlate(train_0, train_1, mode="full")
                curr_lags = signal.correlation_lags(
                    len(train_0), len(train_1), mode="full"
                )
                # Identify optimal lag for alignment
                trains_lag = curr_lags[np.argmax(np.abs(curr_corr))]
                if not np.isscalar(trains_lag):
                    # If there is more than one possible lag, choose the minimum
                    trains_lag = np.amin(trains_lag)
                # Fill matrices
                spikes_corr[unit_ref, unit_test] = np.amax(curr_corr)
                spikes_lag[unit_ref, unit_test] = int(trains_lag)

    # Find most likely pairs by progressively taking the max corr
    pair_idx = []
    pair_lag = []
    while np.any(sum(spikes_corr)):
        idx_max_corr = np.unravel_index(np.argmax(spikes_corr), spikes_corr.shape)
        pair_idx.append(idx_max_corr)
        pair_lag.append(int(spikes_lag[idx_max_corr]))
        spikes_corr[idx_max_corr[0], :] = 0
        spikes_corr[:, idx_max_corr[1]] = 0
        if spike_trains_ref is None:
            spikes_corr[idx_max_corr[1], :] = 0
            spikes_corr[:, idx_max_corr[0]] = 0

    # Compute rate of agreement
    roa = np.empty((len(pair_idx)))
    for i, pair in enumerate(pair_idx):
        # Get corresponding firings and apply optimal lag
        if spike_trains_ref is None:
            firings_0 = np.nonzero(spike_trains_test[:, pair[0]])[0]
        else:
            firings_0 = np.nonzero(spike_trains_ref[:, pair[0]])[0]
        firings_1 = np.nonzero(spike_trains_test[:, pair[1]])[0] + pair_lag[i]

        # Initialise variables
        firings_common = 0
        firings_0_only = 0
        firings_1_only = 0
        # Pair firings
        for firing in firings_0:
            curr_firing_diff = np.abs(firings_1 - firing)
            if np.any(curr_firing_diff <= tol_spike):
                # A common firing
                firings_common += 1
                firings_1 = np.delete(firings_1, np.argmin(curr_firing_diff))
            else:
                # Only in firings 0
                firings_0_only += 1
        firings_1_only = len(firings_1)
        # Compute rate of agreement
        roa[i] = firings_common / (firings_common + firings_0_only + firings_1_only)

    #  Align the indexes to the reference
    first_pair = [pair[1] for pair in pair_idx]
    pairs_sort_idx = np.argsort(first_pair)

    roa_sorted = roa[pairs_sort_idx]
    pair_idx_sorted = [pair_idx[i] for i in pairs_sort_idx]
    pair_lag_sorted = [int(pair_lag[i]) for i in pairs_sort_idx]

    return roa_sorted, pair_idx_sorted, pair_lag_sorted


def pair_ground_truth(
    spikes_gt: np.ndarray,
    spikes_calib: np.ndarray,
    fs: int,
    tol_spike_ms: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Match ground truth spikes to calibration window

    Args:
        spikes_gt (np.ndarray): Ground-truth binary spike train for the full
            recording, with shape (samples, n_gt).
        spikes_calib (np.ndarray): Calibration binary spike train (e.g.
            CBSSResult.spikes), with shape (samples_calib, n_dec).
        fs (int): Sampling frequency in Hz.
        tol_spike_ms (float, optional): Spike-coincidence tolerance in ms,
            forwarded to rate_of_agreement. Defaults to 2.0.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            - gt_full_bin with matched spikes_gt to spikes_calib's units, shape
            (samples, n_matched) (n_matched <= n_dec if rate_of_agreement's
            leaves some decomposed units unmatched))
            - roa_calib: per-matched-unit rate of agreement over the calibration
             window with shape (, n_matched)
        (None, None) if spikes_calib has no units or none is matched.
    """
    if spikes_calib.shape[1] == 0:
        return None, None

    roa_calib, pair_calib, _ = rate_of_agreement(
        spikes_gt[: spikes_calib.shape[0]], spikes_calib, fs=fs, tol_spike_ms=tol_spike_ms,
    )
    if not pair_calib:
        return None, None

    gt_idx = np.array(pair_calib)[:, 0]
    return spikes_gt[:, gt_idx], roa_calib


def spikes_dict_to_binary(
    spikes_dict: Dict[int, np.ndarray],
    n_samples: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert {unit: spike_indices} to a [T, n_units] int32 binary matrix."""
    spike_trains = torch.zeros(n_samples, len(spikes_dict), dtype=torch.int32, device=device)
    for unit, idx in spikes_dict.items():
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        idx_t = idx_t[(idx_t >= 0) & (idx_t < n_samples)]
        if idx_t.numel() > 0:
            spike_trains[idx_t, int(unit)] = 1
    return spike_trains



def remove_duplicates(
    result: Dict,
    fs: float,
    *,
    roa_th: float = 0.3,
    tol_train_ms: float = 40.0,
    tol_spike_ms: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    """Remove duplicate motor units (by RoA), keeping the unit with lower CoV-ISI.

    Operates on result['sources'] [T, n_mu] and result['spikes_dict'].
    Per-unit arrays preserved: sil, cov_isi, spikes_centr, base_centr,
    sep_vectors [dim, n_mu].
    """
    sources = result.get("sources")
    spikes_dict = result.get("spikes_dict", {})
    if sources is None or len(spikes_dict) < 2:
        return result

    _device = torch.device(device)
    sources_t = torch.as_tensor(sources, dtype=dtype, device=_device)
    n_samples = sources_t.shape[0]

    spike_trains = spikes_dict_to_binary(spikes_dict, n_samples, device=_device)
    n_units = spike_trains.shape[1]
    if n_units < 2:
        return result

    timestamps = torch.arange(n_samples, dtype=dtype, device=_device) / fs
    cov_isi = get_coefficient_of_variation(spike_trains, timestamps, None)
    cov_isi_np = torch.where(
        torch.isnan(cov_isi), torch.full_like(cov_isi, torch.inf), cov_isi
    ).cpu().numpy()

    roa_vals, pairs, _ = rate_of_agreement(
        None, spike_trains.cpu().numpy(), fs=int(fs),
        tol_spike_ms=tol_spike_ms, tol_train_ms=tol_train_ms,
    )
    sort_order = np.argsort(roa_vals)[::-1]
    keep = np.ones(n_units, dtype=bool)
    for sort_idx in sort_order:
        score = float(roa_vals[sort_idx])
        if score <= roa_th:
            break
        i, j = pairs[sort_idx]
        if not (keep[i] and keep[j]):
            continue
        if cov_isi_np[i] <= cov_isi_np[j]:
            keep[j] = False
            removed, kept = j, i
        else:
            keep[i] = False
            removed, kept = i, j
        if verbose:
            logger.debug(f"Removed duplicate unit {removed} (RoA={score:.3f}, kept unit {kept})")

    keep_idx = np.where(keep)[0]
    if len(keep_idx) == n_units:
        return result

    result["sources"] = sources_t[:, keep_idx].cpu().numpy()
    result["spikes_dict"] = {
        new_id: spikes_dict[int(old_id)] for new_id, old_id in enumerate(keep_idx)
    }
    for key in ("sil", "cov_isi", "spikes_centr", "base_centr"):
        value = result.get(key)
        if value is not None and len(np.asarray(value)) == n_units:
            result[key] = np.asarray(value)[keep_idx]
    sep_vectors = result.get("sep_vectors")
    if sep_vectors is not None:
        sep_vectors = np.asarray(sep_vectors)
        if sep_vectors.ndim == 2 and sep_vectors.shape[1] == n_units:
            result["sep_vectors"] = sep_vectors[:, keep_idx]

    return result