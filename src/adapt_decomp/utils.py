"""Functions to evaluate motor unit properties"""

import numpy as np
from typing import Union, Optional, Tuple, List
from scipy import signal
import itertools

def firings_to_spikes(firings, ipts, matlab_index=False):
    """
    Convert firings to spikes
    """
    spikes = np.zeros_like(ipts)
    for i, firing in enumerate(firings):
        if matlab_index:
            firing = firing - 1
        spikes[i, firing.astype(int)] = 1

    return spikes

def _check_mu_format(data: np.ndarray) -> np.ndarray:
    """Check data is 2D and return it.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: 2D data array.
    """

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    return data


def get_discharge_rate(
        spike_train: np.ndarray,
        timestamps: Union[list, np.ndarray]
        ) -> np.ndarray:
    """Compute the discharge rate of motor units.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        timestamps (Union[list, np.ndarray]): Array of timestamps corresponding
            to the spike train.

    Returns:
        np.ndarray: Array of discharge rates for each motor unit.
    """

    # Get number of motor units and initialise dr
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    dr = np.zeros(units)

    for unit in range(units):
        # Compute the total number of firings
        n_spikes = np.sum(spike_train[:, unit].astype(int))

        if n_spikes == 0:
            continue

        # Get firing times
        times_spikes = timestamps[spike_train[:, unit]]

        # Get total active period
        total_period = times_spikes[-1] - times_spikes[0]

        if total_period == 0:
            continue

        # Calculate interspike interval (ISI)
        isi = np.diff(times_spikes)

        # Find silent periods (i.e. where the ISI is larger than the minimum
        # motor unit discharge rate). This is 4 Hz according to "Negro F (2016)
        # Multi-channel intramuscular and surface EMG decomposition by
        # convolutive blind source separation."
        silent_period = np.sum(isi[isi > 0.25])

        # Calculate the actual active period of the motor unit
        active_period = total_period - silent_period

        # Return mean discharge rate
        dr[unit] = n_spikes / active_period

    return dr


def get_number_of_spikes(spike_train: np.ndarray) -> np.ndarray:
    """Function to compute the number of spikes.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.

    Returns:
        np.ndarray: number of spikes for each motor unit
    """

    # Compute the number of spikes
    n_spikes = np.sum(spike_train.astype(int), axis=0)

    return n_spikes


def get_inst_discharge_rate(
        spike_train: np.ndarray,
        fs: Optional[int] = 2048
        ) -> np.ndarray:
    """Compute the instantaneous discharge rate of motor units.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to 2048.

    Returns:
        np.ndarray: Array of instantaneous discharge rates for each motor unit.
    """

    # Get number of motor units and initialise ints_DR
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    inst_dr = np.zeros(spike_train.shape)

    # Define hanning window
    dur = 1  # (s) for the moving average
    hann_win = np.hanning(np.round(dur * fs))

    for unit in range(units):
        # Convolve the hanning window and the binary spikes
        inst_dr[:, unit] = np.convolve(
            spike_train[:, unit], hann_win, mode='same'
            ) * 2

    return inst_dr


def get_coefficient_of_variation(
        spike_train: np.ndarray,
        timestamps: Union[list, np.ndarray]
        ) -> np.ndarray:
    """
    Calculate the coefficient of variation (CoV) for each motor unit in a spike
    train.

    Parameters:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        timestamps (Union[list, np.ndarray]): Array of timestamps
            corresponding to each time point in the spike train.

    Returns:
        np.ndarray: Array of CoV values for each motor unit, scaled by 100.

    Notes:
        - The CoV is calculated as the standard deviation of the interspike
          intervals divided by the mean interspike interval.
        - Interspike intervals greater than 0.25 s (or discharge rate less than
          4 Hz) and intervals less than 0.02 s (or discharge rate greater than
          50 Hz) are discarded, based on "Negro F (2016). Multi-channel
          intramuscular and surface EMG decomposition by convolutive blind
          source separation."
    """
    # Function implementation
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    cov = np.zeros(units)
    cov[:] = np.nan

    for unit in range(units):
        if not np.any(spike_train[:, unit]):
            continue

        times_spikes = timestamps[spike_train[:, unit]]
        isi = np.diff(times_spikes)
        isi = isi[isi < 0.25]
        cov[unit] = np.std(isi) / np.mean(isi)

    return cov * 100


def get_pulse_to_noise_ratio(
    spike_train: np.ndarray,
    ips: np.ndarray,
    ext_fact: int = 8
    ) -> np.ndarray:
    """Compute the pulse-to-noise ratio (PNR) for each motor unit.

    Args:
    spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
        where n is the number of time points and m is the number of motor
        units.
    ips (np.ndarray): Innervated pulse trains (IPTs) with shape (n, m),
        where n is the number of time points and m is the number of motor
        units.
    ext_fact (int, optional): Extension factor to discard initial spikes.
        Defaults to 8.

    Returns:
    np.ndarray: Array of PNR values for each motor unit.

    Notes:
    - The PNR is calculated as 20 * log10(spikes_mean / baseline_mean),
      where spikes_mean is the mean of the IPTs corresponding to the spikes
      and baseline_mean is the mean of the IPTs corresponding to the
      baseline.
    - The baseline is defined as the IPTs with amplitude lower than the
      lowest spike.
    - IPTs greater than the extension factor are considered for both spikes
      and baseline.
    """

    # Get number of motor units and initialise PNR
    spike_train = _check_mu_format(spike_train.astype(bool))
    ips = _check_mu_format(ips)
    units = spike_train.shape[-1]
    pnr = np.zeros(units)
    pnr[:] = np.nan

    # Square IPTs
    ipts2 = ips ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline = ipts2[baseline_peaks_idx, unit]

        if len(spikes) == 0:
            continue

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes)
        baseline_mean = np.mean(baseline)

        # Compute PNR
        pnr[unit] = 20 * np.log10(spikes_mean / baseline_mean)

    return pnr


def get_silhouette_measure(
    spike_train: np.ndarray,
    ipts: np.ndarray,
    ext_fact: int = 8
) -> np.ndarray:
    """Compute the silhouette measure for each motor unit.

    Args:
        spike_train (np.ndarray): Binary spike train matrix of shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ips (np.ndarray): Innervated pulse trains (IPTs) with shape (n, m),
            where n is the number of time points and m is the number of motor
            units.
        ext_fact (int, optional): Extension factor to discard initial spikes.
            Defaults to 8.

    Returns:
        np.ndarray: Array of silhouette measures for each motor unit.

    Notes:
        - The silhouette measure is a measure of how well-separated the spikes
          and baseline are in terms of their IPTs.
        - The silhouette measure is calculated as
          (dist_sum_baseline - dist_sum_spikes) / max_dist, where
          dist_sum_baseline is the sum of squared distances between each
          baseline IPT and the mean baseline IPT, dist_sum_spikes is the sum
          of squared distances between each spike IPT and the mean spike IPT,
          and max_dist is the maximum of dist_sum_baseline and dist_sum_spikes.
    """

    # Get number of motor units and initialise sil
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    sil = np.zeros(units)
    sil[:] = np.nan

    # Square IPTs
    ipts2 = ipts ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes_amp = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes_amp)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline_amp = ipts2[baseline_peaks_idx, unit]

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes_amp)
        baseline_mean = np.mean(baseline_amp)

        # Compute distances
        dist_sum_spikes = np.sum(np.power((spikes_amp - spikes_mean), 2))
        dist_sum_baseline = np.sum(np.power((spikes_amp - baseline_mean), 2))

        # Compute sil
        max_dist = np.amax([dist_sum_spikes, dist_sum_baseline])
        if max_dist == 0:
            sil[unit] = 0
        else:
            sil[unit] = (dist_sum_baseline - dist_sum_spikes) / max_dist

    return sil

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
        pair_idx = np.arange(n_units)
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
    
    if spike_trains_ref.shape[0] != spike_trains_test.shape[0]:
        raise ValueError(f'Time dimensionality mismatch between ref {spike_trains_ref.shape} and test {spike_trains_test.shape}.')

    # Put tolerances into samples
    tol_spike = round(tol_spike_ms/1000 * fs)
    tol_train = round(tol_train_ms/1000 * fs)

    # Initialise test variables
    n_units_test = spike_trains_test.shape[1]

    #  If no spike trains to test are provided, return empty RoA
    if not np.any(spike_trains_test):
        pair_idx = np.arange(n_units_test)
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
        # len_firings_1 = len(firings_1)
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