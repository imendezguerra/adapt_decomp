"""Functions to preprocess EMG signals"""

from typing import Optional
import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[list] = None,
    order: Optional[int] = 4,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a bandpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): List with the cutoff frequencies of the filter.
            Default is [20, 500].
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define cutoff frequencies
    if cutoff is None:
        cutoff = [20, 500]

    # Define filter
    sos = signal.butter(order, cutoff, btype='band', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out


def highpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 20,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a highpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): High cutoff frequency of the filter. Default is 20.
        order (Optional[int]): The order of the filter. Default is 2.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out


def lowpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 500,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Apply a lowpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): Low cutoff frequency of the filter. Default is 20.
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.

    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out

 
def remove_powerline(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 50,
    width: Optional[float] = 1,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """
    Remove powerline noise from the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        fs (Optional[int]): The sampling frequency of the data. Default is 2048.
        cutoff (Optional[list]): Cutoff frequency of the filter. Default is 50.
        width (Optional[float]): Width of the filter. Default is 1. 
        order (Optional[int]): The order of the filter. Default is 4.
        filtfilt (Optional[bool]): Whether to use forward-backward filtering.
            Default is True.
    Returns:
        np.ndarray: The filtered data.
    """

    # Build cutoff
    cutoff = [cutoff - width/2, cutoff + width/2]

    # Define filter
    sos = signal.butter(order, cutoff, btype='bandstop', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data)
    else:
        out = signal.sosfilt(sos, data)

    return out