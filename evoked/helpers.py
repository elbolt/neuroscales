import yaml
import numpy as np


def load_config(config_path: str) -> dict:
    """ Load configuration from YAML file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def cluster_signal(
        signal_array: np.ndarray,
        channel_list: list,
        cluster_list: list
) -> np.ndarray:
    """ Calculate the average signal of a cluster of channels

    Parameters
    ----------
    signal_array : np.ndarray
        The signal array of shape (n_channels, n_samples)
    channel_list : list
        The list of channel names
    cluster_list : list
        The list of channel names in the cluster

    Returns
    -------
    np.ndarray
        The average signal of the cluster

    """
    cluster_indices = [channel_list.index(cluster) for cluster in cluster_list]
    cluster_signal = np.mean(signal_array[cluster_indices, :], axis=0)

    return cluster_signal


def find_peak(
    signal_array: np.ndarray,
    signal_times: np.ndarray,
    window: list,
    polarity: str
) -> tuple[float, int]:
    """ Find the peak of a signal within a given window

    Parameters
    ----------
    signal_array : np.ndarray
        The signal array of shape (n_channels, n_samples)
    signal_times : np.ndarray
        The time points of the signal
    window : list
        The window of interest
    polarity : str
        The polarity of the peak (either `positive` or `negative`)

    Returns
    -------
    tuple[float, int]
        The time and index of the peak

    """
    start, end = window[0], window[1]
    start_idx = np.where(signal_times >= start)[0][0]
    end_idx = np.where(signal_times <= end)[0][-1]
    if polarity == 'positive':
        peak_range_idx = np.argmax(signal_array[start_idx:end_idx])
    elif polarity == 'negative':
        peak_range_idx = np.argmin(signal_array[start_idx:end_idx])
    else:
        raise ValueError('Polarity must be either `positive` or `negative`.')
    peak_time = signal_times[start_idx:end_idx][peak_range_idx]

    # get actual idx of peak in the whole signal
    peak_idx = np.where(signal_times == peak_time)[0][0]

    return peak_time, peak_idx


def mean_amplitude_around_peak(
    signal_array: np.ndarray,
    signal_times: np.ndarray,
    peak_time: float,
    sfreq: float,
    window: int,
) -> np.ndarray:
    """ Calculate the mean amplitude around a peak in a signal array.

    Parameters
    ----------
    signal_array : np.ndarray
        The signal array.
    signal_times : np.ndarray
        The times array.
    peak_time : float
        The peak time.
    sampling_rate : float
        The sampling rate.
    window : int
        The window around the peak in seconds.

    """
    window_samples = int(window * sfreq)
    peak_idx = np.where(signal_times == peak_time)[0][0]
    start_idx = peak_idx - window_samples
    end_idx = peak_idx + window_samples
    mean_amplitude = np.mean(signal_array[start_idx:end_idx])

    return mean_amplitude
