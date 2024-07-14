import yaml
import numpy as np
from pathlib import Path
from scipy.signal import hilbert

import brian2 as b2
import brian2hears as b2h
import mne


def load_config(config_path: str) -> dict:
    """ Load configuration from YAML file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def extract_envelope(
        stimulus: Path,
        center_freqs: np.ndarray,
        compression: float,
        sfreq: float,
        sfreq_goal: float,
        alias_dict: dict
) -> np.ndarray:
    """ Extract the envelope of the stimulus using a gammatone filterbank.

    Procedure:
    1. Load the sound.
    2. Apply a gammatone filterbank.
    3. Half-wave rectify and compress the output.
    4. Average the subbands.
    5. Resample envelope after applying anti-aliasing filter.

    Parameters
    ----------
    stimulus : Path
        Path to the stimulus.
    center_freqs : np.ndarray
        Center frequencies of the gammatone filterbank.
    compression : float
        Exponent of the compression function.
    sfreq : float
        Sampling frequency of the stimulus.
    sfreq_goal : float
        Desired sampling frequency of the envelope.
    alias_dict : dict
        Dictionary with the IIR filter parameters.

    Returns
    -------
    envelope : np.ndarray
        Envelope of the stimulus.

    """
    sound = b2h.loadsound(str(stimulus))
    gammatone = b2h.Gammatone(sound, center_freqs)
    filterbank = b2h.FunctionFilterbank(gammatone, lambda x: b2.clip(x, 0, b2.Inf)**(compression))
    subbands = filterbank.process()
    envelope = subbands.mean(axis=1)

    envelope = mne.filter.filter_data(
        data=envelope,
        sfreq=sfreq,
        l_freq=None,
        h_freq=sfreq_goal / 3.0,
        h_trans_bandwidth=sfreq_goal / 10.0,
        method='iir',
        iir_params=alias_dict
    )
    envelope = mne.filter.resample(envelope, down=sfreq / sfreq_goal, npad='auto')

    return envelope


def extract_envelope_phase(
        envelope: np.ndarray,
        sfreq: float,
        sfreq_goal: float,
        freq_min: float,
        freq_max: float,
        iir_params: dict
) -> np.ndarray:
    """ Extract the phase of the envelope at the desired frequency band.

    Procedure:
    1. Band-pass filter the envelope.
    2. Resample the envelope.
    3. Compute the phase of the envelope using the Hilbert transform.

    Parameters
    ----------
    envelope : np.ndarray
        Envelope of the stimulus.
    sfreq : float
        Sampling frequency of the envelope.
    sfreq_goal : float
        Desired sampling frequency of the envelope.
    freq_min : float
        Lower frequency of the band-pass filter.
    freq_max : float
        Upper frequency of the band-pass filter.
    iir_params : dict
        Dictionary with the IIR filter parameters.

    Returns
    -------
    phase_envelope : np.ndarray
        Phase of the envelope at the desired frequency band.

    """
    envelope = mne.filter.filter_data(
        data=envelope,
        sfreq=sfreq,
        l_freq=freq_min,
        h_freq=freq_max,
        method='iir',
        iir_params=iir_params
    )

    envelope = mne.filter.resample(envelope, down=sfreq / sfreq_goal, npad='auto')

    phase_envelope = np.angle(hilbert(envelope))

    return phase_envelope


def extract_eeg_phase(
    epochs: mne.Epochs,
    sfreq: float,
    sfreq_goal: float,
    freq_min: float,
    freq_max: float,
    iir_params: dict,
    tmax: float
) -> np.ndarray:
    """ Extract the phase of the EEG signal at the desired frequency band.

    Procedure:
    1. Band-pass filter the EEG signal.
    2. Decimate the EEG signal.
    3. Crop the EEG signal to the desired length.
    4. Compute the phase of the EEG signal using the Hilbert transform.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs.
    sfreq : float
        Sampling frequency of the EEG epochs.
    sfreq_goal : float
        Desired sampling frequency of the EEG epochs.
    freq_min : float
        Lower frequency of the band-pass filter.
    freq_max : float
        Upper frequency of the band-pass filter.
    iir_params : dict
        Dictionary with the IIR filter parameters.
    tmax : float
        Desired length of the EEG signal in seconds.

    Returns
    -------
    phase_eeg : np.ndarray
        Phase of the EEG signal at the desired frequency band.

    """
    epochs.filter(
        l_freq=freq_min,
        h_freq=freq_max,
        method='iir',
        iir_params=iir_params
    )
    epochs.decimate(sfreq / sfreq_goal)
    epochs.crop(tmin=0, tmax=tmax)
    eeg = epochs.get_data(picks='eeg')

    phase_eeg = np.angle(hilbert(eeg))

    return phase_eeg


def reorder_eeg_data(order_list: list, eeg: np.ndarray):
    """ Reorder the EEG data according to the order of the stimuli.

    Parameters
    ----------
    order_list : list
        Order of the stimuli. Should reflect the order of the stimuli in the experiment.
    eeg : np.ndarray
        EEG data.

    Returns
    -------
    sorted_eeg : np.ndarray
        EEG data reordered according to the order of the stimuli.

    """
    sorted_order = sorted(order_list)
    eeg_order_indices = {name: index for index, name in enumerate(order_list)}
    new_order = [eeg_order_indices[name] for name in sorted_order]
    sorted_eeg = eeg[new_order]

    return sorted_eeg
