import brian2hears as b2h
import brian2 as b2
import numpy as np
from scipy.io import wavfile


def cut_wav_from_cue(stimulus: str) -> np.ndarray:
    """ Cut the wav file from the cue point to remove the silence at the beginning of the file.

    Parameters
    ----------
    stimulus : str
        Path to the wav file.

    Returns
    -------
    speech_train : np.ndarray
        Array of the speech signal (mono train).

    """

    _, data = wavfile.read(stimulus)

    if data.ndim != 2:
        raise ValueError('Soundfile needs to be a stereo file.')

    speech_train = data[:, 0]
    trigger_train = data[:, 1]
    trigger = np.where(trigger_train != 0)[0][0]

    speech_train = speech_train[trigger:]

    return speech_train


def compute_modulation_spectrum(
    stimuli: list,
    sfreq: float,
    mean_samples: int,
    center_freqs: dict,
    compression: float,
    spectrum_limits: list
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the modulation spectrum of a list of stimuli according to the procedure suggested by Ding et al. (2017).
    The code is adapted from Oderbolz et al. (2024).

    Procedure:
    1. Load the stimuli.
    2. Initialize the gammatone filterbank and apply gammatone filtering to half-wave rectified and compressed wave.
    3. On each filterbank output (subband), compute Discrete Fourier Transform (DFT) and average over subbands.
    4. Weight absolute value of DFT coefficients by modulation frequency (since filters are logarithmically spaced).
    5. Average over the stimuli and compute the root mean square (RMS) of modulation spectrum.
    6. Weight the modulation spectrum by frequencies between spectrum limits of interest.

    Parameters
    ----------
    stimuli : list
        List of stimuli to compute the modulation spectrum from.
    sfreq : float
        Sampling frequency of the stimuli.
    mean_samples : int
        Number of samples to average the modulation spectrum over.
    center_freqs : dict
        Dictionary containing the parameters to create the gammatone filterbank.
    compression : float
        Compression factor for the half-wave rectification using the clip function.
    spectrum_limits : list
        List containing the lower and upper limit of the modulation spectrum.

    Returns
    -------
    freqs_weighted : np.ndarray
        Weighted modulation frequencies.
    mod_spectrum : np.ndarray
        Weighted modulation spectrum.

    References
    ----------
    Ding, N., Patel, A. D., Chen, L., Butler, H., Luo, C., & Poeppel, D. (2017). Temporal modulations in speech and
    music. Neuroscience & Biobehavioral Reviews, 81, 181-187. https://doi.org/10.1016/j.neubiorev.2017.02.011

    Oderbolz, J., Sauppe, S., & Meyer, M. (2024). Concurrent processing of the prosodic hierarchy is supported by
    cortical entrainment and phase-amplitude coupling. bioRxiv preprint. https://doi.org/10.1101/2024.01.22.576636

    """
    mean_spectra_list = []

    for stimulus in stimuli:
        sound = b2h.loadsound(str(stimulus))
        gammatone = b2h.Gammatone(sound, center_freqs)
        filterbank = b2h.FunctionFilterbank(gammatone, lambda x: b2.clip(x, 0, b2.Inf)**(compression))
        subbands = filterbank.process()
        subband_spectra_list = []

        for j in range(len(subbands[1])):
            band = subbands[:, j]

            if band.shape[0] < mean_samples:
                pad_length = mean_samples - band.shape[0]
                band = np.pad(band, (0, pad_length), 'constant')
            elif band.shape[0] > mean_samples:
                cut_length = band.shape[0] - mean_samples
                band = band[:-cut_length]

            yf = np.abs(b2.rfft(band))
            xf = b2.rfftfreq(mean_samples, 1 / sfreq)

            abs_yf = np.abs(yf)

            subband_spectra_list.append(abs_yf)

        subband_spectra_list = np.array(subband_spectra_list)
        mean_spectrum = np.sqrt(np.mean(subband_spectra_list**2, axis=0))
        mean_spectrum = np.sqrt(xf) * mean_spectrum
        mean_spectra_list.append(mean_spectrum)

    mean_spectra_list = np.array(mean_spectra_list)
    spectrum_rms = np.sqrt(np.mean(mean_spectra_list**2, axis=0))
    mod_spectrum = spectrum_rms / np.max(spectrum_rms[(xf >= spectrum_limits[0]) & (xf <= spectrum_limits[1])])

    freqs_weighted = xf[(xf >= spectrum_limits[0]) & (xf <= spectrum_limits[1])]
    mod_spectrum = mod_spectrum[(xf >= spectrum_limits[0]) & (xf <= spectrum_limits[1])]

    return freqs_weighted, mod_spectrum


def compute_envelopes(
    stimuli: list,
    mean_samples: int,
    center_freqs: dict,
    compression: float,
) -> np.ndarray:
    """ Compute the envelopes of a list of stimuli using the gammatone filterbank.

    Parameters
    ----------
    stimuli : list
        List of stimuli to compute the envelopes from.
    mean_samples : int
        Number of samples to average the envelopes over.
    center_freqs : dict
        Dictionary containing the parameters to create the gammatone filterbank.
    compression : float
        Compression factor for the half-wave rectification using the clip function.

    """

    envelopes = np.full((len(stimuli), mean_samples), np.nan)

    for idx, stimulus in enumerate(stimuli):
        sound = b2h.loadsound(str(stimulus))
        gammatone = b2h.Gammatone(sound, center_freqs)
        filterbank = b2h.FunctionFilterbank(gammatone, lambda x: b2.clip(x, 0, b2.Inf)**(compression))
        subbands = filterbank.process()

        envelope = []

        for j in range(len(subbands[1])):
            band = subbands[:, j]

            if band.shape[0] < mean_samples:
                pad_length = mean_samples - band.shape[0]
                band = np.pad(band, (0, pad_length), 0)
            elif band.shape[0] > mean_samples:
                cut_length = band.shape[0] - mean_samples
                band = band[:-cut_length]

            envelope.append(band)

        envelopes[idx, :] = np.mean(envelope, axis=0)

        return envelopes

