""" Compute modulation spectrum of the stimuli according to Ding et al. (2017). """

import brian2hears as b2h
import brian2 as b2
import numpy as np
import pandas as pd
from pathlib import Path
from helpers import load_config
from speech_utils import compute_modulation_spectrum


if __name__ == '__main__':
    config = load_config('speech_config.yaml')

    # Create output folder
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Stimuli parameters
    path = Path(config['stimuli_parameters']['stimuli_folder'])
    sfreq = config['stimuli_parameters']['sfreq']
    properties_filename = config['stimuli_parameters']['properties_filename']
    properties_df = pd.read_csv(Path(config['output_folder']) / properties_filename)
    durations = properties_df['duration'].values
    mean_duration = np.mean(durations)
    mean_duration_samples = int(mean_duration * sfreq)
    stimuli_list = [file for file in path.glob('*.wav') if not file.name.startswith('._')]

    # Spectrum parameters
    mod_spectrum_filename = config['spectrum_parameters']['spectrum_filename']
    center_freqs = config['spectrum_parameters']['gammatone_center_freqs']
    center_freqs['low'] = center_freqs['low']*b2.Hz
    center_freqs['high'] = center_freqs['high']*b2.Hz
    center_freqs = b2h.erbspace(**center_freqs)

    compression = config['spectrum_parameters']['compression']

    spectrum_limits = [
        config['spectrum_parameters']['spectrum_limits']['low'],
        config['spectrum_parameters']['spectrum_limits']['high']
    ]

    freqs_weighted, mod_spectrum = compute_modulation_spectrum(
        stimuli=stimuli_list,
        center_freqs=center_freqs,
        sfreq=sfreq,
        mean_samples=mean_duration_samples,
        compression=compression,
        spectrum_limits=spectrum_limits
    )

    np.savez(output_folder / mod_spectrum_filename, freqs_weighted=freqs_weighted, mod_spectrum=mod_spectrum)
