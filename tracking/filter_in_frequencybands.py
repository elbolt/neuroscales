import numpy as np
import pandas as pd
from pathlib import Path
import brian2 as b2
import brian2hears as b2h
import mne

from tracking_utils import (
    load_config,
    extract_envelope,
    extract_envelope_phase,
    extract_eeg_phase,
    reorder_eeg_data
)

mne.set_log_level('WARNING')


if __name__ == '__main__':
    print(f'Running {__file__}')

    config = load_config('tracking_config.yaml')

    eeg_folder = Path(config['eeg_folder'])
    speech_folder = Path(config['speech_folder'])
    logs_folder = Path(config['logs_folder'])
    bands_folder = Path(config['bands_folder'])
    bands_folder.mkdir(exist_ok=True)

    logs_txt_extension = config['files_parameters']['logs_txt_extension']
    epochs_extension = config['files_parameters']['epochs_extension']
    no_participants = config['files_parameters']['no_participants']
    participants = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]

    # Frequency bands
    frequency_bands_dict = config['frequency_bands']
    frequency_bands = list(frequency_bands_dict.keys())

    # Filtering
    sfreq_wav = config['filtering_parameters']['sfreq_wav']
    sfreq_eeg = config['filtering_parameters']['sfreq_eeg']
    sfreq_goal = config['filtering_parameters']['sfreq_goal']
    center_freqs = (config['filtering_parameters']['gammatone_center_freqs'])
    center_freqs['low'] = center_freqs['low']*b2.Hz
    center_freqs['high'] = center_freqs['high']*b2.Hz
    center_freqs = b2h.erbspace(**center_freqs)
    compression = config['filtering_parameters']['compression']
    mean_length_s = config['filtering_parameters']['mean_length_s']
    time = np.arange(0, mean_length_s, 1 / sfreq_goal)
    mean_length_samples = time.shape[0]

    alias_dict = config['iir_parameters']['alias_dict']

    wav_files = sorted(list(Path(speech_folder).rglob('*.wav')), key=lambda x: x.stem)

    for band in frequency_bands:
        print(f'Processing `{band}` band')
        # First, create array of phase envelopes containing all stimuli
        phase_envelopes = np.full((len(wav_files), mean_length_samples), np.nan)

        for wav_idx, wav_file in enumerate(wav_files):
            # 1. Get envelopes at 512 Hz (preprocessed EEG sampling rate)
            envelope = extract_envelope(
                wav_file,
                center_freqs=center_freqs,
                compression=compression,
                sfreq=sfreq_wav,
                sfreq_goal=sfreq_eeg,
                alias_dict=alias_dict
            )

            # 2. Get band-pass filtered envelope phase at 128 Hz (goal sampling rate)
            phase_envelope = extract_envelope_phase(
                envelope,
                sfreq=sfreq_eeg,
                sfreq_goal=sfreq_goal,
                freq_min=frequency_bands_dict[band][0],
                freq_max=frequency_bands_dict[band][1],
                iir_params=alias_dict
            )

            # 3. Padding/cutting to account for different stimuli lengths
            if phase_envelope.shape[0] > mean_length_samples:
                phase_envelope = phase_envelope[:mean_length_samples]
            else:
                phase_envelope = np.pad(
                    phase_envelope,
                    (0, mean_length_samples - phase_envelope.shape[0]),
                    'constant'
                )

            phase_envelopes[wav_idx] = phase_envelope

        # Second, create array of EEG data for each participant
        for participant_id in participants:
            # 1. Get epochs at 512 Hz (preprocessed EEG sampling rate)
            epochs = mne.read_epochs(eeg_folder / f'{participant_id}{epochs_extension}', preload=True)

            # 2. Get band-pass filtered EEG phase at 128 Hz (goal sampling rate)
            eeg = extract_eeg_phase(
                epochs,
                sfreq=sfreq_eeg,
                sfreq_goal=sfreq_goal,
                freq_min=frequency_bands_dict[band][0],
                freq_max=frequency_bands_dict[band][1],
                iir_params=alias_dict,
                tmax=mean_length_s
            )

            # 3. Reorder stimuli in EEG array from randomized participant-specific order to stimuli array order
            log_df = pd.read_csv(logs_folder / f'{participant_id}{logs_txt_extension}', sep='\t')
            random_order = list(log_df.file.values)
            sorted_eeg = reorder_eeg_data(order_list=random_order, eeg=eeg)

            phase_envelopes_dim = np.expand_dims(phase_envelopes, axis=1)
            band_array = np.concatenate((phase_envelopes_dim, sorted_eeg), axis=1)

            np.save(bands_folder / f'{participant_id}_{band}.npy', band_array)
