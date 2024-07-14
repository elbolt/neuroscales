import mne
import numpy as np
import pandas as pd
from pathlib import Path
from tracking_utils import load_config
from mne_connectivity import spectral_connectivity_time
from warnings import simplefilter

# Suppress future warnings
simplefilter(action='ignore', category=FutureWarning)


def main():
    # Load configuration
    config = load_config('tracking_config.yaml')

    # Set up folders and parameters
    bands_folder = Path(config['bands_folder'])
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    frequency_bands_dict = config['frequency_bands']
    frequency_bands = list(frequency_bands_dict.keys())

    sfreq = config['filtering_parameters']['sfreq_goal']
    csv_filename = config['files_parameters']['csv_filename']
    csv_col_names = config['files_parameters']['csv_col_names']
    array_filename = config['files_parameters']['array_filename']

    # Prepare participant and EEG channel information
    no_participants = config['files_parameters']['no_participants']
    participants_list = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]

    montage = mne.channels.make_standard_montage(config['files_parameters']['eeg_montage'])
    channels_list = montage.ch_names

    # Load stimuli information
    filename = config['files_parameters']['sentences_filename']
    df_stimuli = pd.read_excel(filename, header=None, names=['file', 'sentence', 'syllables'])
    stimuli_list = df_stimuli['file'].tolist()

    # Initialize tracking array
    tracking_array = np.full(
        (no_participants, len(frequency_bands), len(stimuli_list), len(channels_list)), np.nan
    )

    # Compute phase-locking values (PLV) for each frequency band
    for b_idx, band in enumerate(frequency_bands):
        files = sorted(list(bands_folder.glob(f'*{band}.npy')))
        files = [file for file in files if not file.name.startswith('.')]

        for f_idx, file in enumerate(files):
            data = np.load(file)
            n_channels = len(channels_list)
            n_cycles = 2 if band == 'phrase_rate' else 7

            envelope_idx = np.zeros(n_channels).astype(int)
            eeg_channel_indices = np.arange(1, n_channels + 1)
            indices = (envelope_idx, eeg_channel_indices)

            frequency_bins = np.linspace(frequency_bands_dict[band][0], frequency_bands_dict[band][1], 10)

            tracking = spectral_connectivity_time(
                data,
                freqs=frequency_bins,
                method='plv',
                average=False,
                indices=indices,
                fmin=frequency_bands_dict[band][0],
                fmax=frequency_bands_dict[band][1],
                sfreq=sfreq,
                faverage=True,
                verbose=False,
                n_cycles=n_cycles,
            )

            tracking_array[f_idx, b_idx, :, :] = tracking.get_data().squeeze()

    # Reshape and save the tracking results
    index = pd.MultiIndex.from_product(
        [participants_list, frequency_bands, stimuli_list],
        names=csv_col_names[:3]
    )

    df = pd.DataFrame(tracking_array.reshape(-1, n_channels), index=index, columns=channels_list)
    df_long = df.stack().reset_index()
    df_long.columns = csv_col_names
    df_long.to_csv(output_folder / csv_filename, index=False)

    np.save(output_folder / array_filename, tracking_array)


if __name__ == '__main__':
    print(f'Running {__file__}')
    main()
