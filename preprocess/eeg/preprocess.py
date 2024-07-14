""" Pipeline for preprocessing EEG data.

Prodecure includes the following steps:
- Load raw data
- Remove bad channels
- Set reference and eog channels
- Anti-alias filter at 1/3 of the goal frequency
- Notch filter to remove power line noise
- Segment data while accounting for delay
- Decimate data to the goal frequency
- Apply ICA to remove EOG artifacts
- Interpolate bad channels
- Apply baseline correction
- Save preprocessed data

"""
from pathlib import Path
from helpers import load_config
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.set_log_level('ERROR')


def run_preprocessing(config: str, segment_to: str = 'audio') -> None:
    """ Run the preprocessing pipeline.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    segment_to : str
        Segment to which the data will be epoched. Options are 'audio' or 'onset'.

    """
    if segment_to not in ['onset', 'target']:
        raise ValueError('segment_to parameter must be either "audio" or "onset"')

    # Paths
    raw_folder = Path(config['raw_folder'])
    preprocessed_folder = Path(config['preprocessed_folder'])
    plots_folder = Path(config['plots_folder'])
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    # File parameters
    no_participants = config['files_parameters']['no_participants']
    raw_fif_extension = config['files_parameters']['raw_fif_extension']
    epochs_fif_extension = config['files_parameters']['epochs_fif_extension']

    # Generate participants list
    participants = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]

    # EEG parameters
    sfreq_goal = config['eeg_parameters']['sfreq_goal']
    baseline = config['eeg_parameters']['baseline']
    notch_frequencies = config['eeg_parameters']['notch_frequencies']
    notch_width = config['eeg_parameters']['notch_width']
    ica_seed = config['eeg_parameters']['ica_seed']

    alias_dict = config['iir_parameters']['alias_dict']
    notch_dict = config['iir_parameters']['notch_dict']

    # Epochs parameters (onset or target)
    if segment_to == 'onset':
        epochs_params = config['onset_epochs_params']
    elif segment_to == 'target':
        epochs_params = config['target_epochs_params']

    preprocessed_folder = preprocessed_folder / epochs_params['folder']
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    delta_t = epochs_params['delta_t']
    trigger_codes = epochs_params['trigger_codes']
    epoch_limits = epochs_params['epoch_limits']

    # Channels processing
    bad_mastoids = config['channels']['bad_mastoids']
    bad_eog = config['channels']['bad_eog']
    montage = mne.channels.make_standard_montage(config['channels']['montage'])

    for participant_id in participants:
        raw_file = raw_folder / (participant_id + raw_fif_extension)
        raw = mne.io.read_raw_fif(raw_file, preload=True)
        raw.set_montage(montage)

        bad_channels = config['channels']['bad_cap'][participant_id]
        raw.info['bads'] = bad_channels
        if participant_id in bad_mastoids:
            reference_channels = config['channels']['mastoids_alt']
        else:
            reference_channels = config['channels']['mastoids']

        if participant_id in bad_eog:
            eog_channels = config['channels']['eog_alt']
        else:
            eog_channels = config['channels']['eog']
        raw.set_eeg_reference(reference_channels)
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})

        if segment_to == 'onset':
            ica_components = config['channels']['onset_ica_components'][participant_id]
        elif segment_to == 'target':
            ica_components = config['channels']['target_ica_components'][participant_id]

        raw.filter(
            l_freq=None,
            h_freq=sfreq_goal / 3.0,
            h_trans_bandwidth=sfreq_goal / 10.0,
            method='iir',
            iir_params=alias_dict
        )
        for f, freq in enumerate(notch_frequencies):
            raw.notch_filter(
                freqs=freq,
                method='iir',
                iir_params=notch_dict,
                notch_widths=notch_width
            )
        events = mne.find_events(
            raw,
            stim_channel='Status',
            min_duration=(1 / raw.info['sfreq']),
            shortest_event=1,
            initial_event=True
        )
        if participant_id == 'p21' and segment_to == 'onset':
            remove_times = config['eeg_parameters']['p21_extra_events_t']
            events = events[np.isin(events[:, 0], remove_times, invert=True)]

        delay_samples = int(delta_t * raw.info['sfreq'])
        mask = np.isin(events[:, 2], trigger_codes)
        audio_events_delta = events[mask, :]
        audio_events_delta[:, 0] = audio_events_delta[:, 0] + delay_samples

        epochs = mne.Epochs(
            raw,
            audio_events_delta,
            event_id=trigger_codes,
            tmin=epoch_limits[0],
            tmax=epoch_limits[1],
            baseline=None,
            preload=True
        )
        del raw
        epochs.decimate(int(epochs.info['sfreq'] / sfreq_goal))

        epochs_ica_copy = epochs.copy()
        epochs_ica_copy.filter(
            l_freq=1.0,
            h_freq=None,
            method='iir',
            iir_params=dict(order=3, ftype='butter', output='sos')
        )
        ica = mne.preprocessing.ICA(
            n_components=0.999,
            method='picard',
            max_iter=1000,
            fit_params=dict(fastica_it=5),
            random_state=ica_seed
        )
        ica.fit(epochs_ica_copy)
        fig_list = ica.plot_components(show=False)

        for f, fig in enumerate(fig_list):
            fig.savefig(plots_folder / f'{participant_id}_ica_{f}.pdf')

        # Now close all the figures
        plt.close('all')

        ica.exclude = ica_components
        print(f'{participant_id}: removing components {ica_components}')
        ica.apply(epochs)

        if participant_id in bad_eog:
            eog_channels = config['channels']['eog_alt']
            epochs.set_channel_types({ch: 'eeg' for ch in eog_channels})

        epochs.interpolate_bads()

        epochs.apply_baseline(baseline)

        preprocessed_file = preprocessed_folder / (participant_id + epochs_fif_extension)
        epochs.save(preprocessed_file, overwrite=True)

        # epochs.filter(1, 12)
        # epochs.crop(tmin=-.200, tmax=.700)
        # evoked = epochs.average()
        # evoked.plot(show=False, time_unit='ms').savefig(plots_folder / f'{participant_id}_evoked.pdf')


if __name__ == '__main__':
    config = load_config('eeg_config.yaml')

    # run_preprocessing(config, segment_to='onset')
    run_preprocessing(config, segment_to='target')
