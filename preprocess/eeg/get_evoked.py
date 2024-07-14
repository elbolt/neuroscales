"""
Pipeline to derive evoked responses from EEG data locked to onset and target events.

Procedure includes the following steps:
- Load preprocessed data
- Apply bandpass filter
- Get evoked response for context and random conditions
- Save evoked responses
"""

from pathlib import Path
from helpers import load_config
import pandas as pd
import mne

mne.set_log_level('ERROR')


def get_evoked(
        epochs: mne.Epochs,
        final_frequencies: list,
        iir_parameters: dict,
        evoked_limits: list,
        by_condition: bool = False
) -> tuple[mne.Evoked, mne.Evoked] | mne.Evoked:
    """Get evoked response for context and random conditions."""
    epochs.filter(
        l_freq=final_frequencies[0],
        h_freq=final_frequencies[1],
        method='iir',
        iir_params=iir_parameters
    )

    if by_condition:
        context_epochs = epochs['condition == "context"']
        context_epochs = context_epochs[context_epochs.metadata['hit'] == 1]
        context_epochs.apply_baseline(baseline=(None, 0))
        context_evoked = context_epochs.crop(tmin=evoked_limits[0], tmax=evoked_limits[1]).average()

        random_epochs = epochs['condition == "random"']
        random_epochs = random_epochs[random_epochs.metadata['hit'] == 1]
        random_epochs.apply_baseline(baseline=(None, 0))
        random_evoked = random_epochs.crop(tmin=evoked_limits[0], tmax=evoked_limits[1]).average()

        return context_evoked, random_evoked

    epochs = epochs[epochs.metadata['hit'] == 1]
    epochs.apply_baseline(baseline=(None, 0))
    evoked = epochs.crop(tmin=evoked_limits[0], tmax=evoked_limits[1]).average()

    return evoked


if __name__ == '__main__':
    config = load_config('eeg_config.yaml')

    # Paths
    preprocessed_folder = Path(config['preprocessed_folder'])
    logs_folder = Path(config['logs_folder'])

    # File parameters
    no_participants = config['files_parameters']['no_participants']
    epochs_fif_extension = config['files_parameters']['epochs_fif_extension']
    logs_txt_extension = config['files_parameters']['logs_txt_extension']
    evoked_context_extension = config['files_parameters']['evoked_context_extension']
    evoked_random_extension = config['files_parameters']['evoked_random_extension']
    evoked_extension = config['files_parameters']['evoked_extension']

    # EEG parameters and participants/analyses
    final_frequencies = config['eeg_parameters']['final_frequencies']
    iir_parameters = config['iir_parameters']['alias_dict']
    participants = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]
    analyses = ['onset', 'target']

    for analysis in analyses:
        print(f'Getting evoked response for {analysis} analysis')
        params = config[f'{analysis}_epochs_params']
        evoked_limits = params['evoked_limits']
        subfolder = params['folder']
        folder_out = Path(config['evoked_folder']) / subfolder
        folder_out.mkdir(parents=True, exist_ok=True)

        for participant in participants:
            print(f'Participant {participant}')

            # Merge log and epochs
            log = pd.read_csv(logs_folder / (participant + logs_txt_extension), sep='\t')
            log['condition'] = [x[:-3] for x in log['file'].values]
            epochs = mne.read_epochs(preprocessed_folder / subfolder / (participant + epochs_fif_extension))
            epochs.metadata = log

            if analysis == 'target':
                context_evoked, random_evoked = get_evoked(
                    epochs,
                    final_frequencies,
                    iir_parameters,
                    evoked_limits,
                    by_condition=True
                )
                context_evoked.save(folder_out / (participant + evoked_context_extension), overwrite=True)
                random_evoked.save(folder_out / (participant + evoked_random_extension), overwrite=True)
            else:
                evoked = get_evoked(
                    epochs,
                    final_frequencies,
                    iir_parameters,
                    evoked_limits,
                    by_condition=False
                )
                evoked.save(folder_out / (participant + evoked_extension), overwrite=True)
