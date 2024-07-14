from pathlib import Path
from helpers import load_config, cluster_signal, find_peak, mean_amplitude_around_peak
import pandas as pd
import numpy as np
import mne
mne.set_log_level('ERROR')


if __name__ == '__main__':
    # Load configuration
    config = load_config('evoked_config.yaml')

    no_participants = config['files_parameters']['no_participants']
    participants = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]

    out_folder = Path(config['output_folder'])
    out_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # # Section: N400 analysis
    # -----------------------------------
    evoked_params = config['evoked_parameters']['N400']
    evoked_folder = Path(config['evoked_folder']) / Path(evoked_params['subfolder'])
    context_file_extension = evoked_params['context_file_extension']
    random_file_extension = evoked_params['random_file_extension']
    cluster = evoked_params['cluster']
    window = evoked_params['window']
    window_area = evoked_params['window_area']
    dataframe_columns = evoked_params['dataframe_columns']
    dataframe_filename = evoked_params['dataframe_filename']

    participant_ids = []
    latencies = []
    random_amplitudes = []
    context_amplitudes = []

    for participant in participants:
        # Load data
        evoked_context = mne.read_evokeds(evoked_folder / (participant + context_file_extension))[0]
        evoked_random = mne.read_evokeds(evoked_folder / (participant + random_file_extension))[0]
        evoked_context_data = evoked_context.get_data()
        evoked_random_data = evoked_random.get_data()
        channel_names = evoked_context.info['ch_names']
        sfreq = evoked_context.info['sfreq']
        times = evoked_context.times

        # Cluster into electrodes of interest
        context_clustered = cluster_signal(evoked_context_data, channel_names, cluster)
        random_clustered = cluster_signal(evoked_random_data, channel_names, cluster)

        # Get the most negative peak in the random signal
        random_peak_time, random_peak_idx = find_peak(random_clustered, times, window, polarity='negative')

        mean_random_amp = mean_amplitude_around_peak(
            random_clustered,
            times,
            random_peak_time,
            sfreq,
            window_area
        )

        mean_context_amp = mean_amplitude_around_peak(
            context_clustered,
            times,
            random_peak_time,
            sfreq,
            window_area
        )

        participant_ids.append(participant)
        latencies.append(random_peak_time)
        random_amplitudes.append(mean_random_amp)
        context_amplitudes.append(mean_context_amp)

    # Convert latencies to milliseconds and voltages to microvolts
    latencies = np.array(latencies) * 1e3
    latencies = latencies.round().astype(int)
    random_amplitudes = np.array(random_amplitudes) * 1e6
    context_amplitudes = np.array(context_amplitudes) * 1e6

    df = pd.DataFrame({
        dataframe_columns[0]: participant_ids,
        dataframe_columns[1]: latencies,
        dataframe_columns[2]: random_amplitudes,
        dataframe_columns[3]: context_amplitudes
    })

    long_df = pd.melt(
        df,
        id_vars=['participant_id', 'N400_latency_ms'],
        value_vars=['random_amplitude_uV', 'context_amplitude_uV'],
        var_name='condition',
        value_name='amplitude_uV'
    )

    long_df['condition'] = long_df['condition'].str.replace('_amplitude_uV', '')
    long_df.to_csv(out_folder / dataframe_filename, index=False)
