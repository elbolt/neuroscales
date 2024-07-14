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
    # Section: Auditory evoked potentials
    # -----------------------------------
    evoked_params = config['evoked_parameters']['AEP']
    evoked_folder = Path(config['evoked_folder']) / Path(evoked_params['subfolder'])
    file_extension = evoked_params['file_extension']
    cluster = evoked_params['cluster']
    P1_window = evoked_params['P1_window']
    N1_window = evoked_params['N1_window']
    P2_window = evoked_params['P2_window']
    window_area = evoked_params['window_area']
    dataframe_columns = evoked_params['dataframe_columns']
    dataframe_filename = evoked_params['dataframe_filename']

    participant_ids = []
    P1_latencies, P1_amplitudes = [], []
    N1_latencies, N1_amplitudes = [], []
    P2_latencies, P2_amplitudes = [], []

    for participant in participants:
        # Load data
        evoked = mne.read_evokeds(evoked_folder / (participant + file_extension))[0]

        evoked_data = evoked.get_data()
        channel_names = evoked.info['ch_names']
        sfreq = evoked.info['sfreq']
        times = evoked.times

        # Cluster into electrodes of interest
        clustered = cluster_signal(evoked_data, channel_names, cluster)

        # Get the most negative/positive peak in the signal
        P1_peak_time, P1_peak_idx = find_peak(clustered, times, P1_window, polarity='positive')
        P1_mean_amp = mean_amplitude_around_peak(
            clustered,
            times,
            P1_peak_time,
            sfreq,
            window_area
        )

        N1_peak_time, N1_peak_idx = find_peak(clustered, times, N1_window, polarity='negative')
        N1_mean_amp = mean_amplitude_around_peak(
            clustered,
            times,
            N1_peak_time,
            sfreq,
            window_area
        )

        P2_peak_time, P2_peak_idx = find_peak(clustered, times, P2_window, polarity='positive')
        P2_mean_amp = mean_amplitude_around_peak(
            clustered,
            times,
            P2_peak_time,
            sfreq,
            window_area
        )

        participant_ids.append(participant)
        P1_latencies.append(P1_peak_time)
        P1_amplitudes.append(P1_mean_amp)
        N1_latencies.append(N1_peak_time)
        N1_amplitudes.append(N1_mean_amp)
        P2_latencies.append(P2_peak_time)
        P2_amplitudes.append(P2_mean_amp)

    # Convert latencies to milliseconds and voltages to microvolts
    P1_latencies = np.array(P1_latencies) * 1e3
    P1_latencies = P1_latencies.round().astype(int)
    P1_amplitudes = np.array(P1_amplitudes) * 1e6

    N1_latencies = np.array(N1_latencies) * 1e3
    N1_latencies = N1_latencies.round().astype(int)
    N1_amplitudes = np.array(N1_amplitudes) * 1e6

    P2_latencies = np.array(P2_latencies) * 1e3
    P2_latencies = P2_latencies.round().astype(int)
    P2_amplitudes = np.array(P2_amplitudes) * 1e6

    df = pd.DataFrame({
        dataframe_columns[0]: participant_ids,
        dataframe_columns[1]: P1_latencies,
        dataframe_columns[2]: P1_amplitudes,
        dataframe_columns[3]: N1_latencies,
        dataframe_columns[4]: N1_amplitudes,
        dataframe_columns[5]: P2_latencies,
        dataframe_columns[6]: P2_amplitudes
    })

    df.to_csv(out_folder / dataframe_filename, index=False)
