import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_tracking_df(config: dict) -> pd.DataFrame:
    """Load tracking dataframe.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Tracking dataframe.
    """
    logs_folder = config['logs_folder']
    tracking_df = pd.read_csv(config['tracking_file'])
    participant_df = pd.read_csv(config['participant_file'])
    logs_txt_extension = config['track_files_parameters']['logs_txt_extension']
    final_dataset_col_order = config['track_files_parameters']['final_dataset_col_order']
    no_participants = config['track_files_parameters']['no_participants']
    eeg_clusters = config['track_files_parameters']['eeg_clusters']

    participants_list = ['p' + str(i).zfill(2) for i in range(1, no_participants + 1)]

    merged_df = pd.merge(tracking_df, participant_df, on='participant_id')
    all_logs_df = pd.DataFrame()

    for participant in participants_list:
        log = pd.read_csv(f'{logs_folder}/{participant}{logs_txt_extension}', sep='\t')
        log['participant_id'] = participant

        all_logs_df = pd.concat([all_logs_df, log])

    all_logs_df.rename(columns={'file': 'stimulus_id', 'hit': 'correct'}, inplace=True)
    final_dataset_df = pd.merge(all_logs_df, merged_df, on=['participant_id', 'stimulus_id'])

    # Add condition column
    final_dataset_df['condition'] = final_dataset_df['stimulus_id'].apply(
        lambda x: 'context' if x.startswith('con') else 'random'
    )

    # Add cluster column and drop rows without a cluster
    final_dataset_df['cluster'] = None
    for cluster, channels in eeg_clusters.items():
        final_dataset_df.loc[final_dataset_df['channel_id'].isin(channels), 'cluster'] = cluster
    final_dataset_df = final_dataset_df[final_dataset_df['cluster'].notna()]

    final_dataset_df = final_dataset_df[final_dataset_col_order]

    return final_dataset_df


if __name__ == '__main__':
    config = load_config('statistics_config.yaml')

    data_folder = Path(config['data_folder'])
    data_folder.mkdir(parents=True, exist_ok=True)

    # Speech tracking data
    tracking_df_filename = config['track_files_parameters']['csv_filename']
    tracking_data = get_tracking_df(config)
    tracking_data.to_csv(data_folder / tracking_df_filename, index=False)

    # Evoked analysis data
    participant_df = pd.read_csv(config['participant_file'])
    AEP_filename = config['AEP_filename']
    AEP_df = pd.read_csv(config['AEP_file'])
    AEP_df = pd.merge(AEP_df, participant_df, on='participant_id')
    AEP_df.to_csv(data_folder / AEP_filename, index=False)

    N400_df = pd.read_csv(config['N400_file'])
    N400_filename = config['N400_filename']
    N400_df = pd.merge(N400_df, participant_df, on='participant_id')
    N400_df.to_csv(data_folder / N400_filename, index=False)
