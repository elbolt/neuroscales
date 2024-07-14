""" Generate txt files for MFA from excel file. """

import numpy as np
import pandas as pd
from pathlib import Path
from helpers import load_config


if __name__ == '__main__':
    config = load_config('speech_config.yaml')
    transcripts_dir = Path(config['stimuli_parameters']['stimuli_folder'])
    transcripts_dir.mkdir(exist_ok=True)
    filename = config['mfa_parameters']['sentences_filename']

    df = pd.read_excel(filename, header=None, names=['file', 'sentence', 'syllables'])

    for idx in range(len(df)):
        file = df.iloc[idx]['file'] + '_70dB.txt'
        sentence = df.iloc[idx]['sentence']

        np.savetxt(transcripts_dir / file, [sentence], fmt='%s')
