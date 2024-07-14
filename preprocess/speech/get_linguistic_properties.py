""" Calculate the rates of phrases, syllables, words and phones for each stimulus. """

import numpy as np
import pandas as pd
import textgrid
from pathlib import Path
from helpers import load_config
from scipy.io import wavfile

if __name__ == '__main__':
    config = load_config('speech_config.yaml')

    # Create output folder
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get forced-alignment files from the MFA output
    tg_dir = Path(config['mfa_parameters']['final_output'])
    filename = config['mfa_parameters']['sentences_filename']
    tg_tier_order = config['mfa_parameters']['tg_tier_order']
    word_idx = tg_tier_order.index('word')
    phone_idx = tg_tier_order.index('phone')

    # Stimuli parameters
    stimuli_dir = Path(config['stimuli_parameters']['stimuli_folder'])
    n_phrases = config['stimuli_parameters']['n_phrases']
    properties_filename = config['stimuli_parameters']['properties_filename']

    df_stimulus = pd.read_excel(filename, header=None, names=['file', 'sentence', 'syllables'])
    stimulus_list = df_stimulus['file'].tolist()

    durations = []
    phrase_rates = []
    word_rates = []
    syllable_rates = []
    phone_rates = []

    for stimulus in stimulus_list:
        fs, speech_stream = wavfile.read(stimuli_dir / f'{stimulus}_70dB.wav')
        tg = textgrid.TextGrid.fromFile(tg_dir / f'{stimulus}_70dB.TextGrid')

        duration = len(speech_stream) / fs
        offset_silence = np.abs(tg[word_idx][-1].minTime - tg[word_idx][-1].maxTime)
        duration = duration - offset_silence

        word_onsets = [entry.mark for entry in tg[word_idx] if entry.mark != '']
        no_words = len(word_onsets)

        no_syllables = df_stimulus.loc[df_stimulus['file'] == stimulus, 'syllables'].values[0]

        phone_onsets = [entry.mark for entry in tg[phone_idx] if entry.mark != '']
        no_phones = len(phone_onsets)

        durations.append(duration)
        phrase_rates.append(n_phrases / duration)
        word_rates.append(no_words / duration)
        syllable_rates.append(no_syllables / duration)
        phone_rates.append(no_phones / duration)

    df_rates = pd.DataFrame({
        'file': stimulus_list,
        'duration': durations,
        'phrase_rate': phrase_rates,
        'word_rate': word_rates,
        'syllable_rate': syllable_rates,
        'phone_rate': phone_rates
    })

    df_rates.to_csv(output_folder / properties_filename, index=False, header=True)
