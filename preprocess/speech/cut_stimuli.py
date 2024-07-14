""" Remove silent part at the beginning of the stimuli. """

import numpy as np
from scipy.io import wavfile
from pathlib import Path
from speech_utils import cut_wav_from_cue
from helpers import load_config


if __name__ == '__main__':
    config = load_config('speech_config.yaml')
    wav_dir = Path(config['raw_stimuli_folder'])
    out_dir = Path(config['stimuli_parameters']['stimuli_folder'])
    out_dir.mkdir(exist_ok=True)

    stimuli_list = [file for file in wav_dir.glob('*.wav') if not file.name.startswith('._')]

    for stimulus in stimuli_list:
        speech_train = cut_wav_from_cue(stimulus)
        fs, _ = wavfile.read(stimulus)
        speech_train = speech_train.astype(np.int16)
        wavfile.write(out_dir / f'{stimulus.stem[:-5]}.wav', fs, speech_train)
