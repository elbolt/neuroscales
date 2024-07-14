#!/bin/bash

###############################
# Speech processing pipeline  #
###############################

eval "$(conda shell.bash hook)"

conda activate neuro

# 1. Run the Python script `get_stimuli_transcripts.py` to generate .txt-files for the MFA from file `stimumatrix_sentences.xlsx`
python get_stimuli_transcripts.py

# 2. Run the Python script `cut_stimuli.py` to remove trigger train and onset silence from the stimuli, as played in the experiment
python cut_stimuli.py

# 3. Activate MFA virtual environment "linfeatures" and run the bash script `run_aligner.sh` to align the transcriptions with the stimuli
conda activate linfeatures
bash run_aligner.sh

# 4. Get linguistic properties of the aligned transcriptions (which needed some manual corrections, hence we are reading from the folder `aligned_transcripts_edited`)
conda activate neuro

# 5. Run the Python script `get_linguistic_properties.py` to get the linguistic properties "linguistic_properties"
python get_linguistic_properties.py

# 6. Compute the modulation spectrum of the stimuli
python compute_modulation_spectrum.py

