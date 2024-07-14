#!/bin/bash

###############################
# Run Montreal Forced Aligner #
###############################

eval "$(conda shell.bash hook)"
conda activate linfeatures

CONFIG_FILE="/Users/elenauzh/Projects/rantext-speech/rantext-speech/speech/speech_config.yaml"

INPUT_FOLDER=$(yq -r '.stimuli_parameters.stimuli_folder' $CONFIG_FILE)
OUTPUT_FOLDER=$(yq -r '.mfa_parameters.output_folder' $CONFIG_FILE)
DICT_FOLDER=$(yq -r '.mfa_parameters.dict_folder' $CONFIG_FILE)

mkdir -p $OUTPUT_FOLDER

mfa model download g2p german_mfa

mfa align --clean "$INPUT_FOLDER" "$DICT_FOLDER" german_mfa "$OUTPUT_FOLDER" --single_speaker
