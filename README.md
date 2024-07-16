# Analyzing neural tracking of speech across linguistic timescales

This repository contains Python and R scripts for preprocessing EEG data, analyzing evoked neurophysiological responses, and computing phase-locking value (PLV) across different linguistic timescales. The statistical analyses are performed using R.

**Manuscript:** The data and analyses presented here are part of the manuscript "*Hearing and cognitive decline in aging differentially impact neural tracking of context-supported versus random speech across linguistic timescales*," currently in preparation.

## Repository Structure

This repository is organized into the following folders:

### 1. Preprocess
This folder contains scripts for preprocessing EEG data and determining timescales in speech stimuli.

### 2. Evoked
This folder contains pipelines for analyzing evoked neurophysiological responses.

### 3. Tracking
This folder contains pipelines for computing the phase-locking value (PLV) across different linguistic timescales.

### 4. Statistics
This folder contains R scripts for performing the statistical analyses reported in the manuscript.

## Data Availability

The data required to run these scripts, including the preprocessed EEG data, speech stimulus material, and participant information, are available in the Open Science Framework (OSF) repository. You can access the data at [OSF.io/5usgp](https://osf.io/5usgp/).

## Environment Setup

The `environment.yml` file included in this repository contains the specifications for the conda environment used in this project. Please note that the code is adapted to our environment and data infrastructure and cannot be executed without adjustments.

## Acknowledgements

We would like to thank Chantal Oderbolz for sharing her code related to her publication:

> Oderbolz, C., Sauppe, S., & Meyer, M. (bioRxiv, 2024). Concurrent processing of the prosodic hierarchy is supported by cortical entrainment and phase-amplitude coupling. [doi:10.1101/2024.01.22.576636](https://doi.org/10.1101/2024.01.22.576636).

Her contributions greatly facilitated our pipeline development.