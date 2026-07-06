# Functional gradients dissociate between cortical layers

This repository contains the complete pipeline to reproduce the analyses and
figures from the manuscript:

> **Functional gradients dissociate between cortical layers**
> Degutis, J.K. *et al.* (2026)

## Repository structure

```
.
├── src/
│   ├── preprocessing/          # fMRI preprocessing pipeline
│   │   ├── fmri-analysis       # submodule for preprocessing from https://github.com/dchaimow/fmri_analysis/tree/main/library
│   ├── restingState/           # main resting-state analysis
│   │   ├── runRestingState_inter.py   # entry point for the main analysis
│   │   └── laminar_rs/         # analysis functions
│   └── robustness/             # robustness / control analyses
├── demo/                       # standalone, minimal demo (see demo/README.md)
├── environment.yml             # conda environment specification
└── requirements.txt            # pip requirements + external-tool provenance
```

A small, self-contained **demo** that reproduces the core analysis on a single
precomputed matrix lives in [`demo/`](demo/) and is documented separately in
[`demo/README.md`](demo/README.md).

## Installation

Create the conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate laminar-analysis
```
Total instalation time may take 30-60 minutes. 

> **Note.** The pipeline also relies on external neuroimaging software
> (FreeSurfer, ANTs, AFNI, LAYNII, SPM12/CAT12, ciftify, Connectome Workbench).
> Install them separately. Required versions are listed at the bottom of [`requirements.txt`](requirements.txt).

## Usage

The main resting-state analysis is run from `src/restingState/`:

This estimates the gradients, computes the inter-areal laminar
dissimilarity, and produces the associated surface maps and figures. The
analysis functions it calls are organised under
`src/restingState/laminar_rs/`.

Preprocessing scripts are provided in `src/preprocessing/`, and the
robustness / control analyses reported in the manuscript are in
`src/robustness/`.

All analyses were run inside a Singularity/Apptainer container bootstrapped 
from the Docker image ubuntu:22.04 (Ubuntu 22.04 LTS, Jammy Jellyfish; amd64).

## Citation

If you use this code, please cite:

> Degutis, J.K. *et al.* (2026). *Functional gradients dissociate between
> cortical layers.*

## Contact

For questions, please contact:

📧 j.karolis.degutis (at) gmail.com