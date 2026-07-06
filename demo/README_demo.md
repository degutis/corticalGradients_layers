# Laminar resting-state gradients demo

Minimal, reproducible demo of the core analysis:

## Contents

- `laminar_demo.ipynb` — main notebook
- `gradients.py` — gradient estimation + laminar dissimilarity
- `surface_maps.py` — cortical surface plotting
- `environment.yml` — conda environment
- `FC_matrix.npy` — precomputed supra-adjacency FC matrix (place here)

## Install

```bash
conda env create -f environment.yml
conda activate laminar-demo
```
Should take around 10 min to install. 

Then run the cells top to bottom. The FC matrix, scree plot, and surface maps
render inline; all figures are also saved under `results/`.