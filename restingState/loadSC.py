import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

output_dir = '../../highRes_resting/derivatives/correlations/structuralMatrix/'
os.makedirs(output_dir, exist_ok=True)

def load_layers_by_vertex(hemi, n_layers=6, base_dir='../../SC_laminarThickness'):

    out = []
    for layer in range(1, n_layers+1):
        fname = f'tpl-bigbrain_hemi-{hemi}_desc-layer{layer}_thickness.txt'
        arr   = np.loadtxt(os.path.join(base_dir, fname))
        out.append(arr)
    return np.vstack(out).T

def parcel_means_from_vertices(hemi, vert_layers, base_dir='../../SC_laminarThickness'):

    # load labels
    lab_fname = f'tpl-bigbrain_hemi-{hemi}_desc-mmp1_parcellation.label.gii'
    labels    = nib.load(os.path.join(base_dir, lab_fname)).darrays[0].data
    # ignore background
    parcels   = np.unique(labels)
    parcels   = parcels[parcels != 0]
    # compute means
    means = []
    for p in parcels:
        idx = labels == p
        means.append( vert_layers[idx].mean(axis=0) )
    return np.vstack(means)   # shape (n_parcels, 6)

means_L = parcel_means_from_vertices('L', load_layers_by_vertex('L'))
means_R = parcel_means_from_vertices('R', load_layers_by_vertex('R'))
thickness_matrix = np.vstack([means_L, means_R])  

covariate = thickness_matrix.mean(axis=1)   # shape (360,)

n_parcels, n_layers = thickness_matrix.shape
residuals = np.zeros_like(thickness_matrix)
lm = LinearRegression(fit_intercept=True)

for i in range(n_layers):
    y = thickness_matrix[:, i]
    X = covariate.reshape(-1, 1)
    lm.fit(X, y)
    residuals[:, i] = y - lm.predict(X)

adj_matrix = np.corrcoef(residuals, rowvar=True)

plt.figure(figsize=(6, 6))
plt.imshow(adj_matrix, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="Partial correlation (Controlled for Total Thickness)")
plt.title(f"Correlation Matrix of Laminar Thickness")
plt.savefig(f"{output_dir}/PartialCorrelation_BBA_Thickness.png", bbox_inches="tight")
plt.close()
np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), adj_matrix)


