import os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import laminarRestingState as lrs
import laminarAnalyses as laman

# (your unused imports removed)
# import laminarRestingState as lrs
# import laminarAnalyses as laman

output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/'
os.makedirs(output_dir, exist_ok=True)

def load_layers_by_vertex(hemi, n_layers=6, base_dir='/home/degutis/repos/SC_laminarThickness'):
    """
    Returns array shape (n_vertices, n_layers) with raw thickness per layer.
    """
    out = []
    for layer in range(1, n_layers+1):
        fname = f'tpl-bigbrain_hemi-{hemi}_desc-layer{layer}_thickness.txt'
        arr   = np.loadtxt(os.path.join(base_dir, fname))
        out.append(arr.astype(np.float64))
    return np.vstack(out).T  # (n_vertices, n_layers)

def normalize_vertices_to_relative(vert_layers, eps=1e-12):
    """
    Convert absolute layer thickness (per vertex) to relative (sums to ~1 per vertex).
    Handles zero totals safely.
    """
    totals = vert_layers.sum(axis=1, keepdims=True)
    totals = np.where(totals <= eps, 1.0, totals)  # avoid divide-by-zero
    return vert_layers / totals

def parcel_means_from_vertices(hemi, vert_layers, base_dir='/home/degutis/repos/SC_laminarThickness'):
    """
    Parcel the (already normalized) vertex-layer array by MMP-360 labels.
    Returns array shape (n_parcels_hemi, n_layers) with parcel-wise means.
    """
    lab_fname = f'tpl-bigbrain_hemi-{hemi}_desc-mmp1_parcellation.label.gii'
    labels    = nib.load(os.path.join(base_dir, lab_fname)).darrays[0].data
    parcels   = np.unique(labels)
    parcels   = parcels[parcels != 0]

    means = []
    for p in parcels:
        idx = labels == p
        means.append( vert_layers[idx].mean(axis=0) )
    return np.vstack(means)   # (n_parcels_hemi, n_layers)

# ---- Load, normalize at vertex-level, then parcel ----
vert_L = load_layers_by_vertex('L')
vert_R = load_layers_by_vertex('R')

rel_vert_L = normalize_vertices_to_relative(vert_L)  # (v_L, 6)
rel_vert_R = normalize_vertices_to_relative(vert_R)  # (v_R, 6)

means_L = parcel_means_from_vertices('L', rel_vert_L)
print(means_L.shape)
means_R = parcel_means_from_vertices('R', rel_vert_R)

# Stack hemispheres => (360, 6) for MMP-360 total
rel_thickness_matrix = np.vstack([means_L, means_R])  # (n_parcels=360, n_layers=6)

# # ---- Partial correlation across parcels controlling the global layer profile ----
# # Global profile g is the across-parcel mean of the 6 relative layers
# g = rel_thickness_matrix.mean(axis=0)  # shape (6,)

# # Build regression (with intercept) once, and residualize all parcels in one shot:
# # For each parcel vector y (length 6), regress y ~ g + intercept; residualize; then corr across parcels.
# G = np.column_stack([g, np.ones_like(g)])  # (6, 2)
# GtG_inv = np.linalg.inv(G.T @ G)
# H = G @ GtG_inv @ G.T                       # Hat matrix (6x6)
# I = np.eye(6)

# Y = rel_thickness_matrix.T                  # (6, 360) — columns are parcels
# Y_res = (I - H) @ Y                         # residuals (6, 360)
# residuals = Y_res.T                         # (360, 6)

# # Pearson correlation across parcels on residual 6-vectors
# r_matrix = np.corrcoef(residuals, rowvar=True)

# g = rel_thickness_matrix.mean(axis=0)  # shape (6,)
g = rel_thickness_matrix.mean(axis=0) 
covariate = np.column_stack([g, np.ones_like(g)])  # (6, 2)

n_parcels, n_layers = rel_thickness_matrix.shape
residuals = np.zeros_like(rel_thickness_matrix)
lm = LinearRegression(fit_intercept=True)

for i in range(n_layers):
    y = rel_thickness_matrix[:, i]
    X = covariate.reshape(-1, 1)
    lm.fit(X, y)
    residuals[:, i] = y - lm.predict(X)

adj_matrix = np.corrcoef(residuals, rowvar=True)




np.fill_diagonal(r_matrix, 0)

# ---- Fisher r→z transform (clipped to avoid infinities) ----
r_clipped = np.clip(r_matrix, -0.999999, 0.999999)
z_matrix = np.arctanh(r_clipped)

# ---- Save & plot ----
# Save both r and z to be explicit
np.save(os.path.join(output_dir, 'adjacency_matrix_raw.npy'), r_matrix)
np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), z_matrix)

plt.figure(figsize=(6, 6))
plt.imshow(z_matrix, cmap="viridis")
plt.colorbar(label="Fisher z (partial corr; relative thickness)")
plt.title("Parcel × Parcel (MMP-360): Relative Laminar Thickness (controlled global profile)")
plt.tight_layout()
plt.savefig(f"{output_dir}/PartialCorrelation_RelativeThickness_FisherZ.png", bbox_inches="tight")
plt.close()



restStateSub = lrs.LaminarRestingState(output_dir, 360, 0, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")    
M = z_matrix

print(M.shape)

n_components = 5
G = laman.run_gradient_analysis(M, n_components=n_components, random_state=13011992)
np.save(os.path.join(output_dir, 'gradients_lamThick.npy'), G)

for i in range(n_components):
    restStateSub.__plot_on_mmhcp_surface_multipleLayers__(G[:,[i]], f'BigBrainLamThick_{i}', "PartialCorr")        

