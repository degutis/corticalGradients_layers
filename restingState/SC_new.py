import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

import laminarRestingState as lrs
import laminarAnalyses as laman


# ---------------- config ----------------
output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/'
base_dir   = '/home/degutis/repos/SC_laminarThickness'
n_layers   = 6
disk_radius_mm = 10.0     # moving-disk radius on inflated surface (uniform average)

# NEW: choose how to partial out nuisance effects in the correlation step.
#   'global'       -> regress out the global mean 6-layer profile (current behavior)
#   'parcel_total' -> regress out each parcel's overall thickness (sum across layers) across parcels
partial_mode = 'parcel_total'   # or 'parcel_total'
# ---------------------------------------- 

os.makedirs(output_dir, exist_ok=True)

# ---------- I/O helpers ----------
def _first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of the candidate files exist:\n" + "\n".join(path_list))

def load_layers_by_vertex(hemi, n_layers=6):
    mats = []
    for li in range(1, n_layers+1):
        fname = f'tpl-bigbrain_hemi-{hemi}_desc-layer{li}_thickness.txt'
        arr = np.loadtxt(os.path.join(base_dir, fname)).astype(np.float64)
        mats.append(arr)
    return np.vstack(mats).T  # (n_vertices, n_layers)

def load_labels(hemi):
    # lab = f'tpl-bigbrain_hemi-{hemi}_desc-mmp1_parcellation.label.gii'
    lab = f'tpl-bigbrain_hemi-{hemi}_desc-schaefer400_parcellation.label.gii'
    return nib.load(os.path.join(base_dir, lab)).darrays[0].data

def load_inflated_coords(hemi):
    # Try some likely names in src; adjust if your local names differ
    cands = [
        os.path.join(base_dir, f'tpl-bigbrain_hemi-{hemi}_desc-mid.surf.inflate.gii'),
        os.path.join(base_dir, f'tpl-bigbrain_hemi-{hemi}_desc-inflated.surf.gii'),
        os.path.join(base_dir, f'tpl-bigbrain_hemi-{hemi}_desc-mid_inflated.surf.gii'),
    ]
    surf_path = _first_existing(cands)
    g = nib.load(surf_path)
    # coords are in the first darray for GIFTI .surf.gii
    return g.darrays[0].data.astype(np.float64)  # (n_vertices, 3)

# ---------- core ops ----------
def moving_disk_smooth(coords_xyz, data_vec, radius_mm):
    """
    Uniform average of all vertices within Euclidean radius_mm on the *inflated* surface.
    coords_xyz: (N,3) inflated coords
    data_vec:   (N,)   scalar per vertex
    """
    tree = BallTree(coords_xyz)  # Euclidean metric
    ind = tree.query_radius(coords_xyz, r=radius_mm, return_distance=False)
    out = np.empty_like(data_vec, dtype=np.float64)
    for i, nbrs in enumerate(ind):
        out[i] = data_vec[nbrs].mean() if nbrs.size else data_vec[i]
    return out

def smooth_layers_on_inflated(hemi, vert_layers, radius_mm):
    coords = load_inflated_coords(hemi)
    out = np.empty_like(vert_layers, dtype=np.float64)
    for li in range(vert_layers.shape[1]):
        out[:, li] = moving_disk_smooth(coords, vert_layers[:, li], radius_mm)
    return out

def to_relative_per_vertex(vert_layers, eps=1e-12):
    totals = vert_layers.sum(axis=1, keepdims=True)
    totals = np.where(totals <= eps, 1.0, totals)
    return vert_layers / totals

def parcel_means_from_vertices(hemi, vert_layers):
    labels = load_labels(hemi)
    parcels = np.unique(labels)
    parcels = parcels[parcels != 0]
    means = []
    for p in parcels:
        idx = labels == p
        means.append(vert_layers[idx].mean(axis=0))
    return np.vstack(means)  # (parcels_hemi, 6)

# ---------- pipeline (per the paper’s LTC) ----------
# 1) Load absolute laminar thickness per vertex
vL = load_layers_by_vertex('L', n_layers=n_layers)
vR = load_layers_by_vertex('R', n_layers=n_layers)

# 2) Smooth *absolute* layer maps on the inflated surface with a 10 mm moving disk (uniform)
svL = smooth_layers_on_inflated('L', vL, disk_radius_mm)
svR = smooth_layers_on_inflated('R', vR, disk_radius_mm)
# (This is the “curvature correction” step in the paper.)  # see citations below

# 3) Normalize to *relative* thickness per vertex (after smoothing)
rvL = to_relative_per_vertex(svL)
rvR = to_relative_per_vertex(svR)

rvL = to_relative_per_vertex(vL)
rvR = to_relative_per_vertex(vR)


# 4) Parcel to atlas (keep all parcels as requested)
means_L = parcel_means_from_vertices('L', rvL)
means_R = parcel_means_from_vertices('R', rvR)
rel_thickness_matrix = np.vstack([means_L, means_R])

# ---------- pipeline (per the paper’s LTC) ----------
# 1) Load absolute laminar thickness per vertex
vL = load_layers_by_vertex('L', n_layers=n_layers)
vR = load_layers_by_vertex('R', n_layers=n_layers)

# 2) Optional smoothing of ABSOLUTE layer maps on inflated surface
# svL = smooth_layers_on_inflated('L', vL, disk_radius_mm)
# svR = smooth_layers_on_inflated('R', vR, disk_radius_mm)

# 3) Normalize to RELATIVE per vertex (after smoothing, if enabled)
rvL = to_relative_per_vertex(vL)   # or to_relative_per_vertex(svL)
rvR = to_relative_per_vertex(vR)   # or to_relative_per_vertex(svR)

# 4) Parcel-average (keep all parcels)
means_L = parcel_means_from_vertices('L', rvL)   # (P_L, 6)
means_R = parcel_means_from_vertices('R', rvR)   # (P_R, 6)
rel_thickness_matrix = np.vstack([means_L, means_R])  # X: (P, 6)

P = rel_thickness_matrix.shape[0]
I_P = np.eye(P)
I_6 = np.eye(6)

# 5) Partial out nuisance, depending on `partial_mode`
if partial_mode == 'global':
    # --- CURRENT BEHAVIOR ---
    # Regress out the global mean 6-layer profile (and intercept) across the LAYER dimension
    g = rel_thickness_matrix.mean(axis=0)           # (6,)
    G = np.column_stack([g, np.ones_like(g)])       # (6,2)
    H_layers = G @ np.linalg.pinv(G.T @ G) @ G.T    # projector in layer-space
    Y = rel_thickness_matrix.T                      # (6, P)
    Y_res = (I_6 - H_layers) @ Y                    # residualize across layers
    residuals = Y_res.T                              # (P, 6)

elif partial_mode == 'parcel_total':
    # --- NEW OPTION ---
    # Regress out each parcel's overall thickness (sum across layers) across the PARCEL dimension.
    # Use *absolute* parcel means to compute the per-parcel total.
    abs_means_L = parcel_means_from_vertices('L', vL)  # (P_L, 6) ABSOLUTE
    abs_means_R = parcel_means_from_vertices('R', vR)  # (P_R, 6) ABSOLUTE
    abs_means = np.vstack([abs_means_L, abs_means_R])  # (P, 6)
    parcel_total = abs_means.sum(axis=1)               # (P,)

    # Design matrix over parcels: [parcel_total, intercept]
    C = np.column_stack([parcel_total, np.ones_like(parcel_total)])  # (P, 2)
    H_parcels = C @ np.linalg.pinv(C.T @ C) @ C.T                    # projector in parcel-space

    # Residualize the RELATIVE layer matrix across parcels (column-wise)
    # For each layer (column), remove variance explained by parcel_total (and intercept)
    residuals = (I_P - H_parcels) @ rel_thickness_matrix            # (P, 6)

elif partial_mode == 'global_and_parcel':
    # --- NEW: do BOTH ---
    # 1) Remove global mean layer profile (across LAYERS)
    g = rel_thickness_matrix.mean(axis=0)                  # (6,)
    G = np.column_stack([g, np.ones_like(g)])              # (6,2)
    H_layers = G @ np.linalg.pinv(G.T @ G) @ G.T           # (6,6)
    M1 = rel_thickness_matrix @ (I_6 - H_layers)           # (P,6)

    # 2) Remove parcel-wise overall absolute thickness (across PARCELS)
    abs_means_L = parcel_means_from_vertices('L', vL)      # (P_L,6) ABSOLUTE
    abs_means_R = parcel_means_from_vertices('R', vR)      # (P_R,6) ABSOLUTE
    abs_means = np.vstack([abs_means_L, abs_means_R])      # (P,6)
    parcel_total = abs_means.sum(axis=1)                   # (P,)
    C = np.column_stack([parcel_total, np.ones_like(parcel_total)])  # (P,2)
    H_parcels = C @ np.linalg.pinv(C.T @ C) @ C.T          # (P,P)

    residuals = (I_P - H_parcels) @ M1                     # (P,6)

else:
    raise ValueError("partial_mode must be 'global', 'parcel_total', or 'global_and_parcel'")

# 6) Pearson corr across parcels on residuals; remove diagonal BEFORE z
r = np.corrcoef(residuals, rowvar=True)
np.fill_diagonal(r, np.nan)

# 7) Fisher z
r = np.clip(r, -0.999999, 0.999999)
adjacency = np.arctanh(r)
np.fill_diagonal(adjacency, 0)


# 8) Save + plot
np.save(os.path.join(output_dir, 'adjacency_matrix_Schaefer.npy'), adjacency)

vmax = np.nanmax(np.abs(adjacency))
plt.figure(figsize=(6, 6))
plt.imshow(adjacency, cmap="viridis", vmin=-vmax, vmax=vmax)
plt.colorbar(label="Fisher z (partial corr; relative laminar thickness)")
plt.title("LTC (MMP-360, L+R) — disk-smoothed 10 mm on inflated surface")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PartialCorrelation_RelativeThickness_FisherZ_Schaefer.png'), bbox_inches="tight")
plt.close()

restStateSub = lrs.LaminarRestingState(output_dir, 360, 0, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")    
M = adjacency

print(M.shape)

n_components = 5
# G = laman.run_gradient_analysis(M, n_components=n_components, random_state=13011992)
G, A = laman.run_gradient_analysis_affinity(M, n_components=n_components, approach="dm", kernel=None, random_state=13011992)
np.save(os.path.join(output_dir, 'gradients_lamThick_Schaefer.npy'), G)

for i in range(n_components):
    laman.plotFlatMap(G[:,[i]], output_dir, f'BigBrainLamThick_{i}.png')
