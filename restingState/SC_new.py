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
# svL = smooth_layers_on_inflated('L', vL, disk_radius_mm)
# svR = smooth_layers_on_inflated('R', vR, disk_radius_mm)
# (This is the “curvature correction” step in the paper.)  # see citations below

# 3) Normalize to *relative* thickness per vertex (after smoothing)
# rvL = to_relative_per_vertex(svL)
# rvR = to_relative_per_vertex(svR)

rvL = to_relative_per_vertex(vL)
rvR = to_relative_per_vertex(vR)


# 4) Parcel to MMP-360 (keep all parcels as requested)
means_L = parcel_means_from_vertices('L', rvL)
means_R = parcel_means_from_vertices('R', rvR)
rel_thickness_matrix = np.vstack([means_L, means_R])  # (360, 6)

# 5) Partial corr between parcels controlling for the average layer profile
g = rel_thickness_matrix.mean(axis=0)                # (6,)
G = np.column_stack([g, np.ones_like(g)])            # (6,2)
GtG_inv = np.linalg.inv(G.T @ G)
H = G @ GtG_inv @ G.T
I = np.eye(6)

Y = rel_thickness_matrix.T                           # (6,360)
Y_res = (I - H) @ Y
residuals = Y_res.T                                  # (360,6)

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
G, A = laman.run_gradient_analysis_affinity(M, n_components=n_components, approach="PCA", kernel=None, random_state=13011992)
np.save(os.path.join(output_dir, 'gradients_lamThick_Schaefer.npy'), G)

for i in range(n_components):
    laman.plotFlatMap(G[:,[i]], output_dir, f'BigBrainLamThick_{i}.png')
