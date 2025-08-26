import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore
import laminarAnalyses as laman


def parcel_layer_adjacency_10(
    subject: str,
    runNum: str,                 # you pass "run1"
    layer_01: bool = True,
    out_base_dir: str = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/HubnessAnalysis",
    force_recompute: bool = False
):
    """
    Returns:
      adjacency (400, 400): corr between parcels of their 10-d layer profiles.
    Also saves per-subject products under {out_base_dir}/{subject}/.
    """
    # --- per-subject output dir
    out_dir = os.path.join(out_base_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    adj_path = os.path.join(out_dir, f"Layer10_{runNum}_adjacency.npy")

    # --- cache: load and return if present
    if (not force_recompute) and os.path.exists(adj_path):
        adj = np.load(adj_path)
        if adj.shape == (400, 400):
            print(f"[{subject}] Loaded cached adjacency: {adj_path}")
            return adj
        else:
            print(f"[{subject}] Cached file has wrong shape {adj.shape}; recomputing.")

    # --- paths
    BOLD_data_path = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/{subject}/merged_residuals_{runNum}.nii"
    layer_path     = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/ln_depths_equivol.nii"
    atlas_right    = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_R_in-func.nii"
    atlas_left     = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_L_in-func.nii"

    # --- load data
    layer_img  = nib.load(layer_path)
    right_img  = nib.load(atlas_right)
    left_img   = nib.load(atlas_left)
    bold_img   = nib.load(BOLD_data_path)

    layer_data = layer_img.get_fdata()
    atlas_data = right_img.get_fdata() + left_img.get_fdata()
    bold_data  = bold_img.get_fdata()  # (X, Y, Z, T)

    # --- sanity checks
    if not layer_01:
        raise ValueError("This function expects 'layer_01=True' (0..1 depth map).")
    if layer_data.shape != atlas_data.shape or layer_data.shape != bold_data.shape[:-1]:
        raise ValueError("Dimension mismatch between layer, atlas, and BOLD volumes")

    # --- parcels list (expect 1..400 present in atlas)
    fs_parcels = np.unique(atlas_data)
    fs_parcels = fs_parcels[(fs_parcels > 0) & (fs_parcels <= 400)]
    fs_parcels = np.sort(fs_parcels).astype(int)
    if fs_parcels.size != 400:
        raise RuntimeError(f"Expected 400 parcels in FS atlas, got {fs_parcels.size}")

    # --- build 10 non-overlapping layer masks in 0..1
    n_layers = 10
    edges = np.linspace(0.0, 1.0, n_layers + 1)  # 0.0, 0.1, ..., 1.0
    layer_masks = [ (layer_data > edges[i]) & (layer_data <= edges[i+1]) for i in range(n_layers) ]

    # --- allocate output tensors
    T = bold_data.shape[-1]
    nP = len(fs_parcels)

    # (400, 10, T) z-scored across time within each (parcel, layer)
    parcel_layer_ts = np.zeros((nP, n_layers, T), dtype=float)

    # --- fill time series
    atlas_int = atlas_data.astype(np.int32, copy=False)
    for pi, parcel in enumerate(fs_parcels):
        parcel_mask = (atlas_int == parcel)
        for li, lmask in enumerate(layer_masks):
            combo = parcel_mask & lmask
            if np.any(combo):
                vox_ts = bold_data[combo]  # (#voxels, T)
                mean_ts = np.nanmean(vox_ts, axis=0)
                parcel_layer_ts[pi, li, :] = zscore(mean_ts, axis=0)

    # clean NaNs/Infs (e.g., constant time series)
    parcel_layer_ts = np.nan_to_num(parcel_layer_ts, nan=0.0, posinf=0.0, neginf=0.0)

    # --- per-parcel average across layers (T,) and corr(layer, avg) -> (400, 10)
    def corr_1d(a, b):
        a = np.asarray(a, dtype=float) - np.mean(a)
        b = np.asarray(b, dtype=float) - np.mean(b)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    parcel_avg = parcel_layer_ts.mean(axis=1)  # (400, T)
    parcel_layer_corr = np.zeros((nP, n_layers), dtype=float)
    for i in range(nP):
        y = parcel_avg[i]
        for j in range(n_layers):
            x = parcel_layer_ts[i, j, :]
            parcel_layer_corr[i, j] = corr_1d(x, y)

    # --- adjacency between parcels via corr of their 10-d profiles -> (400, 400)
    with np.errstate(invalid='ignore'):
        adjacency = np.corrcoef(parcel_layer_corr, rowvar=True)
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)

    # --- save useful intermediates
    np.save(os.path.join(out_dir, f"Layer10_{runNum}_parcel_layer_ts.npy"), parcel_layer_ts)
    np.save(os.path.join(out_dir, f"Layer10_{runNum}_parcel_layer_corr_to_avg.npy"), parcel_layer_corr)
    np.save(os.path.join(out_dir, f"Layer10_{runNum}_adjacency.npy"), adjacency)

    print(f"[{subject}] Shapes: pl_ts {parcel_layer_ts.shape}, corr {parcel_layer_corr.shape}, adj {adjacency.shape}")
    return adjacency


# ---------- driver ----------
output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/HubnessAnalysis/'
os.makedirs(output_dir, exist_ok=True)

adjs = []
subjects = ["sub-LAM001","sub-LAM002","sub-LAM003","sub-LAM004","sub-LAM005","sub-LAM006","sub-LAM009","sub-LAM011"]
for s in subjects:
    adj = parcel_layer_adjacency_10(subject=s, runNum="run1", layer_01=True, out_base_dir=output_dir)
    adjs.append(adj)

stacked = np.stack(adjs, axis=-1)    # (400, 400, N)
mean_adj = stacked.mean(axis=2)      # (400, 400)

n_components = 5
G, A = laman.run_gradient_analysis_affinity(mean_adj, n_components=n_components, approach="dm", kernel="cosine", random_state=13011995)
np.save(os.path.join(output_dir, 'gradients_Hubness_Schaefer.npy'), G)

for i in range(n_components):
    laman.plotFlatMap(G[:, [i]], output_dir, f'Hubness_{i}.png')