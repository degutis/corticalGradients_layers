import os
import numpy as np
import networkx as nx
import laminarRestingState as lrs

# ---------------------
# config
# ---------------------
N = 360
NUM_LAYERS = 3
thresholdRange = np.arange(0, 5)   # drop bottom % per row; keep top (100 - t)%
output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGapGraph/'
os.makedirs(output_dir, exist_ok=True)

data_dirs = [
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM001',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM002',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM003',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM004',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM005',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM006',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM009',
    '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM011'
]

# # ---------------------
# # helpers
# # ---------------------
# def ensure_layer_blocks(A):
#     """
#     Accepts (N,N,3) or (3N,3N). Returns (N,N,3).
#     """
#     if A.ndim == 3 and A.shape[:2] == (N, N) and A.shape[2] == NUM_LAYERS:
#         B = A.copy()
#     elif A.ndim == 2 and A.shape == (N*NUM_LAYERS, N*NUM_LAYERS):
#         B = np.stack([A[i*N:(i+1)*N, i*N:(i+1)*N] for i in range(NUM_LAYERS)], axis=2)
#     else:
#         raise ValueError(f"Unexpected adjacency shape {A.shape}; expected (N,N,3) or (3N,3N)")
#     # symmetry + clean diagonal
#     for l in range(NUM_LAYERS):
#         B[:, :, l] = (B[:, :, l] + B[:, :, l].T) / 2.0
#         np.fill_diagonal(B[:, :, l], 0.0)
#     return B

# def thresh_and_mask(adj3d, setThresh=0):
#     """
#     Row-wise masking per layer: drops the bottom setThresh% edges by |corr|.
#     Preserves sign/weight of the kept edges.
#     """
#     N, _, L = adj3d.shape
#     out = np.empty_like(adj3d, dtype=float)
#     frac = setThresh / 100.0
#     rows = np.arange(N)[:, None]
#     for l in range(L):
#         corr = adj3d[:, :, l].copy()
#         mag = np.abs(corr)
#         idx = np.argsort(mag, axis=1)                 # ascending
#         mask = np.ones_like(mag, dtype=bool)
#         k_drop = int(np.floor(frac * N))
#         mask[rows, idx[:, :k_drop]] = False          # drop smallest by row
#         corr_masked = corr * mask
#         np.fill_diagonal(corr_masked, 0.0)
#         out[:, :, l] = corr_masked
#     return out

# def _build_graph(W):
#     """
#     Non-negative, weighted, undirected graph with 'length' = 1/weight for paths.
#     Negatives are set to 0 to avoid ill-defined path lengths.
#     """
#     W = W.copy()
#     W[W < 0] = 0.0
#     np.fill_diagonal(W, 0.0)
#     G = nx.from_numpy_array(W)
#     eps = 1e-12
#     for u, v, d in G.edges(data=True):
#         w = d.get("weight", 0.0)
#         d["length"] = (1.0 / max(w, eps)) if w > 0 else np.inf
#     return G

# def _layer_metrics(W):
#     G = _build_graph(W)
#     WATTR, LATTR = "weight", "length"
#     strength   = np.array([s for _, s in G.degree(weight=WATTR)])
#     clustering = np.array(list(nx.clustering(G, weight=WATTR).values()))
#     eigvec     = np.array(list(nx.eigenvector_centrality(G, weight=WATTR, max_iter=1000).values()))
#     pagerank   = np.array(list(nx.pagerank(G, weight=WATTR).values()))
#     betw_w     = np.array(list(nx.betweenness_centrality(G, weight=LATTR).values()))
#     harm_close = np.array(list(nx.harmonic_centrality(G, distance=LATTR).values()))
#     # Binary variants from W>0
#     Gbin = nx.from_numpy_array((W > 0).astype(int))
#     deg_bin  = np.array([k for _, k in Gbin.degree()])
#     core_num = np.array(list(nx.core_number(Gbin).values()))
#     return dict(
#         strength=strength,
#         clustering_w=clustering,
#         eigvec=eigvec,
#         pagerank=pagerank,
#         betweenness_w=betw_w,
#         harmonic_closeness_w=harm_close,
#         degree_bin=deg_bin,
#         kcore_bin=core_num,
#     )

# def compute_per_layer_metrics(adj3d, setThresh=0):
#     """
#     Returns dict[metric] -> (N, 3) array; columns = layers 0,1,2.
#     """
#     A = ensure_layer_blocks(adj3d)
#     A = thresh_and_mask(A, setThresh=setThresh)      # density-matched per layer
#     per_layer = [_layer_metrics(A[:, :, l]) for l in range(NUM_LAYERS)]
#     keys = per_layer[0].keys()
#     return {k: np.column_stack([per_layer[0][k], per_layer[1][k], per_layer[2][k]]) for k in keys}

# def add_layer_deltas(metrics):
#     """
#     Adds L2-L1, L3-L1, L3-L2 arrays (shape (N,)) per metric.
#     """
#     out = dict(metrics)
#     for k, M in metrics.items():  # M: (N,3)
#         out[f"{k}_L2minusL1"] = M[:, 1] - M[:, 0]
#         out[f"{k}_L3minusL1"] = M[:, 2] - M[:, 0]
#         out[f"{k}_L3minusL2"] = M[:, 2] - M[:, 1]
#     return out

# # ---------------------
# # main loop
# # ---------------------
# for thresh in thresholdRange:
#     print(f"threshold={thresh}")
#     for data_dir in data_dirs:
#         sub_id = os.path.basename(data_dir.rstrip(os.sep))
#         restStateSub = lrs.LaminarRestingState(
#             data_dir, N, thresh,
#             atlas_dir="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii"
#         )
#         _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()  # (360,360,3) or (1080,1080)

#         metrics = compute_per_layer_metrics(adj_matrix_within_corr, setThresh=thresh)
#         metrics = add_layer_deltas(metrics)

#         # save per subject & threshold
#         out_path = os.path.join(output_dir, f"{sub_id}_graphMetrics_thresh{thresh}.npz")
#         np.savez_compressed(out_path, **metrics)


# ---------- plotting.py ----------
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

N = 360
LAYERS = ["L1", "L2", "L3"]

def _load_subject_mats(files, measure_key):
    """Load a (360,3) matrix for each subject for the given measure."""
    mats = []
    for f in files:
        with np.load(f) as data:
            if measure_key not in data:
                raise KeyError(f"{measure_key} not found in {os.path.basename(f)}")
            M = data[measure_key]  # expected shape (360,3)
            if M.shape != (N, 3):
                raise ValueError(f"{os.path.basename(f)} has shape {M.shape}, expected (360,3)")
            mats.append(M)
    return mats  # list of (360,3)

def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def plot_box_per_layer(mats, measure_key, save_to, thresh):
    """
    Boxplot of parcel values aggregated across subjects × parcels for each layer.
    """
    X = np.concatenate([M.reshape(-1, 3) for M in mats], axis=0)  # shape (S*360, 3)
    fig = plt.figure()
    plt.boxplot([X[:, 0], X[:, 1], X[:, 2]], labels=LAYERS, showfliers=False)
    plt.ylabel(measure_key)
    plt.title(f"{measure_key} • Parcel distribution per layer (thresh={thresh})")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.close(fig)

def plot_subject_means(mats, measure_key, save_to, thresh, subject_labels=None):
    """
    Line plot: for each subject, draw L1→L2→L3 means (across parcels).
    """
    subj_means = np.array([M.mean(axis=0) for M in mats])  # (S,3)
    fig = plt.figure()
    x = np.arange(1, 4)
    for i, y in enumerate(subj_means):
        plt.plot(x, y, marker="o", linewidth=1)
    plt.xticks(x, LAYERS)
    plt.xlabel("Layer")
    plt.ylabel(f"Mean {measure_key} (per subject)")
    if subject_labels:
        # optional: annotate line ends with subject IDs
        for i, y in enumerate(subj_means):
            plt.text(3.02, y[-1], subject_labels[i], va="center", fontsize=8)
    mu = subj_means.mean(axis=0)
    plt.title(f"{measure_key} • Subject means across layers (thresh={thresh})\n"
              f"Group mean: {mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150)
    plt.close(fig)

def plot_all_measures(output_dir, thresh, measures, subject_labels=None):
    """
    For a given threshold, load all subject files and make 3 plots per measure.
    """
    pattern = os.path.join(output_dir, f"*graphMetrics_thresh{thresh}.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    mats_by_measure = {m: _load_subject_mats(files, m) for m in measures}

    outdir = os.path.join(output_dir, f"plots_thresh{thresh}")
    _ensure_outdir(outdir)

    for m in measures:
        mats = mats_by_measure[m]
        # 1) Boxplot across subjects × parcels
        plot_box_per_layer(mats, m, os.path.join(outdir, f"{m}_boxlayers_t{thresh}.png"), thresh)
        # 2) Subject means (lines)
        plot_subject_means(mats, m, os.path.join(outdir, f"{m}_subjectMeans_t{thresh}.png"), thresh, subject_labels)

    print(f"Saved plots to: {outdir}")

# ---------- example call ----------
# Adjust to the metrics you saved in your NPZs:
MEASURES = ["strength", "clustering_w", "eigvec", "pagerank",
            "betweenness_w", "harmonic_closeness_w", "degree_bin", "kcore_bin"]
plot_all_measures(output_dir, thresh=4, measures=MEASURES,
                  subject_labels=["LAM001","LAM002","LAM003","LAM004","LAM005","LAM006","LAM009","LAM011"])