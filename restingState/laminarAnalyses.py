import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from scipy import stats
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps


def run_gradient_analysis(conn_matrix, n_components=10, kernel="cosine", approach="dm", random_state=0):
    
    """
    conn_matrix: (1080 x 1080) supra-adjacency (deep/mid/sup on diagonal blocks, identity couplings off-diagonal).
    Returns G: (1080 x n_components) gradient coordinates in a joint embedding.
    """

    gm = GradientMaps(kernel=kernel, approach=approach, n_components=n_components, random_state=random_state)
    gm.fit(conn_matrix)             # BrainSpace builds the affinity & diffusion map internally
    return gm.gradients_            # shape: (1080, n_components)
    
def _split_layers(G1080, N=360):
    assert G1080.ndim == 2 and G1080.shape[0] == 3*N, f"Expected (3*{N} x k); got {G1080.shape}"
    return G1080[0:N, :], G1080[N:2*N, :], G1080[2*N:3*N, :]

def _l2_normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Y = X / norms
    # keep exact zeros as zeros
    zero_rows = np.isclose(np.linalg.norm(X, axis=1), 0.0)
    if np.any(zero_rows):
        Y[zero_rows, :] = 0.0
    return Y

def inter_areal_dissimilarity(G1080, outputDir, N=360, zscore_within_layer=False):
    """
    D_inter[i] = mean cosine distance between parcel i's laminar profile and all other parcels' profiles.
    G1080: (1080 x k) gradients in a *joint* embedding.
    Returns: (N,) array.
    """
    Gd, Gm, Gs = _split_layers(G1080, N=N)

    if zscore_within_layer:
        Gd = (Gd - Gd.mean(axis=0, keepdims=True)) / (Gd.std(axis=0, keepdims=True) + 1e-12)
        print(Gd.shape)
        Gm = (Gm - Gm.mean(axis=0, keepdims=True)) / (Gm.std(axis=0, keepdims=True) + 1e-12)
        Gs = (Gs - Gs.mean(axis=0, keepdims=True)) / (Gs.std(axis=0, keepdims=True) + 1e-12)

    Ud = _l2_normalize_rows(Gd)
    Um = _l2_normalize_rows(Gm)
    Us = _l2_normalize_rows(Gs)

    # (N x 3k) laminar profile per parcel (unit rows → cosine similarity via dot)
    P = np.concatenate([Ud, Um, Us], axis=1)

    plt.figure(figsize=(6, 6))
    plt.imshow(P, cmap="RdBu_r")
    plt.title("ConcatMatrix P - inter areal dis")
    plt.savefig(f"{outputDir}/ConcatMatrixP_inter.svg", bbox_inches="tight")
    plt.close()


    # cosine distance matrix
    S = P @ P.T              # similarities, diag ~ 1
    D = 1.0 - S              # distances
    np.fill_diagonal(D, 0.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(D, cmap="RdBu_r")
    plt.title("Distance matrix - inter areal dis")
    plt.savefig(f"{outputDir}/Matrix_interArealDis.svg", bbox_inches="tight")
    plt.close()

    distanceSum = D.sum(axis=1) / (N - 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(distanceSum[:, np.newaxis], cmap="RdBu_r")
    plt.title("Distance sum - inter areal dis")
    plt.savefig(f"{outputDir}/Matrix_interArealDisSum.svg", bbox_inches="tight")
    plt.close()

    return distanceSum

def plotMatrix(M, outputDir, name):

    plt.figure(figsize=(6, 6))
    plt.imshow(M, cmap="RdBu_r")
    plt.title("Adjacency matrix")
    plt.savefig(f"{outputDir}/{name}", bbox_inches="tight")
    plt.close()



def intra_areal_dissimilarity(G1080, outputDir, N=360, zscore_within_layer=False, mode="to_mean"):
    """
    D_intra[i] measures laminar heterogeneity within parcel i.
    mode="to_mean": average cosine distance of each layer vector to the parcel's mean *direction*.
    mode="pairwise": mean pairwise cosine distance among the three layers.
    Returns: (N,) array.
    """
    Gd, Gm, Gs = _split_layers(G1080, N=N)

    if zscore_within_layer:
        Gd = (Gd - Gd.mean(axis=0, keepdims=True)) / (Gd.std(axis=0, keepdims=True) + 1e-12)
        Gm = (Gm - Gm.mean(axis=0, keepdims=True)) / (Gm.std(axis=0, keepdims=True) + 1e-12)
        Gs = (Gs - Gs.mean(axis=0, keepdims=True)) / (Gs.std(axis=0, keepdims=True) + 1e-12)

    Ud, Um, Us = _l2_normalize_rows(Gd), _l2_normalize_rows(Gm), _l2_normalize_rows(Gs)

    if mode == "to_mean":
        # Ubar = _l2_normalize_rows((Ud + Um + Us) / 3.0)
        Ubar = _l2_normalize_rows((Gd + Gm + Gs) / 3.0)
        print(Ubar.shape)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(Ubar, cmap="RdBu_r")
        plt.title("Ubar - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_intraArealDis.svg", bbox_inches="tight")
        plt.close()

        d_deep = 1.0 - np.einsum("ij,ij->i", Ud, Ubar)
        print(d_deep.shape)

        plt.figure(figsize=(10, 10))
        plt.imshow(d_deep[:,np.newaxis], cmap="RdBu_r")
        plt.title("D_deep - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_deep_intraArealDis.svg", bbox_inches="tight")
        plt.close()


        d_mid  = 1.0 - np.einsum("ij,ij->i", Um, Ubar)

        plt.figure(figsize=(10, 10))
        plt.imshow(d_mid[:,np.newaxis], cmap="RdBu_r")
        plt.title("D_mid - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_mid_intraArealDis.svg", bbox_inches="tight")
        plt.close()

        d_sup  = 1.0 - np.einsum("ij,ij->i", Us, Ubar)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(d_sup[:,np.newaxis], cmap="RdBu_r")
        plt.title("D_sup - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_sup_intraArealDis.svg", bbox_inches="tight")
        plt.close()


        d_intraMean = (d_deep + d_mid + d_sup) / 3.0
        
        plt.figure(figsize=(10, 10))
        plt.imshow(d_intraMean[:,np.newaxis], cmap="RdBu_r")
        plt.title("D_intraMean - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_mean_intraArealDis.svg", bbox_inches="tight")
        plt.close()


        return d_intraMean, d_deep, d_mid, d_sup

    elif mode == "pairwise":
        d_dm = 1.0 - np.einsum("ij,ij->i", Ud, Um)
        d_ds = 1.0 - np.einsum("ij,ij->i", Ud, Us)
        d_ms = 1.0 - np.einsum("ij,ij->i", Um, Us)
        return (d_dm + d_ds + d_ms) / 3.0

    else:
        raise ValueError("mode must be 'to_mean' or 'pairwise'")


def plotFlatMap(
    M,
    outdir,
    outname,
    inputdir_1="/home/degutis/repos/HCP_WB_parcels",
    inputdir_2="/home/degutis/repos/HumanCorticalParcellations",
    cmap="RdBu_r",
    vmin=None, vmax=None,
    symmetric=False,         # center color range at 0 if True
    rasterize=False          # set True if SVGs get too heavy
):
    """
    M : (360,) or (360,1) array of parcel values (Glasser order: LH 1..180, RH 181..360).
    Saves {outdir}/{outname} (e.g., 'D_interFlatMap.svg' or .png).
    Requires:
      - {inputdir_1}/GlasserAtlas.L.32k_fs_LR.label.gii
      - {inputdir_1}/GlasserAtlas.R.32k_fs_LR.label.gii
      - {inputdir_2}/S1200.L.flat.32k_fs_LR.surf.gii
      - {inputdir_2}/S1200.R.flat.32k_fs_LR.surf.gii
    """
    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    vals = np.asarray(M).reshape(-1)
    assert vals.shape[0] == 360, f"Expected 360 values, got {vals.shape}"

    # ---- Load per-vertex labels (0=medial wall, 1..180 per hemi)
    L_lab = nib.load(os.path.join(inputdir_1, "GlasserAtlas.L.32k_fs_LR.label.gii")).agg_data().astype(int).squeeze()
    R_lab = nib.load(os.path.join(inputdir_1, "GlasserAtlas.R.32k_fs_LR.label.gii")).agg_data().astype(int).squeeze()

    # ---- Map parcel values -> per-vertex metrics
    metric_L = np.full(L_lab.shape, np.nan, float)
    metric_R = np.full(R_lab.shape, np.nan, float)
    mL = L_lab > 0
    mR = R_lab > 0
    metric_L[mL] = vals[L_lab[mL] - 1]            # LH labels 1..180 -> vals[0..179]
    metric_R[mR] = vals[180 + R_lab[mR] - 1]      # RH labels 1..180 -> vals[180..359]

    # ---- Load flat meshes (coords, faces)
    def load_surf_xy(path):
        g = nib.load(path)
        coords = np.asarray(g.darrays[0].data, dtype=float)
        faces  = np.asarray(g.darrays[1].data, dtype=int)
        # Use X,Y for flat map (Z is ~0)
        return coords[:, 0], coords[:, 1], faces

    xL, yL, fL = load_surf_xy(os.path.join(inputdir_2, "S1200.L.flat.32k_fs_LR.surf.gii"))
    xR, yR, fR = load_surf_xy(os.path.join(inputdir_2, "S1200.R.flat.32k_fs_LR.surf.gii"))

    # Sanity checks
    assert metric_L.size == xL.size and metric_R.size == xR.size, "Vertex count mismatch (labels vs surface)."

    # ---- Color range
    data_all = np.concatenate([metric_L[np.isfinite(metric_L)], metric_R[np.isfinite(metric_R)]])
    if data_all.size == 0:
        raise ValueError("All metric values are NaN.")
    if symmetric:
        m = np.nanmax(np.abs(data_all))
        vmin, vmax = -m, m
    else:
        if vmin is None: vmin = np.nanmin(data_all)
        if vmax is None: vmax = np.nanmax(data_all)
        if vmin == vmax:  # avoid degenerate color scale
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

    # ---- Build triangulations
    triL = mtri.Triangulation(xL, yL, fL)
    triR = mtri.Triangulation(xR, yR, fR)

    # Mask triangles touching NaNs (so medial wall doesn't render)
    maskL = np.any(np.isnan(metric_L)[fL], axis=1)
    maskR = np.any(np.isnan(metric_R)[fR], axis=1)
    triL.set_mask(maskL)
    triR.set_mask(maskR)

    # ---- Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax in axes: ax.set_aspect('equal'); ax.axis('off')

    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    # Left hemi
    imL = axes[0].tripcolor(triL, metric_L, **kw)
    if rasterize: imL.set_rasterized(True)
    axes[0].set_title("Left hemisphere")
    # Right hemi
    imR = axes[1].tripcolor(triR, metric_R, **kw)
    if rasterize: imR.set_rasterized(True)
    axes[1].set_title("Right hemisphere")

    # Colorbar (shared)
    cbar = fig.colorbar(imR, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Value")

    # Save (SVG or PNG etc.)
    fig.savefig(outpath, dpi=300 if outpath.lower().endswith(".png") else None)
    plt.close(fig)
    return outpath



def runClusterAnalysis(eigvecs_list, threshold=0.3):

    eigvecs_array = np.array([v / np.linalg.norm(v) for v in eigvecs_list])

    D = squareform(pdist(eigvecs_array, metric=sign_invariant_distance))

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(D)
    
    cluster_groups = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        cluster_groups[cluster_id].append(i)

    return cluster_groups, labels

# Pairwise distance matrix (sign-invariant)
def sign_invariant_distance(u, v):
    return 1 - np.abs(np.dot(u, v))

def convert_eigvals_to_list(eigvecs, eigvals, N, num_layers):
    
    eigvecs_list = []
    eigvalue_list = []
    source_info = []

    for layer_idx in range(num_layers):
        row_start = layer_idx * N
        row_end = row_start + N

        # Get the 360x1080 block for this layer (rows slice, all columns)
        layer_eigvecs = eigvecs[row_start:row_end, :]

        row_idx = layer_idx + 1
        col_idx = layer_idx + 1

        for i in range(layer_eigvecs.shape[1]):
            eigvec = layer_eigvecs[:, i]
            eigvecs_list.append(eigvec / np.linalg.norm(eigvec))
            #source_info.append((layer_idx + 1, i))  # layer 1-based, eigenvector index
            source_info.append((row_idx, col_idx, i))
            eigvalue_list.append(eigvals[i])

    return eigvecs_list, eigvalue_list, source_info


def plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, name):
    
    for cluster_id, indices in cluster_groups.items():
        
        if len(indices) == 1:
            continue
        eigvecs_to_plot = [eigvecs_list[i] for i in indices]
        meta = [source_info[i] for i in indices]
        titles = [f"(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        # Build filename from source info
        name_str = "-".join([f"r{r}_c{c}_e{e}" for (r, c, e) in meta])
        eig_label = f"{name_str}"

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, eig_label, name=name, titles=titles, folder_name="SimilarDissimilar"
            )


    for cluster_id, indices in cluster_groups.items():
        if len(indices) > 1:
            continue  # Only consider singleton clusters

        i = indices[0]
        if not(0 < eigvalue_list[i] < eigenvalue_threshold):
            continue

        r, c, _ = source_info[i]
        v_i = eigvecs_list[i] / np.linalg.norm(eigvecs_list[i])

        # Compare against others from the same region-pair
        similar_found = False
        for j, (rj, cj, _) in enumerate(source_info):
            if (rj, cj) == (r, c) and j != i:
                eigval_j = eigvalue_list[j]
                if not (0 < eigval_j < eigenvalue_threshold):
                    continue
                
                v_j = eigvecs_list[j] / np.linalg.norm(eigvecs_list[j])
                similarity = np.abs(np.dot(v_i, v_j))
                if similarity >= (1 - cluster_threshold):
                    similar_found = True
                    break

        if similar_found:
            continue

        # Passed distinctness check → plot
        eigvecs_to_plot = [eigvecs_list[i]]
        meta = [source_info[i]]
        titles = [f"Distinct_(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        name_str = f"r{r}_c{c}_e{meta[0][2]}"
        eig_label = name_str

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, eig_label, name=name, titles=titles, folder_name="SimilarDissimilar"
        )

def hurst_dfa(ts, min_window=5, max_window=None, n_windows=20):
    """
    Estimate the Hurst exponent via Detrended Fluctuation Analysis (DFA).
    
    Parameters
    ----------
    ts : 1D np.ndarray
        Input time series of length T.
    min_window : int
        Minimum window size (in samples) to start DFA.
    max_window : int, optional
        Maximum window size. Defaults to len(ts)//4.
    n_windows : int
        Number of distinct window sizes (log-spaced) between min and max.
    
    Returns
    -------
    H : float
        Estimated Hurst exponent.
    windows : np.ndarray
        Array of window sizes used.
    flucts : np.ndarray
        Fluctuation function values at each window size.
    """
    ts = np.array(ts, dtype=float)
    T = len(ts)
    if max_window is None:
        max_window = T // 4
    
    # 1) Integrate (cumsum of demeaned signal)
    ts_demeaned = ts - np.mean(ts)
    y = np.cumsum(ts_demeaned)

    # 2) Define window sizes (logarithmically spaced)
    windows = np.floor(np.logspace(np.log10(min_window), np.log10(max_window), n_windows)).astype(int)
    windows = np.unique(windows[windows > 0])

    flucts = []
    for m in windows:
        # split into non-overlapping windows of size m
        n_segments = T // m
        segs = y[:n_segments * m].reshape(n_segments, m)
        # detrend each segment
        local_rms = []
        x = np.arange(m)
        for seg in segs:
            # linear fit
            p = np.polyfit(x, seg, 1)
            trend = np.polyval(p, x)
            # root-mean-square of detrended
            rms = np.sqrt(np.mean((seg - trend) ** 2))
            local_rms.append(rms)
        flucts.append(np.mean(local_rms))
    flucts = np.array(flucts)

    # 3) Fit line in log-log to get scaling exponent
    log_windows = np.log10(windows)
    log_flucts = np.log10(flucts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_flucts)

    H = slope  # for DFA, slope ≈ Hurst exponent
    return H, windows, flucts

def hurst_dfa_bound(ts, min_window=5, max_window=None, n_windows=20,
              slope_bounds=(-0.5, 1.5), intercept_bounds=(0, 10)):
    """
    Estimate the Hurst exponent via bounded DFA.

    Returns
    -------
    H : float
        Estimated Hurst exponent (slope of log-log fit).
    windows : np.ndarray
        Window sizes used.
    flucts : np.ndarray
        Fluctuation function values at each window size.
    """
    ts = np.asarray(ts, dtype=float)
    T = len(ts)
    if max_window is None:
        max_window = T // 2

    # 1) integrate demeaned series
    y = np.cumsum(ts - ts.mean())

    # 2) select windows
    wins = np.floor(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows)
    ).astype(int)
    windows = np.unique(wins[wins > 0])

    # 3) compute F(m) for each window size
    # flucts = []
    # for m in windows:
    #     nseg = T // m
    #     segs = y[:nseg*m].reshape(nseg, m)
    #     rms_vals = []
    #     x = np.arange(m)
    #     for seg in segs:
    #         p = np.polyfit(x, seg, 1)
    #         trend = np.polyval(p, x)
    #         rms_vals.append(np.sqrt(np.mean((seg - trend)**2)))
    #     flucts.append(np.mean(rms_vals))
    # flucts = np.array(flucts)
    # 3) compute fluctuation F(m) with quadratic detrending
    
    flucts = []
    x = None
    for m in windows:
        nseg = T // m
        segs = y[:nseg*m].reshape(nseg, m)
        if x is None or len(x) != m:
            x = np.arange(m)
        rms_vals = []
        for seg in segs:
            # quadratic fit
            p = np.polyfit(x, seg, 2)
            trend = np.polyval(p, x)
            rms_vals.append(np.sqrt(np.mean((seg - trend)**2)))
        flucts.append(np.mean(rms_vals))
    flucts = np.array(flucts)
    
    valid = (flucts > 0) & np.isfinite(flucts)
    if valid.sum() < 2:
        # raise ValueError("Not enough valid (fluct > 0) windows to fit a slope.")
        print("Not enough valid (fluct > 0) windows to fit a slope.")
        print(f"Valid windows: {flucts[valid]}")   
        return np.nan, np.array([], dtype=int), np.array([], dtype=float)
    windows = windows[valid]
    flucts = flucts[valid]


    # 4) bounded linear fit in log-log space
    log_w = np.log10(windows)
    log_f = np.log10(flucts)

    # design matrix: [ log_w , 1 ]
    X = np.vstack([log_w, np.ones_like(log_w)]).T

    # set bounds on [slope, intercept]
    lb = [slope_bounds[0], intercept_bounds[0]]
    ub = [slope_bounds[1], intercept_bounds[1]]

    res = lsq_linear(X, log_f, bounds=(lb, ub))
    # res = lsq_linear(X, log_f)
    slope, intercept = res.x
    return slope, windows, flucts


def dfa_fast(vdata, istart=None, iend=None, L_all=None, min_L=5, max_frac=0.25, num_L=20):

    # taken and edited from: https://github.com/kjamison/fmriclean/blob/master/fmri_hurst.py
    
    vdata = np.array(vdata)
    T, M = vdata.shape

    # default start/end = full series
    if istart is None:
        istart = 0
    if iend is None:
        iend = T - 1

    # compute L_all automatically if not provided
    if L_all is None:
        # maximum window length = floor(max_frac * T)
        max_L = int(np.floor(max_frac * T))
        # choose num_L values log-spaced between min_L and max_L
        L_all = np.unique(
            np.floor(
                np.logspace(np.log10(min_L),
                            np.log10(max_L),
                            num=num_L)
            ).astype(int)
        )
        # ensure we don't include anything < min_L
        L_all = L_all[L_all >= min_L]


    istart = int(round(istart))
    iend   = int(round(iend))
    
    # integrate (cumulative sum) along time
    vdata = np.cumsum(vdata, axis=0)

    # pre-allocate
    FL_all = np.zeros((len(L_all), M))

    # for each window size
    for il, L in enumerate(L_all):
        # design matrix for linear detrending: [t, 1]
        t = np.arange(L)
        X = np.vstack([t, np.ones(L)]).T

        c = 0
        FL = np.zeros(M)
        # slide non-overlapping windows of length L
        for start in range(istart, min(iend+1, T)-L+1, L):
            seg = vdata[start:start+L, :]       # shape (L, M)
            # fit linear trend in each column
            # b has shape (2, M), sse is length-M array of sum-of-squares errors
            b, sse, _, _ = np.linalg.lstsq(X, seg, rcond=None)
            rms = np.sqrt(sse / L)              # rms per channel
            FL += rms
            c  += 1

        FL_all[il, :] = FL / c

    # now fit log F(L) = alpha * log L + log b
    logFL = np.log(FL_all)             # shape (len(L_all), M)
    logL  = np.log(L_all)[:, None]     # shape (len(L_all), 1)
    logX    = np.hstack([logL, np.ones_like(logL)])
    b, _, _, _ = np.linalg.lstsq(logX, logFL, rcond=None)
    alpha = b[0, :]                     # the DFA exponents

    alpha = []
    for i in range(logFL.shape[1]):
        res = lsq_linear(logX, logFL[:,i], bounds=([-0.5,0], [1.5,10]))
        # res = lsq_linear(logX, logFL[:,i], bounds=(-0.5,1.5))
        alpha.append(res.x[0])

    return alpha,FL_all

def plot_cosine_similarity(cosineSim, data_dir, name, thresholds, extraName='CosineSimilarityAcrossThresholds§', labels=None, ylabel='Cosine Similarity', ):
    """
    Plot mean cosine similarity across thresholds with SEM shading.

    Parameters
    ----------
    cosineSim : np.ndarray
        Array of shape (nComparisons, nSubjects, nThresh) containing cosine similarity values.
    thresholds : array-like, optional
        Sequence of threshold values (e.g., np.arange(70, 100)). If None, defaults to 70-99.
    labels : list of str, optional
        Labels for each comparison line. If None, generic labels are used.

    Raises
    ------
    ValueError
        If the length of thresholds does not match the third dimension of cosineSim.
    """
    # Default thresholds 70-99 if not provided
    
    nComparisons, nSubjects, nThresh = cosineSim.shape
    
    # Validate thresholds length
    if len(thresholds) != nThresh:
        raise ValueError(f"Expected thresholds of length {nThresh}, got {len(thresholds)}.")

    # Default labels if none provided
    if labels is None:
        labels = [f'Comparison {i+1}' for i in range(nComparisons)]
        labels = ["Deep vs. Middle", "Deep vs. Superficial", "Superficial vs. Middle"]

    mean_sim = cosineSim.mean(axis=1)
    sem_sim = cosineSim.std(axis=1, ddof=1) / np.sqrt(nSubjects)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    for i in range(nComparisons):
        plt.plot(thresholds, mean_sim[i], label=labels[i])
        plt.fill_between(thresholds, 
                         mean_sim[i] - sem_sim[i], 
                         mean_sim[i] + sem_sim[i], 
                         alpha=0.2)
    
    plt.xlabel('Threshold (%)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{data_dir}/{name}/{extraName}.png", bbox_inches="tight")
    plt.close()


def cosine_similarity_upper(mat1: np.ndarray, mat2: np.ndarray) -> float:

    if mat1.shape != mat2.shape:
        raise ValueError(f"Matrix shapes must match, got {mat1.shape} and {mat2.shape}")
    if mat1.ndim != 2 or mat1.shape[0] != mat1.shape[1]:
        raise ValueError(f"Matrices must be square, but got shape {mat1.shape}")
    
    # Extract upper-triangular indices without the diagonal
    iu = np.triu_indices(mat1.shape[0], k=1)
    
    # Vectorize the upper-triangular parts
    v1 = mat1[iu]
    v2 = mat2[iu]
    
    # Compute cosine similarity
    dot = np.dot(v1, v2)
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0:
        return 0.0  # or np.nan, depending on desired behavior
    return dot / norm_prod
