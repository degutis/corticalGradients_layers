import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from scipy import stats
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps
import matplotlib as mpl




def run_gradient_analysis(conn_matrix, n_components=10, kernel="cosine", approach="dm", random_state=0, sparsity=0.9):
    
    """
    conn_matrix: (1080 x 1080) supra-adjacency (deep/mid/sup on diagonal blocks, identity couplings off-diagonal).
    Returns G: (1080 x n_components) gradient coordinates in a joint embedding.
    """
    gm = GradientMaps(kernel=kernel, approach=approach, n_components=n_components, random_state=random_state)
    # gm.fit(conn_matrix)             # BrainSpace builds the affinity & diffusion map internally
    gm.fit(conn_matrix, sparsity=sparsity)             # BrainSpace builds the affinity & diffusion map internally

    return gm.gradients_, gm.lambdas_           # shape: (1080, n_components)
    

def run_gradient_analysis_affinity(conn_matrix, n_components=10, kernel="cosine", approach="dm", random_state=0, sparsity=0.9):

    from brainspace.gradient.kernels import compute_affinity
    from brainspace.gradient import GradientMaps

    # X: (n_samples × n_features), e.g., regions × connectivity profile
    A = compute_affinity(
        conn_matrix,
        kernel=kernel,       # or 'pearson', 'spearman', 'normalized_angle', 'gaussian'
        sparsity=sparsity,          # set None for no sparsification
        non_negative=True      # set False if you want to keep negatives
    )

    gm = GradientMaps(n_components=n_components, approach=approach, kernel=None, random_state=random_state)
    gm.fit(A)    
    
    return gm.gradients_, A



def _split_layers(G, N=360):
    """
    Split (L*N x k) matrix into L layers of shape (N x k).
    """
    assert G.ndim == 2, f"Expected 2D array; got {G.ndim}D"
    L, rem = divmod(G.shape[0], N)
    assert rem == 0 and L >= 1, f"Expected (L*{N} x k); got {G.shape}"
    return [G[i*N:(i+1)*N, :] for i in range(L)]

def _l2_normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Y = X / norms
    zero_rows = np.isclose(np.linalg.norm(X, axis=1), 0.0)
    if np.any(zero_rows):
        Y[zero_rows, :] = 0.0
    return Y

def _equidistant_layer_indices(L):
    """
    Return three ~equidistant layer indices (0-based) spanning [0, L-1],
    excluding hard endpoints when possible:
      idx = ceil([L/4, L/2, 3L/4]) - 1
    For L=8 -> (1, 3, 5)  (i.e., 2, 4, 6 in 1-based terms).
    """
    if L < 3:
        raise ValueError("Need at least 3 layers to choose superficial/middle/deep.")
    idx = np.ceil(np.array([1, 2, 3]) * L / 4.0).astype(int) - 1
    idx = np.clip(idx, 0, L-1)
    # If duplicates occur for small L, spread them sensibly.
    if len(np.unique(idx)) < 3:
        idx = np.rint(np.linspace(0, L-1, 3)).astype(int)
    return tuple(np.sort(idx).tolist())  # (superficial, middle, deep)


def inter_areal_dissimilarity(G_all, outputDir, N=360, zscore_within_layer=False):
    """
    D_inter[i] = mean cosine distance between parcel i's laminar profile and all other parcels' profiles.
    G_all: (L*N x k) gradients from a joint embedding.
    Returns:
      - If L == 3: (distanceSum, distanceSum_deep, distanceSum_mid, distanceSum_sup)
      - Else:      (distanceSum, distanceSum_layers) where distanceSum_layers has shape (L, N)
    """
    layers = _split_layers(G_all, N=N)   # list of length L; each (N x k)
    L = len(layers)

    if zscore_within_layer:
        layers = [
            (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
            for X in layers
        ]

    # Unit-row embeddings per layer
    U_layers = [_l2_normalize_rows(X) for X in layers]

    # (N x (L*k)) laminar profile per parcel (concatenate along features)
    P = np.concatenate(U_layers, axis=1)

    # Plot concatenated profile matrix
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['text.usetex'] = False
    plt.figure(figsize=(6, 6))
    plt.imshow(P, cmap="viridis")
    plt.title("ConcatMatrix P - inter areal dis")
    plt.savefig(f"{outputDir}/ConcatMatrixP_inter.svg", bbox_inches="tight", format="svg")
    plt.close()

    # Cosine distance across concatenated profiles
    S = P @ P.T          # similarities (diag ~ 1)
    D = 1.0 - S          # distances
    np.fill_diagonal(D, 0.0)

    # Mean distance per parcel (exclude self)
    distanceSum = D.sum(axis=1) / (N - 1)

    # Per-layer distances & means
    D_layers = [1.0 - (U @ U.T) for U in U_layers]
    for Dl in D_layers:
        np.fill_diagonal(Dl, 0.0)
    distanceSum_layers = np.vstack([Dl.sum(axis=1) / (N - 1) for Dl in D_layers])  # (L, N)

    i_sup, i_mid, i_deep = _equidistant_layer_indices(L)
    distanceSum_sup = distanceSum_layers[i_sup]
    distanceSum_mid      = distanceSum_layers[i_mid]
    distanceSum_deep        = distanceSum_layers[i_deep]

    # Plot overall distance matrix and mean-distance column
    plt.figure(figsize=(6, 6))
    plt.imshow(D, cmap="viridis")
    plt.title("Distance matrix - inter areal dis")
    plt.savefig(f"{outputDir}/Matrix_interArealDis.svg", bbox_inches="tight", format="svg")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(distanceSum[:, np.newaxis], cmap="viridis")
    plt.title("Distance sum - inter areal dis")
    plt.savefig(f"{outputDir}/Matrix_interArealDisSum.svg", bbox_inches="tight", format="svg")
    plt.close()

    return distanceSum, distanceSum_deep, distanceSum_mid, distanceSum_sup

def plotMatrix(M, outputDir, name):

    mpl.rcParams['svg.fonttype'] = 'none'   # keep text as <text>, not paths
    mpl.rcParams['text.usetex'] = False     # avoid TeX (which becomes outlines)

    plt.figure(figsize=(6, 6))
    plt.imshow(M, cmap="viridis")
    plt.title("Adjacency matrix")
    plt.savefig(f"{outputDir}/{name}", bbox_inches="tight", format="svg")
    plt.close()


def intra_areal_dissimilarity(G_all, outputDir, N=360, zscore_within_layer=False, mode="to_mean"):
    
    """
    D_intra[i] measures laminar heterogeneity within parcel i.

    mode="to_mean":
        - Average cosine distance of each layer vector to the parcel's mean *direction*.
        - Returns: (d_intraMean, d_superficial, d_middle, d_deep), each shape (N,).

    mode="pairwise":
        - Mean pairwise cosine distance among all layers for each parcel.
        - Returns: (N,) array.
    """

    layers = _split_layers(G_all, N=N)          # list length L; each (N x k)
    L = len(layers)

    if zscore_within_layer:
        layers = [
            (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
            for X in layers
        ]

    # Unit-row embeddings per layer
    U_layers = [_l2_normalize_rows(X) for X in layers]

    mpl.rcParams['svg.fonttype'] = 'none'   # keep text as <text>, not paths
    mpl.rcParams['text.usetex'] = False     # avoid TeX

    if mode == "to_mean":
        # Mean raw vector across layers per parcel, then L2-normalize rowwise
        Ubar = _l2_normalize_rows(sum(layers) / float(L))   # (N x k)

        # Debug/QA plot
        plt.figure(figsize=(6, 6))
        plt.imshow(Ubar, cmap="viridis")
        plt.title("Ubar - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_intraArealDis.svg", bbox_inches="tight", format="svg")
        plt.close()

        # Per-layer cosine distance to Ubar
        d_layers = [1.0 - np.einsum("ij,ij->i", U, Ubar) for U in U_layers]  # list of (N,)
        D_layers = np.vstack(d_layers)  # (L, N)

        # Overall mean across layers
        d_intraMean = D_layers.mean(axis=0)  # (N,)

        # Also save an overview heatmap of per-layer distances (layers x parcels)
        plt.figure(figsize=(8, 6))
        plt.imshow(D_layers, aspect='auto', cmap="viridis")
        plt.title("Per-layer distances to mean direction (layers × parcels)")
        plt.ylabel("Layer")
        plt.xlabel("Parcel")
        plt.savefig(f"{outputDir}/Matrix_intraArealDis_layers.svg", bbox_inches="tight", format="svg")
        plt.close()

        # Save overall column image (backward-compatible filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(d_intraMean[:, np.newaxis], cmap="viridis")
        plt.title("D_intraMean - intra areal dis")
        plt.savefig(f"{outputDir}/Matrix_mean_intraArealDis.svg", bbox_inches="tight", format="svg")
        plt.close()

        # Pick equidistant layers for (superficial, middle, deep)
        i_sup, i_mid, i_deep = _equidistant_layer_indices(L)
        d_superficial = D_layers[i_sup]
        d_middle      = D_layers[i_mid]
        d_deep        = D_layers[i_deep]

        # Optional: save the three selected layers with familiar names
        for name, vec in [("sup", d_superficial), ("mid", d_middle), ("deep", d_deep)]:
            plt.figure(figsize=(10, 10))
            plt.imshow(vec[:, np.newaxis], cmap="viridis")
            plt.title(f"D_{name} - intra areal dis")
            plt.savefig(f"{outputDir}/Matrix_{name}_intraArealDis.svg", bbox_inches="tight", format="svg")
            plt.close()

        return d_intraMean, d_superficial, d_middle, d_deep

    elif mode == "pairwise":
        if L < 2:
            raise ValueError("pairwise mode requires at least 2 layers.")
        # Stack to (L, N, k)
        Ustack = np.stack(U_layers, axis=0)

        # Layer-layer similarities per parcel: S[a,b,i] = dot(U[a,i,:], U[b,i,:])
        S = np.einsum('aik,bik->abi', Ustack, Ustack)  # (L, L, N)

        # Mean over unique off-diagonal pairs a<b
        iu = np.triu_indices(L, k=1)
        mean_S_pairs = S[iu[0], iu[1], :].mean(axis=0)  # (N,)
        d_pairwise = 1.0 - mean_S_pairs                 # (N,)

        return d_pairwise

    else:
        raise ValueError("mode must be 'to_mean' or 'pairwise'")


def plotFlatMap(
    M,
    outdir,
    outname,
    inputdir_1="",
    inputdir_2="/home/degutis/repos/HumanCorticalParcellations",
    cmap="viridis",
    HCP=False,
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
    assert vals.size % 2 == 0, f"Expected even number of parcel values (LH+RH), got {vals.shape}"
    
    if HCP:
        inputdir_1 = "/home/degutis/repos/HCP_WB_parcels"
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

    else:
        inputdir_1 = "/home/degutis/repos/SchaeferAtlas"

        L_lab = nib.load(os.path.join(inputdir_1, "Schaefer400.L.label.gii")).agg_data().astype(int).squeeze()
        R_lab = nib.load(os.path.join(inputdir_1, "Schaefer400.R.label.gii")).agg_data().astype(int).squeeze()

        n_total = vals.size
        n_hemi  = n_total // 2
        assert n_total % 2 == 0, "M should be [LH parcels..., RH parcels...]"

        uL = np.unique(L_lab[L_lab > 0])
        uR = np.unique(R_lab[R_lab > 0])
        assert len(uL) == n_hemi and len(uR) == n_hemi, \
            f"Expected {n_hemi} parcels per hemi; got {len(uL)} (L), {len(uR)} (R)."

        L_rank = {k: i for i, k in enumerate(sorted(uL))}
        R_rank = {k: i for i, k in enumerate(sorted(uR))}

        metric_L = np.full(L_lab.shape, np.nan, float)
        metric_R = np.full(R_lab.shape, np.nan, float)
        mL = L_lab > 0
        mR = R_lab > 0
        metric_L[mL] = vals[[L_rank[k] for k in L_lab[mL]]]
        metric_R[mR] = vals[n_hemi + np.array([R_rank[k] for k in R_lab[mR]])]


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
