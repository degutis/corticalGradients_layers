import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from scipy import stats
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps


def runGradientAnalysis(conn_matrix):
    gm = GradientMaps(kernel="cosine", approach = "dm")
    gm.fit(conn_matrix)
    
    return gm.gradients_

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
