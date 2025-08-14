#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import f_oneway

# If your loader is available:
import laminarRestingState as lrs

# ---------------------
# Configuration
# ---------------------
N = 360
NUM_LAYERS = 3
thresholdRange = np.arange(0,1)  # drop bottom % per row inside LRS (your current use)
atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii"

data_dirs = [
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM001",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM002",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM003",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM004",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM005",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM006",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM009",
    "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap/sub-LAM011",
]

output_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGapGraph2/"
os.makedirs(output_dir, exist_ok=True)

RSN_ASSIGNMENTS_PATH = "cortex_parcel_network_assignments.txt"  # 360 ints in {1..12}

TICK_LABELS = [
    "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
    "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
    "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
]

MEASURES_TO_COMPUTE = [
    "strength", "clustering_w", "eigvec", "pagerank",
    "betweenness_w", "harmonic_closeness_w", "degree_bin", "kcore_bin"
]
MEASURES_WITH_WINNER_COUNTS = {"eigvec", "pagerank"}  # arg-max plots in addition to mean±SEM

# ---------------------
# Helpers
# ---------------------
def ensure_layer_blocks(A):
    """
    Accepts (360,360,3) or (1080,1080) with identity inter-layer blocks.
    Returns (360,360,3) of the three diagonal blocks.
    """
    if A.ndim == 3 and A.shape == (N, N, NUM_LAYERS):
        B = A.copy()
    elif A.ndim == 2 and A.shape == (N * NUM_LAYERS, N * NUM_LAYERS):
        B = np.stack([A[i*N:(i+1)*N, i*N:(i+1)*N] for i in range(NUM_LAYERS)], axis=2)
    else:
        raise ValueError(f"Unexpected adjacency shape {A.shape}; expected (N,N,3) or (3N,3N)")
    for l in range(NUM_LAYERS):
        # enforce symmetry and zero diagonal
        B[:, :, l] = (B[:, :, l] + B[:, :, l].T) / 2.0
        np.fill_diagonal(B[:, :, l], 0.0)
    return B

def build_graph_from_layer(W):
    """
    Non-negative, weighted, undirected graph with 'length' = 1/weight for path metrics.
    Negative correlations set to 0 to keep distances well-defined.
    """
    W = W.copy()
    W[W < 0] = 0.0
    np.fill_diagonal(W, 0.0)
    G = nx.from_numpy_array(W)  # undirected; 'weight' attribute present
    eps = 1e-12
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        d["length"] = (1.0 / max(w, eps)) if w > 0 else float("inf")
    return G

def layer_metrics(W):
    """
    Compute graph measures for one layer (360x360).
    Returns dict[str] -> (360,) arrays.
    """
    G = build_graph_from_layer(W)
    WATTR, LATTR = "weight", "length"

    strength   = np.array([s for _, s in G.degree(weight=WATTR)])
    clustering = np.array(list(nx.clustering(G, weight=WATTR).values()))
    eigvec     = np.array(list(nx.eigenvector_centrality(G, weight=WATTR, max_iter=1000).values()))
    pagerank   = np.array(list(nx.pagerank(G, weight=WATTR).values()))
    betw_w     = np.array(list(nx.betweenness_centrality(G, weight=LATTR).values()))
    harm_close = np.array(list(nx.harmonic_centrality(G, distance=LATTR).values()))
    Gbin       = nx.from_numpy_array((W > 0).astype(int))
    degree_bin = np.array([k for _, k in Gbin.degree()])
    kcore_bin  = np.array(list(nx.core_number(Gbin).values()))
    return {
        "strength": strength,
        "clustering_w": clustering,
        "eigvec": eigvec,
        "pagerank": pagerank,
        "betweenness_w": betw_w,
        "harmonic_closeness_w": harm_close,
        "degree_bin": degree_bin,
        "kcore_bin": kcore_bin,
    }

def compute_measures_per_subject(adj_within_layers):
    """
    Input: (360,360,3) or (1080,1080) adjacency (symmetric).
    Output: dict[measure] -> (360,3) matrix (columns = layers).
    """
    A = ensure_layer_blocks(adj_within_layers)
    per_layer = [layer_metrics(A[:, :, l]) for l in range(NUM_LAYERS)]
    keys = per_layer[0].keys()
    out = {k: np.column_stack([per_layer[0][k], per_layer[1][k], per_layer[2][k]]) for k in keys}
    return out
# --- add these small helpers once (e.g., near the other helpers) ---
def bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR correction.
    pvals: iterable of length m
    returns: (qvals, reject_mask) where qvals are adjusted p-values.
    """
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    # enforce monotonicity from right to left
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    q = np.clip(q, 0.0, 1.0)
    reject = q <= alpha
    return q, reject

def safe_anova(a, b, c):
    """One-way ANOVA p-value with NaN/edge-case safety."""
    import numpy as np
    from scipy.stats import f_oneway
    try:
        _, p = f_oneway(a, b, c)
        if not np.isfinite(p):
            return 1.0
        return float(p)
    except Exception:
        return 1.0
    
def plot_measure_by_rsn(
    measure_3d,
    outdir,
    measure_label,
    do_winner_counts=False,
    additional_name="",
    violin_alpha=0.35,
    line_alpha=0.4,
    jitter=0.06,
    show_group_errorbars=True
):
    """
    measure_3d: (360, 3, S) array for S subjects (per-parcel, per-layer, per-subject).
    Makes TWO figures (if do_winner_counts=True): counts & mean—now as violins + paired lines.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    cats = np.loadtxt(RSN_ASSIGNMENTS_PATH, dtype=int)   # 360 ints in {1..12}
    subs = measure_3d.shape[-1]
    rng = np.random.default_rng(0)  # reproducible jitter

    layer_positions = np.array([1, 2, 3], dtype=float)
    layer_labels = ["Deep", "Middle", "Superficial"]

    # --------------------------------------------------
    # (A) Winner counts (optional): argmax across layers
    # --------------------------------------------------
    if do_winner_counts:
        # build per-subject counts per RSN per layer (integers)
        counts = np.zeros((12, 3, subs), dtype=float)
        for s in range(subs):
            M = measure_3d[:, :, s]      # (360,3)
            one_hot = np.zeros_like(M, dtype=int)
            idx = np.argmax(M, axis=1)
            one_hot[np.arange(360), idx] = 1
            for k in range(1, 13):
                mask = (cats == k)
                counts[k-1, :, s] = one_hot[mask, :].sum(axis=0)

        # p (per RSN) and BH-FDR q across the 12 RSNs
        pvals = [safe_anova(counts[k, 0, :], counts[k, 1, :], counts[k, 2, :]) for k in range(12)]
        qvals, reject = bh_fdr(pvals, alpha=0.05)

        fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
        axes = axes.flatten()

        # global y-lims to keep scale comparable across RSNs (optional)
        global_min = np.floor(counts.min())
        global_max = np.ceil(counts.max() + 0.01 * max(1.0, counts.max()))

        for idx_rsn, ax in enumerate(axes):
            # subject-level layer vectors: three arrays (Deep/Mid/Sup) each of length S
            data_layers = [counts[idx_rsn, i, :] for i in range(3)]

            # violins
            v = ax.violinplot(
                dataset=data_layers,
                positions=layer_positions,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for body in v['bodies']:
                body.set_alpha(violin_alpha)

            # subject scatter + paired lines
            for s in range(subs):
                xs = layer_positions + rng.uniform(-jitter, jitter, size=3)
                ys = [data_layers[i][s] for i in range(3)]
                ax.plot(layer_positions, ys, linewidth=0.8, alpha=line_alpha, color="gray")
                ax.scatter(xs, ys, s=10, alpha=0.8, zorder=3)

            # group mean ± SEM (optional overlay)
            if show_group_errorbars:
                means = np.array([np.mean(d) for d in data_layers])
                sems  = np.array([np.std(d, ddof=1) / np.sqrt(subs) for d in data_layers])
                ax.errorbar(layer_positions, means, yerr=sems, fmt='o', capsize=3, linewidth=1.2, zorder=4)

            # annotations / cosmetics
            sig = "*" if reject[idx_rsn] else ""
            ax.text(0.5, 0.95, f"p = {pvals[idx_rsn]:.3f}, q = {qvals[idx_rsn]:.3f}{sig}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10)
            ax.set_title(TICK_LABELS[idx_rsn])
            ax.set_xticks(layer_positions); ax.set_xticklabels(layer_labels, rotation=0)
            ax.set_ylabel(f"Parcel count (arg-max {measure_label})")
            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(global_min, global_max)

        plt.tight_layout()
        fn = os.path.join(outdir, f"RSN_{measure_label}_CountsViolin_AcrossSubs{additional_name}.png")
        plt.savefig(fn, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # ------------------------------------------
    # (B) Mean per RSN per layer (12 ANOVAs)
    # ------------------------------------------
    # averages[k, l, s] = mean value in RSN k, layer l, subject s
    averages = np.zeros((12, 3, subs), dtype=float)
    for s in range(subs):
        M = measure_3d[:, :, s]
        for k in range(1, 13):
            mask = (cats == k)
            averages[k-1, :, s] = M[mask, :].mean(axis=0)

    pvals = [safe_anova(averages[k, 0, :], averages[k, 1, :], averages[k, 2, :]) for k in range(12)]
    qvals, reject = bh_fdr(pvals, alpha=0.05)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16), sharey=True)
    axes = axes.flatten()

    # global y-lims to match scales across panels (optional but helpful)
    global_min = np.min(averages)
    global_max = np.max(averages)
    pad = 0.03 * (global_max - global_min + 1e-12)
    ylo, yhi = global_min - pad, global_max + pad

    for idx_rsn, ax in enumerate(axes):
        # per-layer data for this RSN across subjects
        data_layers = [averages[idx_rsn, i, :] for i in range(3)]

        # violins
        v = ax.violinplot(
            dataset=data_layers,
            positions=layer_positions,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for body in v['bodies']:
            body.set_alpha(violin_alpha)

        # subject scatter + paired lines
        for s in range(subs):
            xs = layer_positions + rng.uniform(-jitter, jitter, size=3)
            ys = [data_layers[i][s] for i in range(3)]
            ax.plot(layer_positions, ys, linewidth=0.8, alpha=line_alpha, color="gray")
            ax.scatter(xs, ys, s=10, alpha=0.8, zorder=3)

        # group mean ± SEM
        if show_group_errorbars:
            means = np.array([np.mean(d) for d in data_layers])
            sems  = np.array([np.std(d, ddof=1) / np.sqrt(subs) for d in data_layers])
            ax.errorbar(layer_positions, means, yerr=sems, fmt='o', capsize=3, linewidth=1.2, zorder=4)

        # annotations / cosmetics
        sig = "*" if reject[idx_rsn] else ""
        ax.text(0.5, 0.95, f"p = {pvals[idx_rsn]:.3f}, q = {qvals[idx_rsn]:.3f}{sig}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.set_title(TICK_LABELS[idx_rsn])
        ax.set_xticks(layer_positions); ax.set_xticklabels(layer_labels, rotation=0)
        ax.set_ylabel(f"{measure_label} (subject means within RSN)")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(ylo, yhi)

    plt.tight_layout()
    fn = os.path.join(outdir, f"RSN_{measure_label}_MeanViolin_AcrossSubs{additional_name}.png")
    plt.savefig(fn, bbox_inches="tight", dpi=150)
    plt.close(fig)

    
def winner_counts_by_rsn(measure_3d, cats, margin=0.0, zscore_within_parcel=True):
    """
    Returns counts_perc: (12, 3, S) = percentage winners per RSN per layer per subject.
    A "winner" for a parcel is counted only if max - second_best >= margin.
    If zscore_within_parcel=True, z-score each parcel's 3-layer values before argmax.
    """
    import numpy as np

    S = measure_3d.shape[-1]
    counts = np.zeros((12, 3, S), dtype=float)
    # RSN sizes (denominator for percentages)
    rsn_sizes = np.array([(cats == k).sum() for k in range(1, 13)], dtype=float)

    for s in range(S):
        M = measure_3d[:, :, s].copy()  # (360,3)
        if zscore_within_parcel:
            # z-score across the 3 layers per parcel
            mu = M.mean(axis=1, keepdims=True)
            sd = M.std(axis=1, keepdims=True)
            sd[sd == 0] = 1.0
            M = (M - mu) / sd

        # argmax and margin filter
        idx = np.argmax(M, axis=1)                     # (360,)
        sorted_vals = np.sort(M, axis=1)               # ascending
        maxv = sorted_vals[:, -1]
        second = sorted_vals[:, -2]
        ok = (maxv - second) >= margin                 # boolean (360,)
        one_hot = np.zeros_like(M, dtype=int)
        rows = np.arange(M.shape[0])
        one_hot[rows[ok], idx[ok]] = 1                 # no-winner rows remain all zeros

        # accumulate per RSN
        for k in range(1, 13):
            mask = (cats == k)
            win_counts = one_hot[mask, :].sum(axis=0)  # (3,)
            # convert to percentage of parcels in RSN k
            counts[k-1, :, s] = (win_counts / rsn_sizes[k-1]) * 100.0

    return counts  # (12,3,S) percentages

def load_group_npz(npz_path, expected_measures):
    """
    Load the stacked (360, 3, S) arrays for each measure from a saved NPZ.
    Returns dict[measure] -> np.ndarray
    """
    import numpy as np
    with np.load(npz_path) as data:
        stacked = {}
        for m in expected_measures:
            if m not in data:
                raise KeyError(f"'{m}' not found in {npz_path}")
            stacked[m] = data[m]
        return stacked

# ---------------------
# Main: compute + plot per threshold
# ---------------------
def main():
    for thresh in thresholdRange:
        print(f"[INFO] threshold={thresh}")

        # output folder and aggregated file for this threshold
        outdir = os.path.join(output_dir, f"plots_thresh{thresh}")
        os.makedirs(outdir, exist_ok=True)
        npz_path = os.path.join(outdir, f"group_metrics_t{thresh}.npz")

        # -------------------------
        # If aggregated file exists: skip compute, go to plotting
        # -------------------------
        if os.path.exists(npz_path):
            print(f"[INFO] Found existing {npz_path} — skipping computation and replotting.")
            stacked = load_group_npz(npz_path, MEASURES_TO_COMPUTE)

        else:
            # -------------------------
            # Compute per-subject measures and aggregate
            # -------------------------
            buckets = {m: [] for m in MEASURES_TO_COMPUTE}

            for data_dir in data_dirs:
                restStateSub = lrs.LaminarRestingState(
                    data_dir, N, thresh, atlas_dir=atlas_dir
                )
                # Get within-layer correlation matrix: (360,360,3) or (1080,1080)
                _, adj_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()

                subj_meas = compute_measures_per_subject(adj_within_corr)  # dict -> (360,3)
                for m in MEASURES_TO_COMPUTE:
                    buckets[m].append(subj_meas[m])  # list of (360,3)

            # stack into (360,3,S) and save
            stacked = {m: np.stack(buckets[m], axis=-1) for m in MEASURES_TO_COMPUTE}
            np.savez_compressed(npz_path, **stacked)
            print(f"[INFO] Saved aggregated data to: {npz_path}")

        # -------------------------
        # Plotting (always runs)
        # -------------------------
        add_name = f"_t{thresh}"
        for m, arr in stacked.items():
            plot_measure_by_rsn(
                measure_3d=arr,
                outdir=outdir,
                measure_label=m,
                do_winner_counts=(m in MEASURES_WITH_WINNER_COUNTS),
                additional_name=add_name
            )

        print(f"[INFO] Finished plotting for threshold {thresh} → {outdir}")


if __name__ == "__main__":
    main()