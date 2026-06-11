#!/usr/bin/env python3
"""
Split-half robustness analysis for laminar gradients.

- Uses the same connectivity construction as your main pipeline.
- Keeps the original number of gradients (n_components).
- On each iteration:
    * randomly splits subjects into two halves
    * builds group FC for each half
    * runs gradient analysis
    * computes D_inter, D_intra, D_Deep, D_Mid, D_Sup
    * computes |r| between halves for each index
- Produces:
    * split_half_corr_D_*.npy   (distributions of correlations)
    * split_half_summary_nIter*.csv  (mean/variance per index)
    * a single figure with 4 violin plots (one per index)

"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import laminar_rs as lrs


def vector_corr(x, y):
    """
    Pearson correlation between two parcel vectors.

    Computed identically to the empirical r inside
    schaefer_stats.p_spin_corr_schaefer400, which the main analysis uses:
    inputs are coerced to float and squeezed to 1D, required to have the
    same shape, required to be finite (no NaNs/infs), and r is taken from
    np.corrcoef. No NaN masking is performed, so a non-finite parcel
    raises here exactly as it would in the main script rather than being
    silently dropped (which would change the parcel count entering r).
    """
    x = np.asarray(x, float).squeeze()
    y = np.asarray(y, float).squeeze()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D vectors.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite (no NaNs or infs).")

    return float(np.corrcoef(x, y)[0, 1])


def run_split_half_robustness(
    r_matrices_4d,
    N,
    output_dir,
    analysis,
    n_components=15,
    n_iter=500,
    kernel="cosine",
    random_state=13011991,
):
    """
    Split-half robustness:
    - r_matrices_4d: per-subject FCs in r-space, shape (N, N, num_layers, subs)
    - On each iteration, randomly split subjects into two halves,
      compute group FC for each half, run gradient analysis, compute
      D_inter, D_intra, D_Deep, D_Mid, D_Sup, and correlate between halves.
    - Collect |r| across iterations and plot violin plots.
    """
    rng = np.random.default_rng(random_state)

    base_folder = os.path.join(output_dir, analysis, "robustness_split_half")
    os.makedirs(base_folder, exist_ok=True)

    _, _, num_layers, subs = r_matrices_4d.shape
    print(f"[SPLIT-HALF] subs={subs}, layers={num_layers}, n_iter={n_iter}")

    corr_inter = np.zeros(n_iter)
    corr_intra = np.zeros(n_iter)
    corr_Deep  = np.zeros(n_iter)
    corr_Mid   = np.zeros(n_iter)
    corr_Sup   = np.zeros(n_iter)

    for it in range(n_iter):
        # random split of subjects
        idx = np.arange(subs)
        rng.shuffle(idx)
        split = subs // 2
        g1 = idx[:split]
        g2 = idx[split:]

        # mean FC for each half (in r space)
        mean_r_g1 = np.nanmean(r_matrices_4d[:, :, :, g1], axis=3)
        mean_r_g2 = np.nanmean(r_matrices_4d[:, :, :, g2], axis=3)

        # multiplex adjacency for each half
        M1 = lrs.connectivity.build_multiplex_adjacency(mean_r_g1)
        M2 = lrs.connectivity.build_multiplex_adjacency(mean_r_g2)

        # gradients
        G1, eig1 = lrs.gradients.run_gradient_analysis(
            M1, n_components=n_components, kernel=kernel, random_state=random_state
        )
        G2, eig2 = lrs.gradients.run_gradient_analysis(
            M2, n_components=n_components, kernel=kernel, random_state=random_state
        )

        # dissimilarity metrics
        D_inter_1, D_Deep_1, D_Mid_1, D_Sup_1, _ = lrs.gradients.inter_areal_dissimilarity(
            G1, base_folder, N=N, zscore_within_layer=True
        )
        D_inter_2, D_Deep_2, D_Mid_2, D_Sup_2, _ = lrs.gradients.inter_areal_dissimilarity(
            G2, base_folder, N=N, zscore_within_layer=True
        )

        # D_intra_1, D_Deep_1, D_Mid_1, D_Sup_1 = lrs.gradients.intra_areal_dissimilarity(
        #     G1, base_folder, N=N, zscore_within_layer=True
        # )
        # D_intra_2, D_Deep_2, D_Mid_2, D_Sup_2 = lrs.gradients.intra_areal_dissimilarity(
        #     G2, base_folder, N=N, zscore_within_layer=True
        # )

        # correlation between halves, use |r| to get 0–1 reliability
        corr_inter[it] = np.abs(vector_corr(D_inter_1, D_inter_2))
        # corr_intra[it] = np.abs(vector_corr(D_intra_1, D_intra_2))
        corr_Deep[it]  = np.abs(vector_corr(D_Deep_1,  D_Deep_2))
        corr_Mid[it]   = np.abs(vector_corr(D_Mid_1,   D_Mid_2))
        corr_Sup[it]   = np.abs(vector_corr(D_Sup_1,   D_Sup_2))

        if (it + 1) % 50 == 0:
            print(f"[SPLIT-HALF] iteration {it+1}/{n_iter}")

    # Save correlations
    np.save(os.path.join(base_folder, "split_half_corr_D_inter.npy"), corr_inter)
    # np.save(os.path.join(base_folder, "split_half_corr_D_intra.npy"), corr_intra)
    np.save(os.path.join(base_folder, "split_half_corr_D_Deep.npy"),  corr_Deep)
    np.save(os.path.join(base_folder, "split_half_corr_D_Mid.npy"),   corr_Mid)
    np.save(os.path.join(base_folder, "split_half_corr_D_Sup.npy"),   corr_Sup)

    # ---- index distributions used for the summary CSV and raincloud plots ----
    # metrics = [corr_inter, corr_intra, corr_Deep, corr_Mid, corr_Sup]
    # names   = ["D_inter", "D_intra", "D_Deep", "D_Mid", "D_Sup"]
    metrics = [corr_inter, corr_Deep, corr_Mid, corr_Sup]
    names   = ["D_inter", "D_Deep", "D_Mid", "D_Sup"]

    # ---- Summary CSV: mean / variance of each index's |r| distribution ----
    # Statistics are computed over finite iterations only, matching the
    # values shown in the raincloud plot. variance/std use ddof=0
    # (population, i.e. the variance of the observed distribution).
    summary_rows = []
    for vals, name in zip(metrics, names):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            summary_rows.append({
                "index": name,
                "n": 0,
                "mean": np.nan,
                "variance": np.nan,
                "std": np.nan,
                "median": np.nan,
            })
            continue
        summary_rows.append({
            "index": name,
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "variance": float(np.var(v)),
            "std": float(np.std(v)),
            "median": float(np.median(v)),
        })

    summary_csv_path = os.path.join(
        base_folder, f"split_half_summary_nIter{n_iter}.csv"
    )
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "n", "mean", "variance", "std", "median"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SPLIT-HALF] Saved summary statistics to {summary_csv_path}")

    # ---- Raincloud plots ----
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    for ax, vals, name in zip(axes, metrics, names):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            ax.set_title(f"{name}\n(no data)", fontsize=8)
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            continue

        # KDE for the "cloud"
        y_grid = np.linspace(0.0, 1.0, 200)
        kde = gaussian_kde(vals)
        density = kde(y_grid)

        # scale density horizontally to a reasonable width
        if np.max(density) > 0:
            density = density / np.max(density) * 0.4  # width in x

        # half-violin / cloud
        ax.fill_betweenx(y_grid, 0, density, alpha=0.6)

        # "rain" – jittered points
        x_jitter = np.random.uniform(-0.05, 0.05, size=vals.size)
        ax.scatter(
            x_jitter,
            vals,
            s=8,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.3,
            zorder=3,
        )

        # median line across the cloud
        med = np.median(vals)
        ax.plot([0, np.max(density)], [med, med], linewidth=1.5, zorder=4)

        ax.set_title(name, fontsize=9)
        ax.set_xlim(-0.2, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Split-half |r|")
    fig.suptitle(
        f"Split-half robustness (n_iter={n_iter}, n_gradients={n_components})",
        y=0.95
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = os.path.join(base_folder, f"split_half_raincloud_nIter{n_iter}.svg")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SPLIT-HALF] Saved split-half correlations and raincloud plot to {base_folder}")
# =============================
# Main script
# =============================

if __name__ == "__main__":

    # ---- basic config (adapt as needed) ----
    N = 400
    num_layers = 3
    largeGap = False

    BASE = Path('/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations')
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]

    gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
    root = BASE / gap_dir

    data_dirs = [root / f'sub-LAM{s:03d}' for s in SUBJECTS]
    output_dir = root

    subs = len(data_dirs)
    os.makedirs(output_dir, exist_ok=True)

    analysis = "WithinLayer_gradients_kernelCOS_API_interSpecific"

    kernel = None
    n_components = 15       # original number of gradients
    N_ITER = 500            # suggested number of split-half iterations

    subj_fc_path = os.path.join(output_dir, analysis, 'FC_subject_matrices_r.npy')

    # ---- Load or compute subject-level FC matrices (r-space) ----
    if os.path.isfile(subj_fc_path):
        r_matrices_4d = np.load(subj_fc_path)
        print(f"[INFO] loaded subject-level FCs: {subj_fc_path}  shape={r_matrices_4d.shape}")
    else:
        print("[INFO] FC_subject_matrices_r.npy not found, computing from subjects.")
        os.makedirs(os.path.join(output_dir, analysis), exist_ok=True)

        adj_matrices_appended = []
        atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii"

        for iSub, data_dir in enumerate(data_dirs):
            print(f"[INFO] Subject {iSub+1}/{subs}: {data_dir}")

            cfg = lrs.config.LaminarConfig(
                data_dir=data_dir,
                N=N,
                num_layers=3,
            )

            # adj_full is ignored; we only want per-layer Fisher z
            _, corr_layer_z = lrs.connectivity.within_layer_block_matrix(
                cfg, subtract_average=False
            )

            adjMatrix = lrs.connectivity.thresh_and_binarize(
                corr_layer_z,
            )
            adj_matrices_appended.append(adjMatrix)

        # adj_matrices_4d: z-values, shape (N, N, L, subs)
        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        _, _, L, subs = adj_matrices_4d.shape

        # Convert each subject's z-matrix to r
        r_matrices_4d = np.empty_like(adj_matrices_4d)
        for si in range(subs):
            r_matrices_4d[:, :, :, si] = lrs.connectivity.fisher_z_to_r(adj_matrices_4d[:, :, :, si])

        np.save(subj_fc_path, r_matrices_4d)
        print(f"[INFO] saved subject-level FCs to: {subj_fc_path}")

    # ---- Run split-half robustness analysis ----
    run_split_half_robustness(
        r_matrices_4d=r_matrices_4d,
        N=N,
        output_dir=output_dir,
        analysis=analysis,
        n_components=n_components,
        n_iter=N_ITER,
        kernel=kernel,
        random_state=13011991,
    )