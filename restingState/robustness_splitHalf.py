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
    * split_half_corr_D_*.npy  (distributions of correlations)
    * a single figure with 5 violin plots (one per index)
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import laminar_rs as lrs

# =============================
# Helper functions
# =============================


def defineAdj(adjMatrix, interlayer_weight=1.0):
    """
    Construct a multiplex adjacency matrix M from layer-wise adjMatrix.
    adjMatrix: (N, N, L)
    M: (L*N, L*N), with interlayer_weight on off-diagonal layer couplings.
    """
    A = np.asarray(adjMatrix)
    if A.ndim != 3 or A.shape[0] != A.shape[1]:
        raise ValueError("adjMatrix must have shape (N, N, L)")

    N, _, L = A.shape
    dtype = np.result_type(A.dtype, float)

    M = np.zeros((L * N, L * N), dtype=dtype)
    for l in range(L):
        M[l * N:(l + 1) * N, l * N:(l + 1) * N] = A[:, :, l]

    layer_coupling = np.ones((L, L), dtype=dtype) - np.eye(L, dtype=dtype)
    M += interlayer_weight * np.kron(layer_coupling, np.eye(N, dtype=dtype))

    np.fill_diagonal(M, 0)
    return M


def vector_corr(x, y):
    """Correlation between two vectors (any shape), ignoring NaNs."""
    x = np.ravel(x)
    y = np.ravel(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


def run_split_half_robustness(
    r_matrices_4d,
    N,
    output_dir,
    analysis,
    n_components=20,
    n_iter=500,
    kernel=None,
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
        M1 = defineAdj(mean_r_g1)
        M2 = defineAdj(mean_r_g2)

        # gradients
        G1, eig1 = lrs.gradients.run_gradient_analysis(
            M1, n_components=n_components, kernel=kernel, random_state=random_state
        )
        G2, eig2 = lrs.gradients.run_gradient_analysis(
            M2, n_components=n_components, kernel=kernel, random_state=random_state
        )

        # dissimilarity metrics
        D_inter_1, _, _, _ = lrs.gradients.inter_areal_dissimilarity(
            G1, base_folder, N=N, zscore_within_layer=True
        )
        D_inter_2, _, _, _ = lrs.gradients.inter_areal_dissimilarity(
            G2, base_folder, N=N, zscore_within_layer=True
        )

        D_intra_1, D_Deep_1, D_Mid_1, D_Sup_1 = lrs.gradients.intra_areal_dissimilarity(
            G1, base_folder, N=N, zscore_within_layer=True
        )
        D_intra_2, D_Deep_2, D_Mid_2, D_Sup_2 = lrs.gradients.intra_areal_dissimilarity(
            G2, base_folder, N=N, zscore_within_layer=True
        )

        # correlation between halves, use |r| to get 0–1 reliability
        corr_inter[it] = np.abs(vector_corr(D_inter_1, D_inter_2))
        corr_intra[it] = np.abs(vector_corr(D_intra_1, D_intra_2))
        corr_Deep[it]  = np.abs(vector_corr(D_Deep_1,  D_Deep_2))
        corr_Mid[it]   = np.abs(vector_corr(D_Mid_1,   D_Mid_2))
        corr_Sup[it]   = np.abs(vector_corr(D_Sup_1,   D_Sup_2))

        if (it + 1) % 50 == 0:
            print(f"[SPLIT-HALF] iteration {it+1}/{n_iter}")

    # Save correlations
    np.save(os.path.join(base_folder, "split_half_corr_D_inter.npy"), corr_inter)
    np.save(os.path.join(base_folder, "split_half_corr_D_intra.npy"), corr_intra)
    np.save(os.path.join(base_folder, "split_half_corr_D_Deep.npy"),  corr_Deep)
    np.save(os.path.join(base_folder, "split_half_corr_D_Mid.npy"),   corr_Mid)
    np.save(os.path.join(base_folder, "split_half_corr_D_Sup.npy"),   corr_Sup)

    # ---- Violin plots ----
    metrics = [corr_inter, corr_intra, corr_Deep, corr_Mid, corr_Sup]
    names   = ["D_inter", "D_intra", "D_Deep", "D_Mid", "D_Sup"]

    fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharey=True)
    for ax, vals, name in zip(axes, metrics, names):
        parts = ax.violinplot(
            vals,
            positions=[0],
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts['bodies']:
            pc.set_alpha(0.7)

        # median as a dot
        ax.scatter(0, np.median(vals), color='k', s=12, zorder=3)
        ax.set_xticks([0])
        ax.set_xticklabels([name])
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Split-half |r|")
    fig.suptitle(
        f"Split-half robustness (n_iter={n_iter}, n_gradients={n_components})",
        y=0.95
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = os.path.join(base_folder, f"split_half_violin_nIter{n_iter}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[SPLIT-HALF] Saved split-half correlations and violin plot to {base_folder}")


# =============================
# Main script
# =============================

if __name__ == "__main__":

    # ---- basic config (adapt as needed) ----
    N = 400
    setThresh = 0          # % of weakest edges per row to drop
    num_layers = 3
    binarize_flag = False
    subtractAverage_true = False
    largeGap = False

    BASE = Path('/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations')
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]

    gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
    root = BASE / gap_dir

    data_dirs = [root / f'sub-LAM{s:03d}' for s in SUBJECTS]
    output_dir = root

    subs = len(data_dirs)
    os.makedirs(output_dir, exist_ok=True)

    # must match the analysis folder name you use elsewhere
    analysis = "WithinLayer_gradients_kernelNone_21Subs_20Components_API"

    kernel = None
    n_components = 20       # original number of gradients
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
                set_thresh=0,
                num_layers=3,
            )

            # adj_full is ignored; we only want per-layer Fisher z
            _, corr_layer_z = lrs.connectivity.within_layer_block_matrix(cfg, subtract_average=False)

            adjMatrix = lrs.connectivity.thresh_and_binarize(
                corr_layer_z,
                set_thresh=0,
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