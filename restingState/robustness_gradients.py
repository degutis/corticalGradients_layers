#!/usr/bin/env python3
"""
Robustness analysis for laminar gradients:
- Vary number of gradients (n_components) from 5 to 25
- Recompute D_inter, D_intra, D_Deep, D_Mid, D_Sup for each
- Compute correlation matrices across gradient counts
- Save:
    * correlation matrices as .npy
    * one figure with 5 subplots (lower-triangle style)
"""

import os
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import laminarRestingState as lrs
import laminarAnalyses as laman


# =============================
# Helper functions
# =============================

def subtractAverage(adjMatrix):
    """Subtract across-subject average (over 3rd dim) and zero negatives."""
    avg_matrix = np.nanmean(adjMatrix, axis=2)
    for i in range(adjMatrix.shape[2]):
        adjMatrix[:, :, i] -= avg_matrix
    adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)
    return adjMatrix


def thresh_and_binarize(adj, setThresh=0, binarize=False):
    """
    Threshold each row by magnitude and optionally binarize.
    If not binarized, apply Fisher z-transform to retained values.
    adj: (N, N, num_layers)
    """
    N, _, num_layers = adj.shape
    A = np.empty((N, N, num_layers), dtype=float)
    percentThresh = setThresh / 100.0

    for layer in range(num_layers):
        mag = np.abs(adj[:, :, layer])
        sorted_idx = np.argsort(mag, axis=1)  # (N, N)
        mask = np.ones_like(mag, dtype=bool)
        rows = np.arange(N)[:, None]
        k = int(np.floor(percentThresh * N))
        if k > 0:
            mask[rows, sorted_idx[:, :k]] = False

        if binarize:
            A[:, :, layer] = mask.astype(int)
        else:
            corr_masked = adj[:, :, layer] * mask
            eps = 1 - 1e-6
            corr_masked = np.clip(corr_masked, -eps, eps)
            z_transformed = np.arctanh(corr_masked)
            np.fill_diagonal(z_transformed, 0)
            A[:, :, layer] = z_transformed
    return A


def backToR(adj):
    """
    Convert z-values back to correlations, clipping to avoid tanh overflow.
    adj: (N, N, num_layers)
    """
    N, _, num_layers = adj.shape
    A = np.empty((N, N, num_layers), dtype=float)
    for layer in range(num_layers):
        r_matrix = np.clip(adj[:, :, layer], -5, 5)
        z_matrix = np.tanh(r_matrix)
        np.fill_diagonal(z_matrix, 0)
        A[:, :, layer] = z_matrix
    return A


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


def run_gradient_robustness(
    M,
    N,
    output_dir,
    analysis,
    grad_min=5,
    grad_max=25,
    kernel=None,
    random_state=13011991,
):
    """
    Sweep n_components from grad_min to grad_max, recompute
    D_inter, D_intra, D_Deep, D_Mid, D_Sup, and compute correlation matrices.
    Saves:
    - C_D_inter.npy, C_D_intra.npy, C_D_Deep.npy, C_D_Mid.npy, C_D_Sup.npy
    - robustness_gradients_XX-YY.png
    """
    grad_list = list(range(grad_min, grad_max + 1))
    nG = len(grad_list)

    base_folder = os.path.join(output_dir, analysis, "robustness_gradients")
    os.makedirs(base_folder, exist_ok=True)

    D_inter_dict = {}
    D_intra_dict = {}
    D_Deep_dict = {}
    D_Mid_dict = {}
    D_Sup_dict = {}

    for n_comp in grad_list:
        print(f"\n[ROBUSTNESS] Running gradients with n_components={n_comp}")
        G, eig = laman.run_gradient_analysis(
            M, n_components=n_comp, kernel=kernel, random_state=random_state
        )

        subfolder = os.path.join(base_folder, f"nGrad_{n_comp:02d}")
        os.makedirs(subfolder, exist_ok=True)

        # Compute dissimilarity measures
        D_inter, D_inter_deep, D_inter_mid, D_inter_sup = laman.inter_areal_dissimilarity(
            G, subfolder, N=N, zscore_within_layer=True
        )
        D_intra, D_Deep, D_Mid, D_Sup = laman.intra_areal_dissimilarity(
            G, subfolder, N=N, zscore_within_layer=True
        )

        D_inter_dict[n_comp] = D_inter
        D_intra_dict[n_comp] = D_intra
        D_Deep_dict[n_comp] = D_Deep
        D_Mid_dict[n_comp] = D_Mid
        D_Sup_dict[n_comp] = D_Sup

    # --- build correlation matrices ---
    def build_corr_matrix(D_dict):
        C = np.zeros((nG, nG), dtype=float)
        for i, gi in enumerate(grad_list):
            for j, gj in enumerate(grad_list):
                C[i, j] = vector_corr(D_dict[gi], D_dict[gj])
        return C

    C_D_inter = build_corr_matrix(D_inter_dict)
    C_D_intra = build_corr_matrix(D_intra_dict)
    C_D_Deep = build_corr_matrix(D_Deep_dict)
    C_D_Mid = build_corr_matrix(D_Mid_dict)
    C_D_Sup = build_corr_matrix(D_Sup_dict)

    # Save correlation matrices
    np.save(os.path.join(base_folder, "C_D_inter.npy"), C_D_inter)
    np.save(os.path.join(base_folder, "C_D_intra.npy"), C_D_intra)
    np.save(os.path.join(base_folder, "C_D_Deep.npy"), C_D_Deep)
    np.save(os.path.join(base_folder, "C_D_Mid.npy"), C_D_Mid)
    np.save(os.path.join(base_folder, "C_D_Sup.npy"), C_D_Sup)

    # --- plot lower-triangle-style correlation matrices ---
    def lower_triangle(mat):
        lt = mat.copy()
        iu = np.triu_indices_from(lt, k=1)
        lt[iu] = np.nan
        return lt

    mats   = [
        lower_triangle(C_D_inter),
        lower_triangle(C_D_intra),
        lower_triangle(C_D_Deep),
        lower_triangle(C_D_Mid),
        lower_triangle(C_D_Sup),
    ]
    titles = ["D_inter", "D_intra", "D_Deep", "D_Mid", "D_Sup"]

    # slightly smaller, more compact figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, mat, title in zip(axes, mats, titles):
        m = np.ma.masked_invalid(mat)
        im = ax.imshow(m, vmin=0, vmax=1, aspect='equal')
        ax.set_title(title)
        ax.set_xticks(range(nG))
        ax.set_yticks(range(nG))
        ax.set_xticklabels(grad_list, rotation=90)
        ax.set_yticklabels(grad_list)
        ax.tick_params(labelsize=8)

    # Hide last (unused) subplot
    axes[-1].axis("off")

    # Give a bit more room around the axes so y-ticks aren’t cut off
    fig.suptitle(f"Robustness across #gradients ({grad_min}-{grad_max})", y=0.96)
    fig.subplots_adjust(
        left=0.08,   # more space for y-labels
        right=0.88,  # leave room for colorbar
        bottom=0.08,
        top=0.9,
        wspace=0.3,
        hspace=0.3,
    )

    # Slim colorbar on the right
    cbar = fig.colorbar(
        im,
        ax=axes[:-1],      # all used axes
        location="right",
        fraction=0.025,    # thinner bar
        pad=0.02
    )
    cbar.set_label("Pearson r")
    cbar.ax.tick_params(labelsize=8)

    fig_path = os.path.join(
        base_folder, f"robustness_gradients_{grad_min}-{grad_max}.png"
    )
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"[ROBUSTNESS] Saved correlation matrices and figure to {base_folder}")

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
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22]

    gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
    root = BASE / gap_dir

    data_dirs = [root / f'sub-LAM{s:03d}' for s in SUBJECTS]
    output_dir = root

    subs = len(data_dirs)
    os.makedirs(output_dir, exist_ok=True)

    # must match the analysis folder name used when FC_matrix.npy was created
    analysis = "WithinLayer_gradients_kernelNone_21Subs_20Components_32k_withEigVecs"

    kernel = None
    grad_min = 5
    grad_max = 25

    # ---- Load or compute FC matrix M ----
    fc_path = os.path.join(output_dir, analysis, 'FC_matrix.npy')

    if os.path.isfile(fc_path):
        M = np.load(fc_path, allow_pickle=False)
        print(f"[INFO] loaded: {fc_path}  shape={M.shape}")
        print("  max:", np.max(M))
        print("  min:", np.min(M))
    else:
        print("[INFO] FC_matrix.npy not found, recomputing from subjects.")
        os.makedirs(os.path.join(output_dir, analysis), exist_ok=True)

        adj_matrices_appended = []
        atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii"

        for iSub, data_dir in enumerate(data_dirs):
            print(f"[INFO] Subject {iSub+1}/{subs}: {data_dir}")
            restStateSub = lrs.LaminarRestingState(
                data_dir,
                N,
                setThresh,
                num_layers=num_layers,
                atlas_dir=atlas_dir,
            )
            _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()

            if subtractAverage_true:
                adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
                adjMatrix = thresh_and_binarize(
                    adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag
                )
            else:
                adjMatrix = thresh_and_binarize(
                    adj_matrix_within_corr, setThresh=setThresh, binarize=binarize_flag
                )

            adj_matrices_appended.append(adjMatrix)

        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)
        r_matrix = backToR(mean_adj_matrix)

        M = defineAdj(r_matrix)
        print("[INFO] FC matrix constructed.")
        print("  max:", np.max(M))
        print("  min:", np.min(M))
        np.save(fc_path, M)
        print(f"[INFO] saved FC matrix to: {fc_path}")

    # ---- Run robustness analysis ----
    run_gradient_robustness(
        M=M,
        N=N,
        output_dir=output_dir,
        analysis=analysis,
        grad_min=grad_min,
        grad_max=grad_max,
        kernel=kernel,
        random_state=13011991,
    )