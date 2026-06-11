#!/usr/bin/env python3
"""
Robustness analysis for laminar gradients:
- Vary number of gradients (n_components) from 5 to 25
- Recompute D_inter, D_Deep, D_Mid, D_Sup for each
- Compute correlation matrices across gradient counts
- Correlate D_inter/D_Deep/D_Mid/D_Sup with Margulies G1 and G2
  to check how stable the functional similarity is across #gradients
- Save:
    * correlation matrices as .npy
    * correlations with Margulies as .npz
    * one figure with subplots (lower-triangle style)
    * figures of correlation-vs-#gradients with G1/G2

"""

import os
import numpy as np
import matplotlib.pyplot as plt

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


def load_margulies_gradients(output_dir, analysis):
    """Load Margulies G1 and G2 in Schaefer-400 space. Returns (g1, g2) as 1D arrays."""
    from neuromaps.datasets import fetch_annotation
    from netneurotools import datasets as nntdata
    from neuromaps.parcellate import Parcellater
    from neuromaps.images import dlabel_to_gifti

    out = Path(output_dir) / analysis / "MarguliesFunc"
    out.mkdir(parents=True, exist_ok=True)

    g1 = fetch_annotation(source='margulies2016', desc='fcgradient01',
                          space='fsLR', den='32k')
    g2 = fetch_annotation(source='margulies2016', desc='fcgradient02',
                          space='fsLR', den='32k')
    schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
    parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')

    g1_s400 = np.ravel(parc.fit_transform(g1, 'fsLR'))
    g2_s400 = np.ravel(parc.fit_transform(g2, 'fsLR'))
    return g1_s400, g2_s400


def run_gradient_robustness(
    M,
    N,
    output_dir,
    analysis,
    grad_min=5,
    grad_max=15,
    kernel=None,
    random_state=13011991,
):
    """
    Sweep n_components from grad_min to grad_max, recompute
    D_inter, D_Deep, D_Mid, D_Sup, and compute correlation matrices.
    Saves:
    - C_D_inter.npy, C_D_Deep.npy, C_D_Mid.npy, C_D_Sup.npy
    - corr_with_margulies.npz
    - robustness_gradients_XX-YY
    - corr_margulies_gradients_XX-YY
    """
    grad_list = list(range(grad_min, grad_max + 1))
    nG = len(grad_list)

    base_folder = os.path.join(output_dir, analysis, "robustness_gradients")
    os.makedirs(base_folder, exist_ok=True)

    D_inter_dict = {}
    D_Deep_dict = {}
    D_Mid_dict = {}
    D_Sup_dict = {}

    for n_comp in grad_list:
        print(f"\n[ROBUSTNESS] Running gradients with n_components={n_comp}")
        G, eig = lrs.gradients.run_gradient_analysis(
            M, n_components=n_comp, kernel=kernel, random_state=random_state
        )

        subfolder = os.path.join(base_folder, f"nGrad_{n_comp:02d}")
        os.makedirs(subfolder, exist_ok=True)

        # Compute dissimilarity measures
        D_inter, D_Deep, D_Mid, D_Sup, _ = lrs.gradients.inter_areal_dissimilarity(
            G, subfolder, N=N, zscore_within_layer=True
        )

        D_inter_dict[n_comp] = D_inter
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
    C_D_Deep = build_corr_matrix(D_Deep_dict)
    C_D_Mid = build_corr_matrix(D_Mid_dict)
    C_D_Sup = build_corr_matrix(D_Sup_dict)

    # Save correlation matrices
    np.save(os.path.join(base_folder, "C_D_inter.npy"), C_D_inter)
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
        lower_triangle(C_D_Deep),
        lower_triangle(C_D_Mid),
        lower_triangle(C_D_Sup),
    ]
    titles = ["D_inter", "D_Deep", "D_Mid", "D_Sup"]

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

    # Give a bit more room around the axes so y-ticks aren't cut off
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
        base_folder, f"robustness_gradients_{grad_min}-{grad_max}.svg"
    )
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # --- NEW: correlation with Margulies G1 / G2 across #gradients ---
    g1, g2 = load_margulies_gradients(output_dir, analysis)

    indices_dict = {
        "D_inter": D_inter_dict,
        "D_Deep":  D_Deep_dict,
        "D_Mid":   D_Mid_dict,
        "D_Sup":   D_Sup_dict,
    }

    # One signed r value per gradient count. The dissimilarity indices
    # are row-means of cosine-distance matrices (positive scalars per
    # parcel), so unlike the raw gradient axes they are not sign-ambiguous.
    corr_g1 = {name: np.array([vector_corr(d[g], g1) for g in grad_list])
               for name, d in indices_dict.items()}
    corr_g2 = {name: np.array([vector_corr(d[g], g2) for g in grad_list])
               for name, d in indices_dict.items()}

    np.savez(
        os.path.join(base_folder, "corr_with_margulies.npz"),
        n_gradients=np.array(grad_list),
        **{f"g1_{k}": v for k, v in corr_g1.items()},
        **{f"g2_{k}": v for k, v in corr_g2.items()},
    )

    # palette consistent with the paper's Fig 1D
    colors = {
        "D_inter": "0.4",
        "D_Deep":  "#4fc3f7",
        "D_Mid":   "#8D6E63",
        "D_Sup":   "#2278b5",
    }
    markers = {"D_inter": "s", "D_Deep": "o", "D_Mid": "o", "D_Sup": "o"}

    # all four indices, G1 and G2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, corr_dict, title in zip(
        axes, [corr_g1, corr_g2],
        ["Correlation with G1 (sensory-association)",
         "Correlation with G2 (visual-somatomotor)"],
    ):
        for name, vals in corr_dict.items():
            ax.plot(grad_list, vals, marker=markers[name],
                    label=name, color=colors[name], linewidth=1.8)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("# gradients (n_components)")
        ax.set_xticks(grad_list)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        ax.set_ylim(-0.25, 1)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder, f"corr_margulies_gradients_{grad_min}-{grad_max}.svg"
    ), dpi=300)
    plt.close(fig)

    # superficial vs deep only (clearer comparison)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    pair = ({"D_Deep": corr_g1["D_Deep"], "D_Sup": corr_g1["D_Sup"]},
            {"D_Deep": corr_g2["D_Deep"], "D_Sup": corr_g2["D_Sup"]})

    for ax, corr_dict, title in zip(
        axes, pair,
        ["Correlation with G1 (sensory-association)",
         "Correlation with G2 (visual-somatomotor)"],
    ):
        for name, vals in corr_dict.items():
            ax.plot(grad_list, vals, marker="o",
                    label=name, color=colors[name], linewidth=2.2,
                    markersize=6)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("# gradients (n_components)")
        ax.set_xticks(grad_list)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        ax.set_ylim(-0.25, 1)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder,
        f"corr_margulies_gradients_supDeep_{grad_min}-{grad_max}.svg"
    ), dpi=300)
    plt.close(fig)

    print(f"[ROBUSTNESS] Saved correlation matrices, Margulies correlations, "
          f"and figures to {base_folder}")

# =============================
# Main script
# =============================

if __name__ == "__main__":

    # ---- basic config (adapt as needed) ----
    N = 400
    setThresh = 0          # % of weakest edges per row to drop
    num_layers = 3
    binarize_flag = False
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

    kernel = "cosine"
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

        for iSub, data_dir in enumerate(data_dirs):
            print(f"[INFO] Subject {iSub+1}/{subs}: {data_dir}")
            cfg = lrs.config.LaminarConfig(
                data_dir=data_dir,
                N=N,
                set_thresh=0,
                num_layers=3,
            )

            _, corr_layer_z = lrs.connectivity.within_layer_block_matrix(
                cfg, subtract_average=False
            )

            adjMatrix = lrs.connectivity.thresh_and_binarize(
                corr_layer_z,
                set_thresh=0,
            )
            adj_matrices_appended.append(adjMatrix)

        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)
        r_matrix = lrs.connectivity.fisher_z_to_r(mean_adj_matrix)

        M = lrs.connectivity.build_multiplex_adjacency(r_matrix)
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