#!/usr/bin/env python3
"""
Robustness analysis for laminar gradients across sparsity:
- Vary `sparsity` passed to run_gradient_analysis
- Recompute D_inter, D_intra, D_Deep, D_Mid, D_Sup for each
- Compute correlation matrices across sparsity values
- Save:
    * correlation matrices as .npy
    * one figure with subplots (lower-triangle style)

"""

import os
from pathlib import Path

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


def run_sparsity_robustness(
    M,
    N,
    output_dir,
    analysis,
    sparsity_values=None,
    kernel=None,
    random_state=13011991,
):
    if sparsity_values is None:
        sparsity_values = [round(x, 2) for x in np.arange(0.0, 0.95, 0.1)]
    sparsity_values = list(sparsity_values)
    nS = len(sparsity_values)

    base_folder = os.path.join(output_dir, analysis, "robustness_sparsity")
    os.makedirs(base_folder, exist_ok=True)

    D_inter_dict, D_Deep_dict, D_Mid_dict, D_Sup_dict = {}, {}, {}, {}

    for s in sparsity_values:
        G, eig, all_l, frac, cum, n_keep = lrs.run_gradient_analysis_auto(
            M, outputDir=output_dir, max_components=50,
            kernel="cosine", var_threshold=0.85, sparsity=s,
        )
        subfolder = os.path.join(base_folder, f"sparsity_{s:.2f}")
        os.makedirs(subfolder, exist_ok=True)

        D_inter, D_Deep, D_Mid, D_Sup, _ = lrs.gradients.inter_areal_dissimilarity(
            G, subfolder, N=N, zscore_within_layer=True
        )
        D_inter_dict[s] = D_inter
        D_Deep_dict[s] = D_Deep
        D_Mid_dict[s] = D_Mid
        D_Sup_dict[s] = D_Sup

    # ---- existing pairwise sparsity correlation matrices ----
    def build_corr_matrix(D_dict):
        C = np.zeros((nS, nS), dtype=float)
        for i, si in enumerate(sparsity_values):
            for j, sj in enumerate(sparsity_values):
                C[i, j] = vector_corr(D_dict[si], D_dict[sj])
        return C

    C_D_inter = build_corr_matrix(D_inter_dict)
    C_D_Deep  = build_corr_matrix(D_Deep_dict)
    C_D_Mid   = build_corr_matrix(D_Mid_dict)
    C_D_Sup   = build_corr_matrix(D_Sup_dict)

    np.save(os.path.join(base_folder, "C_D_inter.npy"), C_D_inter)
    np.save(os.path.join(base_folder, "C_D_Deep.npy"),  C_D_Deep)
    np.save(os.path.join(base_folder, "C_D_Mid.npy"),   C_D_Mid)
    np.save(os.path.join(base_folder, "C_D_Sup.npy"),   C_D_Sup)

    def lower_triangle(mat):
        lt = mat.copy()
        lt[np.triu_indices_from(lt, k=1)] = np.nan
        return lt

    mats   = [lower_triangle(C_D_inter), lower_triangle(C_D_Deep),
              lower_triangle(C_D_Mid),   lower_triangle(C_D_Sup)]
    titles = ["D_inter", "D_Deep", "D_Mid", "D_Sup"]
    tick_labels = [f"{s:.2f}" for s in sparsity_values]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, mat, title in zip(axes, mats, titles):
        m = np.ma.masked_invalid(mat)
        im = ax.imshow(m, vmin=0, vmax=1, aspect='equal')
        ax.set_title(title)
        ax.set_xticks(range(nS)); ax.set_yticks(range(nS))
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(labelsize=8)
    for ax in axes[len(mats):]:
        ax.axis("off")

    s_min, s_max = min(sparsity_values), max(sparsity_values)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.9,
                        wspace=0.3, hspace=0.3)
    cbar = fig.colorbar(im, ax=axes[:len(mats)], location="right",
                        fraction=0.025, pad=0.02)
    cbar.set_label("Pearson r"); cbar.ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(
        base_folder, f"robustness_sparsity_{s_min:.2f}-{s_max:.2f}.svg"
    ), dpi=300)
    plt.close(fig)

    # ---- NEW: correlation with Margulies G1 / G2 across sparsity ----
    g1, g2 = load_margulies_gradients(output_dir, analysis)

    indices_dict = {
        "D_inter": D_inter_dict,
        "D_Deep":  D_Deep_dict,
        "D_Mid":   D_Mid_dict,
        "D_Sup":   D_Sup_dict,
    }

    corr_g1 = {name: np.array([vector_corr(d[s], g1) for s in sparsity_values])
               for name, d in indices_dict.items()}
    corr_g2 = {name: np.array([vector_corr(d[s], g2) for s in sparsity_values])
               for name, d in indices_dict.items()}

    np.savez(
        os.path.join(base_folder, "corr_with_margulies.npz"),
        sparsity=np.array(sparsity_values),
        **{f"g1_{k}": v for k, v in corr_g1.items()},
        **{f"g2_{k}": v for k, v in corr_g2.items()},
    )

    # palette consistent with the paper's Fig 1D
    colors = {
        "D_inter": "0.4",
        "D_Deep":  "#4FC3F7",
        "D_Mid":   "#8D6E63",
        "D_Sup":   "#1976D2",
    }
    markers = {"D_inter": "s", "D_Deep": "o", "D_Mid": "o", "D_Sup": "o"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, corr_dict, title in zip(
        axes, [corr_g1, corr_g2],
        ["Correlation with G1 (sensory–association)",
         "Correlation with G2 (visual–somatomotor)"],
    ):
        for name, vals in corr_dict.items():
            ax.plot(sparsity_values, vals, marker=markers[name],
                    label=name, color=colors[name], linewidth=1.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0.67, color="r", lw=0.8, ls="--", alpha=0.5,
                   label="effective floor (0.67)")
        ax.set_xlabel("Sparsity")
        ax.set_title(title)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder, f"corr_margulies_sparsity_{s_min:.2f}-{s_max:.2f}.svg"
    ), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    pair = {"D_Deep": corr_g1["D_Deep"], "D_Sup": corr_g1["D_Sup"]}, \
           {"D_Deep": corr_g2["D_Deep"], "D_Sup": corr_g2["D_Sup"]}

    for ax, corr_dict, title in zip(
        axes, pair,
        ["Correlation with G1 (sensory–association)",
         "Correlation with G2 (visual–somatomotor)"],
    ):
        for name, vals in corr_dict.items():
            ax.plot(sparsity_values, vals, marker="o",
                    label=name, color=colors[name], linewidth=2.2,
                    markersize=6)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0.67, color="r", lw=0.8, ls="--", alpha=0.5,
                   label="effective floor (0.67)")
        ax.set_xlabel("Sparsity")
        ax.set_title(title)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder,
        f"corr_margulies_sparsity_supDeep_{s_min:.2f}-{s_max:.2f}.svg"
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
    # sweep 0.0, 0.1, ..., 0.9
    sparsity_values = [round(x, 2) for x in np.arange(0.67, 1, 0.01)]

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

    # ---- Run sparsity robustness analysis ----
    run_sparsity_robustness(
        M=M,
        N=N,
        output_dir=output_dir,
        analysis=analysis,
        sparsity_values=sparsity_values,
        kernel=kernel,
        random_state=13011991,
    )