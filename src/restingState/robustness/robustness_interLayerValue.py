#!/usr/bin/env python3
"""
Robustness analysis for laminar gradients: inter-layer coupling weight.

Instead of varying the number of gradients, this script varies the
inter-layer coupling weight used in build_multiplex_adjacency() from
0.0 to 1.0 in 0.1 steps. The off-diagonal blocks of the multilayer
adjacency matrix couple the same parcel across cortical depths; the
main analysis uses a weight of 1.0, and this script checks how
sensitive the downstream dissimilarity indices are to that choice.

For each inter-layer weight:
    - rebuild the multilayer (multiplex) adjacency matrix M
    - run the gradient embedding (fixed n_components)
    - recompute D_inter, D_Deep, D_Mid, D_Sup
Then:
    - compute correlation matrices of each index across weights
    - correlate each index with Margulies G1 and G2
Save:
    * correlation matrices as .npy
    * correlations with Margulies as .npz
    * one figure with subplots (lower-triangle style)
    * figures of correlation-vs-weight with G1/G2

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


def make_weight_list(weight_min, weight_max, weight_step):
    """Build an inclusive list of weights, robust to floating-point drift."""
    n_steps = int(round((weight_max - weight_min) / weight_step)) + 1
    return [round(weight_min + i * weight_step, 2) for i in range(n_steps)]


def run_interlayer_weight_robustness(
    r_matrix,
    N,
    output_dir,
    analysis,
    n_components=15,
    weight_min=0.0,
    weight_max=1.0,
    weight_step=0.1,
    kernel=None,
    random_state=13011991,
):
    """
    Sweep interlayer_weight from weight_min to weight_max, rebuild the
    multiplex adjacency matrix, recompute D_inter, D_Deep, D_Mid, D_Sup,
    and compute correlation matrices.

    Saves:
    - C_D_inter.npy, C_D_Deep.npy, C_D_Mid.npy, C_D_Sup.npy
    - corr_with_margulies.npz
    - robustness_interlayer_weight_XX-YY.svg
    - corr_margulies_interlayer_weight_XX-YY.svg
    - corr_margulies_interlayer_weight_supDeep_XX-YY.svg

    Parameters
    ----------
    r_matrix : np.ndarray
        Per-layer FC matrix (the `per_layer_matrix` argument of
        build_multiplex_adjacency), i.e. FC computed BEFORE multiplexing.
    n_components : int
        Number of gradient components, held fixed across the sweep
        (matches the value used in the main analysis).
    """
    weight_list = make_weight_list(weight_min, weight_max, weight_step)
    nW = len(weight_list)

    base_folder = os.path.join(output_dir, analysis, "robustness_interlayer_weight")
    os.makedirs(base_folder, exist_ok=True)

    D_inter_dict = {}
    D_Deep_dict = {}
    D_Mid_dict = {}
    D_Sup_dict = {}

    for w in weight_list:
        print(f"\n[ROBUSTNESS] Running gradients with interlayer_weight={w:.1f} "
              f"(n_components={n_components})")

        # Rebuild the multilayer adjacency matrix with this coupling weight.
        M_w = lrs.connectivity.build_multiplex_adjacency(
            r_matrix, interlayer_weight=w
        )

        G, eig = lrs.gradients.run_gradient_analysis(
            M_w, n_components=n_components, kernel=kernel,
            random_state=random_state
        )

        subfolder = os.path.join(base_folder, f"weight_{w:.1f}")
        os.makedirs(subfolder, exist_ok=True)

        # Compute dissimilarity measures
        D_inter, D_Deep, D_Mid, D_Sup, _ = lrs.gradients.inter_areal_dissimilarity(
            G, subfolder, N=N, zscore_within_layer=True
        )

        D_inter_dict[w] = D_inter
        D_Deep_dict[w] = D_Deep
        D_Mid_dict[w] = D_Mid
        D_Sup_dict[w] = D_Sup

    # --- build correlation matrices (index stability across weights) ---
    def build_corr_matrix(D_dict):
        C = np.zeros((nW, nW), dtype=float)
        for i, wi in enumerate(weight_list):
            for j, wj in enumerate(weight_list):
                C[i, j] = vector_corr(D_dict[wi], D_dict[wj])
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

    mats = [
        lower_triangle(C_D_inter),
        lower_triangle(C_D_Deep),
        lower_triangle(C_D_Mid),
        lower_triangle(C_D_Sup),
    ]
    titles = ["D_inter", "D_Deep", "D_Mid", "D_Sup"]
    weight_labels = [f"{w:.1f}" for w in weight_list]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, mat, title in zip(axes, mats, titles):
        m = np.ma.masked_invalid(mat)
        im = ax.imshow(m, vmin=0, vmax=1, aspect='equal')
        ax.set_title(title)
        ax.set_xticks(range(nW))
        ax.set_yticks(range(nW))
        ax.set_xticklabels(weight_labels, rotation=90)
        ax.set_yticklabels(weight_labels)
        ax.set_xlabel("interlayer weight")
        ax.set_ylabel("interlayer weight")
        ax.tick_params(labelsize=8)

    # Hide unused subplots (4 indices in a 2x3 grid)
    for ax in axes[len(mats):]:
        ax.axis("off")

    fig.suptitle(
        f"Robustness across inter-layer weight ({weight_min:.1f}-{weight_max:.1f})",
        y=0.96,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.9,
        wspace=0.3,
        hspace=0.4,
    )

    # Slim colorbar on the right (attached to the used axes only)
    cbar = fig.colorbar(
        im,
        ax=list(axes[:len(mats)]),
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Pearson r")
    cbar.ax.tick_params(labelsize=8)

    fig_path = os.path.join(
        base_folder,
        f"robustness_interlayer_weight_{weight_min:.1f}-{weight_max:.1f}.svg",
    )
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # --- correlation with Margulies G1 / G2 across inter-layer weights ---
    g1, g2 = load_margulies_gradients(output_dir, analysis)

    indices_dict = {
        "D_inter": D_inter_dict,
        "D_Deep":  D_Deep_dict,
        "D_Mid":   D_Mid_dict,
        "D_Sup":   D_Sup_dict,
    }

    # One signed r value per inter-layer weight. The dissimilarity indices
    # are row-means of cosine-distance matrices (positive scalars per
    # parcel), so unlike the raw gradient axes they are not sign-ambiguous.
    corr_g1 = {name: np.array([vector_corr(d[w], g1) for w in weight_list])
               for name, d in indices_dict.items()}
    corr_g2 = {name: np.array([vector_corr(d[w], g2) for w in weight_list])
               for name, d in indices_dict.items()}

    np.savez(
        os.path.join(base_folder, "corr_with_margulies.npz"),
        interlayer_weights=np.array(weight_list),
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
            ax.plot(weight_list, vals, marker=markers[name],
                    label=name, color=colors[name], linewidth=1.8)
        ax.set_xlabel("inter-layer coupling weight")
        ax.set_xticks(weight_list)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder,
        f"corr_margulies_interlayer_weight_{weight_min:.1f}-{weight_max:.1f}.svg"
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
            ax.plot(weight_list, vals, marker="o",
                    label=name, color=colors[name], linewidth=2.2,
                    markersize=6)
        ax.set_xlabel("inter-layer coupling weight")
        ax.set_xticks(weight_list)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(
        base_folder,
        f"corr_margulies_interlayer_weight_supDeep_"
        f"{weight_min:.1f}-{weight_max:.1f}.svg"
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

    # n_components is held FIXED here (the gradient-count sweep is a
    # separate robustness analysis). 15 matches the main analysis.
    n_components = 15
    weight_min = 0.0
    weight_max = 1.0
    weight_step = 0.1

    # ---- Load or compute per-layer FC matrix r_matrix ----
    # NOTE: this is the per-layer matrix BEFORE multiplexing, i.e. the
    # `per_layer_matrix` input to build_multiplex_adjacency. We cannot
    # reuse a cached FC_matrix.npy here, because that is the already-
    # assembled multiplex matrix with the inter-layer weight baked in.
    r_path = os.path.join(output_dir, analysis, 'r_matrix.npy')

    if os.path.isfile(r_path):
        r_matrix = np.load(r_path, allow_pickle=False)
        print(f"[INFO] loaded: {r_path}  shape={r_matrix.shape}")
        print("  max:", np.max(r_matrix))
        print("  min:", np.min(r_matrix))
    else:
        print("[INFO] r_matrix.npy not found, recomputing from subjects.")
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

        # Per-layer FC matrix (NOT multiplexed). build_multiplex_adjacency
        # is applied per weight inside run_interlayer_weight_robustness.
        r_matrix = lrs.connectivity.fisher_z_to_r(mean_adj_matrix)
        print("[INFO] per-layer FC matrix constructed.")
        print("  max:", np.max(r_matrix))
        print("  min:", np.min(r_matrix))
        np.save(r_path, r_matrix)
        print(f"[INFO] saved per-layer FC matrix to: {r_path}")

    # ---- Run inter-layer weight robustness analysis ----
    run_interlayer_weight_robustness(
        r_matrix=r_matrix,
        N=N,
        output_dir=output_dir,
        analysis=analysis,
        n_components=n_components,
        weight_min=weight_min,
        weight_max=weight_max,
        weight_step=weight_step,
        kernel=kernel,
        random_state=13011991,
    )