from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from laminar_rs.config import LaminarConfig
from laminar_rs.connectivity import (
    within_layer_block_matrix,
    fisher_z_to_r,
    build_multiplex_adjacency,
    thresh_and_binarize
)
from laminar_rs.gradients import (
    run_gradient_analysis_auto,
    inter_areal_dissimilarity,
    intra_areal_dissimilarity,
    run_gradient_analysis
)
from laminar_rs.plots_embedding import (
    plot_scatter3D_with_plane,
    plot_network_centroids3D,
    plot_scatter_with_global_correlation,
    plot_scatter_centroids,
    plot_rsn_distributions,
    plot_rsn_distributions_by_network
)
from laminar_rs.flatmaps import plotFlatMap
from laminar_rs.models import run_ff_fb_models, plot_horizontal_correlation_bar
from laminar_rs.surface_maps import plotSurfaceMap, plotSurfaceMap_LH_gradients

import laminar_rs.schaefer_stats as stats


# ----------------- Parameters -----------------

ATLAS = "schaefer"
# ATLAS = "glasser"
NUM_LAYERS = 3
LARGE_GAP = False
DATA_SET = "huppi"


if DATA_SET=="huppi":
    BASE = Path("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations")
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
elif DATA_SET=="kd":
    BASE = Path("/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/correlations")
    SUBJECTS = [1, 2, 4]

if ATLAS=="schaefer":
    N = 400
    GAP_DIR = f'{"large" if LARGE_GAP else "small"}Gap_Schaefer'
    HCP = False

else:
    N=360
    GAP_DIR = f'{"large" if LARGE_GAP else "small"}Gap_Glasser'
    HCP = True

ROOT = BASE / GAP_DIR


if DATA_SET=="huppi":
    DATA_DIRS = [ROOT / f"sub-LAM{s:03d}" for s in SUBJECTS]
elif DATA_SET=="kd":
    DATA_DIRS = [ROOT / f"sub-{s:02d}" for s in SUBJECTS]

OUTPUT_DIR = ROOT

ANALYSIS_NAME = "WithinLayer_gradients_kernelCOS_API"
RESULT_PATH = OUTPUT_DIR / ANALYSIS_NAME / "FC_matrix.npy"


# ----------------- Main -----------------

def main() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESULT_PATH.is_file():
        M = np.load(RESULT_PATH, allow_pickle=False)
        print(f"[INFO] loaded: {RESULT_PATH}  shape={M.shape}")
        print("max:", np.max(M))
        print("min:", np.min(M))

    else:
        per_subject_corr_z = []

        for data_dir in DATA_DIRS:
            print(f"[INFO] Processing {data_dir}")

            cfg = LaminarConfig(
                data_dir=data_dir,
                N=N,
                num_layers=NUM_LAYERS,
            )

            # adj_full is ignored; we only want per-layer Fisher z
            _, corr_layer_z = within_layer_block_matrix(cfg, subtract_average=False)

            adjMatrix = thresh_and_binarize(
                corr_layer_z
        )

            per_subject_corr_z.append(adjMatrix)

        # Stack over subjects: (N, N, L, n_subjects)
        corr_z_all = np.stack(per_subject_corr_z, axis=3)
        # Mean Fisher z across subjects, then convert back to r
        mean_corr_z = np.nanmean(corr_z_all, axis=3)  # (N, N, L)
        mean_r = fisher_z_to_r(mean_corr_z)

        # Build multiplex adjacency and save
        M = build_multiplex_adjacency(mean_r)
        print("max:", np.max(M))
        print("min:", np.min(M))

        np.save(RESULT_PATH, M)
        print(f"[INFO] Saved FC matrix to {RESULT_PATH}")

    # ---------------------------------------------------------------------
    #                           Gradient analysis
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME

    G, eig, all_l, frac, cum, n_keep = run_gradient_analysis_auto(
        M, outputDir=output, max_components=50, kernel="cosine", var_threshold=0.85,
    )
    print(f"Using {n_keep} gradients")

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient"

    D_inter, D_inter_deep, D_inter_mid, D_inter_sup, D = inter_areal_dissimilarity(G, output, N=N)

    D_intra, D_Deep, D_Mid, D_Sup = intra_areal_dissimilarity(G, output, N=N)

    D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
    D_intra_standard = (D_intra - np.min(D_intra)) / ((np.max(D_intra) - np.min(D_intra)))
    D_Deep_standard = (D_Deep - np.min(D_Deep)) / ((np.max(D_Deep) - np.min(D_Deep)))
    D_Mid_standard = (D_Mid - np.min(D_Mid)) / ((np.max(D_Mid) - np.min(D_Mid)))
    D_Sup_standard = (D_Sup - np.min(D_Sup)) / ((np.max(D_Sup) - np.min(D_Sup)))

    plotSurfaceMap_LH_gradients(G, outdir=output, outname="gradients.png", HCP=HCP, cmap="PRGn")
    plotSurfaceMap(np.arange(N), output, "Surface_Schaefer", cmap = "gray", HCP=HCP)

    # ---------------------------------------------------------------------
    #                           Flatmaps and surface maps
    # ---------------------------------------------------------------------

    out_inter = plotSurfaceMap(D_inter, output, "SurfaceMap_interFlatMap.png", cmap = "viridis", HCP=HCP)
    out_intra = plotSurfaceMap(D_intra, output, "SurfaceMap_intraFlatMap.png", cmap = "cividis", HCP=HCP)
    out_intraD = plotSurfaceMap(D_Deep, output, "SurfaceMap_intraDeep_FlatMap.png", vmin=0,vmax=0.4, cmap = "cividis", HCP=HCP)
    out_intraM = plotSurfaceMap(D_Mid, output, "SurfaceMap_intraMid_FlatMap.png", vmin=0,vmax=0.4, cmap = "cividis", HCP=HCP)
    out_intraS = plotSurfaceMap(D_Sup, output, "SurfaceMap_intraSup_FlatMap.png", vmin=0,vmax=0.4, cmap = "cividis", HCP=HCP)

    plot_rsn_distributions([D_Deep, D_Mid, D_Sup], out_dir=output, name="RSN_intraLayers", 
                           array_labels=["Deep", "Middle", "Superficial"], atlas=ATLAS, y_label="Intra-regional Distance")

    plot_rsn_distributions_by_network([D_Deep, D_Mid, D_Sup], out_dir=output, name="RSN_intraLayers", 
                           array_labels=["Deep", "Middle", "Superficial"], atlas=ATLAS, y_label="Intra-regional Distance")

    plot_rsn_distributions([D_inter, D_intra], out_dir=output, name="RSN", 
                           array_labels=["Inter", "Intra"], atlas=ATLAS, y_label="Distance", share_yaxis = False)

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Flatmap"

    plotFlatMap(D_inter, output, "Flatmap_interFlatMap.png", HCP=HCP)
    plotFlatMap(D_intra, output, "Flatmap_intraFlatMap.png", HCP=HCP)
    plotFlatMap(D_Deep, output, "Flatmap_intraDeep_FlatMap.png", vmin=0,vmax=0.5, HCP=HCP)
    plotFlatMap(D_Mid, output, "Flatmap_intraMid_FlatMap.png", vmin=0,vmax=0.5, HCP=HCP)
    plotFlatMap(D_Sup, output, "Flatmap_intraSup_FlatMap.png", vmin=0,vmax=0.5, HCP=HCP)

    # ---------------------------------------------------------------------
    #                           Scatter plots
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Scatter"

    plot_scatter3D_with_plane(D_Sup[:,np.newaxis], output, Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_uncorrected.svg", atlas=ATLAS)

    plot_network_centroids3D(D_Sup[:,np.newaxis], output, Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_uncorrected.svg", atlas=ATLAS)


    plot_scatter3D_with_plane(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_corrected.svg", atlas=ATLAS)

    plot_network_centroids3D(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_corrected.svg", atlas=ATLAS)

    plot_scatter_with_global_correlation(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra.svg", atlas=ATLAS)

    plot_scatter_centroids(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra_centroid.svg", atlas=ATLAS)

    # ---------------------------------------------------------------------
    #                           Scatter plots layer-specific 2D
    # ---------------------------------------------------------------------

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: middle", fname="Scatter_SupMid.svg", atlas=ATLAS)

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: middle", fname="Scatter_SupMid_centroid.svg", atlas=ATLAS)

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: deep", fname="Scatter_SupDeep.svg", atlas=ATLAS)

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: deep", fname="Scatter_SupDeep_centroid.svg", atlas=ATLAS)

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: middle", y_label ="Intraparcel laminar difference: deep", fname="Scatter_MidDeep.svg", atlas=ATLAS)

    plot_scatter_centroids(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: middle", y_label ="Intraparcel laminar difference: deep", fname="Scatter_MidDeep_centroid.svg", atlas=ATLAS)

    # ---------------------------------------------------------------------
    #      Scatter plots BigBrain Gradient. Will only run for Schaefer 400
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "ENIGMA"

    G_bigBrain = np.loadtxt(
        "/home/degutis/repos/ENIGMA/enigmatoolbox/histology/bb_gradient_schaefer_400.csv",
        delimiter=",",
    )
    G_bigBrain_standard = (G_bigBrain - np.min(G_bigBrain)) / ((np.max(G_bigBrain) - np.min(G_bigBrain)))

    plot_scatter_with_global_correlation(np.concatenate([D_intra_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_inter_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_InterLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Deep_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: deep", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraDeepLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: middle", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraMidLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraSupLamThick.svg")

    plot_horizontal_correlation_bar([D_Sup, D_Mid, D_Deep], G_bigBrain, output, "Layers_G_BigBrain.svg")

    bigBrain_surf = plotSurfaceMap(G_bigBrain, output, "SurfaceMap_BigBrain_grad.png", cmap = "PRGn", HCP=HCP)

    # ---------------------------------------------------------------------
    #                           rDCM Analysis
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "DCM"

    rDCM_matrix = loadmat("/home/degutis/repos/rDCM/hcp_rDCM_sch400.mat", squeeze_me=True, struct_as_record=False)
    rDCM_eff = rDCM_matrix["results"].Strength_Efferent_wholeBrain   
    rDCM_aff = rDCM_matrix["results"].Strength_Afferent_wholeBrain   
    rDCM_grad = rDCM_eff - rDCM_aff

    rDCM_grad_surf = plotSurfaceMap(rDCM_grad, output, "SurfaceMap_rDCM_grad.png", cmap = "PRGn", HCP=HCP)

    rDCM_grad = (rDCM_grad - np.min(rDCM_grad)) / ((np.max(rDCM_grad) - np.min(rDCM_grad)))
    rDCM_eff = (rDCM_eff - np.min(rDCM_eff)) / ((np.max(rDCM_eff) - np.min(rDCM_eff)))
    rDCM_aff = (rDCM_aff - np.min(rDCM_aff)) / ((np.max(rDCM_aff) - np.min(rDCM_aff)))

    df, fig_path, csv_path = run_ff_fb_models(
        [D_Sup, D_Mid, D_Deep],
        rDCM_eff,
        rDCM_aff,
        output,
        "ff_fb_partial_bars_ENIGMA.svg",
        spin_n_perm=5000,
        spin_random_state=123,
    )    
    
    print("saved:", fig_path, "and", csv_path)

    # ---------------------------------------------------------------------
    #                           SNR
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "SNR"

    SNR_deep = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR/all/all_tSNRorig_layer1_tSNR_byParcel.npy")
    SNR_mid = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR/all/all_tSNRorig_layer2_tSNR_byParcel.npy")
    SNR_sup = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR/all/all_tSNRorig_layer3_tSNR_byParcel.npy")
    
    SNR_deep = (SNR_deep - np.min(SNR_deep)) / ((np.max(SNR_deep) - np.min(SNR_deep)))
    SNR_mid = (SNR_mid - np.min(SNR_mid)) / ((np.max(SNR_mid) - np.min(SNR_mid)))
    SNR_sup = (SNR_sup - np.min(SNR_sup)) / ((np.max(SNR_sup) - np.min(SNR_sup)))


    plot_scatter_with_global_correlation(np.concatenate([D_Deep_standard[:,np.newaxis], SNR_deep[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: deep", y_label ="SNR Deep", fname="Scatter_SNR_IntraDeep.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], SNR_mid[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: middle", y_label ="SNR Mid", fname="Scatter_SNR_IntraMid.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], SNR_sup[:,np.newaxis]], axis=1), output, 
                                        x_label="Intraparcel laminar difference: superficial", y_label ="SNR Sup", fname="Scatter_SNR_IntraSup.svg")


    # ---------------------------------------------------------------------
    #                           Stats
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "Stats"
    output.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Network × layer interactions
    # ----------------------------

    intra_int_spin = stats.network_layer_interaction_general(
        D_layers=[D_Deep, D_Mid, D_Sup],
        layer_names=["Deep", "Middle", "Superficial"],
        n_perm=5000,
        random_state=1,
    )

    stats.save_interaction_to_csv(
        intra_int_spin,
        out_csv=str(output / "intra_interaction_spin.csv"),
    )

    inter_intra_int_spin = stats.network_layer_interaction_general(
        D_layers=[D_inter, D_intra],
        layer_names=["Inter", "Intra"],
        n_perm=5000,
        random_state=2,
    )

    stats.save_interaction_to_csv(
        inter_intra_int_spin,
        out_csv=str(output / "interintra_interaction_spin_ENIGMA.csv"),
    )

    # --------------------------------------------------
    # Spin-based correlation tests (ENIGMA-style p-spin)
    # --------------------------------------------------

    import csv  # local is fine here

    corr_rows = []

    def add_spin_corr(name: str, x: np.ndarray, y: np.ndarray, seed: int) -> None:
        """Helper to compute p-spin correlation and append a row."""
        r_emp, p_spin = stats.p_spin_corr_schaefer400(
            x=x,
            y=y,
            n_perm=5000,
            corr_type="pearson",
            random_state=seed,
        )
        corr_rows.append(
            {
                "contrast": name,
                "r_emp": float(r_emp),
                "p_spin": float(p_spin),
                "n_perm": 5000,
                "seed": int(seed),
            }
        )

    if ATLAS == "schaefer":
        # 1) Inter vs intra laminar difference
        add_spin_corr(
            name="Inter_vs_Intra",
            x=D_inter_standard,
            y=D_intra_standard,
            seed=10,
        )

        # 2) Within-intra layer pairs
        add_spin_corr(
            name="Intra_Sup_vs_Mid",
            x=D_Sup_standard,
            y=D_Mid_standard,
            seed=11,
        )
        add_spin_corr(
            name="Intra_Sup_vs_Deep",
            x=D_Sup_standard,
            y=D_Deep_standard,
            seed=12,
        )
        add_spin_corr(
            name="Intra_Mid_vs_Deep",
            x=D_Mid_standard,
            y=D_Deep_standard,
            seed=13,
        )

        # 3) Laminar dissociation vs BigBrain histology gradient (G1)
        add_spin_corr(
            name="Intra_vs_BigBrain_G1",
            x=D_intra_standard,
            y=G_bigBrain_standard,
            seed=20,
        )
        add_spin_corr(
            name="Inter_vs_BigBrain_G1",
            x=D_inter_standard,
            y=G_bigBrain_standard,
            seed=21,
        )
        add_spin_corr(
            name="Deep_intra_vs_BigBrain_G1",
            x=D_Deep_standard,
            y=G_bigBrain_standard,
            seed=22,
        )
        add_spin_corr(
            name="Mid_intra_vs_BigBrain_G1",
            x=D_Mid_standard,
            y=G_bigBrain_standard,
            seed=23,
        )
        add_spin_corr(
            name="Sup_intra_vs_BigBrain_G1",
            x=D_Sup_standard,
            y=G_bigBrain_standard,
            seed=24,
        )

        # 4) Laminar dissociation vs SNR
        add_spin_corr(
            name="Deep_intra_vs_SNR_deep",
            x=D_Deep_standard,
            y=SNR_deep,
            seed=30,
        )
        add_spin_corr(
            name="Mid_intra_vs_SNR_mid",
            x=D_Mid_standard,
            y=SNR_mid,
            seed=31,
        )
        add_spin_corr(
            name="Sup_intra_vs_SNR_sup",
            x=D_Sup_standard,
            y=SNR_sup,
            seed=32,
        )

        corr_csv_path = output / "spin_correlations_schaefer_enigma.csv"
        with open(corr_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["contrast", "r_emp", "p_spin", "n_perm", "seed"],
            )
            writer.writeheader()
            writer.writerows(corr_rows)

        print(f"[INFO] Saved spin-based correlation stats to {corr_csv_path}")

        # --------------------------------------------------------
        # Partial correlations: layer-specific intra vs BigBrain G1
        # Each layer vs G1, controlling the other two layers
        # --------------------------------------------------------

        partial_rows = []

        # Superficial vs G1 | Middle, Deep
        Z_sup = np.column_stack([D_Mid_standard, D_Deep_standard])
        r_sup, p_sup = stats.p_spin_partial_corr_schaefer400(
            x=D_Sup_standard,
            y=G_bigBrain_standard,
            Z=Z_sup,
            n_perm=5000,
            corr_type="pearson",
            random_state=401,
        )
        partial_rows.append(
            {
                "layer": "Superficial",
                "r_partial_emp": float(r_sup),
                "p_spin": float(p_sup),
                "n_perm": 5000,
                "seed": 401,
            }
        )

        # Middle vs G1 | Superficial, Deep
        Z_mid = np.column_stack([D_Sup_standard, D_Deep_standard])
        r_mid, p_mid = stats.p_spin_partial_corr_schaefer400(
            x=D_Mid_standard,
            y=G_bigBrain_standard,
            Z=Z_mid,
            n_perm=5000,
            corr_type="pearson",
            random_state=402,
        )
        partial_rows.append(
            {
                "layer": "Middle",
                "r_partial_emp": float(r_mid),
                "p_spin": float(p_mid),
                "n_perm": 5000,
                "seed": 402,
            }
        )

        # Deep vs G1 | Superficial, Middle
        Z_deep = np.column_stack([D_Sup_standard, D_Mid_standard])
        r_deep, p_deep = stats.p_spin_partial_corr_schaefer400(
            x=D_Deep_standard,
            y=G_bigBrain_standard,
            Z=Z_deep,
            n_perm=5000,
            corr_type="pearson",
            random_state=403,
        )
        partial_rows.append(
            {
                "layer": "Deep",
                "r_partial_emp": float(r_deep),
                "p_spin": float(p_deep),
                "n_perm": 5000,
                "seed": 403,
            }
        )

        partial_csv_path = output / "partial_corr_bigbrain_layers_spin_enigma.csv"
        with open(partial_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "r_partial_emp", "p_spin", "n_perm", "seed"],
            )
            writer.writeheader()
            writer.writerows(partial_rows)

        print(
            f"[INFO] Saved partial spin correlations (layers vs BigBrain G1) to {partial_csv_path}"
        )



if __name__ == "__main__":
    main()