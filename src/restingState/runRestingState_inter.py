from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from laminar_rs.config import LaminarConfig
from laminar_rs.connectivity import (
    within_layer_block_matrix,
    fisher_z_to_r,
    build_multiplex_adjacency,
    thresh_and_binarize,
)
from laminar_rs.gradients import (
    run_gradient_analysis_auto,
    inter_areal_dissimilarity,
)

from laminar_rs.plots_embedding import (
    plot_scatter3D_with_plane,
    plot_network_centroids3D,
    plot_scatter_with_global_correlation,
    plot_scatter_centroids,
    plot_rsn_distributions_by_network
)
from laminar_rs.flatmaps import plotFlatMap
from laminar_rs.models import plot_horizontal_correlation_bar, plot_horizontal_correlation_bar_partial
from laminar_rs.surface_maps import plotSurfaceMap, plotSurfaceMap_LH_gradients

import laminar_rs.schaefer_stats as stats


# ----------------- Parameters -----------------

ATLAS = "schaefer"
# ATLAS = "glasser"
YEO_N = 7   # or 17 
NUM_LAYERS = 3
LARGE_GAP = False
DATA_SET = "huppi"


if DATA_SET=="huppi":
    BASE = Path("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations")
    # SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31]

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

ANALYSIS_NAME = "WithinLayer_gradients_kernelCOS_API_interSpecific_newSubs"
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
        M, outputDir=output, max_components=50, kernel="cosine", var_threshold=0.85
    )
    print(f"Using {n_keep} gradients")

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient"

    D_inter, D_inter_deep, D_inter_mid, D_inter_sup, D = inter_areal_dissimilarity(G, output, N=N)

    D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
    D_Deep_standard = (D_inter_deep - np.min(D_inter_deep)) / ((np.max(D_inter_deep) - np.min(D_inter_deep)))
    D_Mid_standard = (D_inter_mid - np.min(D_inter_mid)) / ((np.max(D_inter_mid) - np.min(D_inter_mid)))
    D_Sup_standard = (D_inter_sup - np.min(D_inter_sup)) / ((np.max(D_inter_sup) - np.min(D_inter_sup)))

    plotSurfaceMap_LH_gradients(G, outdir=output, outname="gradients.png", HCP=HCP, cmap="PRGn")
    plotSurfaceMap(np.arange(N), output, "Surface_Schaefer", cmap = "gray", HCP=HCP)

    # ---------------------------------------------------------------------
    #                           Flatmaps and surface maps
    # ---------------------------------------------------------------------
    
    out_inter = plotSurfaceMap(D_inter, output, "SurfaceMap_interFlatMap.png", vmin=0.95,vmax=1.06, cmap = "viridis", HCP=HCP)
    out_interD = plotSurfaceMap(D_inter_deep, output, "SurfaceMap_interFlatMap_deep.png", vmin=0.95,vmax=1.06, cmap = "viridis", HCP=HCP)
    out_interM = plotSurfaceMap(D_inter_mid, output, "SurfaceMap_interFlatMap_mid.png", vmin=0.95,vmax=1.06, cmap = "viridis", HCP=HCP)
    out_interS = plotSurfaceMap(D_inter_sup, output, "SurfaceMap_interFlatMap_sup.png", vmin=0.95,vmax=1.06, cmap = "viridis", HCP=HCP)


    plot_rsn_distributions_by_network([D_inter_deep, D_inter_mid, D_inter_sup], out_dir=output, name="RSN_interLayers", 
                           array_labels=["Deep", "Middle", "Superficial"], atlas=ATLAS, yeo_n=YEO_N, y_label="Inter-regional Distance")


    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Flatmap"

    plotFlatMap(D_inter, output, "Flatmap_interFlatMap.png", HCP=HCP)
    plotFlatMap(D_inter_deep, output, "Flatmap_intraDeep_FlatMap.png", vmin=0.95,vmax=1.06, HCP=HCP)
    plotFlatMap(D_inter_mid, output, "Flatmap_intraMid_FlatMap.png", vmin=0.95,vmax=1.06, HCP=HCP)
    plotFlatMap(D_inter_sup, output, "Flatmap_intraSup_FlatMap.png", vmin=0.95,vmax=1.06, HCP=HCP)

    # ---------------------------------------------------------------------
    #                           Scatter plots
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Scatter"

    plot_scatter3D_with_plane(D_inter_sup[:,np.newaxis], output, Y=D_inter_mid[:,np.newaxis], Z=D_inter_deep[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_uncorrected.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_network_centroids3D(D_inter_sup[:,np.newaxis], output, Y=D_inter_mid[:,np.newaxis], Z=D_inter_deep[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_uncorrected.svg", atlas=ATLAS, yeo_n=YEO_N)


    plot_scatter3D_with_plane(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_corrected.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_network_centroids3D(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_corrected.svg", atlas=ATLAS, yeo_n=YEO_N)


    # ---------------------------------------------------------------------
    #                           Scatter plots layer-specific 2D
    # ---------------------------------------------------------------------

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Interparcel laminar difference: superficial", y_label ="Interparcel laminar difference: middle", fname="Scatter_SupMid.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Interparcel laminar difference: superficial", y_label ="Interparcel laminar difference: middle", fname="Scatter_SupMid_centroid.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Interparcel laminar difference: superficial", y_label ="Interparcel laminar difference: deep", fname="Scatter_SupDeep.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Interparcel laminar difference: superficial", y_label ="Interparcel laminar difference: deep", fname="Scatter_SupDeep_centroid.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Interparcel laminar difference: middle", y_label ="Interparcel laminar difference: deep", fname="Scatter_MidDeep.svg", atlas=ATLAS, yeo_n=YEO_N)

    plot_scatter_centroids(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Interparcel laminar difference: middle", y_label ="Interparcel laminar difference: deep", fname="Scatter_MidDeep_centroid.svg", atlas=ATLAS, yeo_n=YEO_N)


    # ---------------------------------------------------------------------
    #           Comaprison with G1 G2. Will only run for Schaefer 400
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "MarguliesFunc"

    from neuromaps.datasets import fetch_annotation
    from netneurotools import datasets as nntdata
    from neuromaps.parcellate import Parcellater
    from neuromaps.images import dlabel_to_gifti

    g1 = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
    g2 = fetch_annotation(source='margulies2016', desc='fcgradient02', space='fsLR', den='32k')
    schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
    parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')
    g1_s400 = parc.fit_transform(g1, 'fsLR')
    g2_s400 = parc.fit_transform(g2, 'fsLR')

    g1_surf = plotSurfaceMap(g1_s400, output, "SurfaceMap_G1.png", cmap = "PRGn", HCP=HCP)
    g2_surf = plotSurfaceMap(g2_s400, output, "SurfaceMap_G2.png", cmap = "PRGn", HCP=HCP)

    g1_s400 = (g1_s400 - np.min(g1_s400)) / ((np.max(g1_s400) - np.min(g1_s400)))
    g2_s400 = (g2_s400 - np.min(g2_s400)) / ((np.max(g2_s400) - np.min(g2_s400)))

    plot_horizontal_correlation_bar([D_inter_sup, D_inter_mid, D_inter_deep], g1_s400, output, "Layers_G1.svg")
    plot_horizontal_correlation_bar([D_inter_sup, D_inter_mid, D_inter_deep], g2_s400, output, "Layers_G2.svg")

    # ---------------------------------------------------------------------
    #      Scatter plots BigBrain Gradient. Will only run for Schaefer 400
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "ENIGMA"

    G_bigBrain = np.loadtxt(
        "/home/degutis/repos/ENIGMA/enigmatoolbox/histology/bb_gradient_schaefer_400.csv",
        delimiter=",",
    )

    bigBrain_surf = plotSurfaceMap(G_bigBrain, output, "SurfaceMap_BigBrain_grad.png", cmap = "PRGn", HCP=HCP)

    G_bigBrain_standard = (G_bigBrain - np.min(G_bigBrain)) / ((np.max(G_bigBrain) - np.min(G_bigBrain)))

    plot_scatter_with_global_correlation(np.concatenate([D_inter_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_InterLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Deep_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: deep", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraDeepLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: middle", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraMidLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraSupLamThick.svg")

    plot_horizontal_correlation_bar([D_inter_sup, D_inter_mid, D_inter_deep], G_bigBrain, output, "Layers_G_BigBrain.svg")

    plot_horizontal_correlation_bar_partial([D_inter_sup, D_inter_deep], G_bigBrain, output, "Layers_G_BigBrain_partial.svg")


    # ---------------------------------------------------------------------
    #                           rDCM Analysis
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "DCM"

    rDCM_matrix = loadmat("/home/degutis/repos/rDCM/hcp_rDCM_sch400.mat", squeeze_me=True, struct_as_record=False)
    rDCM_eff = rDCM_matrix["results"].Strength_Efferent_wholeBrain   
    rDCM_aff = rDCM_matrix["results"].Strength_Afferent_wholeBrain   
    rDCM_grad = rDCM_eff - rDCM_aff

    rDCM_grad_surf = plotSurfaceMap(rDCM_grad, output, "SurfaceMap_rDCM_grad.png", cmap = "PRGn", HCP=HCP)
    rDCM_grad_surf = plotSurfaceMap(rDCM_eff, output, "SurfaceMap_rDCM_eff_send.png", cmap = "PRGn", HCP=HCP)
    rDCM_grad_surf = plotSurfaceMap(rDCM_aff, output, "SurfaceMap_rDCM_aff_receive.png", cmap = "PRGn", HCP=HCP)

    # rDCM_grad_standard = (rDCM_grad - np.min(rDCM_grad)) / ((np.max(rDCM_grad) - np.min(rDCM_grad)))

    plot_horizontal_correlation_bar([D_inter_sup, D_inter_mid, D_inter_deep], rDCM_grad, output, "Layers_rDCM.svg")
    plot_horizontal_correlation_bar_partial([D_inter_sup, D_inter_deep], rDCM_grad, output, "Layers_rDCM_partial.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Deep_standard[:,np.newaxis], rDCM_grad[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: deep", y_label ="rDCM grad", fname="Scatter_ENIGMABigBrain_IntraDeepLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], rDCM_grad[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: superficial", y_label ="rDCM grad", fname="Scatter_ENIGMABigBrain_IntraMidLamThick.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraSupLamThick.svg")


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
                                        x_label="Interparcel laminar difference: deep", y_label ="SNR Deep", fname="Scatter_SNR_IntraDeep.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], SNR_mid[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: middle", y_label ="SNR Mid", fname="Scatter_SNR_IntraMid.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], SNR_sup[:,np.newaxis]], axis=1), output, 
                                        x_label="Interparcel laminar difference: superficial", y_label ="SNR Sup", fname="Scatter_SNR_IntraSup.svg")


    # ---------------------------------------------------------------------
    #                           Stats
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "Stats"
    output.mkdir(parents=True, exist_ok=True)
    import csv 

        # --------------------------------------------------------
        # Partial correlations: layer-specific inter vs rDCM
        # Each layer vs G1, controlling the other two layers
        # --------------------------------------------------------

    partial_rows = []

    # Superficial vs rDCM | Deep
    Z_sup = np.column_stack([D_inter_deep])
    r_sup, p_sup = stats.p_spin_partial_corr_schaefer400(
        x=D_inter_sup,
        y=rDCM_grad,
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

    # Deep vs rDCM | Superficial
    Z_deep = np.column_stack([D_inter_sup])
    r_deep, p_deep = stats.p_spin_partial_corr_schaefer400(
        x=D_inter_deep,
        y=rDCM_grad,
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

    partial_csv_path = output / "partial_corr_rDCM_layers_spin_rDCM.csv"
    with open(partial_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "r_partial_emp", "p_spin", "n_perm", "seed"],
        )
        writer.writeheader()
        writer.writerows(partial_rows)

    print(
        f"[INFO] Saved partial spin correlations (layers vs DCM) to {partial_csv_path}"
    )


        # --------------------------------------------------------
        # Partial correlations: layer-specific intra vs BigBrain G1
        # Each layer vs G1, controlling the other two layers
        # --------------------------------------------------------

    partial_rows = []

    # Superficial vs BigBrain | Deep
    Z_sup = np.column_stack([D_inter_deep])
    r_sup, p_sup = stats.p_spin_partial_corr_schaefer400(
        x=D_inter_sup,
        y=G_bigBrain,
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
            "seed": 501,
        }
    )

    # Deep vs rDCM | Superficial
    Z_deep = np.column_stack([D_inter_sup])
    r_deep, p_deep = stats.p_spin_partial_corr_schaefer400(
        x=D_inter_deep,
        y=G_bigBrain,
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
            "seed": 503,
        }
    )

    partial_csv_path = output / "partial_corr_BigBrain_layers_spin.csv"
    with open(partial_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "r_partial_emp", "p_spin", "n_perm", "seed"],
        )
        writer.writeheader()
        writer.writerows(partial_rows)

    print(
        f"[INFO] Saved partial spin correlations (layers vs DCM) to {partial_csv_path}"
    )

    # ----------------------------
    # Network × layer interactions
    # ----------------------------
    inter_int_spin_two = stats.network_layer_interaction_general(
        D_layers=[D_inter_deep, D_inter_mid, D_inter_sup],
        layer_names=["Deep", "Superficial"],
        n_perm=5000,
        random_state=1,
    )

    stats.save_interaction_to_csv(
        inter_int_spin_two,
        out_csv=str(output / "inter_interaction_spin_deep_sup.csv"),
    )


    # ------------------------------
    # Spin-based correlation tests
    # ------------------------------

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

        # 1) G1 G2 Func

        add_spin_corr(
            name="Inter_vs_G1",
            x=D_inter_standard,
            y=g1_s400,
            seed=1,
        )
        add_spin_corr(
            name="Deep_inter_vs_G1",
            x=D_Deep_standard,
            y=g1_s400,
            seed=2,
        )
        add_spin_corr(
            name="Mid_inter_vs_G1",
            x=D_Mid_standard,
            y=g1_s400,
            seed=3,
        )

        add_spin_corr(
            name="Sup_inter_vs_G1",
            x=D_Sup_standard,
            y=g1_s400,
            seed=4,
        )

        # G2 Func

        add_spin_corr(
            name="Inter_vs_G2",
            x=D_inter_standard,
            y=g2_s400,
            seed=5,
        )
        add_spin_corr(
            name="Deep_inter_vs_G2",
            x=D_Deep_standard,
            y=g2_s400,
            seed=6,
        )
        add_spin_corr(
            name="Mid_inter_vs_G2",
            x=D_Mid_standard,
            y=g2_s400,
            seed=7,
        )

        add_spin_corr(
            name="Sup_inter_vs_G2",
            x=D_Sup_standard,
            y=g2_s400,
            seed=8,
        )


        # 2) Within-inter layer pairs
        add_spin_corr(
            name="inter_Sup_vs_Mid",
            x=D_Sup_standard,
            y=D_Mid_standard,
            seed=11,
        )
        add_spin_corr(
            name="inter_Sup_vs_Deep",
            x=D_Sup_standard,
            y=D_Deep_standard,
            seed=12,
        )
        add_spin_corr(
            name="inter_Mid_vs_Deep",
            x=D_Mid_standard,
            y=D_Deep_standard,
            seed=13,
        )


        # 3) Laminar dissociation vs SNR
        add_spin_corr(
            name="Deep_inter_vs_SNR_deep",
            x=D_Deep_standard,
            y=SNR_deep,
            seed=30,
        )
        add_spin_corr(
            name="Mid_inter_vs_SNR_mid",
            x=D_Mid_standard,
            y=SNR_mid,
            seed=31,
        )
        add_spin_corr(
            name="Sup_inter_vs_SNR_sup",
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


if __name__ == "__main__":
    main()