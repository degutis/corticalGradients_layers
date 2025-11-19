from pathlib import Path
import numpy as np
from scipy.io import loadmat

from laminar_rs.config import LaminarConfig
from laminar_rs.connectivity import (
    within_layer_block_matrix,
    fisher_z_to_r,
    build_multiplex_adjacency,
    thresh_and_binarize
)
from laminar_rs.gradients import (
    run_gradient_analysis,
    inter_areal_dissimilarity,
    intra_areal_dissimilarity
)
from laminar_rs.plots_embedding import (
    plot_on_mmhcp_surface_multipleLayers,
    plot_scatter3D_with_plane,
    plot_network_centroids3D,
    plot_scatter_with_global_correlation,
    plot_scatter_centroids
)
from laminar_rs.flatmaps import plotFlatMap
from laminar_rs.models import run_ff_fb_models


# ----------------- Parameters -----------------

N = 400
SET_THRESH = 0.0
NUM_LAYERS = 3
LARGE_GAP = False
N_COMPONENTS = 10

BASE = Path("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations")
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]

GAP_DIR = f'{"large" if LARGE_GAP else "small"}Gap_Schaefer'
ROOT = BASE / GAP_DIR

DATA_DIRS = [ROOT / f"sub-LAM{s:03d}" for s in SUBJECTS]
OUTPUT_DIR = ROOT

ANALYSIS_NAME = "WithinLayer_gradients_kernelNone_21Subs_20Components_API"
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
            # print(f"[INFO] Processing {data_dir}")

            cfg = LaminarConfig(
                data_dir=data_dir,
                N=N,
                set_thresh=SET_THRESH,
                num_layers=NUM_LAYERS,
            )

            # adj_full is ignored; we only want per-layer Fisher z
            _, corr_layer_z = within_layer_block_matrix(cfg, subtract_average=False)

            adjMatrix = thresh_and_binarize(
                corr_layer_z,
                set_thresh=SET_THRESH,
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

    G, eig = run_gradient_analysis(M, n_components=N_COMPONENTS)
   
    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient"

    D_inter, D_inter_deep, D_inter_mid, D_inter_sup = inter_areal_dissimilarity(G, output, N=N)
    D_intra, D_Deep, D_Mid, D_Sup = intra_areal_dissimilarity(G, output, N=N)

    D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
    D_intra_standard = (D_intra - np.min(D_intra)) / ((np.max(D_intra) - np.min(D_intra)))
    D_Deep_standard = (D_Deep - np.min(D_Deep)) / ((np.max(D_Deep) - np.min(D_Deep)))
    D_Mid_standard = (D_Mid - np.min(D_Mid)) / ((np.max(D_Mid) - np.min(D_Mid)))
    D_Sup_standard = (D_Sup - np.min(D_Sup)) / ((np.max(D_Sup) - np.min(D_Sup)))


    # ---------------------------------------------------------------------
    #                           Flatmaps and surface maps
    # ---------------------------------------------------------------------

    plot_on_mmhcp_surface_multipleLayers(D_inter[:,np.newaxis], output, "D_Inter")        
    plot_on_mmhcp_surface_multipleLayers(D_intra[:,np.newaxis], output, "D_Intra")        
    plot_on_mmhcp_surface_multipleLayers(D_Deep[:,np.newaxis], output, "D_Deep", vmin=0,vmax=0.5)        
    plot_on_mmhcp_surface_multipleLayers(D_Mid[:,np.newaxis], output, "D_Mid", vmin=0,vmax=0.5)        
    plot_on_mmhcp_surface_multipleLayers(D_Sup[:,np.newaxis], output, "D_Sup", vmin=0,vmax=0.5)       

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Flatmap"

    plotFlatMap(D_inter, output, "Flatmap_interFlatMap.png")
    plotFlatMap(D_intra, output, "Flatmap_intraFlatMap.png")
    plotFlatMap(D_Deep, output, "Flatmap_intraDeep_FlatMap.png", vmin=0,vmax=0.5)
    plotFlatMap(D_Mid, output, "Flatmap_intraMid_FlatMap.png", vmin=0,vmax=0.5)
    plotFlatMap(D_Sup, output, "Flatmap_intraSup_FlatMap.png", vmin=0,vmax=0.5)

    # ---------------------------------------------------------------------
    #                           Scatter plots
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient" / "Scatter"

    plot_scatter3D_with_plane(D_Sup[:,np.newaxis], output, Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_uncorrected.svg")

    plot_network_centroids3D(D_Sup[:,np.newaxis], output, Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_uncorrected.svg")


    plot_scatter3D_with_plane(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], layer_labels="AcrossLayers", 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_corrected.svg")

    plot_network_centroids3D(D_Sup_standard[:,np.newaxis], output, Y=D_Mid_standard[:,np.newaxis], Z=D_Deep_standard[:,np.newaxis], 
                                        x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_corrected.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra.svg")

    plot_scatter_centroids(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra_centroid.svg")

    # ---------------------------------------------------------------------
    #                           Scatter plots layer-specific 2D
    # ---------------------------------------------------------------------

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: middle", fname="Scatter_SupMid.svg")

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Mid_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: middle", fname="Scatter_SupMid_centroid.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: deep", fname="Scatter_SupDeep.svg")

    plot_scatter_centroids(np.concatenate([D_Sup_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: superficial", y_label ="Intraparcel laminar difference: deep", fname="Scatter_SupDeep_centroid.svg")

    plot_scatter_with_global_correlation(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                                    x_label="Intraparcel laminar difference: middle", y_label ="Intraparcel laminar difference: deep", fname="Scatter_MidDeep.svg")

    plot_scatter_centroids(np.concatenate([D_Mid_standard[:,np.newaxis], D_Deep_standard[:,np.newaxis]], axis=1), output, 
                        x_label="Intraparcel laminar difference: middle", y_label ="Intraparcel laminar difference: deep", fname="Scatter_MidDeep_centroid.svg")


    # ---------------------------------------------------------------------
    #                           Scatter plots BigBrain Gradient
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


    # ---------------------------------------------------------------------
    #                           Scatter plots BigBrain Gradient
    # ---------------------------------------------------------------------

    output = OUTPUT_DIR / ANALYSIS_NAME / "DCM"

    rDCM_matrix = loadmat("/home/degutis/repos/rDCM/hcp_rDCM_sch400.mat", squeeze_me=True, struct_as_record=False)
    rDCM_eff = rDCM_matrix["results"].Strength_Efferent_wholeBrain   
    rDCM_aff = rDCM_matrix["results"].Strength_Afferent_wholeBrain   
    rDCM_grad = rDCM_eff - rDCM_aff

    rDCM_grad = (rDCM_grad - np.min(rDCM_grad)) / ((np.max(rDCM_grad) - np.min(rDCM_grad)))
    rDCM_eff = (rDCM_eff - np.min(rDCM_eff)) / ((np.max(rDCM_eff) - np.min(rDCM_eff)))
    rDCM_aff = (rDCM_aff - np.min(rDCM_aff)) / ((np.max(rDCM_aff) - np.min(rDCM_aff)))

    df, fig_path, csv_path = run_ff_fb_models([D_Sup, D_Mid, D_Deep], rDCM_eff, rDCM_aff, output, "ff_fb_partial_bars.svg")
    print("saved:", fig_path, "and", csv_path)



    G_hubness = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/HubnessAnalysis/gradients_Hubness_Schaefer.npy")
    G_hubness_00_standard = (G_hubness[:,0] - np.min(G_hubness[:,0])) / ((np.max(G_hubness[:,0]) - np.min(G_hubness[:,0])))
    G_hubness_01_standard = (G_hubness[:,1] - np.min(G_hubness[:,1])) / ((np.max(G_hubness[:,1]) - np.min(G_hubness[:,1])))



if __name__ == "__main__":
    main()