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
    run_gradient_analysis_auto,
    inter_areal_dissimilarity,
    intra_areal_dissimilarity
)
from laminar_rs.plots_embedding import (
    plot_on_mmhcp_surface_multipleLayers,
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


# ----------------- Parameters -----------------

ATLAS = "schaefer"
# ATLAS = "glasser"
SET_THRESH = 0.0
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



    G, eig, all_l, frac, cum, n_keep = run_gradient_analysis_auto(
        M, outputDir=output, max_components=50, kernel="cosine", var_threshold=0.85,
    )
    print(f"Using {n_keep} gradients")

    output = OUTPUT_DIR / ANALYSIS_NAME / "dissimilarityGradient"

    D_inter, D_inter_deep, D_inter_mid, D_inter_sup = inter_areal_dissimilarity(G, output, N=N)
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

    out_inter = plotSurfaceMap(D_inter, output, "SurfaceMap_interFlatMap.png", cmap = "magma", HCP=HCP)
    out_intra = plotSurfaceMap(D_intra, output, "SurfaceMap_intraFlatMap.png", cmap = "cividis", HCP=HCP)
    out_intraD = plotSurfaceMap(D_Deep, output, "SurfaceMap_intraDeep_FlatMap.png", vmin=0,vmax=0.5, cmap = "cividis", HCP=HCP)
    out_intraM = plotSurfaceMap(D_Mid, output, "SurfaceMap_intraMid_FlatMap.png", vmin=0,vmax=0.5, cmap = "cividis", HCP=HCP)
    out_intraS = plotSurfaceMap(D_Sup, output, "SurfaceMap_intraSup_FlatMap.png", vmin=0,vmax=0.5, cmap = "cividis", HCP=HCP)

    # plot_on_mmhcp_surface_multipleLayers(D_inter[:,np.newaxis], output, "D_Inter")        
    # plot_on_mmhcp_surface_multipleLayers(D_intra[:,np.newaxis], output, "D_Intra")        
    # plot_on_mmhcp_surface_multipleLayers(D_Deep[:,np.newaxis], output, "D_Deep", vmin=0,vmax=0.5)        
    # plot_on_mmhcp_surface_multipleLayers(D_Mid[:,np.newaxis], output, "D_Mid", vmin=0,vmax=0.5)        
    # plot_on_mmhcp_surface_multipleLayers(D_Sup[:,np.newaxis], output, "D_Sup", vmin=0,vmax=0.5)       

    plot_rsn_distributions([D_Deep, D_Mid, D_Sup], out_dir=output, name="RSN_intraLayers", 
                           array_labels=["Deep", "Middle", "Superficial"], atlas=ATLAS, y_label="Intra-regional Distance")


    plot_rsn_distributions_by_network([D_Deep, D_Mid, D_Sup], out_dir=output, name="RSN_intraLayers", 
                           array_labels=["Deep", "Middle", "Superficial"], atlas=ATLAS, y_label="Intra-regional Distance")

    plot_rsn_distributions([D_inter, D_intra], out_dir=output, name="RSN", 
                           array_labels=["Inter", "Intra"], atlas=ATLAS, y_label="Distance", share_yaxis = False)

    # plot_rsn_distributions([D_intra], out_dir=output, name="RSN_intra", 
    #                        array_labels=["Intra"], atlas=ATLAS, y_label="Intra-regional Distance")

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

    df, fig_path, csv_path = run_ff_fb_models([D_Sup, D_Mid, D_Deep], rDCM_eff, rDCM_aff, output, "ff_fb_partial_bars.svg")
    print("saved:", fig_path, "and", csv_path)



    G_hubness = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/HubnessAnalysis/gradients_Hubness_Schaefer.npy")
    G_hubness_00_standard = (G_hubness[:,0] - np.min(G_hubness[:,0])) / ((np.max(G_hubness[:,0]) - np.min(G_hubness[:,0])))
    G_hubness_01_standard = (G_hubness[:,1] - np.min(G_hubness[:,1])) / ((np.max(G_hubness[:,1]) - np.min(G_hubness[:,1])))



if __name__ == "__main__":
    main()