import laminarRestingState as lrs
import numpy as np
import os
import laminarAnalyses as laman
# import bimodularityAnalysis as bam
from numpy.linalg import inv
import scipy
from pathlib import Path
import stats



N = 400
setThresh = 0
thresholdRange = np.arange(88, 90)
num_layers=3
binarize_flag = False
subtractAverage_true = False
invert_flag = False
hcplabels = True
gradients_flag = True
# kernel = "normalized_angle"
kernel = None
largeGap = False
HCP=False
atlas = "schaefer"
n_components = 20
n_perm = 5000

BASE = Path('/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations')
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22] 
# SUBJECTS = [1, 4, 10, 13, 14, 15, 17, 18, 21] # group 0
# SUBJECTS = [2, 3, 5, 6, 7, 8, 9, 11, 12, 16, 19, 22] # group 1
# BASE = Path('/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/correlations')
# SUBJECTS = [1, 2, 4] 

# gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
# gap_dir = f'{"large" if largeGap else "small"}Gap_Glasser'
# gap_dir = "eightLayers_Schaefer"
root = BASE / gap_dir

data_dirs = [root / f'sub-LAM{s:03d}' for s in SUBJECTS]
# data_dirs = [root / f'sub-{s:02d}' for s in SUBJECTS]

output_dir = root

subs = len(data_dirs)
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer_gradients_kernelNone_21Subs_20Components_test"


if analysis=="WithinLayer_gradients_kernelNone_21Subs_20Components_test":

    def subtractAverage(adjMatrix):
        avg_matrix = np.nanmean(adjMatrix, axis=2)
        for i in range(adjMatrix.shape[2]):
            adjMatrix[:, :, i] -= avg_matrix
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)
        return adjMatrix

    def thresh_and_binarize(adj, setThresh=setThresh, binarize=False):
        N, _, num_layers = adj.shape
        A = np.empty((N, N, num_layers), dtype=float)
        percentThresh = setThresh/100

        for layer in range(num_layers):
            mag = np.abs(adj[:, :, layer])
            sorted_idx = np.argsort(mag, axis=1)  # shape (N, N)
            mask = np.ones_like(mag, dtype=bool)
            rows = np.arange(N)[:, None]
            setThresh = int(np.floor(percentThresh * N))
            mask[rows, sorted_idx[:, :setThresh]] = False

            if binarize:
                A[:, :, layer] = mask.astype(int)
            else:
                # apply mask to original signed correlations
                corr_masked = adj[:, :, layer] * mask
                eps = 1 - 1e-6
                corr_masked = np.clip(corr_masked, -eps, eps)
                z_transformed = np.arctanh(corr_masked)
                np.fill_diagonal(z_transformed, 0)
                A[:, :, layer] = z_transformed
        return A
    
    def backToR(adj):
        N, _, num_layers = adj.shape
        A = np.empty((N, N, num_layers), dtype=float)
        for layer in range(num_layers):
            r_matrix = np.clip(adj[:,:,layer], -5, 5)  # prevents tanh overflow
            z_matrix = np.tanh(r_matrix)
            np.fill_diagonal(z_matrix, 0)
            A[:,:,layer] = z_matrix
        return A


    def defineAdj(adjMatrix, interlayer_weight=1.0):
        A = np.asarray(adjMatrix)
        if A.ndim != 3 or A.shape[0] != A.shape[1]:
            raise ValueError("adjMatrix must have shape (N, N, L)")

        N, _, L = A.shape
        dtype = np.result_type(A.dtype, float)

        M = np.zeros((L * N, L * N), dtype=dtype)
        for l in range(L):
            M[l*N:(l+1)*N, l*N:(l+1)*N] = A[:, :, l]

        layer_coupling = np.ones((L, L), dtype=dtype) - np.eye(L, dtype=dtype)  # 1 off-diagonal, 0 on-diagonal
        M += interlayer_weight * np.kron(layer_coupling, np.eye(N, dtype=dtype))

        np.fill_diagonal(M, 0)
        return M

    if os.path.isfile(os.path.join(output_dir, analysis, 'FC_matrix.npy')):
        M = np.load(os.path.join(output_dir, analysis, 'FC_matrix.npy'), allow_pickle=False)
        print(f"[INFO] loaded: {os.path.join(output_dir, analysis, 'FC_matrix.npy')}  shape={M.shape}")
        print(np.max(M))
        print(np.min(M))

         
    else:
        os.makedirs(os.path.join(output_dir,analysis), exist_ok=True)
        
        adj_matrices_appended = []
        centrality_list = []
        centrality_one_hot_list = []

        for iSub, data_dir in enumerate(data_dirs):
            restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, num_layers = num_layers, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            # restStateSub.plotReliability(TR=3.3) # Run this once per fHC
            _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()


            if subtractAverage_true:
                adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
                adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
            else:
                adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=setThresh, binarize=binarize_flag)

            adj_matrices_appended.append(adjMatrix)

        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)
        print(mean_adj_matrix[1,5])
        r_matrix = backToR(mean_adj_matrix)
        print(r_matrix[1,5])


        M = defineAdj(r_matrix)
        print(np.max(M))
        print(np.min(M))
        np.save(os.path.join(output_dir, analysis, 'FC_matrix.npy'), M)


    if gradients_flag:
        
        rDCM_matrix = scipy.io.loadmat("/home/degutis/repos/rDCM/hcp_rDCM_sch400.mat", squeeze_me=True, struct_as_record=False)
        rDCM_eff = rDCM_matrix["results"].Strength_Efferent_wholeBrain   
        rDCM_aff = rDCM_matrix["results"].Strength_Afferent_wholeBrain   

        laman.plotMatrix(M, os.path.join(output_dir, analysis), "AdjMatrix.svg")
        print(np.size(M))
        restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        print(M.shape)
        G, eig = laman.run_gradient_analysis(M, n_components=n_components, kernel=kernel, random_state=13011991)
        # G, affinity_matrix = laman.run_gradient_analysis_affinity(M, n_components=n_components, random_state=13011991)

        restStateSub.plotScree(eig, analysis)

        mins = np.nanmin(G, axis=0, keepdims=True)
        maxs = np.nanmax(G, axis=0, keepdims=True)
        rng  = np.where((maxs - mins) == 0, 1, (maxs - mins))
        G_standard = (G - mins) / rng

        laman.plotMatrix(G_standard, os.path.join(output_dir, analysis), "Matrix_Gradients.svg")        
        # laman.plotMatrix(affinity_matrix, os.path.join(output_dir, analysis), "Matrix_Affinity_matrix.svg")        

        # restStateSub.eigvecs_to_nifti(G, analysis, hcp_atlas=hcplabels)

        additionalFolder = "dissimilarityGradient"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)

        D_inter, D_inter_deep, D_inter_mid, D_inter_sup = laman.inter_areal_dissimilarity(G, os.path.join(output_dir, analysis, additionalFolder), N=N, zscore_within_layer=True)
        D_intra, D_Deep, D_Mid, D_Sup = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis, additionalFolder), N=N, zscore_within_layer=True)
        D_intra_pairwise = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis, additionalFolder), N=N, zscore_within_layer=True, mode = "pairwise")

        D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
        D_intra_standard = (D_intra_pairwise - np.min(D_intra_pairwise)) / ((np.max(D_intra_pairwise) - np.min(D_intra_pairwise)))

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Deep[:,np.newaxis], "D_Deep_15_zscore", os.path.join(analysis, additionalFolder),vmin=0,vmax=0.5, atlas=atlas)        
        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Mid[:,np.newaxis], "D_Mid_15_zscore", os.path.join(analysis, additionalFolder),vmin=0,vmax=0.5, atlas=atlas)        
        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Sup[:,np.newaxis], "D_Sup_15_zscore", os.path.join(analysis, additionalFolder),vmin=0,vmax=0.5, atlas=atlas)       

        laman.plotFlatMap(D_inter, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_interFlatMap.png",HCP=HCP)
        laman.plotFlatMap(D_intra, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_intraFlatMap.png", vmin=0,vmax=0.5, HCP=HCP)

        # laman.plotFlatMap(D_inter_deep, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_interFlatMap_deep.png")
        # laman.plotFlatMap(D_inter_mid, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_interFlatMap_mid.png")
        # laman.plotFlatMap(D_inter_sup, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_interFlatMap_sup.png")

        laman.plotFlatMap(D_Deep, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_intraDeepFlatMap.png", vmin=0,vmax=0.5, HCP=HCP)
        laman.plotFlatMap(D_Mid, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_intraMidFlatMap.png", vmin=0,vmax=0.5, HCP=HCP)
        laman.plotFlatMap(D_Sup, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_intraSupFlatMap.png", vmin=0,vmax=0.5, HCP=HCP)

        restStateSub.plotScatter3DWithPlane(X=D_Sup[:,np.newaxis], Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers", 
                                                      x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_DeepMidSup_uncorrected.svg", atlas=atlas)
        
        # spin_results, spin_rsn_names = stats.spin_anova_layers(D_Deep, D_Mid, D_Sup,
        #                                     n_perm=n_perm, random_state=1)
        # for lyr, r in spin_results.items():
        #     print(f"\n{lyr}: F={r['F_obs']:.3f}, p_spin={r['p_spin']:.4g}")
        #     for k, (m, n) in enumerate(zip(r['net_means'], r['net_ns'])):
        #         print(f"  {spin_rsn_names[k]:>16s}: mean={m: .4f} (n={n})")

        # stats.save_spin_results_csv(spin_results, spin_rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"spin_anova_layers.csv"))


        # ---- Layer-wise ANOVAs with SPIN (2–4 maps) ----
        # layer_res, rsn_names = stats.layerwise_network_anova(
        #     [D_Deep, D_Mid, D_Sup],                     # or 2 or 4 maps
        #     layer_names=["Deep","Middle","Superficial"],
        #     method='spin',                               # or 'msr'
        #     n_perm=n_perm, random_state=19910113, batch_size=50, spin_unique=False
        # )
        # stats.save_layerwise_results_csv(layer_res, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"spin_anova_layers.csv"))

        # # ---- Interaction (2 or 3 layers) with MSR ----
        # res_int = stats.network_layer_interaction_general(
        #     [D_Deep, D_Mid, D_Sup],                            # or [D_deep, D_mid, D_sup]
        #     layer_names=["Deep","Middle", "Superficial"],
        #     method='spin',                               # or 'spin'
        #     n_perm=n_perm, random_state=19910114
        # )
        # stats.save_interaction_to_csv(res_int, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"IntraLayers_anova.csv"))


        # res_int2 = stats.network_layer_interaction_general(
        #     [D_inter, D_intra],                            # or [D_deep, D_mid, D_sup]
        #     layer_names=["D_inter","D_intra"],
        #     method='spin',                               # or 'spin'
        #     n_perm=n_perm, random_state=19910115
        # )
        # stats.save_interaction_to_csv(res_int2, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"InterIntra_anova.csv"))



        restStateSub.plotNetworkCentroids3D(X=D_Sup[:,np.newaxis], Y=D_Mid[:,np.newaxis], Z=D_Deep[:,np.newaxis], name=os.path.join(analysis,additionalFolder), 
                                                      x_label="Sup", y_label ="Middle", z_label="Deep", fname="Scatter_Centroid_DeepMidSup_uncorrected.svg", atlas=atlas)


        D_Deep = (D_Deep - np.min(D_Deep)) / ((np.max(D_Deep) - np.min(D_Deep)))
        D_Mid = (D_Mid - np.min(D_Mid)) / ((np.max(D_Mid) - np.min(D_Mid)))
        D_Sup = (D_Sup - np.min(D_Sup)) / ((np.max(D_Sup) - np.min(D_Sup)))

        G_SC = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/gradients_lamThick_Schaefer.npy")
        G_SC_standard = (G_SC[:,0] - np.min(G_SC[:,0])) / ((np.max(G_SC[:,0]) - np.min(G_SC[:,0])))

        G_hubness = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/HubnessAnalysis/gradients_Hubness_Schaefer.npy")
        G_hubness_00_standard = (G_hubness[:,0] - np.min(G_hubness[:,0])) / ((np.max(G_hubness[:,0]) - np.min(G_hubness[:,0])))
        G_hubness_01_standard = (G_hubness[:,1] - np.min(G_hubness[:,1])) / ((np.max(G_hubness[:,1]) - np.min(G_hubness[:,1])))

        G_bigBrain = np.loadtxt(
            "/home/degutis/repos/ENIGMA/enigmatoolbox/histology/bb_gradient_schaefer_400.csv",
            delimiter=",",
        )
        G_bigBrain_standard = (G_bigBrain - np.min(G_bigBrain)) / ((np.max(G_bigBrain) - np.min(G_bigBrain)))

        ## Inter intra

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra.svg", atlas=atlas)

        restStateSub.plotScatterCentroids(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                            x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="Scatter_InterIntra_centroid.svg", atlas=atlas)

        ## Intra of each layer compared to one another

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: deep", y_label ="Intraparcel laminar difference: middle", fname="Scatter_DeepMid.svg", atlas=atlas)

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: deep", y_label ="Intraparcel laminar difference: superficial", fname="Scatter_DeepSup.svg", atlas=atlas)

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: middle", y_label ="Intraparcel laminar difference: superficial", fname="Scatter_MidSup.svg", atlas=atlas)

        restStateSub.plotScatter3DWithPlane(X=D_Deep[:,np.newaxis], Y=D_Mid[:,np.newaxis], Z=D_Sup[:,np.newaxis], name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers", 
                                                      x_label="Deep", y_label ="Middle", z_label="Superficial", fname="Scatter_DeepMidSup_corrected.svg", atlas=atlas)



        additionalFolder = "ENIGMA"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)

        from enigmatoolbox.histology import bb_gradient_plot
        ax_intra = bb_gradient_plot(D_intra, parcellation='schaefer_400')
        ax_intra.figure.savefig(os.path.join(output_dir, analysis, additionalFolder,"Intra_bb_gradient_plot.png"), dpi=300, bbox_inches="tight")        
        
        ax_inter = bb_gradient_plot(D_inter, parcellation='schaefer_400')
        ax_inter.figure.savefig(os.path.join(output_dir, analysis, additionalFolder, "Inter_bb_gradient_plot.png"), dpi=300, bbox_inches="tight")        
        
        from enigmatoolbox.histology import economo_koskinas_spider
        # Stratify cortical atrophy based on Economo-Koskinas classes
        class_mean, ax = economo_koskinas_spider(D_intra_standard, axis_range=(np.min(D_intra_standard), np.max(D_intra_standard)))
        ax.figure.savefig(os.path.join(output_dir, analysis, additionalFolder, "Intra_vonEconomo_SpinderPlot.png"), dpi=300, bbox_inches="tight")        

        class_mean_2, ax2 = economo_koskinas_spider(D_inter_standard, axis_range=(np.min(D_inter_standard), np.max(D_inter_standard)))
        ax2.figure.savefig(os.path.join(output_dir, analysis, additionalFolder, "Inter_vonEconomo_SpinderPlot.png"), dpi=300, bbox_inches="tight")        

        ## Inter intra and ENIGMA BigBrain gradient

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_intra_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_IntraLamThick.svg")

        # res_intra_bigbrain = stats.network_layer_interaction_general(
        #     [D_intra, G_bigBrain], 
        #     layer_names=["IntraRegional","BigBrainENIGMA"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910116
        # )
        # stats.save_interaction_to_csv(res_intra_bigbrain, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"IntraBigBrain_anova.csv"))


        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], G_bigBrain_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_ENIGMABigBrain_InterLamThick.svg")

        # res_inter_bigbrain = stats.network_layer_interaction_general(
        #     [D_inter, G_bigBrain], 
        #     layer_names=["InterRegional","BigBrainENIGMA"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910117
        # )
        # stats.save_interaction_to_csv(res_inter_bigbrain, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"InterBigBrain_anova.csv"))



        additionalFolder = "BigBrain"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)

        ## Inter intra and laminar thickness gradient from my G1

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_intra_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_G1BB_IntraLamThick.svg")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="Scatter_G1BB_InterLamThick.svg")
        

        ## Intra of each layer and laminar thickness gradient from my G1

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: deep", y_label ="Laminar Thickness G1", fname="Scatter_G1BB_DeepLamThick.svg")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: middle", y_label ="Laminar Thickness G1", fname="Scatter_G1BB_MidLamThick.svg")

        # res_intraMid_bigbrainG = stats.network_layer_interaction_general(
        #     [D_Mid, G_SC_standard], 
        #     layer_names=["IntraMid","BigBrainSC"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910120
        # )
        # stats.save_interaction_to_csv(res_intraMid_bigbrainG, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"IntraMidBigBrainG_anova.csv"))


        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Sup[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="Scatter_G1BB_SupLamThick.svg")


        ## SNR

        additionalFolder = "SNR"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)
        SNR_deep = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR_layers/group_layer1_tSNR_byParcel.npy")
        SNR_mid = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR_layers/group_layer2_tSNR_byParcel.npy")
        SNR_sup = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR_layers/group_layer3_tSNR_byParcel.npy")

        laman.plotFlatMap(SNR_deep, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_layer1.png",)
        laman.plotFlatMap(SNR_mid, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_layer2.png",)
        laman.plotFlatMap(SNR_sup, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_layer3.png",)




        additionalFolder = "DCM"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)

        ## DCM 

        rDCM_grad = rDCM_eff - rDCM_aff

        laman.plotFlatMap(rDCM_eff, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_rDCM_eff.png",)
        laman.plotFlatMap(rDCM_aff, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_rDCM_aff.png",)
        laman.plotFlatMap(rDCM_grad, os.path.join(output_dir, analysis, additionalFolder), "Flatmap_rDCM_grad.png",)

        rDCM_grad = (rDCM_grad - np.min(rDCM_grad)) / ((np.max(rDCM_grad) - np.min(rDCM_grad)))
        rDCM_eff = (rDCM_eff - np.min(rDCM_eff)) / ((np.max(rDCM_eff) - np.min(rDCM_eff)))
        rDCM_aff = (rDCM_aff - np.min(rDCM_aff)) / ((np.max(rDCM_aff) - np.min(rDCM_aff)))

        restStateSub.plot_horizontal_correlation_bar([D_Sup[:,np.newaxis], D_Mid[:,np.newaxis], D_Deep[:,np.newaxis]], rDCM_grad[:,np.newaxis], os.path.join(output_dir, analysis, additionalFolder), "rDCM_bar.png", layer_names=["Sup","Middle","Deep"])

        restStateSub.plot_horizontal_correlation_bar([D_Sup[:,np.newaxis], D_Mid[:,np.newaxis], D_Deep[:,np.newaxis]], rDCM_eff[:,np.newaxis], os.path.join(output_dir, analysis, additionalFolder), "rDCM_bar_eff.png", layer_names=["Sup","Middle","Deeo"],
                                                     xlabel="Correlation with efferent effective conn. gradient",)
        restStateSub.plot_horizontal_correlation_bar([D_Sup[:,np.newaxis], D_Mid[:,np.newaxis], D_Deep[:,np.newaxis]], rDCM_aff[:,np.newaxis], os.path.join(output_dir, analysis, additionalFolder), "rDCM_bar_aff.png", layer_names=["Sup","Middle","Deep"],
                                                     xlabel="Correlation with afferent effective conn. gradient",)

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_eff[:,np.newaxis], rDCM_aff[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM efferent", y_label ="rDCM afferent", fname="Scatter_rDCMeff_aff.svg")

        # Eff
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_eff[:,np.newaxis], D_Deep[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM efferent", y_label ="Deep intraparcel difference", fname="Scatter_rDCMeff_Deep.svg")
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_eff[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM efferent", y_label ="Middle intraparcel difference", fname="Scatter_rDCMeff_Mid.svg")
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_eff[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM efferent", y_label ="Superficial intraparcel difference", fname="Scatter_rDCMeff_Sup.svg")
        # Aff
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_aff[:,np.newaxis], D_Deep[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM afferent", y_label ="Deep intraparcel difference", fname="Scatter_rDCMaff_Deep.svg")
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_aff[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM afferent", y_label ="Middle intraparcel difference", fname="Scatter_rDCMaff_Mid.svg")
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_aff[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM afferent", y_label ="Superficial intraparcel difference", fname="Scatter_rDCMaff_Sup.svg")
        # Grad
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_grad[:,np.newaxis], D_Deep[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM grad", y_label ="Deep intraparcel difference", fname="Scatter_rDCMgrad_Deep.svg")
        
        
        # res_deep_rDCMgrad = stats.network_layer_interaction_general(
        #     [D_Deep, rDCM_grad], 
        #     layer_names=["IntraDeep","rDCM"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910118
        # )
        # stats.save_interaction_to_csv(res_deep_rDCMgrad, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"Ddeep_rDCMgrad_anova.csv"))
        
        
        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_grad[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM grad", y_label ="Middle intraparcel difference", fname="Scatter_rDCMgrad_Mid.svg")



        # res_mid_rDCMgrad = stats.network_layer_interaction_general(
        #     [D_Mid, rDCM_grad], 
        #     layer_names=["IntraMid","rDCM"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910119
        # )
        # stats.save_interaction_to_csv(res_mid_rDCMgrad, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"Dmid_rDCMgrad_anova.csv"))


        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([rDCM_grad[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="rDCM grad", y_label ="Superficial intraparcel difference", fname="Scatter_rDCMgrad_Sup.svg")


        # res_sup_rDCMgrad = stats.network_layer_interaction_general(
        #     [D_Sup, rDCM_grad], 
        #     layer_names=["IntraSup","rDCM"],
        #     method='spin',
        #     n_perm=n_perm, random_state=19910121
        # )
        # stats.save_interaction_to_csv(res_sup_rDCMgrad, rsn_names, out_csv=os.path.join(output_dir, analysis, additionalFolder,"Dsup_rDCMgrad_anova.csv"))


        df, fig_path, csv_path = restStateSub.run_ff_fb_models(
            layers=[D_Sup, D_Mid, D_Deep],
            y_send=rDCM_eff,
            y_recv=rDCM_aff,
            outdir=os.path.join(output_dir, analysis, additionalFolder),
            fname="ff_fb_partial_bars.svg",
        )
        print("saved:", fig_path, "and", csv_path)


        additionalFolder = "Hubness"
        os.makedirs(os.path.join(output_dir,analysis, additionalFolder), exist_ok=True)

        ## Hubness (Renzo measure)

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([G_hubness_00_standard[:,np.newaxis], G_hubness_01_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Hubness 1", y_label ="Hubness 2", fname="Hubness12.svg")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([G_hubness_00_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Hubness 1", y_label ="Intra-distance", fname="Hubness1Intra1.svg")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([G_hubness_00_standard[:,np.newaxis], D_inter_standard[:,np.newaxis]], axis=1), name=os.path.join(analysis,additionalFolder), layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Hubness 1", y_label ="Inter-distance", fname="Hubness1Inter1.svg")

       

    else:
        eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
        restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)


    # centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
    # restStateSub.eigenvector_centrality_plot_avg(centrality, centrality_oneHot, analysis)
    
    # # restStateSub.runDegreeDistribution(adjMatrix[:,:,2], analysis, "Superficial")
    # # restStateSub.runDegreeDistribution(adjMatrix[:,:,1], analysis, "Middle")
    # # restStateSub.runDegreeDistribution(adjMatrix[:,:,0], analysis, "Deep")
    
    # restStateSub.plotConnectogram_allInOne(adjMatrix[:,:,0], adjMatrix[:,:,1], adjMatrix[:,:,2], analysis, percent=2)

    # # rich_nodes1 = restStateSub.calculateRichClub(adjMatrix[:,:,0], analysis, "Deep")
    # # rich_nodes2 = restStateSub.calculateRichClub(adjMatrix[:,:,1], analysis, "Middle")
    # # rich_nodes3 = restStateSub.calculateRichClub(adjMatrix[:,:,2], analysis, "Sup")
    # # restStateSub.plotRichClub(rich_nodes1, rich_nodes2, rich_nodes3, analysis)

    # eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
    
    # restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis)
    # restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[2, 3])
    # restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[3, 4])
    # restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[4, 5])
    # restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[5, 6])

    # restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
    # # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
    # # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    # # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)
    # restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
    # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)

    # restStateSub.plotScree(eigvals_within, analysis)
    # crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)



    # cosineSim = np.zeros((3, subs, len(thresholdRange)))
    # centrality_list_3D = np.zeros((N*num_layers, subs, len(thresholdRange)))
    # centrality_one_hot_list_3D = np.zeros((N, num_layers, subs, len(thresholdRange)))
    # modularity_across_sub = np.zeros((num_layers, subs, len(thresholdRange)))

    # phi_rc = np.zeros((num_layers, subs, len(thresholdRange)))
    # members_rc = [[ [None]*len(thresholdRange) for _ in range(subs)]
    #                 for _ in range(num_layers)]

    # for ithresh, thresh in enumerate(thresholdRange):
    #     print(thresh)
    #     for iSub, data_dir in enumerate(data_dirs):
    #         restStateSub = lrs.LaminarRestingState(data_dir, N, thresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    #         _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()

    #         if subtractAverage_true:
    #             adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
    #             adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=thresh, binarize=binarize_flag)
    #         else:
    #             adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=thresh, binarize=binarize_flag)
            
    #         M = defineAdj(adjMatrix)
    #         np.fill_diagonal(M,0)

    #         cosineSim[0, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,0], adjMatrix[:,:,1]) #Deep-Middle
    #         cosineSim[1, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,0], adjMatrix[:,:,2]) #Deep-Sup
    #         cosineSim[2, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,1], adjMatrix[:,:,2]) #Middle-Sup

    #         # for each layer separately
    #         # for layer_idx in range(num_layers):
    #         #     conn = adjMatrix[:, :, layer_idx]
                
    #         #     phi_auc, members = restStateSub.rich_club_sweep(
    #         #         conn,
    #         #         deg_cutoff_percentile=90,
    #         #         normalized=True,
    #         #         seed=thresh*(layer_idx+1)
    #         #     )
    #         #     phi_rc[layer_idx, iSub, ithresh] = phi_auc
    #         #     members_rc[layer_idx][iSub][ithresh] = members

    #         #     modularity_across_sub[layer_idx, iSub, ithresh] = restStateSub.modularity(conn)

    #         if binarize_flag:
    #             centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
    #             centrality_list_3D[:, iSub, ithresh] = centrality
    #             centrality_one_hot_list_3D[:, :, iSub, ithresh] = centrality_oneHot
    #         else:
    #             centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M, weight="weight")
    #             centrality_list_3D[:, iSub, ithresh] = centrality
    #             centrality_one_hot_list_3D[:, :, iSub, ithresh] = centrality_oneHot


    # freq_rich_club = np.zeros((N, num_layers, subs))
    # for layer_idx in range(num_layers):
    #     for iSub in range(subs):
    #         members_over_thresh = members_rc[layer_idx][iSub]  # list of T lists
    #         _, freq_rich_club[:,layer_idx, iSub] = restStateSub.most_common_members(members_over_thresh, N, min_frac=0.8)


    # restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    # restStateSub.__plot_on_mmhcp_surface_multipleLayers__(np.mean(freq_rich_club, axis=-1), "RichClub", analysis, vmin=0, vmax=1)        
    
    # centrality_AUC = np.trapz(centrality_list_3D, x=thresholdRange, axis=-1)
    # centrality_one_hot_AUC = np.trapz(centrality_one_hot_list_3D, x=thresholdRange, axis=-1)
    # restStateSub.eigenvector_centrality_plot(centrality_AUC, centrality_one_hot_AUC, analysis)
    # laman.plot_cosine_similarity(cosineSim, output_dir, analysis, thresholds=thresholdRange)
    # laman.plot_cosine_similarity(modularity_across_sub, output_dir, analysis, thresholds=thresholdRange, labels = ["Deep","Middle","Superficial"], extraName="ModularityAcrossThresh", ylabel="Modularity")



elif analysis=="FullLayer_kernelNone_sparsity01":

    def runPrecisionMatrix(adjMatrix, gamma):
        p = adjMatrix.shape[0]
        return inv(adjMatrix + gamma * np.eye(p))

    def thresh_and_binarize(adj, setThresh=setThresh, binarize=False):
        N, _ = adj.shape
        A = np.empty((N, N), dtype=float)
        percentThresh = setThresh/100

        mag = np.abs(adj)
        sorted_idx = np.argsort(mag, axis=1)  # shape (N, N)
        mask = np.ones_like(mag, dtype=bool)
        rows = np.arange(N)[:, None]
        setThresh = int(np.floor(percentThresh * N))
        mask[rows, sorted_idx[:, :setThresh]] = False

        if binarize:
            A = mask.astype(int)
        else:
            # apply mask to original signed correlations
            corr_masked = adj * mask
            np.fill_diagonal(corr_masked, 0)
            A = corr_masked
        return A

    if os.path.isfile(os.path.join(output_dir, analysis, 'FC_matrix.npy')):
        M = np.load(os.path.join(output_dir, analysis, 'FC_matrix.npy'), allow_pickle=False)
        print(f"[INFO] loaded: {os.path.join(output_dir, analysis, 'FC_matrix.npy')}  shape={M.shape}")
         
    else:
        os.makedirs(os.path.join(output_dir,analysis), exist_ok=True)

        adj_matrices = []
        for data_dir in data_dirs:
            restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns()
            adj_matrices.append(adj_matrix_within_corr)

        adj_matrices_3d = np.stack(adj_matrices, axis=2)
        adjMatrix_mean = np.mean(adj_matrices_3d, axis=2)

        if invert_flag:
            adjMatrix = runPrecisionMatrix(adjMatrix_mean, gamma=0.2)
        else:
            M = adjMatrix_mean
            # thresh_and_binarize(adj_matrix_within_corr, setThresh=0, binarize=binarize_flag)

        if gradients_flag:
            
            rDCM_matrix = scipy.io.loadmat("/home/degutis/repos/rDCM/hcp_rDCM_sch400.mat", squeeze_me=True, struct_as_record=False)
            rDCM_eff = rDCM_matrix["results"].Strength_Efferent_wholeBrain   
            rDCM_aff = rDCM_matrix["results"].Strength_Afferent_wholeBrain   

            laman.plotMatrix(M, os.path.join(output_dir, analysis), "AdjMatrix.svg")

            restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            # G = laman.run_gradient_analysis(M, n_components=15, kernel=None, random_state=13011991)
            G, affinity_matrix = laman.run_gradient_analysis_affinity(M, n_components=15, kernel=None, sparsity=0.9, random_state=13011991)

            mins = np.nanmin(G, axis=0, keepdims=True)
            maxs = np.nanmax(G, axis=0, keepdims=True)
            rng  = np.where((maxs - mins) == 0, 1, (maxs - mins))
            G_standard = (G - mins) / rng

            laman.plotMatrix(G_standard, os.path.join(output_dir, analysis), "Gradients.svg")        
            laman.plotMatrix(affinity_matrix, os.path.join(output_dir, analysis), "Affinity_matrix.svg")        

            # restStateSub.eigvecs_to_nifti(G, analysis, hcp_atlas=hcplabels)
            D_inter, D_inter_deep, D_inter_mid, D_inter_sup = laman.inter_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=N, zscore_within_layer=True)
            D_intra, D_Deep, D_Mid, D_Sup = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=N, zscore_within_layer=True)
            D_intra_pairwise = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=N, zscore_within_layer=True, mode = "pairwise")

            D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
            D_intra_standard = (D_intra_pairwise - np.min(D_intra_pairwise)) / ((np.max(D_intra_pairwise) - np.min(D_intra_pairwise)))

            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Deep[:,np.newaxis], "D_Deep_15_zscore", analysis, atlas=atlas)        
            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Mid[:,np.newaxis], "D_Mid_15_zscore", analysis, atlas=atlas)        
            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Sup[:,np.newaxis], "D_Sup_15_zscore", analysis, atlas=atlas)        

            laman.plotFlatMap(D_inter, os.path.join(output_dir, analysis), "D_interFlatMap.png")
            laman.plotFlatMap(D_intra, os.path.join(output_dir, analysis), "D_intraFlatMap.png")

            laman.plotFlatMap(D_inter_deep, os.path.join(output_dir, analysis), "D_interFlatMap_deep.png")
            laman.plotFlatMap(D_inter_mid, os.path.join(output_dir, analysis), "D_interFlatMap_mid.png")
            laman.plotFlatMap(D_inter_sup, os.path.join(output_dir, analysis), "D_interFlatMap_sup.png")

            laman.plotFlatMap(D_Deep, os.path.join(output_dir, analysis), "D_intraDeepFlatMap.png")
            laman.plotFlatMap(D_Mid, os.path.join(output_dir, analysis), "D_intraMidFlatMap.png")
            laman.plotFlatMap(D_Sup, os.path.join(output_dir, analysis), "D_intraSupFlatMap.png")

            D_Deep = (D_Deep - np.min(D_Deep)) / ((np.max(D_Deep) - np.min(D_Deep)))
            D_Mid = (D_Mid - np.min(D_Mid)) / ((np.max(D_Mid) - np.min(D_Mid)))
            D_Sup = (D_Sup - np.min(D_Sup)) / ((np.max(D_Sup) - np.min(D_Sup)))

            if plotD:
                restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_inter[:,np.newaxis], "D_inter_15_zscore", analysis, atlas=atlas)        
                restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_intra[:,np.newaxis], "D_intra_15_zscore", analysis, atlas=atlas)        

                restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_intra_pairwise[:,np.newaxis], "D_intra_pairwise_15_zscore", analysis)        
                restStateSub.plotTwoDimEmbedding(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference")

            G_SC = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/gradients_lamThick_Schaefer.npy")
            G_SC_standard = (G_SC[:,0] - np.min(G_SC[:,0])) / ((np.max(G_SC[:,0]) - np.min(G_SC[:,0])))


            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], rDCM_eff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Deep", y_label ="rDCM efferent", fname="rDCM_Deep_Eff.png")
            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], rDCM_eff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Mid", y_label ="rDCM efferent", fname="rDCM_Mid_Eff.png")
            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Sup[:,np.newaxis], rDCM_eff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Sup", y_label ="rDCM efferent", fname="rDCM_Sup_Eff.png")


            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], rDCM_aff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Deep", y_label ="rDCM afferent", fname="rDCM_Deep_Aff.png")
            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], rDCM_aff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Mid", y_label ="rDCM afferent", fname="rDCM_Mid_Aff.png")
            # restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Sup[:,np.newaxis], rDCM_aff[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
            #                                               x_label="Intraparcel laminar difference: Sup", y_label ="rDCM afferent", fname="rDCM_Sup_Aff.png")

            rDCM_grad = rDCM_eff - rDCM_aff
            restStateSub.plot_horizontal_correlation_bar([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], rDCM_grad[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_bar.png", layer_names=["Deep","Middle","Sup"])

            restStateSub.plot_horizontal_correlation_bar([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], rDCM_eff[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_bar_eff.png", layer_names=["Deep","Middle","Sup"],
                                                        xlabel="Correlation with efferent effective conn. gradient",)
            restStateSub.plot_horizontal_correlation_bar([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], rDCM_aff[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_bar_aff.png", layer_names=["Deep","Middle","Sup"],
                                                        xlabel="Correlation with afferent effective conn. gradient",)


            restStateSub.plot_horizontal_correlation_bar([D_inter_deep[:,np.newaxis], D_inter_mid[:,np.newaxis], D_inter_sup[:,np.newaxis]], rDCM_grad[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_inter_bar.png", layer_names=["Deep","Middle","Sup"],
                                                        title="Effective connectivity and interlaminar difference gradients")

            restStateSub.plot_horizontal_correlation_bar([D_inter_deep[:,np.newaxis], D_inter_mid[:,np.newaxis], D_inter_sup[:,np.newaxis]], rDCM_eff[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_inter_bar_eff.png", layer_names=["Deep","Middle","Sup"],
                                                        title="Effective connectivity and interlaminar difference gradients", xlabel="Correlation with efferent effective conn. gradient",)
            restStateSub.plot_horizontal_correlation_bar([D_inter_deep[:,np.newaxis], D_inter_mid[:,np.newaxis], D_inter_sup[:,np.newaxis]], rDCM_aff[:,np.newaxis], os.path.join(output_dir, analysis), "rDCM_inter_bar_aff.png", layer_names=["Deep","Middle","Sup"],
                                                        title="Effective connectivity and interlaminar difference gradients", xlabel="Correlation with afferent effective conn. gradient",)


            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_intra_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Intraparcel laminar difference", y_label ="Laminar Thickness G1", fname="IntraLamThick.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="InterLamThick.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="InterIntra.svg")


            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Intraparcel laminar difference: deep", y_label ="Laminar Thickness G1", fname="DeepLamThick.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Intraparcel laminar difference: middle", y_label ="Laminar Thickness G1", fname="MidLamThick.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Sup[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Intraparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="SupLamThick.svg")


            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Interparcel laminar difference: deep", y_label ="Interparcel laminar difference: middle", fname="DeepMid.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Interparcel laminar difference: deep", y_label ="Interparcel laminar difference: superficial", fname="DeepSup.svg")

            restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                        x_label="Interparcel laminar difference: middle", y_label ="Interparcel laminar difference: superficial", fname="MidSup.svg")

        else:

            # bicomLam = bam.BimodularityAnalysis(adjMatrix, output_dir, N, setThresh, analysis, num_layers=num_layers, atlas_dir ="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            # sending_communities, receiving_communities, _, _ = bicomLam.runBimod(analysis, vector_id_max=4, n_kmeans=10, startFrom=0)
            # bicomLam.plotBicoms(sending_communities, analysis, "Sending")
            # bicomLam.plotBicoms(receiving_communities, analysis, "Receiving")
            restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")

            # restStateSub.plotConnectogram(adjMatrix[:360,:360], analysis, "Deep-Deep", color="red", percent=1)
            # restStateSub.plotConnectogram(adjMatrix[:360,360:720], analysis, "Deep-Middle", color="green", percent=1)
            # restStateSub.plotConnectogram(adjMatrix[:360,720:1080], analysis, "Deep-Sup", color="blue", percent=1)
            # restStateSub.plotConnectogram(adjMatrix[360:720,360:720], analysis, "Middle-Middle", color="red", percent=1)
            # restStateSub.plotConnectogram(adjMatrix[360:720,720:1080], analysis, "Middle-Sup", color="green", percent=1)
            # restStateSub.plotConnectogram(adjMatrix[720:1080,720:1080], analysis, "Sup-Sup", color="red", percent=1)

            # rich_nodes1 = restStateSub.calculateRichClub(adjMatrix[:360,:360], analysis, "Deep-Deep")
            # rich_nodes2 = restStateSub.calculateRichClub(adjMatrix[:360,360:720], analysis, "Deep-Middle")
            # rich_nodes3 = restStateSub.calculateRichClub(adjMatrix[:360,720:1080], analysis, "Deep-Sup")
            # rich_nodes4 = restStateSub.calculateRichClub(adjMatrix[360:720,360:720], analysis, "Middle-Middle")
            # rich_nodes5 = restStateSub.calculateRichClub(adjMatrix[360:720,720:1080], analysis, "Middle-Sup")
            # rich_nodes6 = restStateSub.calculateRichClub(adjMatrix[720:1080,720:1080], analysis, "Sup-Sup")

            M = adjMatrix_mean
            eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
            gradients = laman.runGradientAnalysis(M)
            restStateSub.eigvecs_to_nifti(gradients, analysis, hcp_atlas=hcplabels)

            restStateSub.plotTwoDimEmbedding_byNetwork(gradients, analysis)
            restStateSub.plotTwoDimEmbedding_byNetwork(gradients, analysis, eigvecs_to_plot=[2, 3])
            restStateSub.plotTwoDimEmbedding_byNetwork(gradients, analysis, eigvecs_to_plot=[3, 4])
            restStateSub.plotTwoDimEmbedding_byNetwork(gradients, analysis, eigvecs_to_plot=[4, 5])
            restStateSub.plotTwoDimEmbedding_byNetwork(gradients, analysis, eigvecs_to_plot=[5, 6])

            restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
            restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
            crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
            restStateSub.plotScree(eigvals_within, analysis)
            # restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)

elif analysis=="LaggedLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir ="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        adj_M = restStateSub.lagged_multilayer_fc()
        adj_matrices.append(adj_M)

    adj_matrices_3d = np.stack(adj_matrices, axis=2)
    adjMatrix = np.mean(adj_matrices_3d, axis=2)

    # if binarize:
    #     # threshold = np.percentile(np.abs(adjMatrix), setThresh)
    #     # M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
    #     threshold = np.percentile(adjMatrix, setThresh)
    #     M = np.where(adjMatrix >= threshold, adjMatrix, 0)
    #     M[M != 0] = 1
    # else:
    M = adjMatrix
    #     M[M != 0] = 1
    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")

    eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
    restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
    restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
    crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)
    restStateSub.plotScree(eigvals_within, analysis)

    restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)





elif analysis=="SingleLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns()
        adj_matrices.append(adj_matrix_within_corr)

    adj_matrices_3d = np.stack(adj_matrices, axis=2)
    adjMatrix = np.mean(adj_matrices_3d, axis=2)

    if binarize:
        threshold = np.percentile(np.abs(adjMatrix), setThresh)
        M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
    else:
        M = adjMatrix

    row_start = (row_idx - 1) * N
    row_end = row_start + N
    col_start = (col_idx - 1) * N
    col_end = col_start + N

    M_single = M[row_start:row_end, col_start:col_end]    

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, num_layers=1, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M_single, analysis, num_components=20, convert_to_binary=False, full=True)
    restStateSub.plotScree(eigvals_full, analysis)
    crossingsFull = restStateSub.run_plot_zeroCrossings(M, eigvecs_full, analysis)
    restStateSub.eigvecs_to_nifti(eigvecs_full, analysis, hcp_atlas=hcplabels)

elif analysis=="SingleLayerComparison":
    
    eigvecs_list = []
    eigvalue_list = []
    source_info = [] 

    eigenvalue_threshold = 0.3
    cluster_threshold = 0.3

    for lc in layerComparisons:
        row_idx, col_idx = lc
        adj_matrices = []

        for data_dir in data_dirs:
            restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, 
                                                   atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns()
            adj_matrices.append(adj_matrix_within_corr)

        adj_matrices_3d = np.stack(adj_matrices, axis=2)
        adjMatrix = np.mean(adj_matrices_3d, axis=2)

        if binarize:
            threshold = np.percentile(np.abs(adjMatrix), setThresh)
            M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
        elif setThresh > 0:
            threshold = np.percentile(np.abs(adjMatrix), setThresh)
            M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
        else:
            M = adjMatrix

        row_start = (row_idx - 1) * N
        row_end = row_start + N
        col_start = (col_idx - 1) * N
        col_end = col_start + N

        M_single = M[row_start:row_end, col_start:col_end]    

        restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, num_layers=1, 
                                               atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        
        eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M_single, analysis, num_components=20, 
                                                                        convert_to_binary=False, full=True, addName=f"_{lc[0]}_{lc[1]}")
        for i in range(eigvecs_full.shape[1]):
            eigvec = eigvecs_full[:, i]
            eigvecs_list.append(eigvec/np.linalg.norm(eigvec))
            source_info.append((row_idx, col_idx, i))
            eigvalue_list.append(eigvals_full[i])   

    cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)


    for cluster_id, indices in cluster_groups.items():
        
        if len(indices) == 1:
            continue
        eigvecs_to_plot = [eigvecs_list[i] for i in indices]
        meta = [source_info[i] for i in indices]
        titles = [f"(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        # Build filename from source info
        name_str = "-".join([f"r{r}_c{c}_e{e}" for (r, c, e) in meta])
        eig_label = f"{name_str}"

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(Xp, eig_label, name="SingleLayerComparison", titles=titles)


    for cluster_id, indices in cluster_groups.items():
        if len(indices) > 1:
            continue  # Only consider singleton clusters

        i = indices[0]
        if not(0 < eigvalue_list[i] < eigenvalue_threshold):
            continue

        r, c, _ = source_info[i]
        v_i = eigvecs_list[i] / np.linalg.norm(eigvecs_list[i])

        # Compare against others from the same region-pair
        similar_found = False
        for j, (rj, cj, _) in enumerate(source_info):
            if (rj, cj) == (r, c) and j != i:
                eigval_j = eigvalue_list[j]
                if not (0 < eigval_j < eigenvalue_threshold):
                    continue
                
                v_j = eigvecs_list[j] / np.linalg.norm(eigvecs_list[j])
                similarity = np.abs(np.dot(v_i, v_j))
                if similarity >= (1 - cluster_threshold):
                    similar_found = True
                    break

        if similar_found:
            continue

        # Passed distinctness check → plot
        eigvecs_to_plot = [eigvecs_list[i]]
        meta = [source_info[i]]
        titles = [f"Distinct_(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        name_str = f"r{r}_c{c}_e{meta[0][2]}"
        eig_label = name_str

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
            Xp, eig_label, name="SingleLayerComparison", titles=titles
        )