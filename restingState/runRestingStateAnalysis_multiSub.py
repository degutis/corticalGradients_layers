import laminarRestingState as lrs
import numpy as np
import os
import laminarAnalyses as laman
import bimodularityAnalysis as bam
from numpy.linalg import inv


N = 360
setThresh = 0
thresholdRange = np.arange(88, 90)
num_layers=3
binarize_flag = False
subtractAverage_true = False
invert_flag = False
hcplabels = True
gradients_flag = True
plotD = True

data_dirs = ['/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM001', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM002', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM003', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM004',
             '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM005', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM006', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM009', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/sub-LAM011']

subs = len(data_dirs)
output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Gifti/'
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer_gradients_noThresh_DistanceMetric"

cluster_threshold = 0.3
eigenvalue_threshold = 0.7

row_idx, col_idx = 2,3
layerComparisons = [(1,1), (1,3), (3,2)]


if analysis=="WithinLayer_gradients_noThresh_DistanceMetric":

    def subtractAverage(adjMatrix):
        avg_matrix = np.nanmean(adjMatrix, axis=2)
        for i in range(adjMatrix.shape[2]):
            adjMatrix[:, :, i] -= avg_matrix
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)
        return adjMatrix

    def thresh_and_binarize_old(adj, setThresh=setThresh, binarize=binarize_flag):
        A = np.empty((N,N,num_layers))
        for layer in range(num_layers):
            mat = np.abs(adj[:,:,layer])
            cutoff = np.percentile(mat, setThresh)
            if binarize:
                A[:,:,layer] = (mat > cutoff).astype(int)
            else:
                A[:,:,layer] = np.where(mat > cutoff, mat, 0)
        return A


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
                np.fill_diagonal(corr_masked, 0)
                # eps = 1 - 1e-6
                # corr_masked = np.clip(corr_masked, -eps, eps)
                # Fisher r-to-z transform; at zeros gives 0
                A[:, :, layer] = corr_masked

        return A
        
    def defineAdj(adjMatrix):
        I_N = np.eye(N)
        M = np.block([
            [adjMatrix[:,:,0], I_N, I_N],
            [I_N, adjMatrix[:,:,1], I_N],
            [I_N, I_N, adjMatrix[:,:,2]]
        ])
        np.fill_diagonal(M, 0)
        return M

    if os.path.isfile(os.path.join(output_dir, analysis, 'FC_matrix.npy')):
        M = np.load(os.path.join(output_dir, analysis, 'FC_matrix.npy'), allow_pickle=False)
        print(f"[INFO] loaded: {os.path.join(output_dir, analysis, 'FC_matrix.npy')}  shape={M.shape}")
         
    else:
        os.makedirs(os.path.join(output_dir,analysis), exist_ok=True)
        
        adj_matrices_appended = []
        centrality_list = []
        centrality_one_hot_list = []

        for iSub, data_dir in enumerate(data_dirs):
            restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
            # restStateSub.plotReliability(TR=3.3) # Run this once per subject
            _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()
            adj_matrices_appended.append(adj_matrix_within_corr)

            if subtractAverage_true:
                adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
                adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
            else:
                adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=setThresh, binarize=binarize_flag)

            M = defineAdj(adjMatrix)

            if binarize_flag:
                centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
            else:
                centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M, weight="weight")

            centrality_list.append(centrality)
            centrality_one_hot_list.append(centrality_oneHot)

        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)

        if subtractAverage_true:
            adjMatrix_SA = subtractAverage(mean_adj_matrix)
            adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
        else:
            adjMatrix = thresh_and_binarize(mean_adj_matrix, setThresh=setThresh, binarize=binarize_flag)

        M = defineAdj(adjMatrix)
        np.save(os.path.join(output_dir, analysis, 'FC_matrix.npy'), M)


    if gradients_flag:
        
        laman.plotMatrix(M, os.path.join(output_dir, analysis), "AdjMatrix.svg")

        restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        G = laman.run_gradient_analysis(M, n_components=15, random_state=13011991)

        mins = np.nanmin(G, axis=0, keepdims=True)
        maxs = np.nanmax(G, axis=0, keepdims=True)
        rng  = np.where((maxs - mins) == 0, 1, (maxs - mins))
        G_standard = (G - mins) / rng

        laman.plotMatrix(G_standard, os.path.join(output_dir, analysis), "Gradients.svg")        

        # restStateSub.eigvecs_to_nifti(G, analysis, hcp_atlas=hcplabels)
        D_inter = laman.inter_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=360, zscore_within_layer=True)
        D_intra, D_Deep, D_Mid, D_Sup = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=360, zscore_within_layer=True)
        D_intra_pairwise = laman.intra_areal_dissimilarity(G, os.path.join(output_dir, analysis), N=360, zscore_within_layer=True, mode = "pairwise")

        D_inter_standard = (D_inter - np.min(D_inter)) / ((np.max(D_inter) - np.min(D_inter)))
        D_intra_standard = (D_intra_pairwise - np.min(D_intra_pairwise)) / ((np.max(D_intra_pairwise) - np.min(D_intra_pairwise)))

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Deep[:,np.newaxis], "D_Deep_15_zscore", analysis)        
        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Mid[:,np.newaxis], "D_Mid_15_zscore", analysis)        
        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_Sup[:,np.newaxis], "D_Sup_15_zscore", analysis)        

        D_Deep = (D_Deep - np.min(D_Deep)) / ((np.max(D_Deep) - np.min(D_Deep)))
        D_Mid = (D_Mid - np.min(D_Mid)) / ((np.max(D_Mid) - np.min(D_Mid)))
        D_Sup = (D_Sup - np.min(D_Sup)) / ((np.max(D_Sup) - np.min(D_Sup)))


        if plotD:
            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_inter[:,np.newaxis], "D_inter_15_zscore", analysis)        
            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_intra[:,np.newaxis], "D_intra_15_zscore", analysis)        

            restStateSub.__plot_on_mmhcp_surface_multipleLayers__(D_intra_pairwise[:,np.newaxis], "D_intra_pairwise_15_zscore", analysis)        
            restStateSub.plotTwoDimEmbedding(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference")

        G_SC = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/gradients_lamThick.npy")
        G_SC_standard = (G_SC[:,0] - np.min(G_SC[:,0])) / ((np.max(G_SC[:,0]) - np.min(G_SC[:,0])))

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_intra_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference", y_label ="Laminar Thickness G1", fname="IntraLamThick.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference", y_label ="Laminar Thickness G1", fname="InterLamThick.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_inter_standard[:,np.newaxis], D_intra_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference", y_label ="Intraparcel laminar difference", fname="InterIntra.png")


        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: deep", y_label ="Laminar Thickness G1", fname="DeepLamThick.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: middle", y_label ="Laminar Thickness G1", fname="MidLamThick.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Sup[:,np.newaxis], G_SC_standard[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Intraparcel laminar difference: superficial", y_label ="Laminar Thickness G1", fname="SupLamThick.png")



        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Mid[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference: deep", y_label ="Interparcel laminar difference: middle", fname="DeepMid.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Deep[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference: deep", y_label ="Interparcel laminar difference: superficial", fname="DeepSup.png")

        restStateSub.plotScatterWithGlobalCorrelation(np.concatenate([D_Mid[:,np.newaxis], D_Sup[:,np.newaxis]], axis=1), name=analysis, layer_labels="AcrossLayers",eigvecs_to_plot=(0, 1), 
                                                      x_label="Interparcel laminar difference: middle", y_label ="Interparcel laminar difference: superficial", fname="MidSup.png")




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



elif analysis=="FullLayer":

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
        adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=90, binarize=binarize_flag)


    # bicomLam = bam.BimodularityAnalysis(adjMatrix, output_dir, N, setThresh, analysis, num_layers=num_layers, atlas_dir ="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    # sending_communities, receiving_communities, _, _ = bicomLam.runBimod(analysis, vector_id_max=4, n_kmeans=10, startFrom=0)
    # bicomLam.plotBicoms(sending_communities, analysis, "Sending")
    # bicomLam.plotBicoms(receiving_communities, analysis, "Receiving")
    print(output_dir)
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
    print(gradients.shape)
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
    print(adjMatrix)

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