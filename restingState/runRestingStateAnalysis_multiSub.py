import laminarRestingState as lrs
import numpy as np
import os
import laminarAnalyses as laman
import bimodularityAnalysis as bam
from numpy.linalg import inv


N = 360
setThresh = 5
thresholdRange = np.arange(75, 100)
num_layers=3
binarize_flag = False
subtractAverage_true = False
invert_flag = False
hcplabels = True

# data_dirs = ['../../highRes_resting/derivatives/correlations/sub-01/Multiple_Runs/smallGap', '../../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs/smallGap', '../../highRes_resting/derivatives/correlations/sub-03/Multiple_Runs/smallGap']
data_dirs = ['../../highRes_resting/derivatives/correlations/sub-50/Multiple_Runs/smallGap']
subs = len(data_dirs)
output_dir = '../../highRes_resting/derivatives/correlations/sub-50/smallGap'
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer"

cluster_threshold = 0.3
eigenvalue_threshold = 0.7

row_idx, col_idx = 2,3
layerComparisons = [(1,1), (1,3), (3,2)]


if analysis=="WithinLayer":

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


    def thresh_and_binarize(adj, setThresh, binarize=False):
        N, _, num_layers = adj.shape
        A = np.empty((N, N, num_layers), dtype=float)

        for layer in range(num_layers):
            # 1) magnitude matrix for ranking
            mag = np.abs(adj[:, :, layer])
            # 2) argsort each row; mask p smallest
            sorted_idx = np.argsort(mag, axis=1)  # shape (N, N)
            mask = np.ones_like(mag, dtype=bool)
            rows = np.arange(N)[:, None]
            mask[rows, sorted_idx[:, :setThresh]] = False

            if binarize:
                # keep only topology
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
    
    adj_matrices_appended = []
    centrality_list = []
    centrality_one_hot_list = []

    for iSub, data_dir in enumerate(data_dirs):
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-50/HCP-MMP1_in-func.nii")
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


    # cosineSim = np.zeros((3, subs, len(thresholdRange)))
    # centrality_list_3D = np.zeros((N*num_layers, subs, len(thresholdRange)))
    # centrality_one_hot_list_3D = np.zeros((N, num_layers, subs, len(thresholdRange)))
    # modularity_across_sub = np.zeros((num_layers, subs, len(thresholdRange)))

    # phi_rc = np.zeros((num_layers, subs, len(thresholdRange)))
    # members_rc = [[ [None]*len(thresholdRange) for _ in range(subs)]
    #                 for _ in range(num_layers)]

    # for ithresh, thesh in enumerate(thresholdRange):
    #     print(thesh)
    #     for iSub, data_dir in enumerate(data_dirs):
    #         restStateSub = lrs.LaminarRestingState(data_dir, N, thesh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-50/HCP-MMP1_in-func.nii")
    #         _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()

    #         if subtractAverage_true:
    #             adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
    #             adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=thesh, binarize=binarize_flag)
    #         else:
    #             adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=thesh, binarize=binarize_flag)
            
    #         M = defineAdj(adjMatrix)

    #         cosineSim[0, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,0], adjMatrix[:,:,1]) #Deep-Middle
    #         cosineSim[1, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,0], adjMatrix[:,:,2]) #Deep-Sup
    #         cosineSim[2, iSub, ithresh] = laman.cosine_similarity_upper(adjMatrix[:,:,1], adjMatrix[:,:,2]) #Middle-Sup

    #         # for each layer separately
    #         for layer_idx in range(num_layers):
    #             conn = adjMatrix[:, :, layer_idx]
                
    #             phi_auc, members = restStateSub.rich_club_sweep(
    #                 conn,
    #                 deg_cutoff_percentile=90,
    #                 normalized=True,
    #                 seed=thesh*(layer_idx+1)
    #             )
    #             phi_rc[layer_idx, iSub, ithresh] = phi_auc
    #             members_rc[layer_idx][iSub][ithresh] = members

    #             modularity_across_sub[layer_idx, iSub, ithresh] = restStateSub.modularity(conn)

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


    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    # restStateSub.__plot_on_mmhcp_surface_multipleLayers__(np.mean(freq_rich_club, axis=-1), "RichClub", analysis, vmin=0, vmax=1)        
    
    # centrality_AUC = np.trapz(centrality_list_3D, x=thresholdRange, axis=-1)
    # centrality_one_hot_AUC = np.trapz(centrality_one_hot_list_3D, x=thresholdRange, axis=-1)
    # restStateSub.eigenvector_centrality_plot(centrality_AUC, centrality_one_hot_AUC, analysis)
    # laman.plot_cosine_similarity(cosineSim, output_dir, analysis, thresholds=thresholdRange)
    # laman.plot_cosine_similarity(modularity_across_sub, output_dir, analysis, thresholds=thresholdRange, labels = ["Deep","Middle","Superficial"], extraName="ModularityAcrossThresh", ylabel="Modularity")


    ## Averaged across subs

    adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
    mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)

    # if subtractAverage_true:
    #     adjMatrix_SA = subtractAverage(mean_adj_matrix)
    #     adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
    # else:
    #     adjMatrix = thresh_and_binarize(mean_adj_matrix, setThresh=setThresh, binarize=binarize_flag)

    # M = defineAdj(adjMatrix)
    M = defineAdj(mean_adj_matrix)

    gradients = laman.runGradientAnalysis(M)
    print(gradients.shape)
    eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)

    restStateSub.eigvecs_to_nifti(gradients, analysis, hcp_atlas=hcplabels)



    # np.save(os.path.join(output_dir, analysis, 'FC_matrix.npy'), M)

    # centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
    # restStateSub.eigenvector_centrality_plot_avg(centrality, centrality_oneHot, analysis)
    
    # restStateSub.runDegreeDistribution(adjMatrix[:,:,2], analysis, "Superficial")
    # restStateSub.runDegreeDistribution(adjMatrix[:,:,1], analysis, "Middle")
    # restStateSub.runDegreeDistribution(adjMatrix[:,:,0], analysis, "Deep")
    
    # restStateSub.plotConnectogram_allInOne(adjMatrix[:,:,0], adjMatrix[:,:,1], adjMatrix[:,:,2], analysis, percent=5)

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
    # # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    # # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)

    # restStateSub.plotScree(eigvals_within, analysis)
    # crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    # restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)





elif analysis=="FullLayer":

    def runPrecisionMatrix(adjMatrix, gamma):
        p = adjMatrix.shape[0]
        return inv(adjMatrix + gamma * np.eye(p))

    adj_matrices = []
    setThresh=0
    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns() # do I z-transform the full matrix?
        adj_matrices.append(adj_matrix_within_corr)

    adj_matrices_3d = np.stack(adj_matrices, axis=2)
    adjMatrix = np.mean(adj_matrices_3d, axis=2)

    if invert_flag:
        adjMatrix = runPrecisionMatrix(adjMatrix, gamma=0.2)


    # bicomLam = bam.BimodularityAnalysis(adjMatrix, output_dir, N, setThresh, analysis, num_layers=num_layers, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    # sending_communities, receiving_communities, _, _ = bicomLam.runBimod(analysis, vector_id_max=4, n_kmeans=10, startFrom=0)
    # bicomLam.plotBicoms(sending_communities, analysis, "Sending")
    # bicomLam.plotBicoms(receiving_communities, analysis, "Receiving")

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")

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

    M = adjMatrix
    eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)

    restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis)
    restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[2, 3])
    restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[3, 4])
    restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[4, 5])
    restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[5, 6])


    restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
    restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
    # crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)
    restStateSub.plotScree(eigvals_within, analysis)

    # restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    # restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)

    # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
    # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)

elif analysis=="LaggedLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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
    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")

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
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, num_layers=1, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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
                                                   atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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
                                               atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        
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