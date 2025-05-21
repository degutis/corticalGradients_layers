import laminarRestingState as lrs
import numpy as np
import os
import laminarAnalyses as laman

N = 360
setThresh = 90
num_layers=3
binarize_true = True
subtractAverage_true = True
hcplabels = True

data_dirs = ['../../highRes_resting/derivatives/correlations/sub-01/Multiple_Runs/largeGap', '../../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs/largeGap', '../../highRes_resting/derivatives/correlations/sub-03/Multiple_Runs/largeGap']
output_dir = '../../highRes_resting/derivatives/correlations/combo/largeGap'
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer"

cluster_threshold = 0.3
eigenvalue_threshold = 0.7

row_idx, col_idx = 2,3
layerComparisons = [(1,1), (1,3), (3,2)]


if analysis=="WithinLayer":

    def subtractAverage(adjMatrix):
        print(adjMatrix.shape)
        avg_matrix = np.nanmean(adjMatrix, axis=2)
        for i in range(adjMatrix.shape[2]):
            adjMatrix[:, :, i] -= avg_matrix
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)
        return adjMatrix

    def binarize(adj, setThresh=setThresh):
        A = np.empty((N,N,num_layers))
        for layer in range(num_layers):
            currentLayer = adj[:,:,layer]
            threshold = np.percentile(currentLayer, setThresh)
            adj_matrix = np.where(currentLayer > threshold, currentLayer, 0)
            A[:,:,layer] = adj_matrix
            A[A != 0] = 1
        return A
    
    def defineAdj(adjMatrix):
        I_N = np.eye(N)
        M = np.block([
            [adjMatrix[:,:,0], I_N, I_N],
            [I_N, adjMatrix[:,:,1], I_N],
            [I_N, I_N, adjMatrix[:,:,2]]
        ])
        # np.fill_diagonal(M, 0)
        return M

    adj_matrices = []
    centrality_list = []
    centrality_one_hot_list = []

    for iSub, data_dir in enumerate(data_dirs):
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        # restStateSub.plotReliability() # Run this once per subject
        _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()
        adj_matrices.append(adj_matrix_within_corr)

        if subtractAverage_true:
            print("We are inside...")
            adjMatrix = subtractAverage(adj_matrix_within_corr)
        
        if binarize_true:
            adjMatrix = binarize(adj_matrix_within_corr)
        else:
            adjMatrix = np.where(adj_matrix_within_corr > 0, adj_matrix_within_corr, 0)

        M = defineAdj(adjMatrix)

        centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
        centrality_list.append(centrality)
        centrality_one_hot_list.append(centrality_oneHot)


    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")

    centrality_list_3D = np.stack(centrality_list, axis=1)
    centrality_one_hot_list_3D = np.stack(centrality_one_hot_list, axis=2)
    restStateSub.eigenvector_centrality_plot(centrality_list_3D, centrality_one_hot_list_3D, analysis)


    adj_matrices_4d = np.stack(adj_matrices, axis=3)
    mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)

    if subtractAverage:
        adjMatrix = subtractAverage(mean_adj_matrix)
    
    if binarize:
        adjMatrix = binarize(mean_adj_matrix)
    else:
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)

    M = defineAdj(adjMatrix)


    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
    print(centrality.shape)
    restStateSub.eigenvector_centrality_plot_avg(centrality, centrality_oneHot, analysis)
    
    restStateSub.runDegreeDistribution(adjMatrix[:,:,2], analysis, "Superficial")
    restStateSub.runDegreeDistribution(adjMatrix[:,:,1], analysis, "Middle")
    restStateSub.runDegreeDistribution(adjMatrix[:,:,0], analysis, "Deep")
    
    # restStateSub.modularity(adjMatrix[:,:,2], analysis, "Superficial") # only makes sense to run on a non-subtracted matrix. 
    # restStateSub.modularity(adjMatrix[:,:,1], analysis, "Middle")
    # restStateSub.modularity(adjMatrix[:,:,0], analysis, "Deep")
    

    # restStateSub.plotConnectogram(adjMatrix[:,:,0], analysis, "Deep", color="red", percent=1)
    # restStateSub.plotConnectogram(adjMatrix[:,:,1], analysis, "Middle", color="green", percent=1)
    # restStateSub.plotConnectogram(adjMatrix[:,:,2], analysis, "Sup", color="blue", percent=1)
    # restStateSub.plotConnectogram_allInOne(adjMatrix[:,:,0], adjMatrix[:,:,1], adjMatrix[:,:,2], analysis, percent=1)

    # rich_nodes1 = restStateSub.calculateRichClub(adjMatrix[:,:,0], analysis, "Deep")
    # rich_nodes2 = restStateSub.calculateRichClub(adjMatrix[:,:,1], analysis, "Middle")
    # rich_nodes3 = restStateSub.calculateRichClub(adjMatrix[:,:,2], analysis, "Sup")
    # restStateSub.plotRichClub(rich_nodes1, rich_nodes2, rich_nodes3, analysis)

    # eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
    # restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
    # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
    # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)
    # restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
    # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    # restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)

    # restStateSub.plotScree(eigvals_within, analysis)
    # crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    # restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    # restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)


elif analysis=="FullLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns()
        adj_matrices.append(adj_matrix_within_corr)

    adj_matrices_3d = np.stack(adj_matrices, axis=2)
    adjMatrix = np.mean(adj_matrices_3d, axis=2)

    if subtractAverage:
    
        adj_matrices = []
        for data_dir in data_dirs:
            restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
            adj_matrix_within, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()
            adj_matrices.append(adj_matrix_within_corr)

        adj_matrices_4d = np.stack(adj_matrices, axis=3)
        mean_adj_matrix = np.mean(adj_matrices_4d, axis=3)
        avg_matrix = np.mean(mean_adj_matrix, axis=2)
        avg_matrix_ext = np.tile(avg_matrix, (3, 3))

        adjMatrix = adjMatrix - avg_matrix_ext
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)


    if binarize:
        # threshold = np.percentile(np.abs(adjMatrix), setThresh)
        # M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
        threshold = np.percentile(adjMatrix, setThresh)
        M = np.where(adjMatrix >= threshold, adjMatrix, 0)
        M[M != 0] = 1

    else:
        M = adjMatrix
        M[M != 0] = 1


    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")

    restStateSub.plotConnectogram(adjMatrix[:360,:360], analysis, "Deep-Deep", color="red", percent=1)
    restStateSub.plotConnectogram(adjMatrix[:360,360:720], analysis, "Deep-Middle", color="green", percent=1)
    restStateSub.plotConnectogram(adjMatrix[:360,720:1080], analysis, "Deep-Sup", color="blue", percent=1)
    restStateSub.plotConnectogram(adjMatrix[360:720,360:720], analysis, "Middle-Middle", color="red", percent=1)
    restStateSub.plotConnectogram(adjMatrix[360:720,720:1080], analysis, "Middle-Sup", color="green", percent=1)
    restStateSub.plotConnectogram(adjMatrix[720:1080,720:1080], analysis, "Sup-Sup", color="red", percent=1)

    rich_nodes1 = restStateSub.calculateRichClub(adjMatrix[:360,:360], analysis, "Deep-Deep")
    # rich_nodes2 = restStateSub.calculateRichClub(adjMatrix[:360,360:720], analysis, "Deep-Middle")
    # rich_nodes3 = restStateSub.calculateRichClub(adjMatrix[:360,720:1080], analysis, "Deep-Sup")
    # rich_nodes4 = restStateSub.calculateRichClub(adjMatrix[360:720,360:720], analysis, "Middle-Middle")
    # rich_nodes5 = restStateSub.calculateRichClub(adjMatrix[360:720,720:1080], analysis, "Middle-Sup")
    # rich_nodes6 = restStateSub.calculateRichClub(adjMatrix[720:1080,720:1080], analysis, "Sup-Sup")


    eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
    restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
    restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)
    crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis)
    restStateSub.identifyEigvecActivityPartOfRS(eigvecs_within, analysis, adjustSize=False)
    restStateSub.plotScree(eigvals_within, analysis)

    restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)

    eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
    cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)





    # restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    # eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True)
    
    # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_full, eigvals_full, N, num_layers)
    # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)

    
    # restStateSub.plotScree(eigvals_full, analysis)
    # crossingsFull = restStateSub.run_plot_zeroCrossings(M, eigvecs_full, analysis)
    # restStateSub.eigvecs_to_nifti(eigvecs_full, analysis, hcp_atlas=hcplabels)
    # restStateSub.plotTwoDimEmbedding(eigvecs_full, analysis)


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