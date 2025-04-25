import laminarRestingState as lrs
import numpy as np
import os

from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

import laminarAnalyses as laman

N = 360
setThresh = 0
num_layers=3
binarize = False
hcplabels = True

data_dirs = ['../highRes_resting/derivatives/correlations/sub-01/Multiple_Runs', '../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs']
output_dir = '../highRes_resting/derivatives/correlations/combo/NoThresh'
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer"

cluster_threshold = 0.45
eigenvalue_threshold = 0.85

row_idx, col_idx = 2,3
layerComparisons = [(1,1), (2,2), (3,3)]

if analysis=="WithinLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        adj_matrix_within, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()
        adj_matrices.append(adj_matrix_within_corr)

    adj_matrices_4d = np.stack(adj_matrices, axis=3)
    mean_adj_matrix = np.mean(adj_matrices_4d, axis=3)

    if binarize:
        adjMatrix = np.empty((N,N,num_layers))
        for layer in range(num_layers):
            currentLayer = mean_adj_matrix[:,:,layer]
            threshold = np.percentile(np.abs(currentLayer), setThresh)
            adj_matrix = np.where(np.abs(currentLayer) >= threshold, currentLayer, 0)
            adjMatrix[:,:,layer] = np.abs(adj_matrix)
    else:
        adjMatrix = mean_adj_matrix

    I_N = np.eye(N)
    M = np.block([
        [adjMatrix[:,:,0], I_N, I_N],
        [I_N, adjMatrix[:,:,1], I_N],
        [I_N, I_N, adjMatrix[:,:,2]]
    ])

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")

    restStateSub.plotConnectogram(adjMatrix[:,:,0], analysis, "Deep", color="red", percent=1)
    restStateSub.plotConnectogram(adjMatrix[:,:,1], analysis, "Middle", color="green", percent=1)
    restStateSub.plotConnectogram(adjMatrix[:,:,2], analysis, "Sup", color="blue", percent=1)
    restStateSub.plotConnectogram_allInOne(adjMatrix[:,:,0], adjMatrix[:,:,1], adjMatrix[:,:,2], analysis, percent=1)

    # eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=False)
    # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
    # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)

    #restStateSub.plotScree(eigvals_within, analysis)
    #crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    #restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    #restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)


elif analysis=="FullLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        adj_matrix_full, fullTimeCourse, adj_matrix_within_corr = restStateSub.get_adj_matrix_full_multRuns()
        adj_matrices.append(adj_matrix_within_corr)

    adj_matrices_3d = np.stack(adj_matrices, axis=2)
    adjMatrix = np.mean(adj_matrices_3d, axis=2)

    if binarize:
        threshold = np.percentile(np.abs(adjMatrix), setThresh)
        M = np.where(np.abs(adjMatrix) >= threshold, adjMatrix, 0)
    else:
        M = adjMatrix

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True)
    
    eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_full, eigvals_full, N, num_layers)
    cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
    laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)

    
    # restStateSub.plotScree(eigvals_full, analysis)
    # crossingsFull = restStateSub.run_plot_zeroCrossings(M, eigvecs_full, analysis)
    # restStateSub.eigvecs_to_nifti(eigvecs_full, analysis, hcp_atlas=hcplabels)
    # restStateSub.plotTwoDimEmbedding(eigvecs_full, analysis)

elif analysis=="SingleLayer":

    adj_matrices = []

    for data_dir in data_dirs:
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, num_layers=1, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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
                                                   atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
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
                                               atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
        
        eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M_single, analysis, num_components=20, 
                                                                        convert_to_binary=False, full=True)
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