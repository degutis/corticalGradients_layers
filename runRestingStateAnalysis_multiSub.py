import laminarRestingState as lrs
import numpy as np
import os

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


N = 360
setThresh = 85
num_layers=3
binarize = False
hcplabels = True

data_dirs = ['../highRes_resting/derivatives/correlations/sub-01/Multiple_Runs', '../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs']
output_dir = '../highRes_resting/derivatives/correlations/combo/Multiple_Runs_full'
os.makedirs(output_dir, exist_ok=True)

analysis = "SingleLayerComparison"
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
    eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True)
    restStateSub.plotScree(eigvals_within, analysis)
    crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
    restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
    restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)


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
    eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(M, analysis, num_components=20, convert_to_binary=False, full=False)
    restStateSub.plotScree(eigvals_full, analysis)
    crossingsFull = restStateSub.run_plot_zeroCrossings(M, eigvecs_full, analysis)
    restStateSub.eigvecs_to_nifti(eigvecs_full, analysis, hcp_atlas=hcplabels)
    restStateSub.plotTwoDimEmbedding(eigvecs_full, analysis)

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
    source_info = [] 

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
            eigvecs_list.append(eigvec / np.linalg.norm(eigvec))
            source_info.append((row_idx, col_idx, i))   

    eigvecs_array = np.array([v / np.linalg.norm(v) for v in eigvecs_list])

    # Pairwise distance matrix (sign-invariant)
    def sign_invariant_distance(u, v):
        return 1 - np.abs(np.dot(u, v))

    D = squareform(pdist(eigvecs_array, metric=sign_invariant_distance))

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=0.4,
        n_clusters=None
    )
    labels = clustering.fit_predict(D)
    print(len(labels))
    # Group by cluster label
    cluster_groups = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        cluster_groups[cluster_id].append(i)
    
    for cluster_id, indices in cluster_groups.items():
        if len(indices) <= 1:
            continue

        eigvecs_to_plot = [eigvecs_list[i] for i in indices]
        meta = [source_info[i] for i in indices]
        titles = [f"(r{r},c{c}) eig{e}" for (r, c, e) in meta]
        Xp = np.stack(eigvecs_to_plot, axis=1)

        # Build filename from source info
        name_str = "-".join([f"r{r}_c{c}_e{e}" for (r, c, e) in meta])
        eig_label = f"{name_str}"

        restStateSub.__plot_on_mmhcp_surface_multipleLayers__(Xp, eig_label, name="SingleLayerComparison", titles=titles)