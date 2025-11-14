import laminarRestingState as lrs
import numpy as np
import os
import laminarAnalyses as laman
import bimodularityAnalysis as bam
from numpy.linalg import inv


N = 360
setThresh = 0
num_layers=3
binarize_flag = False
subtractAverage_true = False
hcplabels = True
gradients = True

data_dirs = ['/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM001', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM002', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM003', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM004',
             '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM005', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM006', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM009', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/sub-LAM011']

subs = len(data_dirs)
output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/'
os.makedirs(output_dir, exist_ok=True)

analysis = "WithinLayer_10_bin0"

if analysis=="WithinLayer_10_bin0":

    def subtractAverage(adjMatrix):
        avg_matrix = np.nanmean(adjMatrix, axis=2)
        for i in range(adjMatrix.shape[2]):
            adjMatrix[:, :, i] -= avg_matrix
        adjMatrix = np.where(adjMatrix > 0, adjMatrix, 0)
        return adjMatrix


    def thresh_and_binarize(adj, setThresh=5, binarize=False):
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
    
    adj_matrices_appended = []
    centrality_list = []
    centrality_one_hot_list = []

    for iSub, data_dir in enumerate(data_dirs):
        restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
        _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()
        adj_matrices_appended.append(adj_matrix_within_corr)

    adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
    mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)

    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")

    if subtractAverage_true:
        adjMatrix_SA = subtractAverage(mean_adj_matrix)
        adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
    else:
        adjMatrix = thresh_and_binarize(mean_adj_matrix, setThresh=setThresh, binarize=binarize_flag)

    M = defineAdj(adjMatrix)

    if gradients:
        eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
        gradients, eig = laman.run_gradient_analysis(M)
        print(gradients.shape)
        restStateSub.eigvecs_to_nifti(gradients, analysis, hcp_atlas=hcplabels)
    
    else:
        eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)

        np.save(os.path.join(output_dir, analysis, 'FC_matrix.npy'), M)

        centrality, centrality_oneHot = restStateSub.eigenvector_centrality_calc(M)
        restStateSub.eigenvector_centrality_plot_avg(centrality, centrality_oneHot, analysis)
            
        restStateSub.plotConnectogram_allInOne(adjMatrix[:,:,0], adjMatrix[:,:,1], adjMatrix[:,:,2], analysis, percent=2)

        eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(M, analysis, convert_to_binary=False, full=True, vMax=1)
        
        restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis)
        restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[2, 3])
        restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[3, 4])
        restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[4, 5])
        restStateSub.plotTwoDimEmbedding_byNetwork(eigvecs_within, analysis, eigvecs_to_plot=[5, 6])

        restStateSub.run_plot_FstatComp(eigvecs_within,analysis)
        # eigvecs_list, eigvalue_list, source_info = laman.convert_eigvals_to_list(eigvecs_within, eigvals_within, N, num_layers)
        # cluster_groups, labels = laman.runClusterAnalysis(eigvecs_list, threshold=cluster_threshold)
        # laman.plotEigvectors_similar_distinct(eigvecs_list, eigvalue_list, source_info, cluster_groups, restStateSub, eigenvalue_threshold, cluster_threshold, analysis)
        restStateSub.plotEigenvectorCorrelation(eigvecs_within, analysis)

        restStateSub.plotScree(eigvals_within, analysis)
        crossingsWithin = restStateSub.run_plot_zeroCrossings(M, eigvecs_within, analysis)
        restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)

