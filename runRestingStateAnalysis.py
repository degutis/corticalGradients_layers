import laminarRestingState as lrs

#data_dir = '../highRes_resting/derivatives/correlations/sub-01/Gap_new'

data_dir = '../kenshu_dataset/derivatives/sub-02/correlations/Glasser_residuals'
N = 360
setThresh = 85
hcplabels = True
atlas_dir = "../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_extended_distance.nii"
#atlas_dir = "../kenshu_dataset/derivatives/sub-02/columns/dwscaled_columns10000.nii"


#restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh)
restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = atlas_dir)

adj_matrix_within = restStateSub.get_adj_matrix_withinLayers()
eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(adj_matrix_within, "WithinLayer", full=True)
restStateSub.plotScree(eigvals_within, "WithinLayer")
crossingsWithin = restStateSub.run_plot_zeroCrossings(adj_matrix_within, eigvecs_within, "WithinLayer")
restStateSub.eigvecs_to_nifti(eigvecs_within, "WithinLayer", hcp_atlas=hcplabels)
restStateSub.plotTwoDimEmbedding(eigvecs_within, "WithinLayer")


#adj_matrix_full, fullTimeCourse = restStateSub.get_adj_matrix_full()
#eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(adj_matrix_full, "FullLayer", num_components=20)

#print(fullTimeCourse.shape)
#print(eigvals_full.shape)
#print(eigvecs_full.shape)

#restStateSub.plotScree(eigvals_full, "FullLayer")
#crossingsFull = restStateSub.run_plot_zeroCrossings(adj_matrix_full, eigvecs_full, "FullLayer")
#restStateSub.eigvecs_to_nifti(eigvecs_full, "FullLayer", hcp_atlas=hcplabels)
#restStateSub.plotTwoDimEmbedding(eigvecs_full, "FullLayer")