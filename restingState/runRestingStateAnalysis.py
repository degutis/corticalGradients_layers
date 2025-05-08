import restingState.laminarRestingState as lrs

data_dir = '../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs'

N = 360
setThresh = 85
hcplabels = True

#analysis_types = ["FeedforwardFeedback", "WithinLayer", "BetweenLayers", "FullLayer"]
analysis_types = ["WithinLayer"]
restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "../highRes_resting/derivatives/ref_anat/sub-02/HCP-MM1_in-func.nii")

for analysis in analysis_types:

    if analysis == "FeedforwardFeedback":

        adj_matrix_between = restStateSub.get_adj_matrix_betweenLayers_threshForEach(fullInterconnect=False)
        eigvals_between, eigvecs_between = restStateSub.runLaplacianEmbedding(adj_matrix_between, analysis, full=True)
        restStateSub.plotScree(eigvals_between, analysis)
        crossingsWithin = restStateSub.run_plot_zeroCrossings(adj_matrix_between, eigvecs_between, analysis)
        restStateSub.eigvecs_to_nifti(eigvecs_between, analysis, hcp_atlas=hcplabels)
        restStateSub.plotTwoDimEmbedding(eigvecs_between, analysis)

    elif analysis == "BetweenLayers":

        adj_matrix_between = restStateSub.get_adj_matrix_betweenLayers_threshForEach(fullInterconnect=True)
        eigvals_between, eigvecs_between = restStateSub.runLaplacianEmbedding(adj_matrix_between, analysis, full=True)
        restStateSub.plotScree(eigvals_between, analysis)
        crossingsWithin = restStateSub.run_plot_zeroCrossings(adj_matrix_between, eigvecs_between, analysis)
        restStateSub.eigvecs_to_nifti(eigvecs_between, analysis, hcp_atlas=hcplabels)
        restStateSub.plotTwoDimEmbedding(eigvecs_between, analysis)


    elif analysis == "WithinLayer":

        adj_matrix_within = restStateSub.get_adj_matrix_withinLayers_multRuns()
        eigvals_within, eigvecs_within = restStateSub.runLaplacianEmbedding(adj_matrix_within, analysis, num_components=20)
        restStateSub.plotScree(eigvals_within, analysis)
        crossingsWithin = restStateSub.run_plot_zeroCrossings(adj_matrix_within, eigvecs_within, analysis)
        restStateSub.eigvecs_to_nifti(eigvecs_within, analysis, hcp_atlas=hcplabels)
        restStateSub.plotTwoDimEmbedding(eigvecs_within, analysis)

    elif analysis == "FullLayer":

        adj_matrix_full, fullTimeCourse = restStateSub.get_adj_matrix_full_multRuns()
        eigvals_full, eigvecs_full = restStateSub.runLaplacianEmbedding(adj_matrix_full, analysis, num_components=20)
        restStateSub.plotScree(eigvals_full, analysis)
        crossingsFull = restStateSub.run_plot_zeroCrossings(adj_matrix_full, eigvecs_full, analysis)
        restStateSub.eigvecs_to_nifti(eigvecs_full, analysis, hcp_atlas=hcplabels)
        restStateSub.plotTwoDimEmbedding(eigvecs_full, analysis)