import laminarRestingState as lrs
import numpy as np
import os
from pathlib import Path
from infomap import Infomap



N = 400
setThresh = 95
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
n_components = 15

BASE = Path('/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations')
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22] 
# BASE = Path('/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/correlations')
# SUBJECTS = [1, 2, 4] 

gap_dir = f'{"large" if largeGap else "small"}Gap_Schaefer'
# gap_dir = "eightLayers_Schaefer"
root = BASE / gap_dir

data_dirs = [root / f'sub-LAM{s:03d}' for s in SUBJECTS]
# data_dirs = [root / f'sub-{s:02d}' for s in SUBJECTS]

output_dir = root

subs = len(data_dirs)
os.makedirs(output_dir, exist_ok=True)

analysis = "Infomap"


if analysis=="Infomap":

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
            # restStateSub.plotReliability(TR=3.3) # Run this once per subject
            _, adj_matrix_within_corr = restStateSub.get_adj_matrix_withinLayers_multRuns()

            if subtractAverage_true:
                adjMatrix_SA = subtractAverage(adj_matrix_within_corr)
                adjMatrix = thresh_and_binarize(adjMatrix_SA, setThresh=setThresh, binarize=binarize_flag)
            else:
                adjMatrix = thresh_and_binarize(adj_matrix_within_corr, setThresh=setThresh, binarize=binarize_flag)

            adj_matrices_appended.append(adjMatrix)

        adj_matrices_4d = np.stack(adj_matrices_appended, axis=3)
        mean_adj_matrix = np.nanmean(adj_matrices_4d, axis=3)
        r_matrix = backToR(mean_adj_matrix)

        M = defineAdj(r_matrix)
        np.save(os.path.join(output_dir, analysis, 'FC_matrix.npy'), M)


    # M_current = M[:N,:N]
    # M_current = M[1*N:2*N,1*N:2*N]
    M_current = M[2*N:3*N,2*N:3*N]

    print(M_current.shape)
    print(np.max(M_current))
    print(np.min(M_current))

    X = 0.1
    W = np.copy(M_current)
    W[W < 0] = 0
    print(W.shape)
    # zero diagonal
    np.fill_diagonal(W, 0.0)
    # global threshold
    thr = np.percentile(W[W > 0], 100*(1 - X))
    mask = W >= thr

    im = Infomap("--num-trials 100")
    
    for i in range(N):
        im.add_node(i)

    # Add links for surviving edges
    ii, jj = np.where(np.triu(mask, 1))
    for i, j in zip(ii, jj):
        im.add_link(int(i), int(j), float(W[i, j]))

    im.run()

    modules = im.get_modules()  # dict: {node_id -> top-level module id}
    labels = np.array([modules[i] for i in range(N)], dtype=int).reshape(N, 1)

    print(labels.shape)  # (400, 1)
    # Optional: check module sizes
    from collections import Counter
    print(Counter(labels[:,0]))

    additionalFolder = "modules"
    restStateSub = lrs.LaminarRestingState(output_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    restStateSub.__plot_on_mmhcp_surface_multipleLayers__(labels, "D_Sup_15_zscore", os.path.join(analysis, additionalFolder),vmin=1,vmax=20, cm="tab20")        
