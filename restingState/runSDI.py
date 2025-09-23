from nigsp.operations.laplacian import decomposition
from nigsp.operations.timeseries import graph_fourier_transform, median_cutoff_frequency_idx, graph_filter
from nigsp.operations.metrics import sdi

import numpy as np
import laminarRestingState as lrs
import laminarAnalyses as laman
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats



N = 400
setThresh = 0
num_layers=3
t=138
useHCP_SC=False
subjectLoop=False

data_dirs = ['/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM001', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM002', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM003', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM004',
             '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM005', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM006', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM009', '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/sub-LAM011']

output_dir = '/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/'
os.makedirs(output_dir, exist_ok=True)
analysis = "WithinLayer_gradients_noThresh_DistanceMetric_kernelCosine"


numSubs = len(data_dirs)

fullTimeCourseMatrix = np.empty((N,t,num_layers, numSubs))

for iSub, data_dir in enumerate(data_dirs):

    restStateSub = lrs.LaminarRestingState(data_dir, N, setThresh, atlas_dir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/sub-LAM001/HCP-MMP1_in-func.nii")
    layer_groups = defaultdict(list)
            
    for file in restStateSub.npy_files:

        try:
            layer_str = file.split('_')[-1].replace('.npy', '')
            layer_num = int(layer_str)
            layer_groups[layer_num].append(file)
        except Exception as e:
            print(f"Could not extract layer number from filename: {file}")
            continue

        sorted_layers = sorted(layer_groups.items())
        adj_matrix_within = np.empty((restStateSub.N, restStateSub.N, restStateSub.num_layers))
        adj_matrix_within_noThresh = np.empty((restStateSub.N,restStateSub.N,restStateSub.num_layers))

        for i, (layer_num, files) in enumerate(sorted_layers):
            all_time_series = []

            for file in files:
                file_path = os.path.join(restStateSub.data_dir, file)
                print(file_path)
                time_series = np.load(file_path)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            fullTimeCourseMatrix[:,:,layer_num-1,iSub] = concatenated

if useHCP_SC==True:
    from enigmatoolbox.datasets import load_sc
    sc, sc_labels, _, _ = load_sc('schaefer_400')  # sc_ctx is 400x400
    addString="_HCP_SC"
else:
    sc = np.load("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/BigBrainMatrix/adjacency_matrix_Schaefer.npy")
    sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    sc[sc < 0] = 0.0
    addString="_BigBrainLam"

degree_matrix = np.diag(np.sum(sc, axis=1))
laplacian_matrix = degree_matrix - sc  
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(sc, axis=1) + 1e-10))
L_norm = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt
evals, evecs = decomposition(L_norm)

if subjectLoop==True:
    sdi_full = np.empty((N,num_layers,numSubs))
    for iSub in range(numSubs):
        for layer in range(num_layers):
            ts = fullTimeCourseMatrix[:,:,layer,iSub]
            X_hat = graph_fourier_transform(ts, evecs)                      # GFT of fMRI
            power = (X_hat**2).sum(axis=1)                                  # spectral power
            k = median_cutoff_frequency_idx(power)                          # split index (low vs high graph freq)
            (_, _), proj = graph_filter(ts, evecs, k, keys=["low","high"])  # reconstruct low/high components
            sdi_vals = sdi(proj)                                            # per-region SDI (||high|| / ||low||)
            sdi_full[:,layer,iSub] = sdi_vals

    SDI_acrossSubj = np.mean(sdi_full,axis=-1)

else:
    SDI_acrossSubj = np.empty((N, num_layers), dtype=float)
    for layer in range(num_layers):
        ts_L = fullTimeCourseMatrix[:, :, layer, :]  # (T, N, S)
        ts_ext = np.concatenate([ts_L[:, :, s] for s in range(numSubs)], axis=1)
        print(ts_ext.shape)
        X_hat = graph_fourier_transform(ts_ext, evecs)         # (N x T*S) or (freq x time) per your implementation
        power = (X_hat**2).sum(axis=1)                         # sum over time to get per-frequency power
        k = median_cutoff_frequency_idx(power)                 # index separating low/high graph frequencies
        (_, _), proj = graph_filter(ts_ext, evecs, k, keys=["low", "high"])
        sdi_vals = sdi(proj)                                   # per-region SDI (||high|| / ||low||)
        SDI_acrossSubj[:, layer] = sdi_vals


laman.plotFlatMap(SDI_acrossSubj[:,0], os.path.join(output_dir, analysis), f'SDI_deep{addString}_extSub.png',vmin=-1, vmax=1)
laman.plotFlatMap(SDI_acrossSubj[:,1], os.path.join(output_dir, analysis), f'SDI_mid{addString}_extSub.png',vmin=-1, vmax=1)
laman.plotFlatMap(SDI_acrossSubj[:,2], os.path.join(output_dir, analysis), f'SDI_sup{addString}_extSub.png',vmin=-1, vmax=1)


fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
bins = np.linspace(-1, 1, 21)  # 40 bins

axes[0].hist(SDI_acrossSubj[:,0], bins=bins, range=(-1, 1))
axes[0].set_ylabel('Count')
axes[0].set_title('Deep')

axes[1].hist(SDI_acrossSubj[:,1], bins=bins, range=(-1, 1))
axes[1].set_ylabel('Count')
axes[1].set_title('Middle')

axes[2].hist(SDI_acrossSubj[:,2], bins=bins, range=(-1, 1))
axes[2].set_ylabel('Count')
axes[2].set_xlabel('SDI values for Schaefer 400')
axes[2].set_title('Superficial')

# Nice-to-have: limits + light grid
for ax in axes:
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 70)
    ax.grid(True, linestyle='--', alpha=0.3)

# --- One-way ANOVA across the three vectors ---
F, p = stats.f_oneway(SDI_acrossSubj[:,0], SDI_acrossSubj[:,1], SDI_acrossSubj[:,2])

fig.text(
    0.98, 0.98,
    f"One-way ANOVA\nF = {F:.2f}\np = {p:.3g}",
    ha="right", va="top",
    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85)
)



plt.tight_layout()
plt.savefig(os.path.join(output_dir, analysis, f'SDI_laminarHist{addString}_extSub.png'),dpi=300, bbox_inches="tight")