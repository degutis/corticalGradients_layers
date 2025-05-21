import laminarRestingState as lrs
import laminarAnalyses as laman
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore
import statsmodels.api as sm
from hurst import compute_Hc


data_dirs = ['../../highRes_resting/derivatives/correlations/sub-01/Multiple_Runs/smallGap_noPreproc', 
             '../../highRes_resting/derivatives/correlations/sub-02/Multiple_Runs/smallGap_noPreproc', 
             '../../highRes_resting/derivatives/correlations/sub-03/Multiple_Runs/smallGap_noPreproc']

output_dir = '../../highRes_resting/derivatives/MRSI/smallGap_Zscore_noPreproc'
os.makedirs(output_dir, exist_ok=True)

subs=len(data_dirs)
layer_names = ["Deep", "Middle", "Superficial","Average"]


h_file_path = os.path.join(output_dir, "H_matrix.npy")

neuroTransName="Glutamine"
mrs_path = os.path.join(f"../../highRes_resting/derivatives/MRSI/Parcel_mean_{neuroTransName}.npy")
neuroTrans = np.load(mrs_path).flatten()

if os.path.exists(h_file_path):
    print("File exists. Loading H_matrix...")
    H_matrix = np.load(h_file_path)
else:

    H_matrix = np.empty((360, 4, subs))

    for subIndx, data_dir in enumerate(data_dirs):

        npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

        layer_groups = defaultdict(list)

        for file in npy_files:
            try:
                layer_str = file.split('_')[-1].replace('.npy', '')
                layer_num = int(layer_str)
                layer_groups[layer_num].append(file)
            except Exception as e:
                raise ValueError(f"Could not extract layer number from filename: {file}") from e

        sorted_layers = sorted(layer_groups.items())

        all_time_series_acrossLayers = []
        for i, (layer_num, files) in enumerate(sorted_layers):
            all_time_series = []

            for file in files:
                file_path = os.path.join(data_dir, file)
                time_series = np.load(file_path)
                time_series = time_series - np.mean(time_series,axis=1, keepdims=True) # demean each voxel so we don't have large jumps
                # time_series = zscore(time_series, axis=1)
                all_time_series.append(time_series)

            concatenated = np.concatenate(all_time_series, axis=1)
            all_time_series_acrossLayers.append(concatenated)
            
            for parcel_idx in range(concatenated.shape[0]):
                H, windows, flucts = laman.hurst_dfa_bound(concatenated[parcel_idx, :])
                # H, c, data = compute_Hc(concatenated[parcel_idx, :], kind='price', simplified=True)
                H_matrix[parcel_idx, layer_num-1, subIndx] = H
            
        mean_across_layer = np.nanmean(np.stack(all_time_series_acrossLayers,axis=2),axis=2)
        print(f'Shape mean: {mean_across_layer.shape}')
            
        for parcel_idx in range(concatenated.shape[0]):
            H, windows, flucts = laman.hurst_dfa_bound(mean_across_layer[parcel_idx, :])
            H_matrix[parcel_idx, 3, subIndx] = H
    
    np.save(os.path.join(output_dir, "H_matrix.npy"), H_matrix)

# H, _ = laman.dfa_fast(np.mean(np.stack(all_time_series_acrossLayers,axis=2),axis=2).transpose())



for sub in range(subs):
    restStateSub = lrs.LaminarRestingState(output_dir, 360, 0, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
    print("Min Max Hurst Exponent")
    print(np.nanmin(H_matrix[:,:,sub]))
    print(np.nanmax(H_matrix[:,:,sub]))
    restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
        H_matrix[:,:,sub], f"HurstExponent_{sub}", name='Hurst', folder_name="Maps", cm="YlOrRd"
        )


meanMatrixH = np.nanmean(H_matrix, axis=2)

restStateSub = lrs.LaminarRestingState(output_dir, 360, 0, atlas_dir = "../../highRes_resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii")
restStateSub.__plot_on_mmhcp_surface_multipleLayers__(
    meanMatrixH, "HurstExponent", name='Hurst', folder_name="Maps", cm="YlOrRd"
    )


titles = [f"Deep H vs {neuroTransName}", f"Middle H vs {neuroTransName}", f"Superficial H vs {neuroTransName}", f"Average of all layer H vs {neuroTransName}"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for i in range(4):
    h_col = meanMatrixH[:, i]
    # h_col = H_matrix[:,i,1]

    # Remove NaN pairs
    valid_idx = ~np.isnan(h_col) & ~np.isnan(neuroTrans)
    h_valid = h_col[valid_idx]
    neuroTrans_valid = neuroTrans[valid_idx]

    slope, intercept, r_value, p_value, std_err = linregress(neuroTrans_valid, h_valid)

    x_vals = np.linspace(neuroTrans.min(), neuroTrans.max(), 100)
    y_vals = slope * x_vals + intercept

    ax = axes[i]
    ax.scatter(neuroTrans, h_col, alpha=0.7, label='Data points')
    ax.plot(x_vals, y_vals, color='red', label=f'r={r_value:.2f}, p={p_value:.3g}')
    ax.set_title(titles[i])
    ax.set_xlabel(f'{neuroTransName}')
    if i == 0:
        ax.set_ylabel("Hurst")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/Correlation_H_{neuroTransName}.png", facecolor="white")



vector_folder = '../../highRes_resting/derivatives/MRSI/' 
GABA = np.load(f"../../highRes_resting/derivatives/MRSI/Parcel_mean_GABA.npy").flatten()
Glu  = np.load(f"../../highRes_resting/derivatives/MRSI/Parcel_mean_GlutamicAcid.npy").flatten()
Gln  = np.load(f"../../highRes_resting/derivatives/MRSI/Parcel_mean_Glutamine.npy").flatten()

# Dictionary for easy looping
data = {'GABA': GABA, 'Gln': Gln, 'Glu': Glu}
pairs = [('GABA', 'Gln'), ('GABA', 'Glu'), ('Gln', 'Glu')]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

for idx, (x_name, y_name) in enumerate(pairs):
    x = data[x_name]
    y = data[y_name]

    # Remove NaNs
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid]
    y_valid = y[valid]

    # Linear regression
    slope, intercept, r_value, p_value, _ = linregress(x_valid, y_valid)
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_line = slope * x_line + intercept

    # Plot
    ax = axes[idx]
    ax.scatter(x_valid, y_valid, alpha=0.7, label='Data points')
    ax.plot(x_line, y_line, color='red', label=f'r = {r_value:.2f}, p = {p_value:.3g}')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f'{y_name} vs {x_name}')
    ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/Correlation_allNT.png", facecolor="white")
plt.close()


## Run partial correlation

def partial_corr(x, y, z):
    """
    Compute the partial correlation between x and y, controlling for z.
    x, y, z should be 1D arrays of the same length.
    """
    # Add constant for intercept in the regressions
    Z = sm.add_constant(z)
    # Residuals of x ~ z
    res_x = sm.OLS(x, Z).fit().resid
    # Residuals of y ~ z
    res_y = sm.OLS(y, Z).fit().resid
    # Corr(res_x, res_y)
    return np.corrcoef(res_x, res_y)[0, 1]

A_label = "Hurst Component"
B_label = "Glutamic Acid"
C_label = "GABA"

partial_AB = []
partial_AC = []

for i in range(4):

    A_init = meanMatrixH[:,i]
    B_init = Glu
    C_init = GABA

    valid_idx = ~np.isnan(A_init) & ~np.isnan(B_init) & ~np.isnan(C_init)
    A = A_init[valid_idx]
    B = B_init[valid_idx]
    C = C_init[valid_idx]

    # 1. Partial correlations
    r_AB_C = partial_corr(A, B, C)
    r_AC_B = partial_corr(A, C, B)

    partial_AB.append(r_AB_C)
    partial_AC.append(r_AC_B)

layers = layer_names
x = np.arange(len(layers))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, partial_AB, width,
               label=f"{A_label}–{B_label} | {C_label}")
bars2 = ax.bar(x + width/2, partial_AC, width,
               label=f"{A_label}–{C_label} | {B_label}")

# zero line
ax.axhline(0, color='gray', linewidth=0.8)

# labels & legend
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.set_ylim(-1, 1)
ax.set_ylabel("Partial Correlation")
ax.set_title("Partial Correlations Across Layers")
ax.legend()

# annotate each bar
for bar_list in (bars1, bars2):
    for bar in bar_list:
        val = bar.get_height()
        y = val + 0.02*np.sign(val)
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top")

plt.tight_layout()
plt.savefig(f"{output_dir}/PartialCorrelation_{A_label}_{B_label}_{C_label}.png", facecolor="white")
plt.close()