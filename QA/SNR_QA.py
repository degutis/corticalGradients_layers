#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

#######################################
# USER SETTINGS
#######################################

subjects = [
    # fill with all subject IDs you want in the group analysis
    "sub-LAM001",
    "sub-LAM002",
    "sub-LAM003",
    "sub-LAM004",
    "sub-LAM005",
    "sub-LAM006",
    "sub-LAM007",
    "sub-LAM008",
    "sub-LAM009",
    "sub-LAM010",
    "sub-LAM011",
    "sub-LAM012",
    "sub-LAM013",
    "sub-LAM014",
    "sub-LAM015",
    "sub-LAM016",
    "sub-LAM017",
    "sub-LAM018",
    "sub-LAM019",
    "sub-LAM021",
    "sub-LAM022",
]

# if subjects have multiple runs (e.g. run1, run2) list them here
runNums = [
    "run1"
]

# which laminar binning rule to use (has to match your project logic)
analysis_type = "smallGap"  # or "smallGap_Schaefer" / "largeGap_Schaefer" / etc.
layer_01 = True          # same meaning as in your function

# base paths (adapt if needed)
FUNC_BASE = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func"
ANAT_BASE = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat"

# output dir for group results / figs
GROUP_OUT = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/tSNR_layers"
os.makedirs(GROUP_OUT, exist_ok=True)

#######################################
# HELPER: build layer mask 1/2/3
#######################################

def make_layer_binary(layer_data, analysis_type, layer_01=True):
    """
    Recreates the layer_binary logic from averageVoxels_parcels_with_fs.
    Returns integer array same shape as layer_data with values {0,1,2,3,...}.
    """
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)

    if layer_01:
        if analysis_type == "smallGap":
            layer_binary[(layer_data > 0)   & (layer_data <= 0.3)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.7) & (layer_data <  1.0)] = 3

        elif analysis_type == "noGap":
            layer_binary[(layer_data > 0)   & (layer_data <= 0.4)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.6) & (layer_data <  1.0)] = 3

        elif analysis_type == "largeGap":
            layer_binary[(layer_data > 0)   & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.8) & (layer_data <  1.0)] = 3

        elif analysis_type == "eightLayers":
            layer_binary[(layer_data > 0.1) & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.2) & (layer_data <  0.3)] = 2
            layer_binary[(layer_data > 0.3) & (layer_data <  0.4)] = 3
            layer_binary[(layer_data > 0.4) & (layer_data <  0.5)] = 4
            layer_binary[(layer_data > 0.5) & (layer_data <  0.6)] = 5
            layer_binary[(layer_data > 0.6) & (layer_data <  0.7)] = 6
            layer_binary[(layer_data > 0.7) & (layer_data <  0.8)] = 7
            layer_binary[(layer_data > 0.8) & (layer_data <  0.9)] = 8

    else:
        # alt binning (integer depths)
        layer_binary[(layer_data > 1)  & (layer_data <= 3)]  = 1
        layer_binary[(layer_data > 5)  & (layer_data <= 7)]  = 2
        layer_binary[(layer_data > 9)  & (layer_data <= 11)] = 3

    return layer_binary


#######################################
# HELPER: compute tSNR from 4D data
#######################################

def compute_tsnr_4d(data_4d):
    """
    data_4d: (X,Y,Z,T)

    Returns:
      tsnr_3d after outlier clipping (10 SD rule similar to yours),
      mean_signal_3d
    """
    mean_signal = np.mean(data_4d, axis=-1)
    std_signal  = np.std(data_4d, axis=-1)

    # avoid div by zero
    tsnr = np.where(std_signal > 0, mean_signal / std_signal, 0)

    # clip extreme tSNR outliers:
    tsnr_nonzero = tsnr[tsnr > 0]
    if tsnr_nonzero.size > 0:
        tsnr_mean = np.mean(tsnr_nonzero)
        tsnr_std  = np.std(tsnr_nonzero)

        # NOTE: your comment says "3 SD threshold" but the code uses +10*std.
        # I'll keep it consistent with code: +10*std.
        tsnr_upper_threshold = tsnr_mean + 10 * tsnr_std
        tsnr[tsnr > tsnr_upper_threshold] = 0

    return tsnr, mean_signal


#######################################
# MAIN ANALYSIS
#######################################

# We'll collect per-subject layer x parcel means.
# For each subject we'll end up with:
#   subj_layer_parcel_tsnr[layer_idx, parcel_id-1] = mean tSNR
#
# layer_idx in {0,1,2} for layers {1,2,3}
# parcel_id runs 1..400
#
# If subject has multiple runs, we average across runs within subject.

all_subject_layer_parcel_tsnr = []  # list of arrays shape (3,400)
subject_ids_used = []              # to track which ones succeeded

for subj in subjects:
    print(f"==== Subject {subj} ====")
    # load layer + atlas (assume doesn't change across runs, so load once)
    layer_path = os.path.join(
        ANAT_BASE,
        subj,
        "ln_depths_equivol.nii"
    )
    atlas_path_right = os.path.join(
        ANAT_BASE,
        subj,
        "schaefer_R_in-func.nii"
    )
    atlas_path_left = os.path.join(
        ANAT_BASE,
        subj,
        "schaefer_L_in-func.nii"
    )

    if (not os.path.exists(layer_path) or
        not os.path.exists(atlas_path_right) or
        not os.path.exists(atlas_path_left)):
        print(f"[WARN] Missing atlas/layer for {subj}, skipping.")
        continue

    layer_img = nib.load(layer_path)
    layer_data = layer_img.get_fdata()

    atlas_img_right = nib.load(atlas_path_right)
    atlas_img_left  = nib.load(atlas_path_left)

    atlas_data_right = atlas_img_right.get_fdata()
    atlas_data_left  = atlas_img_left.get_fdata()

    atlas_data = atlas_data_right + atlas_data_left  # parcel IDs 1..400
    fs_parcels = np.unique(atlas_data)
    fs_parcels = fs_parcels[(fs_parcels > 0) & (fs_parcels <= 400)]
    fs_parcels = np.sort(fs_parcels).astype(int)

    # if we don't actually have all 400 labels in this subject
    # we'll still create a 400-length vector and fill missing with 0.
    if fs_parcels.size != 400:
        print(f"[INFO] {subj}: expected 400 parcels but got {fs_parcels.size}. Will fill missing with 0.")
    full_parcel_list = np.arange(1,401)

    # Make layer mask
    layer_binary = make_layer_binary(layer_data, analysis_type, layer_01)
    unique_layers = np.unique(layer_binary[layer_binary > 0])
    # We'll only consider the first three layers (1,2,3). If 8-layer scheme,
    # we'll still just take 1,2,3 for consistency with your request.
    target_layers = [1,2,3]
    # storage for each run, will average later
    subj_run_layer_parcel = []

    for runNum in runNums:
        bold_path = os.path.join(
            FUNC_BASE,
            subj,
            f"merged_residuals_{runNum}.nii"
        )
        if not os.path.exists(bold_path):
            print(f"[WARN] Missing BOLD {bold_path}, skipping run.")
            continue

        bold_img = nib.load(bold_path)
        bold_data = bold_img.get_fdata()  # 4D

        # shape check
        if (layer_data.shape != atlas_data.shape or
            layer_data.shape != bold_data.shape[:-1]):
            print(f"[ERROR] Dim mismatch in {subj} {runNum}, skipping run.")
            continue

        # compute tSNR volume for THIS run
        tsnr_vol, mean_signal = compute_tsnr_4d(bold_data)

        # now compute layer X parcel means of tSNR
        # We'll fill layer x 400 with np.nan then populate
        layer_parcel_tsnr = np.full((3, 400), np.nan, dtype=float)

        for li, layer_id in enumerate(target_layers):
            if layer_id not in unique_layers:
                # leave this layer row as NaN
                continue

            layer_mask = (layer_binary == layer_id)

            for parcel_id in full_parcel_list:
                parcel_mask = (atlas_data == parcel_id)
                combo_mask = layer_mask & parcel_mask

                if np.any(combo_mask):
                    vals = tsnr_vol[combo_mask]
                    if vals.size > 0:
                        layer_parcel_tsnr[li, parcel_id-1] = np.nanmean(vals)

        subj_run_layer_parcel.append(layer_parcel_tsnr)

    # after looping runs, average across runs for this subject
    if len(subj_run_layer_parcel) == 0:
        print(f"[WARN] {subj}: no valid runs, skipping subject.")
        continue

    subj_run_layer_parcel = np.stack(subj_run_layer_parcel, axis=0)  # (nRunsValid,3,400)
    subj_layer_parcel_mean = np.nanmean(subj_run_layer_parcel, axis=0)  # (3,400)

    all_subject_layer_parcel_tsnr.append(subj_layer_parcel_mean)
    subject_ids_used.append(subj)

# convert to array subj x layer x parcel
if len(all_subject_layer_parcel_tsnr) == 0:
    raise RuntimeError("No valid subjects found. Check paths / subject list.")

all_subject_layer_parcel_tsnr = np.stack(all_subject_layer_parcel_tsnr, axis=0)  # (Nsubj,3,400)
Nsubj = all_subject_layer_parcel_tsnr.shape[0]
print(f"Final dataset shape: {all_subject_layer_parcel_tsnr.shape} (subjects x layers x parcels)")
print(f"Subjects used: {subject_ids_used}")

#######################################
# SAVE GROUP-AVERAGE PARCEL VECTORS
#######################################

# mean across subjects for each layer, parcel-wise
group_mean_layer_parcel = np.nanmean(all_subject_layer_parcel_tsnr, axis=0)  # (3,400)

# layer indices 0,1,2 -> layers 1,2,3
layer1_vec = group_mean_layer_parcel[0, :]  # (400,)
layer2_vec = group_mean_layer_parcel[1, :]
layer3_vec = group_mean_layer_parcel[2, :]

np.save(os.path.join(GROUP_OUT, "group_layer1_tSNR_byParcel.npy"), layer1_vec)
np.save(os.path.join(GROUP_OUT, "group_layer2_tSNR_byParcel.npy"), layer2_vec)
np.save(os.path.join(GROUP_OUT, "group_layer3_tSNR_byParcel.npy"), layer3_vec)

print("Saved group-level layer-by-parcel tSNR vectors to:")
print(os.path.join(GROUP_OUT, "group_layer1_tSNR_byParcel.npy"))
print(os.path.join(GROUP_OUT, "group_layer2_tSNR_byParcel.npy"))
print(os.path.join(GROUP_OUT, "group_layer3_tSNR_byParcel.npy"))

#######################################
# VIOLIN PLOT + ANOVA
#######################################

# For each subject and each layer:
#   collapse across parcels (mean over parcels 1..400)
#   → gives us one tSNR number per layer per subject
#
# We'll then:
#   (1) violin these 3 distributions
#   (2) run one-way ANOVA (layer1 vs layer2 vs layer3)

layer1_subj_means = np.nanmean(all_subject_layer_parcel_tsnr[:,0,:], axis=1)  # (Nsubj,)
layer2_subj_means = np.nanmean(all_subject_layer_parcel_tsnr[:,1,:], axis=1)
layer3_subj_means = np.nanmean(all_subject_layer_parcel_tsnr[:,2,:], axis=1)

# ANOVA across the three layers
Fval, pval = f_oneway(layer1_subj_means,
                      layer2_subj_means,
                      layer3_subj_means)

print("One-way ANOVA across layers (subject-mean tSNR):")
print(f"F = {Fval:.4f}, p = {pval:.4e}")

# Make violin plot
fig, ax = plt.subplots(figsize=(6,6))

data_for_plot = [
    layer1_subj_means,
    layer2_subj_means,
    layer3_subj_means
]

parts = ax.violinplot(
    data_for_plot,
    showmeans=True,
    showmedians=False,
    showextrema=True
)

ax.set_xticks([1,2,3])
ax.set_xticklabels(["Layer 1", "Layer 2", "Layer 3"])
ax.set_ylabel("Mean tSNR (per subject, avg over parcels)")
ax.set_title(f"Laminar tSNR across subjects\nANOVA: F={Fval:.2f}, p={pval:.2e}")

# overlay the individual subject points for each layer
x_jitter = 0.08
for i, layer_vals in enumerate(data_for_plot, start=1):
    # jitter the x slightly for visibility
    xs = np.random.uniform(low=i - x_jitter, high=i + x_jitter, size=len(layer_vals))
    ax.scatter(xs, layer_vals, alpha=0.6, edgecolor='k', linewidth=0.5)

plt.tight_layout()
fig_path = os.path.join(GROUP_OUT, "violin_tSNR_layers.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved violin plot to {fig_path}")

#######################################
# OPTIONAL: also save per-subject summary table
#######################################
summary_mat = np.column_stack([
    layer1_subj_means,
    layer2_subj_means,
    layer3_subj_means
])
np.save(os.path.join(GROUP_OUT, "subject_layer_mean_tSNR.npy"), summary_mat)
print("Saved per-subject layer means (Nsubj x 3)")

print("Done.")