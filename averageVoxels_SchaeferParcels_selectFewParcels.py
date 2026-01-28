import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore

# Default to your requested four, but nothing is hard-coded to 4 anymore.
SELECTED_PARCELS = [1, 2, 201, 202]   # LH_Vis_1, LH_Vis_2, RH_Vis_1, RH_Vis_2

def _make_layer_binary(layer_data, analysis_type="smallGap_Schaefer", layer_01=True):
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)
    if not layer_01:
        return layer_binary

    if analysis_type == "smallGap_Schaefer":
        layer_binary[(layer_data > 0)   & (layer_data <= 0.3)] = 1
        layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
        layer_binary[(layer_data > 0.7) & (layer_data <  1.0)] = 3
    elif analysis_type == "noGap":
        layer_binary[(layer_data > 0)   & (layer_data <= 0.4)] = 1
        layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
        layer_binary[(layer_data > 0.6) & (layer_data <  1.0)] = 3
    elif analysis_type == "largeGap_Schaefer":
        layer_binary[(layer_data > 0) & (layer_data <= 0.2)] = 1
        layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
        layer_binary[(layer_data > 0.8) & (layer_data < 1.0)] = 3
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
    return layer_binary


def averageVoxels_selected_parcels(
    subject,
    runNum="run1",
    analysis_type="smallGap_Schaefer",
    layer_01=True,
    parcels=SELECTED_PARCELS,
):
    """
    Returns array with shape (P parcels, L layers, T timepoints).
    """
    # Paths
    BOLD_data_path   = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/{subject}/merged_residuals_{runNum}.nii"
    layer_path       = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/ln_depths_equivol.nii"
    atlas_path_right = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_R_in-func.nii"
    atlas_path_left  = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_L_in-func.nii"

    output_dir = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/{analysis_type}/{subject}/"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    layer_img = nib.load(layer_path)
    atlas_img_right = nib.load(atlas_path_right)
    atlas_img_left  = nib.load(atlas_path_left)
    bold_img  = nib.load(BOLD_data_path)

    layer_data = layer_img.get_fdata(dtype=np.float32)
    atlas_R    = atlas_img_right.get_fdata(dtype=np.float32)
    atlas_L    = atlas_img_left.get_fdata(dtype=np.float32)
    bold_data  = bold_img.get_fdata(dtype=np.float32)  # (X,Y,Z,T)
    atlas_data = np.rint(atlas_L + atlas_R).astype(np.int32)

    if (layer_data.shape != atlas_data.shape) or (layer_data.shape != bold_data.shape[:-1]):
        raise ValueError(
            f"Shape mismatch: layer {layer_data.shape}, atlas {atlas_data.shape}, bold {bold_data.shape}"
        )

    n_tp = bold_data.shape[-1]

    # Layers: dynamic (sorted unique nonzero)
    layer_binary = _make_layer_binary(layer_data, analysis_type=analysis_type, layer_01=layer_01)
    unique_layers = np.array(sorted(np.unique(layer_binary[layer_binary > 0]).tolist()), dtype=int)

    P = len(parcels)
    L = len(unique_layers)
    out_PLT = np.zeros((P, L, n_tp), dtype=np.float32)

    # Compute mean timecourses per (parcel, layer)
    for li, layer_val in enumerate(unique_layers):
        layer_mask = (layer_binary == layer_val)
        if not np.any(layer_mask):
            continue

        for pi, parcel in enumerate(parcels):
            parcel_mask   = (atlas_data == parcel)
            combined_mask = layer_mask & parcel_mask
            if np.any(combined_mask):
                vox_ts  = bold_data[combined_mask, :]
                mean_ts = np.nanmean(vox_ts, axis=0)
                out_PLT[pi, li, :] = zscore(mean_ts, axis=0).astype(np.float32)
            # else leave zeros

    subj_file = os.path.join(output_dir, f"SelectedParcels_{runNum}_P{P}xL{L}xT.npy")
    np.save(subj_file, out_PLT)
    print(f"[{subject}] saved {out_PLT.shape} to {subj_file}")

    return out_PLT


def build_group_matrix(
    subjects,
    runNum="run1",
    analysis_type="smallGap_Schaefer",
    parcels=SELECTED_PARCELS,
):
    """
    Aggregates subjects into (P parcels, L layers, T time, S subjects).
    """
    first = averageVoxels_selected_parcels(
        subjects[0],
        runNum=runNum,
        analysis_type=analysis_type,
        parcels=parcels,
    )
    P, L, T = first.shape
    S = len(subjects)

    group = np.zeros((P, L, T, S), dtype=np.float32)
    group[..., 0] = first

    for si, subj in enumerate(subjects[1:], start=1):
        arr = averageVoxels_selected_parcels(
            subj,
            runNum=runNum,
            analysis_type=analysis_type,
            parcels=parcels,
        )
        if arr.shape != (P, L, T):
            raise ValueError(f"Time/shape mismatch for {subj}: expected {(P, L, T)}, got {arr.shape}")
        group[..., si] = arr

    group_dir = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/{analysis_type}/group/"
    os.makedirs(group_dir, exist_ok=True)
    group_file = os.path.join(group_dir, f"SelectedParcels_{runNum}_P{P}xL{L}xTxS.npy")
    np.save(group_file, group)
    print(f"[GROUP] saved {group.shape} to {group_file}")
    return group


subjects = ["sub-LAM001","sub-LAM002","sub-LAM003","sub-LAM004","sub-LAM005","sub-LAM006","sub-LAM009","sub-LAM011"]
group = build_group_matrix(subjects, runNum="run1", analysis_type="smallGap_Schaefer")
group.shape