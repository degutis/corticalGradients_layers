import nibabel as nib
import numpy as np
import os
from scipy.stats import zscore


def averageVoxels_parcels_with_fs(
    subject,
    runNum,
    analysis_type,
    layer_01=True
):
    """
    Extract layer-wise parcel-averaged time series using Glasser FS atlas.
    Produces per-layer matrices of shape (360 parcels, n_timepoints).
    Final concatenated matrix is (n_timepoints, 360 * n_layers) = (t, 1080) for 3 layers.
    """

    # -------------------------------------------------------------------------
    # 0) Paths
    # -------------------------------------------------------------------------
    BOLD_data_path = (
        f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
        f"derivatives/func/{subject}/merged_residuals_{runNum}.nii"
    )
    layer_path = (
        f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
        f"derivatives/ref_anat/{subject}/ln_depths_equivol.nii"
    )
    atlas_path_right = (
        f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
        f"derivatives/ref_anat/{subject}/glasser_R_in-func.nii"
    )
    atlas_path_left = (
        f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
        f"derivatives/ref_anat/{subject}/glasser_L_in-func.nii"
    )

    output_path = (
        f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
        f"derivatives/correlations/{analysis_type}/{subject}/"
    )
    os.makedirs(output_path, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Load images
    # -------------------------------------------------------------------------
    layer_img = nib.load(layer_path)
    atlas_img_right = nib.load(atlas_path_right)
    atlas_img_left = nib.load(atlas_path_left)
    bold_img = nib.load(BOLD_data_path)

    layer_data = layer_img.get_fdata()
    right = atlas_img_right.get_fdata()
    left = atlas_img_left.get_fdata()
    bold_data = bold_img.get_fdata()  # shape: (X, Y, Z, T)

    # Offset labels only for non-zero voxels so that hemispheres don't collide
    atlas_data_right = np.where(right > 0, right + 1000, 0)
    atlas_data_left = np.where(left > 0, left + 2000, 0)
    atlas_data = atlas_data_right + atlas_data_left

    # -------------------------------------------------------------------------
    # 2) Check shapes
    # -------------------------------------------------------------------------
    if (
        layer_data.shape != atlas_data.shape or
        layer_data.shape != bold_data.shape[:-1]
    ):
        raise ValueError(
            "Dimension mismatch between layer, atlas, and BOLD volumes: "
            f"layer={layer_data.shape}, atlas={atlas_data.shape}, "
            f"bold_spatial={bold_data.shape[:-1]}"
        )

    # -------------------------------------------------------------------------
    # 3) Build layer binary
    # -------------------------------------------------------------------------
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)
    if layer_01:
        if analysis_type == "smallGap_Glasser":
            layer_binary[(layer_data > 0) & (layer_data <= 0.3)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.7) & (layer_data < 1.0)] = 3

        elif analysis_type == "noGap":
            layer_binary[(layer_data > 0) & (layer_data <= 0.4)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.6) & (layer_data < 1.0)] = 3

        # NOTE: this condition duplicates "smallGap_Glasser" in your original code.
        # Leaving it as-is, but it will never be reached.
        elif analysis_type == "smallGap_Glasser":
            layer_binary[(layer_data > 0) & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.8) & (layer_data < 1)] = 3

        elif analysis_type == "eightLayers_Schaefer":
            layer_binary[(layer_data > 0.1) & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.2) & (layer_data < 0.3)] = 2
            layer_binary[(layer_data > 0.3) & (layer_data < 0.4)] = 3
            layer_binary[(layer_data > 0.4) & (layer_data < 0.5)] = 4
            layer_binary[(layer_data > 0.5) & (layer_data < 0.6)] = 5
            layer_binary[(layer_data > 0.6) & (layer_data < 0.7)] = 6
            layer_binary[(layer_data > 0.7) & (layer_data < 0.8)] = 7
            layer_binary[(layer_data > 0.8) & (layer_data < 0.9)] = 8

    else:
        layer_binary[(layer_data > 1) & (layer_data <= 3)] = 1
        layer_binary[(layer_data > 5) & (layer_data <= 7)] = 2
        layer_binary[(layer_data > 9) & (layer_data <= 11)] = 3

    unique_layers = np.unique(layer_binary[layer_binary > 0])

    # -------------------------------------------------------------------------
    # 4) Define expected FS parcels and see which ones are present
    # -------------------------------------------------------------------------
    # Expected Glasser FS labels in this scheme:
    # Right hemisphere: 1001–1180, Left hemisphere: 2001–2180
    expected_fs_parcels = np.concatenate([
        np.arange(1001, 1181),  # 180 parcels
        np.arange(2001, 2181)   # 180 parcels
    ]).astype(int)

    # What parcels actually exist in the current functional atlas
    func_parcels = np.unique(atlas_data)
    func_parcels = func_parcels[
        (func_parcels > 0) & (func_parcels <= 2180)
    ].astype(int)
    func_set = set(func_parcels)

    fs_parcels = expected_fs_parcels  # always 360 entries

    # Optional warning if some parcels are missing in this functional space
    missing = sorted(set(fs_parcels) - func_set)
    if missing:
        print(
            RuntimeError(
                f"Expected 360 parcels in FS atlas, but only "
                f"{len(func_set)} present in functional space. "
                f"Missing {len(missing)} parcels."
            )
        )

    print(f"Total expected FS parcels: {len(fs_parcels)}")

    # -------------------------------------------------------------------------
    # 5) Loop layers and parcels, extract time series
    # -------------------------------------------------------------------------
    layer_parcels_flat = []
    n_tp = bold_data.shape[-1]

    for layer in unique_layers:
        print(f"Processing layer {layer}")
        layer_mask = (layer_binary == layer)

        # Initialize output (360 parcels × timepoints)
        out = np.full((len(fs_parcels), n_tp), 0.0, dtype=float)

        for i, parcel in enumerate(fs_parcels):
            if parcel in func_set:
                parcel_mask = (atlas_data == parcel)
                combined_mask = layer_mask & parcel_mask

                if np.any(combined_mask):
                    vox_ts = bold_data[combined_mask]      # shape: (n_voxels, n_tp)
                    mean_ts = np.nanmean(vox_ts, axis=0)   # shape: (n_tp,)
                    out[i, :] = zscore(mean_ts, axis=0)
                # else: no voxels in this layer for this parcel → row stays zeros

        # Clean up NaNs/Infs just in case
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        fname = f"Layer_{runNum}_{int(layer):d}.npy"
        print(f"has_nan={np.isnan(out).any()}")
        np.save(os.path.join(output_path, fname), out)
        print(f"  → wrote {fname}")

        # Transpose to (timepoints, parcels) for concatenation later
        layer_parcels_flat.append(out.T)

    # -------------------------------------------------------------------------
    # 6) Concatenate across layers → (timepoints, parcels * n_layers)
    # -------------------------------------------------------------------------
    parcel_time_all = np.concatenate(layer_parcels_flat, axis=1)
    np.save(
        os.path.join(output_path, f"Layer_{runNum}_parcels_all_layers.npy"),
        parcel_time_all
    )

    print(parcel_time_all.shape)
    print("All done")


averageVoxels_parcels_with_fs("sub-LAM001","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM002","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM003","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM004","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM005","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM006","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM009","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM011","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM010","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM007","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM012","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM013","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM015","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM016","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM017","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM018","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM019","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM021","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM022","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM014","run1","smallGap_Glasser")
averageVoxels_parcels_with_fs("sub-LAM008","run1","smallGap_Glasser")

# averageVoxels_parcels_with_fs("sub-01","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-01","run2","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-01","run3","smallGap_Schaefer")