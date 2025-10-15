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
    
    BOLD_data_path=f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/{subject}/merged_residuals_{runNum}.nii"
    layer_path=f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/ln_depths_equivol.nii"
    atlas_path_right=f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_R_in-func.nii"
    atlas_path_left=f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subject}/schaefer_L_in-func.nii"

    output_path = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/{analysis_type}/{subject}/"

    # BOLD_data_path=f"/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/func/{subject}/merged_residuals_{runNum}.nii"
    # layer_path=f"/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/ref_anat/{subject}/ln_depths_equivol.nii"
    # atlas_path_right=f"/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/ref_anat/{subject}/schaefer_R_in-func.nii"
    # atlas_path_left=f"/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/ref_anat/{subject}/schaefer_L_in-func.nii"

    # output_path = f"/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/correlations/{analysis_type}/{subject}/"



    os.makedirs(output_path, exist_ok=True)

    # 1) Load images
    layer_img = nib.load(layer_path)
    atlas_img_right = nib.load(atlas_path_right)
    atlas_img_left = nib.load(atlas_path_left)
    bold_img  = nib.load(BOLD_data_path)

    layer_data = layer_img.get_fdata()
    atlas_data_right = atlas_img_right.get_fdata()
    atlas_data_left = atlas_img_left.get_fdata()

    bold_data  = bold_img.get_fdata()
    atlas_data = atlas_data_right + atlas_data_left

    # 2) Check shapes
    if (layer_data.shape != atlas_data.shape or
        layer_data.shape != bold_data.shape[:-1]):
        raise ValueError("Dimension mismatch between layer, atlas, and BOLD volumes")

    # 3) Build layer binary
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)
    if layer_01:
        
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
        layer_binary[(layer_data > 1)  & (layer_data <= 3)]  = 1
        layer_binary[(layer_data > 5)  & (layer_data <= 7)]  = 2
        layer_binary[(layer_data > 9)  & (layer_data <= 11)] = 3

    unique_layers = np.unique(layer_binary[layer_binary > 0])

    fs_parcels = np.unique(atlas_data)
    fs_parcels = fs_parcels[
        (fs_parcels > 0)  & (fs_parcels <= 400)
    ]
    fs_parcels = np.sort(fs_parcels).astype(int)


    if fs_parcels.size != 400:
        print(RuntimeError(f"Expected 400 parcels in FS atlas, got {fs_parcels.size}"))
        fs_parcels = np.arange(1,401)

    func_parcels = np.unique(atlas_data)
    func_parcels = func_parcels[
        (func_parcels > 0)  & (func_parcels <= 400)
    ].astype(int)

    func_set = set(func_parcels)
    layer_parcels_flat = []
    
    # 6) Loop layers and parcels
    for layer in unique_layers:
        print(f"Processing layer {layer}")
        layer_mask = (layer_binary == layer)

        # Initialize output with 0s
        n_tp = bold_data.shape[-1]
        out = np.full((len(fs_parcels), n_tp), 0, dtype=float)

        for i, parcel in enumerate(fs_parcels):
            if parcel in func_set:
                parcel_mask   = (atlas_data == parcel)
                combined_mask = layer_mask & parcel_mask

                if np.any(combined_mask):
                    vox_ts    = bold_data[combined_mask]
                    mean_ts   = np.nanmean(vox_ts, axis=0)
                    out[i, :] = zscore(mean_ts, axis=0)
                    # out[i, :] = mean_ts
                else:
                    print(parcel)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        fname = f"Layer_{runNum}_{int(layer):d}.npy"
        print(f"has_nan={np.isnan(out).any()}, ")
        np.save(os.path.join(output_path, fname), out)
        print(f"  → wrote {fname}")
        
        layer_parcels_flat.append(out.T)

    parcel_time_all = np.concatenate(layer_parcels_flat, axis=1)

    np.save(os.path.join(output_path, f"Layer_{runNum}_parcels_all_layers.npy"),
            parcel_time_all)
    print(parcel_time_all.shape)
    print("All done")


# averageVoxels_parcels_with_fs("sub-LAM001","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM002","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM003","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM004","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM005","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM006","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM009","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM011","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM010","run1","largeGap_Schaefer")
averageVoxels_parcels_with_fs("sub-LAM007","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM012","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM013","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM015","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM016","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM017","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM018","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM019","run1","largeGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM021","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM022","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-LAM014","run1","smallGap_Schaefer")

# averageVoxels_parcels_with_fs("sub-01","run1","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-01","run2","smallGap_Schaefer")
# averageVoxels_parcels_with_fs("sub-01","run3","smallGap_Schaefer")