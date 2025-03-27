import nibabel as nib
import numpy as np
import os
from scipy.stats import zscore


def averageVoxels_parcels(layer_path, atlas_path, BOLD_data_path, output_path):

    os.makedirs(output_path, exist_ok=True)  

    layer_img = nib.load(layer_path)
    atlas_img = nib.load(atlas_path)
    bold_img = nib.load(BOLD_data_path)
    bold_data = bold_img.get_fdata()
   
    layer_data = layer_img.get_fdata()

    layer_binary = np.zeros_like(layer_data)
#    layer_binary[(layer_data > 0) & (layer_data <= 0.33)] = 1
#    layer_binary[(layer_data > 0.33) & (layer_data <= 0.66)] = 2
#    layer_binary[(layer_data > 0.66) & (layer_data <= 0.999)] = 3

#    layer_binary[(layer_data > 0) & (layer_data <= 0.2)] = 1
#    layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
#    layer_binary[(layer_data > 0.8) & (layer_data <= 0.999)] = 3

    layer_binary[(layer_data > 1) & (layer_data <= 3)] = 1
    layer_binary[(layer_data > 5) & (layer_data <= 7)] = 2
    layer_binary[(layer_data > 9) & (layer_data <= 11)] = 3


    atlas_data = atlas_img.get_fdata()
    
    if layer_data.shape != atlas_data.shape:
        raise ValueError("Atlas files must have the same shape")
        
    unique_layers = np.unique(layer_binary)
    unique_layers = unique_layers[unique_layers != 0]  # Exclude outside of the brains
    print(f"Unique layers: {unique_layers}")
    
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[(unique_parcels >= 1001) & (unique_parcels <= 3000) & (unique_parcels != 2000)]  
    print(f"Unique parcels: {len(unique_parcels)}")

    for layer in unique_layers:
        print(f"Processing layer {layer}")

        layer_mask = np.zeros(layer_binary.shape)
        layer_mask[layer_binary == layer] = 1
        
        meaned_parcel = np.zeros((len(unique_parcels), bold_data.shape[-1]))

        for index_p, parcel in enumerate(unique_parcels):
            print(f"    Processing parcel {parcel}")

            parcel_mask = np.zeros(atlas_data.shape)
            parcel_mask[atlas_data == parcel] = 1
            combined_mask = np.logical_and(layer_mask, parcel_mask)

            masked_data = np.zeros_like(bold_data)
            for vol in range(bold_data.shape[-1]):
                #masked_data[..., vol] = combined_mask * bold_data[..., vol]
                masked_data[..., vol] = np.where(combined_mask, bold_data[..., vol], np.nan)

            #meaned_parcel[index_p, ...] = np.mean(masked_data, axis=(0, 1, 2))
            meaned_parcel[index_p, :] = zscore(np.nanmean(masked_data, axis=(0, 1, 2)),axis=0)

        np.save(f"{output_path}/Layer_{int(layer)}.npy", meaned_parcel)

#averageVoxels_parcels('../highRes_resting/derivatives/ref_anat/sub-01/ln_depths_equivol.nii', '../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii', '../highRes_resting/sub-01/func/derivatives/merged_residuals.nii', "../highRes_resting/derivatives/correlations/sub-01/Gap")

#averageVoxels_parcels('../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii', '../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func.nii', '../kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_VASO_across_days.nii', "../kenshu_dataset/derivatives/sub-02/correlations/Glasser_Gap")

#'sub-02/layerification/sub-02_layers_interpolated.nii', '../kenshu_dataset/derivatives/sub-02/anatomical/columns_100_in-fs.nii', '../kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_VASO_across_days.nii', '../kenshu_dataset/derivatives/sub-02/correlations/100Columns')


def averageVoxels_parcels_optimized(layer_path, atlas_path, BOLD_data_path, output_path, layer_01=True):
    os.makedirs(output_path, exist_ok=True)  

    # Load data
    layer_img = nib.load(layer_path)
    atlas_img = nib.load(atlas_path)
    bold_img = nib.load(BOLD_data_path)
    
    bold_data = bold_img.get_fdata()
    layer_data = layer_img.get_fdata()
    atlas_data = atlas_img.get_fdata()

    # Ensure matching shapes
    if layer_data.shape != atlas_data.shape or layer_data.shape != bold_data.shape[:-1]:
        raise ValueError("Mismatch in dimensions between atlas, layer, and BOLD data.")

    # Define layer binary segmentation
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)


    if layer_01:
        layer_binary[(layer_data > 0) & (layer_data <= 0.2)] = 1
        layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
        layer_binary[(layer_data > 0.8) & (layer_data <= 0.999)] = 3

    else:
        layer_binary[(layer_data > 1) & (layer_data <= 3)] = 1
        layer_binary[(layer_data > 5) & (layer_data <= 7)] = 2
        layer_binary[(layer_data > 9) & (layer_data <= 11)] = 3


    # Get unique layer & parcel labels
    unique_layers = np.unique(layer_binary[layer_binary > 0])
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[(unique_parcels > 0)]
    unique_parcels = unique_parcels[(unique_parcels >= 1001) & (unique_parcels <= 3000) & (unique_parcels != 2000)]

    print(f"Unique layers: {unique_layers}")
    print(f"Unique parcels: {len(unique_parcels)}")

    for layer in unique_layers:
        print(f"Processing layer {layer}")
        layer_mask = (layer_binary == layer)  # Boolean mask

        meaned_parcel = np.zeros((len(unique_parcels), bold_data.shape[-1]))  # Default 0

        for index_p, parcel in enumerate(unique_parcels):

            parcel_mask = (atlas_data == parcel)  # Boolean mask
            combined_mask = np.logical_and(layer_mask, parcel_mask)  # Final mask

            if np.any(combined_mask):  # If there is overlap, compute mean
                voxel_data = bold_data[combined_mask]  # Extract time series
                meaned_parcel[index_p, :] = zscore(np.nanmean(voxel_data, axis=0), axis=0)

        np.save(os.path.join(output_path, f"Layer_{int(layer)}.npy"), meaned_parcel)

#averageVoxels_parcels_optimized('../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii', '../kenshu_dataset/derivatives/sub-02/columns/dwscaled_columns10000.nii', '../kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_VASO_across_days.nii', "../kenshu_dataset/derivatives/sub-02/correlations/10kColumns")
#averageVoxels_parcels_optimized('../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii', '../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_extended_distance.nii', '../kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_VASO_across_days.nii', "../kenshu_dataset/derivatives/sub-02/correlations/Glasser_extended_distance")

#averageVoxels_parcels_optimized('../highRes_resting/derivatives/ref_anat/sub-01/ln_depths_equivol.nii', '../highRes_resting/derivatives/ref_anat/sub-01/HCP-MM1_in-func.nii', '../highRes_resting/sub-01/func/derivatives/merged_residuals.nii', "../highRes_resting/derivatives/correlations/sub-01/Gap_new")

averageVoxels_parcels_optimized('../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii', '../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_extended_distance.nii', '../kenshu_dataset/derivatives/sub-02/VASO_func/preprocessed/sub-02_VASO_across_days_residuals.nii', "../kenshu_dataset/derivatives/sub-02/correlations/Glasser_residuals", layer_01=False)
