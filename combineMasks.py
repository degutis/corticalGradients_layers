# outdated

import nibabel as nib
import numpy as np

def combine_atlas_voxels(atlas1_path, atlas2_path, output_path):
    # Load the atlas files
    atlas1_img = nib.load(atlas1_path)
    atlas2_img = nib.load(atlas2_path)
    
    # Get the data from the atlas files
    atlas1_data = atlas1_img.get_fdata()
    atlas2_data = atlas2_img.get_fdata()
    
    # Ensure the atlases have the same shape
    if atlas1_data.shape != atlas2_data.shape:
        raise ValueError("Atlas files must have the same shape")
    
    # Create a new array to store the combined voxel values
    combined_data = np.zeros(atlas1_data.shape)
    
    # Iterate over unique values in atlas1 and atlas2
    unique_values1 = np.unique(atlas1_data)
    unique_values2 = np.unique(atlas2_data)
    
    for val1 in unique_values1:
        for val2 in unique_values2:
            # Find the voxels where both atlas1 and atlas2 have the specific values
            mask = (atlas1_data == val1) & (atlas2_data == val2)
            combined_data[mask] = val1 * 1000 + val2  # Example combination strategy
    
    # Create a new NIfTI image
    combined_img = nib.Nifti1Image(combined_data, atlas1_img.affine, atlas1_img.header)
    
    # Save the new image
    nib.save(combined_img, output_path)

combine_atlas_voxels('../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii', '../kenshu_dataset/derivatives/sub-02/columns/dwscaled_columns10000.nii', '../kenshu_dataset/derivatives/sub-02/layerification/layers_columns_combo.nii')