import nibabel as nib
import numpy as np
from collections import defaultdict

import nibabel as nib
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

def extend_parcels_distance(nifti_360_path, nifti_10000_path, output_path):
    # Load the NIfTI files
    img_360 = nib.load(nifti_360_path)
    img_10000 = nib.load(nifti_10000_path)
    
    data_360 = img_360.get_fdata()
    data_10000 = img_10000.get_fdata()
    
    # Get unique parcel IDs (excluding zero if it represents background)
    parcels_10000 = np.unique(data_10000[data_10000 > 0])
    
    # Compute distance transform from 360 parcels
    binary_360 = data_360 > 0
    dist_map, indices = distance_transform_edt(~binary_360, return_indices=True)
    
    # Assign each 10,000 parcel to the nearest 360 parcel
    new_data_360 = data_360.copy()
    for parcel_10000 in parcels_10000:
        mask_10000 = data_10000 == parcel_10000
        nearest_voxels = indices[:, mask_10000]
        nearest_360_parcels = data_360[nearest_voxels[0], nearest_voxels[1], nearest_voxels[2]]
        nearest_360_parcels = nearest_360_parcels[nearest_360_parcels > 0]
        if len(nearest_360_parcels) > 0:
            closest_parcel = np.bincount(nearest_360_parcels.astype(int)).argmax()
            new_data_360[mask_10000] = closest_parcel
    
    # Save the updated NIfTI file
    new_img = nib.Nifti1Image(new_data_360, img_360.affine, img_360.header)
    nib.save(new_img, output_path)
    print(f"Updated parcel NIfTI saved to {output_path}")

# Example usage
# extend_parcels('parcels_360.nii.gz', 'parcels_10000.nii.gz', 'updated_parcels_360.nii.gz')

# Example usage
extend_parcels_distance('../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func.nii', '../kenshu_dataset/derivatives/sub-02/columns/dwscaled_columns10000.nii', '../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_extended_distance.nii')
