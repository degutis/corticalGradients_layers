import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Load the NIfTI file
nii_file = "../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers.nii" 

nii_img = nib.load(nii_file)
nii_data = nii_img.get_fdata()

# Create a mask for zeros
zero_mask = (nii_data == 0)

# Normalize only nonzero values from [1, 11] -> [0, 1]
nii_data_norm = (nii_data - 1) / (11 - 1)
nii_data_norm[zero_mask] = 0  # Keep zeros unchanged

# Scale to [1, 3]
nii_data_scaled = 1 + 2 * nii_data_norm

# Round to nearest integer (ensuring only 1, 2, or 3)
nii_data_rounded = np.round(nii_data_scaled).astype(np.int32)

# Restore original zeros
nii_data_rounded[zero_mask] = 0

# Save the new NIfTI file
new_nii = nib.Nifti1Image(nii_data_rounded, affine=nii_img.affine)

nib.save(new_nii, "../kenshu_dataset/derivatives/sub-02/layerification/sub-02_layers_interpolated.nii")