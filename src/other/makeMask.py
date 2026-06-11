

import nibabel as nib
import numpy as np

input_nii = '/Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-02/HCP-MMP1_in-func.nii'  # your input NIfTI file with 5000 labels
output_mask = '/Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-02/DMN.nii'  # output binary mask

selected_labels = [1030, 1031, 1032, 1033, 1034, 1035, 2030, 2031, 2032, 2033, 2034, 2035]

nii = nib.load(input_nii)
data = nii.get_fdata()
mask = np.isin(data, selected_labels).astype(np.uint8)  # 1 where label is selected, 0 elsewhere

# === Save the new mask ===
mask_nii = nib.Nifti1Image(mask, affine=nii.affine, header=nii.header)
nib.save(mask_nii, output_mask)

print(f"Mask saved to {output_mask}")