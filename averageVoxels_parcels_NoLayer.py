import nibabel as nib
import numpy as np
import os
from scipy.stats import zscore


def averageVoxels_parcels_with_fs(
    BOLD_data_path,
    atlas_path,
    output_path,
    runNum
    ):
    
    os.makedirs(output_path, exist_ok=True)

    # 1) Load images
    atlas_img = nib.load(atlas_path)
    bold_img  = nib.load(BOLD_data_path)

    atlas_data = atlas_img.get_fdata()
    bold_data  = bold_img.get_fdata()

    func_parcels = np.unique(atlas_data)
    func_parcels = func_parcels[
        (func_parcels >= 1001) & (func_parcels <= 3000) & (func_parcels != 2000)
    ].astype(int)

    out = np.full((len(func_parcels), 1), np.nan, dtype=float)

    for i, parcel in enumerate(func_parcels):
        parcel_mask   = (atlas_data == parcel)
        combined_mask = parcel_mask

        if np.any(combined_mask):
            vox_ts    = bold_data[combined_mask]
            out[i,:]   = np.nanmean(vox_ts, axis=0)

        fname = f"Parcel_mean_{runNum}.npy"
        np.save(os.path.join(output_path, fname), out)

    print("All done.")

# averageVoxels_parcels_with_fs('../highRes_resting/derivatives/ref_anat/sub-99/mrs_GABA-in_t1.nii', 
#                               '../highRes_resting/derivatives/freesurfer/sub-99/mri/HCP-MMP1.nii.gz', 
#                               "../highRes_resting/derivatives/MRSI/",
#                               runNum="GABA")

# averageVoxels_parcels_with_fs('../highRes_resting/derivatives/ref_anat/sub-99/mrs_Glu-in_t1_appliedTrans.nii', 
#                               '../highRes_resting/derivatives/freesurfer/sub-99/mri/HCP-MMP1.nii.gz', 
#                               "../highRes_resting/derivatives/MRSI/",
#                               runNum="GlutamicAcid")

# averageVoxels_parcels_with_fs('../highRes_resting/derivatives/ref_anat/sub-99/mrs_Gln-in_t1_appliedTrans.nii', 
#                               '../highRes_resting/derivatives/freesurfer/sub-99/mri/HCP-MMP1.nii.gz', 
#                               "../highRes_resting/derivatives/MRSI/",
#                               runNum="Glutamine")


averageVoxels_parcels_with_fs("sub-01","run1","smallGap")
averageVoxels_parcels_with_fs("sub-01","run2","smallGap")
averageVoxels_parcels_with_fs("sub-01","run3","smallGap")

averageVoxels_parcels_with_fs("sub-02","run1","smallGap")
averageVoxels_parcels_with_fs("sub-02","run2","smallGap")
averageVoxels_parcels_with_fs("sub-02","run3","smallGap")
averageVoxels_parcels_with_fs("sub-02","run4","smallGap")

averageVoxels_parcels_with_fs("sub-03","run1","smallGap")
averageVoxels_parcels_with_fs("sub-03","run2","smallGap")
averageVoxels_parcels_with_fs("sub-03","run3","smallGap")
averageVoxels_parcels_with_fs("sub-03","run4","smallGap")
averageVoxels_parcels_with_fs("sub-03","run5","smallGap")