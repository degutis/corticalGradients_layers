#!/bin/bash

subject=sub-04
fs_dir=/Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/${subject}/
bold_file=${subject}_bold_SMSEPI_mc_MEANED_full.nii
bold_file_withoutExt=${subject}_bold_SMSEPI_mc_MEANED_full
func_dir=/Users/karolis/Desktop/highRes_Resting/${subject}/func/derivatives

cd ${func_dir}

# fslmerge -t all_func_runs.nii sub-04_run-01_bold_SMSEPI_mc.nii sub-04_run-02_bold_SMSEPI_mc.nii sub-04_run-03_bold_SMSEPI_mc.nii
# fslmaths all_func_runs.nii.gz -Tmean sub-04_MEAN.nii.gz

mri_convert ${fs_dir}/mri/brain.mgz fs_T1.nii

n4bold_file=${bold_file_withoutExt}_n4.nii
N4BiasFieldCorrection -i ${bold_file} -o ${n4bold_file}
bold_file=${n4bold_file}

# bold_brain_file=${bold_file_withoutExt}_n4_brain.nii
# bet ${bold_file} ${bold_brain_file} -f 0.07 
# bold_file=${bold_brain_file}
# gunzip ${bold_file}.gz

# mask_file=mask_le1500.nii.gz
# fslmaths ${bold_file} -uthr 1000 -bin ${mask_file}

# threshold_file=${bold_file_withoutExt}_n4_brain_thresh.nii
# fslmaths ${bold_file} -mas ${mask_file} ${threshold_file}.gz
# gunzip ${threshold_file}.gz

# bold_brain_file=${bold_file_withoutExt}_n4_brain2.nii
# bet ${threshold_file} ${bold_brain_file} -f 0.07 
# bold_file=${bold_brain_file}
# gunzip ${bold_file}.gz


antsRegistration \
    --verbose 1 \
    --dimensionality 3  \
    --float 0  \
    --collapse-output-transforms 1  \
    --interpolation BSpline[5] \
    --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii]  \
    --use-histogram-matching 0  \
    --winsorize-image-intensities [0.005,0.995]  \
    --transform Rigid[0.1]  \
    --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii \
    --transform Affine[0.1]  \
    --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii \
    --transform SyN[0.1,3,0]  \
    --metric CC[${bold_file},fs_T1.nii,1,4]  \
    --convergence [50x50x70x50x20,1e-6,10]  \
    --shrink-factors 10x6x4x2x1  \
    --smoothing-sigmas 5x3x2x1x0vox  \
    -x mask.nii > antsRegistration.log 2>&1

cp fs_to_func_Warped.nii fs_t1_in-func.nii
fslcpgeom ${bold_file} fs_t1_in-func.nii # correct for possible small affine changes