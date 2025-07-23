#!/bin/bash

subject=sub-99
fs_dir=/Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/${subject}/
neuroModName=Gln
bold_file=OrigRes_${neuroModName}_reo.nii
bold_file_withoutExt=OrigRes_${neuroModName}_reo
mrs_dir=/Users/karolis/Desktop/highRes_Resting/${subject}/reo/

cd ${mrs_dir}

mri_convert ${fs_dir}/mri/brain.mgz fs_T1.nii

gunzip ${bold_file}

# antsRegistration \
#     --verbose 1 \
#     --dimensionality 3  \
#     --float 0  \
#     --collapse-output-transforms 1  \
#     --interpolation BSpline[5] \
#     --output [fs_to_mrs_,fs_to_mrs_Warped.nii,fs_to_mrs_InverseWarped.nii]  \
#     --use-histogram-matching 0  \
#     --winsorize-image-intensities [0.005,0.995]  \
#     --transform Rigid[0.1]  \
#     --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
#     --convergence [1000x500x250x100,1e-6,10]  \
#     --shrink-factors 12x8x4x2  \
#     --smoothing-sigmas 4x3x2x1vox  \
#     -x mask.nii \
#     --transform Affine[0.1]  \
#     --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
#     --convergence [1000x500x250x100,1e-6,10]  \
#     --shrink-factors 12x8x4x2  \
#     --smoothing-sigmas 4x3x2x1vox  \
#     -x mask.nii \
#     --transform SyN[0.1,3,0]  \
#     --metric CC[${bold_file},fs_T1.nii,1,4]  \
#     --convergence [50x50x70x50x20,1e-6,10]  \
#     --shrink-factors 10x6x4x2x1  \
#     --smoothing-sigmas 5x3x2x1x0vox  \
#     -x mask.nii > antsRegistration.log 2>&1

antsRegistration \
    --verbose 1 \
    --dimensionality 3  \
    --float 0  \
    --collapse-output-transforms 1  \
    --interpolation BSpline[5] \
    --output [mrs_to_fs_,mrs_to_fs_Warped.nii,mrs_to_fs_InverseWarped.nii]  \
    --use-histogram-matching 0  \
    --winsorize-image-intensities [0.005,0.995]  \
    --transform Rigid[0.1]  \
    --metric MI[fs_T1.nii,${bold_file},1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii \
    --transform Affine[0.1]  \
    --metric MI[fs_T1.nii,${bold_file},1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii \
    --transform SyN[0.1,3,0]  \
    --metric CC[fs_T1.nii,${bold_file},1,4]  \
    --convergence [50x50x70x50x20,1e-6,10]  \
    --shrink-factors 10x6x4x2x1  \
    --smoothing-sigmas 5x3x2x1x0vox  \
    -x mask.nii > antsRegistration_${neuroModName}.log 2>&1

cp mrs_to_fs_Warped.nii mrs_${neuroModName}-in_t1.nii
# fslcpgeom ${bold_file} fs_t1_in-${neuroModName}.nii # correct for possible small affine changes