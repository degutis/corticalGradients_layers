#!/bin/bash
set -euo pipefail

subject=sub-LAM011
# fs_dir=/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/freesurfer/${subject}/
fs_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer/${subject}/
bold_file=${subject}_MEAN.nii
bold_file_withoutExt=${subject}_MEAN
func_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/${subject}
ref_anat_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/${subject}

cd ${func_dir}

if [[ -f "${subject}_MEAN.nii" ]]; then
    echo "${subject}_MEAN.nii already exists. Skipping computation."
else
    echo "Merging all motion-corrected runs for ${subject}…"
    # fslmerge -t all_func_runs.nii.gz *_mc.nii*
    fslmerge -t all_func_runs.nii.gz *NORDIC_mc.nii*

    echo "Computing temporal mean…"
    fslmaths all_func_runs.nii.gz -Tmean "${subject}_MEAN.nii.gz"

    echo "Removing intermediate 4D file…"
    rm all_func_runs.nii.gz

    gunzip "${subject}_MEAN.nii.gz"  # -k keeps .gz, -f overwrites
    echo "Done! Output: ${subject}_MEAN.nii.gz"
fi

mri_convert ${fs_dir}/mri/brain.mgz fs_T1.nii

n4bold_file=${bold_file_withoutExt}_n4.nii
N4BiasFieldCorrection -i ${bold_file} -o ${n4bold_file}
bold_file=${n4bold_file}

bold_brain_file=${bold_file_withoutExt}_n4_brain.nii
# bet ${bold_file} ${bold_brain_file} -f 0.15
bet ${bold_file} ${bold_brain_file} -f 0.2
bold_file=${bold_brain_file}
gunzip ${bold_file}.gz -f

# bold_brain_file=${bold_file_withoutExt}_n4_brain.nii
# bold_brain_file_mask=${bold_file_withoutExt}_n4_brain_mask.nii

# 3dAutomask -clfrac 0.8 -peels 1 \
#            -prefix       ${bold_brain_file_mask} \
#            -apply_prefix ${bold_brain_file} \
#            ${bold_file}



mask_file=mask_le1500.nii.gz
fslmaths ${bold_file} -uthr 800 -bin ${mask_file}

threshold_file=${bold_file_withoutExt}_n4_brain_thresh.nii
fslmaths ${bold_file} -mas ${mask_file} ${threshold_file}.gz
gunzip ${threshold_file}.gz -f

bold_brain_file=${bold_file_withoutExt}_n4_brain2.nii
bet ${threshold_file} ${bold_brain_file} -f 0.1
bold_file=${bold_brain_file}
gunzip ${bold_file}.gz -f


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
    --metric MI[${bold_file},fs_T1.nii,1,64,Regular,0.25]  \
    --convergence [50x50x70x50x20,1e-6,10]  \
    --shrink-factors 10x6x4x2x1  \
    --smoothing-sigmas 5x3x2x1x0vox  \
    -x mask.nii > antsRegistration.log 2>&1

cp fs_to_func_Warped.nii fs_t1_in-func.nii
fslcpgeom ${bold_file} fs_t1_in-func.nii # correct for possible small affine changes

mkdir -p $ref_anat_dir

mv fs_to_func_Warped.nii $ref_anat_dir/fs_to_func_Warped.nii
mv fs_to_func_InverseWarped.nii $ref_anat_dir/fs_to_func_InverseWarped.nii
mv fs_to_func_0GenericAffine.mat $ref_anat_dir/fs_to_func_0GenericAffine.mat
mv fs_to_func_1Warp.nii.gz $ref_anat_dir/fs_to_func_1Warp.nii.gz
mv fs_to_func_1InverseWarp.nii.gz $ref_anat_dir/fs_to_func_1InverseWarp.nii.gz
mv fs_t1_in-func.nii $ref_anat_dir/fs_t1_in-func.nii

    # --metric CC[${bold_file},fs_T1.nii,1,4]  \
