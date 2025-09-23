#!/bin/bash

export SUBJECTS_DIR=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer/

subject=sub-LAM001

# Bring Glasser into subject space


# Bring Glasser into subject space

mri_surf2surf --srcsubject fsaverage \
mri_surf2surf --srcsubject fsaverage \
            --trgsubject ${subject} \
            --hemi lh \
            --sval-annot /home/degutis/repos/HCP_surfaces/lh.HCP-MMP1.annot \
            --tval ${SUBJECTS_DIR}/${subject}/label/lh.HCP-MMP1.annot

mri_surf2surf --srcsubject fsaverage \
            --trgsubject ${subject} \
            --hemi rh \
            --sval-annot /home/degutis/repos/HCP_surfaces/rh.HCP-MMP1.annot \
            --tval ${SUBJECTS_DIR}/${subject}/label/rh.HCP-MMP1.annot

mri_aparc2aseg --s ${subject} --annot HCP-MMP1 --o ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz --labelwm

antsApplyTransforms --interpolation BSpline[5] \
                    -d 3 \
                    -i ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz \
                    -r ${refAnatDir}/fs_t1_in-func.nii \
                    -t ${refAnatDir}/fs_to_func_1Warp.nii.gz \
                    -t ${refAnatDir}/fs_to_func_0GenericAffine.mat \
                    -o ${refAnatDir}/HCP-MMP1_in-func.nii \
                    -n NearestNeighbor

### Maybe this is not necessary:

#fslmaths -dt int \
#        ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz \
#        -thr 999 \
#        -mul 1 \
#        ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1_GM.nii.gz

#flirt -in ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1_GM.nii.gz \
#    -ref ${refAnatDir}/${subject}_bold_SMSEPI_mc_MEAN_n4_brain.nii \
#    -out ${refAnatDir}/Glasser_in_anat.nii \
#    -applyxfm -interp nearestneighbour -usesqform
