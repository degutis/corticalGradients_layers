#!/bin/bash

export SUBJECTS_DIR=/Users/karolis/Desktop/kenshu_dataset/derivatives/freesurfer/

anatFileDir=../kenshu_dataset/derivatives/sub-02/anatomical/
anatFileName=T1.nii
subject=sub-02

mkdir $SUBJECTS_DIR

echo "mris_inflate -n 100" > expert.opts

recon-all -i ${anatFileDir}/${anatFileName} \
          -hires \
          -autorecon1 \
          -noskullstrip \
          -sd ${SUBJECTS_DIR} \
          -s ${subject} \
          -parallel    

cp ${anatFileName}/T1.nii ${SUBJECTS_DIR}/${subject}/mri/T1.nii
mri_convert ${SUBJECTS_DIR}/${subject}/mri/T1.nii ${SUBJECTS_DIR}/${subject}/mri/T1.mgz 

cp ${anatFileName}/mask.nii ${SUBJECTS_DIR}/${subject}/mri/brainmask_mask.nii
mri_convert ${SUBJECTS_DIR}/${subject}/mri/brainmask_mask.nii ${SUBJECTS_DIR}/${subject}/mri/brainmask_mask.mgz 

cp ${anatFileName}/T1.nii ${SUBJECTS_DIR}/${subject}/mri/brainmask.nii
mri_convert ${SUBJECTS_DIR}/${subject}/mri/brainmask.nii ${SUBJECTS_DIR}/${subject}/mri/brainmask.mgz 

recon-all -hires \
          -autorecon2 \
          -autorecon3 \
          -sd ${SUBJECTS_DIR} \
          -s ${subject} \
          -expert expert.opts \
          -xopts-overwrite \
          -parallel   

# Bring Glasser into subject space
# need to cd to the palce where lh.HCP is stored. 
mri_surf2surf --srcsubject fsaverage \
            --trgsubject ${subject} \
            --hemi lh \
            --sval-annot lh.HCP-MMP1.annot \
            --tval ${SUBJECTS_DIR}/${subject}/label/lh.HCP-MMP1.annot

mri_surf2surf --srcsubject fsaverage \
            --trgsubject ${subject} \
            --hemi rh \
            --sval-annot rh.HCP-MMP1.annot \
            --tval ${SUBJECTS_DIR}/${subject}/label/rh.HCP-MMP1.annot

mri_aparc2aseg --s ${subject} --annot HCP-MMP1 --o ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz --labelwm

### Maybe this is not necessary:

fslmaths -dt int \
        ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz \
        -thr 999 \
        -mul 1 \
        ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1_GM.nii.gz


antsApplyTransforms -i /Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/sub-01/mri/HCP-MMP1.nii.gz \
                    -r /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/sub-01_bold_SMSEPI_mc_MEAN.nii \
                    -o /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/HCP-MMP1_in-func.nii \
                    -n NearestNeighbor \
                    -t /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/fs_to_func_0GenericAffine.mat /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/fs_to_func_1Warp.nii.gz\
                    -v

flirt -in /Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/sub-01/mri/HCP-MMP1.nii \
    -ref /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/sub-01_bold_SMSEPI_mc_MEAN.nii \
    -out /Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/sub-01/ses-02/HCP-MMP1_in_func_full.nii \
    -applyxfm -interp nearestneighbour -usesqform

#flirt -in /Users/karolis/Desktop/kenshu_dataset/derivatives/freesurfer/sub-02/mri/HCP-MMP1.nii \
#    -ref /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_mean_func.nii \
#    -out /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_full.nii \
#    -applyxfm -interp nearestneighbour -usesqform


flirt -in /Users/karolis/Desktop/highRes_resting/derivatives/freesurfer/sub-01/mri/HCP-MMP1_GM.nii.gz \
    -ref /Users/karolis/Desktop/highRes_resting/derivatives/sub-01/layerification/sub-02_layers_interpolated.nii \
    -out /Users/karolis/Desktop/highRes_resting/derivatives/sub-01/atlas/Glasser_in_func.nii \
    -applyxfm -interp nearestneighbour -usesqform