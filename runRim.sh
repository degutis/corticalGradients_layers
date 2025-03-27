#!/bin/bash
#
# import-fs-ribbon.sh <freesurferDir> <analysisDir> <vaso_t1>
#
# - imports ribbon file from freesurfer
# - tranaforms it to func space
# - makes values compatible with LAYNII

freesurferDir=/Users/karolis/Desktop/kenshu_dataset/derivatives/freesurfer/sub-02
analysisDir=/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical
funcDir=/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/VASO_func

export FSLOUTPUTTYPE=NIFTI

mri_convert ${freesurferDir}/mri/ribbon.mgz ${analysisDir}/fs_ribbon.nii

antsApplyTransforms --interpolation BSpline[5] \
                    -d 3 \
                    -i ${analysisDir}/fs_ribbon.nii \
                    -o ${analysisDir}/fs_ribbon_highRes.nii \
                    -n NearestNeighbor

fslmaths ${analysisDir}/fs_ribbon_in-func.nii -rem 39 -max 1 \
         ${analysisDir}/rim.nii.gz
gunzip ${analysisDir}/rim.nii.gz

fslmaths ${funcDir}/sub-02_VASO_across_days.nii -Tmean ${funcDir}/sub-02_mean_func.nii.gz
gunzip ${analysisDir}/sub-02_mean_func.nii.gz

LN2_LAYERS -rim ${analysisDir}/rim.nii -nr_layers 3
LN2_COLUMNS -rim ${analysisDir}/rim.nii -midgm ${analysisDir}/rim_midGM_equidist.nii -nr_columns 1000

gunzip ${analysisDir}/layers_3_equidist_in-fs.nii.gz

flirt -in ${analysisDir}/rim_columns100.nii -ref ${funcDir}/sub-02_VASO_across_days.nii -out ${analysisDir}/columns_100_in-fs.nii.gz -applyxfm -usesqform
gunzip ${analysisDir}/columns_100_in-fs.nii.gz