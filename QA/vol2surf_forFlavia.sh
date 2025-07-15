#!/usr/bin/env bash

# Usage: parcel_ts.sh <func4D.nii.gz> <fs_subjID> <atlas.dlabel.nii>
# Requires: FreeSurfer (bbregister, mri_vol2surf, mri_convert),
#           Connectome Workbench (wb_command),
#           FSL (fslval)

FUNC=$1
SUBJ=$2
ATLAS_DLABEL=$3
# script assumes you've already run recon-all in FreeSurfer
# and your functional data is preprocessed/motion corrected

# you need to have your altas saved as .annot files under $SUBJECTS_DIR/fsaverage/label/.
# either use the HCP Glasser parcels or Schaefer2018_400Parcels

export SUBJECTS_DIR=${SUBJECTS_DIR:-/path/to/freesurfer/subjects}
OUTPREFIX=${SUBJ}_native

# register your functional data to the anatomical space
# https://surfer.nmr.mgh.harvard.edu/fswiki/bbregister
bbregister --s "$SUBJ" \
           --mov "$FUNC" \
           --bold \
           --init-fsl \
           --reg func2anat.dat


# volume to surface mapping for each hemisphere of the brain
# projfrac just gives you the mapping to the middle of the gray matter ribbon (shouldn't matter much for low-res data)
# https://surfer.nmr.mgh.harvard.edu/fswiki/mri_vol2surf
for HEMI in lh rh; do
  mri_vol2surf \
    --mov    "$FUNC" \
    --reg    func2anat.dat \
    --hemi   $HEMI \
    --projfrac 0.5 \
    --interp trilinear \
    --o      ${HEMI}.bold_ribbon.mgz
done

# convert to gifti format from freesurfer format for easier processing in the next steps
# https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert
for HEMI in lh rh; do
  mri_convert ${HEMI}.bold_ribbon.mgz ${HEMI}.bold.func.gii
done

# get number of timepoints and TR - should work if input well defined. Would print echo in the beginning to make sure it's getting the right values
# https://www.humanconnectome.org/software/workbench-command/-cifti-create-dense-timeseries
NTR=$(fslval "$FUNC" dim4)
TR=$(fslval "$FUNC" pixdim4)
wb_command -cifti-create-dense-timeseries \
    ${OUTPREFIX}.dtseries.nii \
    -left-metric  lh.bold.func.gii \
    -right-metric rh.bold.func.gii \
    -tseries-length $NTR \
    -tseries-spacing $TR

# parcelate the dense timeseries into atlas-based timecourses. 
# https://www.humanconnectome.org/software/workbench-command/-cifti-parcellate
wb_command -cifti-parcellate \
    ${OUTPREFIX}.dtseries.nii \
    $ATLAS_DLABEL \
    COLUMN \
    ${OUTPREFIX}_ptseries.nii

# Visualization: you can open the dtseries on Connectome Workbench (wb_view).
# take a look at some tutorials on youtube to see how it works. 

