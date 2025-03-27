#!/bin/zsh

# run CAT12 manually
# take the output of CAT12 p0ANAT.nii and make a brainmask

fslmaths /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical/mri/p0ANAT.nii -bin /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical/mri/brainmask.nii
mri_convert /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical/mri/brainmask.nii /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/anatomical/mri/brainmask.mgz

export SUBJECTS_DIR=/Users/karolis/Desktop/kenshu_dataset/derivatives/freesurfer/

# run recon-all

recon-all -s sub-02 -i sub-02/anatomical/ANAT.nii  \                            
    -xmask sub-02/anatomical/mri/brainmask.mgz -all -noskullstrip