#!/bin/bash

export SUBJECTS_DIR=/Users/karolis/Desktop/highRes_resting/derivatives/freesurfer

subject=sub-99
refAnatDir=/Users/karolis/Desktop/highRes_Resting/derivatives/ref_anat/${subject}
funcDir=/Users/karolis/Desktop/highRes_Resting/${subject}/reo/

# Bring Glasser into subject space

# mri_surf2surf --srcsubject fsaverage \
#             --trgsubject ${subject} \
#             --hemi lh \
#             --sval-annot /Users/karolis/Desktop/repos/HCP_surfaces/lh.HCP-MMP1.annot \
#             --tval ${SUBJECTS_DIR}/${subject}/label/lh.HCP-MMP1.annot

# mri_surf2surf --srcsubject fsaverage \
#             --trgsubject ${subject} \
#             --hemi rh \
#             --sval-annot /Users/karolis/Desktop/repos/HCP_surfaces/rh.HCP-MMP1.annot \
#             --tval ${SUBJECTS_DIR}/${subject}/label/rh.HCP-MMP1.annot

# mri_aparc2aseg --s ${subject} --annot HCP-MMP1 --o ${SUBJECTS_DIR}/${subject}/mri/HCP-MMP1.nii.gz --labelwm


antsApplyTransforms --interpolation BSpline[5] \
                    -d 3 \
                    -i ${funcDir}/OrigRes_Glu_reo.nii \
                    -r ${refAnatDir}/fs_T1.nii \
                    -t ${refAnatDir}/mrs_to_fs_1Warp.nii.gz \
                    -t ${refAnatDir}/mrs_to_fs_0GenericAffine.mat \
                    -o ${refAnatDir}/mrs_Glu-in_t1_appliedTrans.nii \
                    -n BSpline[5]

antsApplyTransforms --interpolation BSpline[5] \
                    -d 3 \
                    -i ${funcDir}/OrigRes_Gln_reo.nii \
                    -r ${refAnatDir}/fs_T1.nii \
                    -t ${refAnatDir}/mrs_to_fs_1Warp.nii.gz \
                    -t ${refAnatDir}/mrs_to_fs_0GenericAffine.mat \
                    -o ${refAnatDir}/mrs_Gln-in_t1_appliedTrans.nii \
                    -n BSpline[5]

