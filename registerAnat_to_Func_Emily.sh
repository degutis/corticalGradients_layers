T2_file=SEME15_rT1W.nii
T2_file_withoutExt=SEME15_rT1W
T1_file=T1W_rSEME1.nii
T1_file_withoutExt=T1W_rSEME1

seg_dir=/Users/karolis/Downloads/registration/

cd ${seg_dir}

T2_file_brainStrip=${T2_file_withoutExt}_brain.nii
mri_synthstrip -i ${T2_file} -o ${T2_file_brainStrip}

T1_file_brainStrip=${T1_file_withoutExt}_brain.nii
mri_synthstrip -i ${T1_file} -o ${T1_file_brainStrip} --no-csf


antsRegistration \
    --verbose 1 \
    --dimensionality 3  \
    --float 0  \
    --collapse-output-transforms 1  \
    --interpolation BSpline[5] \
    --output [T2_to_T1_,T2_to_T1.nii,T2_to_T1_inverse.nii]  \
    --use-histogram-matching 0  \
    --winsorize-image-intensities [0.005,0.995]  \
    --transform Rigid[0.1]  \
    --metric MI[${T1_file_brainStrip},${T2_file_brainStrip},1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii \
    --transform Affine[0.1]  \
    --metric MI[${T1_file_brainStrip},${T2_file_brainStrip},1,32,Regular,0.25]  \
    --convergence [1000x500x250x100,1e-6,10]  \
    --shrink-factors 12x8x4x2  \
    --smoothing-sigmas 4x3x2x1vox  \
    -x mask.nii > antsRegistration.log 2>&1


    #    --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii]  \
    #    --metric MI[${bold_file},${T1_file},1,32,Regular,0.25]  \


#    --transform SyN[0.1,3,0]  \
#    --metric CC[${bold_file},${T1_file},1,4]  \
#    --convergence [50x50x70x50x20,1e-6,10]  \
#    --shrink-factors 10x6x4x2x1  \
#    --smoothing-sigmas 5x3x2x1x0vox  \
