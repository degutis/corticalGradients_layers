export SUBJECTS_DIR=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer

subject=sub-LAM021
anat_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/${subject}/anat

recon-all -s ${subject} \
          -i ${anat_dir}/${subject}_acq-denoised_T1w.nii \
          -hires \
          -all