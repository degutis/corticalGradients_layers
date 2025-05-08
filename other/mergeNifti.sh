for ses in 04 05 06 07 08 09 10 11 12 13; do
    files=$(ls /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_ses-${ses}_task-movie_run-*_VASO.nii | sort)
    3dTcat -prefix /Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/VASO_func/sub-02_ses-${ses}_concat.nii $files
done