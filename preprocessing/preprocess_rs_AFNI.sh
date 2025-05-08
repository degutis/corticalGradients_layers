#!/bin/bash

# Define paths
funcDir="/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/VASO_func"
atlasDir="/Users/karolis/Desktop/kenshu_dataset/derivatives/sub-02/atlas"
atlas_file="${atlasDir}/Glasser_in_func_full.nii"

# List of functional scans
name_func_full=("sub-02_ses-13_task-movie_run-01_VASO" "sub-02_ses-13_task-movie_run-02_VASO" "sub-02_ses-13_task-movie_run-03_VASO" "sub-02_ses-13_task-movie_run-04_VASO" "sub-02_ses-13_task-movie_run-05_VASO")  # Add more scans as needed

# Output directory
mkdir -p "${funcDir}/preprocessed"

# Generate White Matter (WM) and CSF masks once (only need to do this once)
3dcalc -a "$atlas_file" -expr 'step(a-3000)' -prefix "${funcDir}/WM_mask.nii"
3dcalc -a "$atlas_file" -expr 'amongst(a,4,15,14,43)' -prefix "${funcDir}/CSF_mask.nii"

# Loop over each functional scan
for name_func in "${name_func_full[@]}"; do

    timecourse_file="${funcDir}/${name_func}.nii"

    echo "Processing: $name_func"

    # Extract mean signals from White Matter and CSF
    3dmaskave -quiet -mask "${funcDir}/WM_mask.nii" "$timecourse_file" > "${funcDir}/WM_signal_${name_func}.txt"
    3dmaskave -quiet -mask "${funcDir}/CSF_mask.nii" "$timecourse_file" > "${funcDir}/CSF_signal_${name_func}.txt"

    # Get timepoints for the functional scan
    timesteps=$(3dinfo -nt "$timecourse_file")

    # Create nuisance regressors (baseline + linear trend + physiological signals)
    yes "1" | head -n "$timesteps" > "${funcDir}/baseline_${name_func}.txt"
    seq 1 "$timesteps" > "${funcDir}/linearTrend_${name_func}.txt"

    # Combine regressors into a single matrix
    paste "${funcDir}/baseline_${name_func}.txt" \
          "${funcDir}/linearTrend_${name_func}.txt" \
          "${funcDir}/WM_signal_${name_func}.txt" \
          "${funcDir}/CSF_signal_${name_func}.txt" > "${funcDir}/nuisance_regressors_${name_func}.txt"

    # Perform nuisance regression and remove drifts
    3dTproject  -input "$timecourse_file" \
                -ort "${funcDir}/nuisance_regressors_${name_func}.txt" \
                -polort 2 \
                -passband 0.01 0.1 \
                -prefix "${funcDir}/preprocessed/${name_func}_residuals.nii" \
                -overwrite

    echo "Finished processing: $name_func"

    # Cleanup temporary files for this iteration
    rm "${funcDir}/WM_signal_${name_func}.txt"
    rm "${funcDir}/CSF_signal_${name_func}.txt"
    rm "${funcDir}/baseline_${name_func}.txt"
    rm "${funcDir}/linearTrend_${name_func}.txt"
    rm "${funcDir}/nuisance_regressors_${name_func}.txt"

done

# Final cleanup (mask files)
rm "${funcDir}/WM_mask.nii" "${funcDir}/CSF_mask.nii"

echo "All preprocessing complete! Residuals saved in ${funcDir}/preprocessed/"