import os
import numpy as np
import nibabel as nib
import scipy.io
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def preprocess_fmri(func_dir, timecourse_name, atlas_file, motion_file, tr, use_motion=True, highpass_cutoff=0.01):
    """
    Preprocess fMRI data by extracting signals, constructing nuisance regressors, and applying a high-pass filter.
    """
    os.makedirs(f'{func_dir}/preprocessed', exist_ok=True)
    
    # Load fMRI timecourse data
    timecourse_file = os.path.join(func_dir, f'{timecourse_name}.nii')
    img_t = nib.load(timecourse_file)
    Y_t = img_t.get_fdata()
    timecourse_dim = Y_t.shape[-1]

    # Load Atlas data
    img_a = nib.load(atlas_file)
    Y_a = img_a.get_fdata()

    timeDim = Y_t.shape[-1]

    # --- White Matter Extraction ---
    white_matter_mask = (Y_a > 3000)  # Shape: (X, Y, Z)
    white_matter_mask_4d = np.expand_dims(white_matter_mask, axis=-1)  # Shape: (X, Y, Z, 1)
    white_matter_data = np.where(white_matter_mask_4d, Y_t, np.nan)  # Shape: (X, Y, Z, Time)
    white_matter_signal = np.nanmean(white_matter_data, axis=(0, 1, 2))  # Shape: (Time,)    
    
    # Extract CSF signal (defined by specific values in the atlas)
    csf_values = [4, 15, 14, 43]
    csf_mask = np.isin(Y_a, csf_values)  # Shape: (X, Y, Z)
    csf_mask_4d = np.expand_dims(csf_mask, axis=-1)  # Shape: (X, Y, Z, 1)

    csf_data = np.where(csf_mask_4d, Y_t, np.nan)  # Shape: (X, Y, Z, Time)
    csf_signal = np.nanmean(csf_data, axis=(0, 1, 2))  # Shape: (Time,)
    
    baseline = np.ones((timecourse_dim, 1))
    linear_trend = np.arange(1, timecourse_dim + 1).reshape(-1, 1)
    quadratic_trend = linear_trend ** 2

    nuisance_regressors = [baseline, linear_trend, quadratic_trend, 
                           white_matter_signal.reshape(-1, 1), 
                           csf_signal.reshape(-1, 1)]

    # Process motion regressors if use_motion is True
    if use_motion:
        motion_params = np.loadtxt(motion_file)
        motion_params = motion_params[:, 1:-2]  # Remove first column (time) and last two columns
        motion_derivatives = np.diff(motion_params, axis=0)
        zero_index = np.where(motion_params[:, 0] == 0)[0]
        motion_derivatives = np.vstack((motion_derivatives[:24, :], np.zeros((1, 6)), motion_derivatives[zero_index + 1:, :]))
        nuisance_regressors.extend([motion_params, motion_derivatives])

    # Combine all nuisance regressors into one matrix
    nuisance_regressors = np.hstack(nuisance_regressors)

    # High-pass filter (zero-frequency component) using scipy.signal
    if highpass_cutoff > 0:
        nyquist = 0.5 * tr  # Nyquist frequency
        lowcut = highpass_cutoff / nyquist  # Normalize cutoff
        b, a = butter(1, lowcut, btype='high')
        
        # Apply the filter to each voxel's timecourse
        for voxel in range(Y_t.shape[0]):
            for row in range(Y_t.shape[1]):
                for col in range(Y_t.shape[2]):
                    Y_t[voxel, row, col, :] = filtfilt(b, a, Y_t[voxel, row, col, :])

    # Fit linear model (nuisance regressors to timecourse)
    regressor_matrix = nuisance_regressors
    residuals = np.zeros_like(Y_t)

    # Loop through each voxel to compute residuals
    for voxel in range(Y_t.shape[0]):
        for row in range(Y_t.shape[1]):
            for col in range(Y_t.shape[2]):
                voxel_timecourse = Y_t[voxel, row, col, :]

                # Fit model to the voxel timecourse
                betas, _, _, _ = np.linalg.lstsq(regressor_matrix, voxel_timecourse, rcond=None)
                predicted_signal = regressor_matrix @ betas
                residuals[voxel, row, col, :] = voxel_timecourse - predicted_signal

    # Save the residuals as a NIfTI file
    residual_img = nib.Nifti1Image(residuals, img_t.affine)
    residual_file = os.path.join(func_dir, 'preprocessed',f'{timecourse_name}_residuals.nii')
    nib.save(residual_img, residual_file)

    # Save the nuisance regressors as a text file
    #regressor_file = os.path.join(func_dir, 'nuisance_regressors.txt')
    #np.savetxt(regressor_file, nuisance_regressors, fmt='%f')
    

# Example usage
#preprocess_fmri('../kenshu_dataset/derivatives/sub-02/VASO_func', 'sub-02_VASO_across_days', '../kenshu_dataset/derivatives/freesurfer/sub-02/mri/HCP-MMP1.nii', '', 5.15, use_motion=False)
preprocess_fmri('../kenshu_dataset/derivatives/sub-02/VASO_func', 'sub-02_VASO_across_days', '../kenshu_dataset/derivatives/sub-02/atlas/Glasser_in_func_full.nii', '', 5.15, use_motion=False)