import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

# Define input and output directories
# studyDataDir = "/media/miplab-nas2/Data/Karolis/high_res_resting/"  # Set your study data directory
studyDataDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"  # Set your study data directory
subject = "sub-LAM001"

# input_dir = f"{studyDataDir}/{subject}/func/derivatives/"
input_dir = f"{studyDataDir}/derivatives/func/{subject}"
output_dir = f"{studyDataDir}/derivatives/func/tsnr/{subject}"
figure_dir = f"{studyDataDir}/derivatives/func/tsnr/{subject}"

os.makedirs(output_dir, exist_ok=True)

# Get list of NIfTI files
# nifti_files = [f for f in os.listdir(input_dir) if f.endswith("mc.nii") or f.endswith("mc.nii.gz")]
nifti_files = [f for f in os.listdir(input_dir) if f.endswith("mc.nii") or f.endswith("mc.nii.gz")]

for file in nifti_files:
    file_path = os.path.join(input_dir, file)
    
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()  # Get the 4D image data

    # Compute mean and standard deviation along time axis (axis=-1 for last axis)
    mean_signal = np.mean(data, axis=-1)
    std_signal = np.std(data, axis=-1)
    
    # Compute temporal SNR (avoid division by zero)
    tsnr = np.where(std_signal > 0, mean_signal / std_signal, 0)


    # Compute threshold for extreme tSNR values
    tsnr_mean = np.mean(tsnr[tsnr > 0])  # Exclude zeros from mean computation
    tsnr_std = np.std(tsnr[tsnr > 0])    # Exclude zeros from std computation
    tsnr_upper_threshold = tsnr_mean + 10 * tsnr_std  # 3 SD threshold

    # Remove values above 3 SD threshold
    tsnr[tsnr > tsnr_upper_threshold] = 0  # Set extreme values to zero

    # Save filtered tSNR map as a new NIfTI file
    tsnr_img = nib.Nifti1Image(tsnr, affine=img.affine, header=img.header)
    tsnr_file = os.path.join(output_dir, f"tsnr_{file}")
    nib.save(tsnr_img, tsnr_file)

    vmin, vmax = np.percentile(tsnr, [0.5, 99.5])  # Robust scaling
    print(vmin, vmax)

    # Plot the tSNR overlaid on the mean functional image
    fig, ax = plt.subplots(figsize=(8, 6))
    plotting.plot_stat_map(
        tsnr_img,
        bg_img=nib.Nifti1Image(mean_signal, affine=img.affine),
        title=f"tSNR - {file}",
        display_mode="ortho",
        draw_cross=False,
        # vmin=0, 
        vmax=50,
        cut_coords=(0, 0, 0),  # Adjust coordinates if needed
        # cmap="afmhot",  # Heatmap-like color map for better visualization
        axes=ax
    )
    
    # Save figure
    figure_file = os.path.join(figure_dir, f"tSNR_{file}.png")
    print(f"Saving figure to {figure_file}")
    plt.savefig(figure_file, dpi=300)
    plt.close()
    
    print(f"tSNR computed and saved for: {file}")

print("All tSNR computations complete.")