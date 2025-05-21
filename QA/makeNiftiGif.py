import nibabel as nib
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.transform import resize
from pathlib import Path

studyDataDir = "/Users/karolis/Desktop/highRes_Resting"  # Set your study data directory
subject = "sub-99"
outputDir = f"/Users/karolis/Desktop/highRes_Resting/derivatives/GIFs/{subject}"
Path(outputDir).mkdir(parents=True, exist_ok=True)

# Load the NIfTI images
nii1 = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/fs_t1.nii").get_fdata()
nii2 = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/mrs_Glu-in_t1_appliedTrans.nii").get_fdata()

#nii2 = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/ln_depths_equivol.nii").get_fdata()
#nii2 = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/sub-01_bold_SMSEPI_mc_MEAN_n4_brain.nii").get_fdata()

# Ensure both images have the same shape
if nii1.shape != nii2.shape:
    raise ValueError("NIfTI images must have the same shape.")

# Choose a middle slice for visualization
slice_index = nii1.shape[2] // 2  # Middle axial slice

# Normalize images to [0, 1] for better contrast
nii1_slice = (nii1[:, :, slice_index] - np.min(nii1)) / (np.max(nii1) - np.min(nii1))
nii2_slice = (nii2[:, :, slice_index] - np.min(nii2)) / (np.max(nii2) - np.min(nii2))

# Convert grayscale image to 8-bit (uint8)
nii1_slice_gray = (nii1_slice * 255).astype(np.uint8)
nii1_rgb = np.stack([nii1_slice_gray] * 3, axis=-1)  # Shape (H, W, 3)

nii2_slice_gray = (nii2_slice * 255).astype(np.uint8)
nii2_rgb = np.stack([nii2_slice_gray] * 3, axis=-1)  # Shape (H, W, 3)



# Generate a random colormap
num_colors = 256
rand_colors = np.random.rand(num_colors, 3)  # Random RGB values
rand_colors[0] = [0, 0, 0]  # Force zero values to be black
cmap = mcolors.ListedColormap(rand_colors)


#cmap = plt.get_cmap("jet")  # Choose colormap
#nii2_colored = cmap(nii2_slice)[:, :, :3]  # Drop alpha channel
#nii2_colored = (nii2_colored * 255).astype(np.uint8)  # Convert to 8-bit

# Apply the random colormap to the second image
nii2_colored = cmap(nii2_slice)[:, :, :3]  # Drop alpha channel
nii2_colored = (nii2_colored * 255).astype(np.uint8)  # Convert to 8-bit
nii2_colored[nii2_slice == 0] = [0, 0, 0]  # Set zero pixels to black

# Ensure nii2_colored has the same size as nii1_rgb
nii2_colored_resized = resize(nii2_colored, nii1_rgb.shape, anti_aliasing=True)
nii2_colored_resized = (nii2_colored_resized * 255).astype(np.uint8)

# Create frames for the GIF
# frames = [nii1_rgb, nii2_colored_resized] * 100  # Flashing effect
frames = [nii1_rgb, nii2_rgb] * 100  # Flashing effect

# Save as GIF
imageio.mimsave(f"{outputDir}/T1_Glu_contrast.gif", frames, duration=500)  # Adjust duration as needed