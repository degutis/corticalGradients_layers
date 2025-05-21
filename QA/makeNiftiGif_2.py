import nibabel as nib
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.transform import resize
from pathlib import Path

# Paths to input NIfTI images
studyDataDir = "/Users/karolis/Desktop/highRes_Resting"  # Set your study data directory
subject = "sub-99"
outputDir = f"/Users/karolis/Desktop/highRes_Resting/derivatives/GIFs/{subject}"
Path(outputDir).mkdir(parents=True, exist_ok=True)

# Get image data
background_data = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/fs_t1.nii").get_fdata()
overlay_data = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/mrs_GABA-in_t1.nii").get_fdata()

# Choose a middle slice along the z-axis
slice_idx = background_data.shape[2] // 2

# Determine symmetric color limits for the overlay
overlay_max = np.max(np.abs(overlay_data))-80 # Max absolute value
print(overlay_max)
vmin, vmax = -overlay_max, overlay_max      # Make colormap symmetric

# Create a mask where overlay is nonzero
alpha_mask = overlay_data[:, :, slice_idx] != 0

# Plot
fig, ax = plt.subplots()
ax.imshow(background_data[:, :, slice_idx], cmap="gray", interpolation="none")
ax.imshow(overlay_data[:, :, slice_idx], cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=alpha_mask.astype(np.float32), interpolation="none")
ax.axis("off")


plt.savefig(f"{outputDir}/T1_GABA_contrast.png", dpi=300, bbox_inches="tight", pad_inches=0)
