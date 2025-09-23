# pip install brainspace  # if needed

from brainspace.datasets import load_group_fc
from brainspace.gradient import GradientMaps
import numpy as np

# 1) Load mean group FC and Schaefer-400 labels
fc = load_group_fc('schaefer', scale=400)                  # (400 x 400)

# 2) Compute gradients (use diffusion maps + normalized-angle kernel)
gm = GradientMaps(n_components=1, approach='dm', kernel='normalized_angle', random_state=0)
gm.fit(fc)

# 3) First gradient as a 400-long vector
g1 = gm.gradients_[:, 0]    # sign is arbitrary; flip if you prefer
print("First 5 values:", np.round(g1[:5], 4))

# 4) Quick surface plot
# mask = labels != 0
# g1_on_surface = map_to_labels(g1, labels, mask=mask, fill=np.nan)
# plot_hemispheres(surf_lh, surf_rh, array_name=g1_on_surface, color_bar=True,
#                  label_text=['Grad 1'])