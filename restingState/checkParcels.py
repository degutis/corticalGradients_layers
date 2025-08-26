from nilearn.datasets import fetch_atlas_schaefer_2018

# 7- or 17-network version
atl = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)
labels = list(atl["labels"])                  # length = 400, e.g. 'LH_Vis_1', ...
id_to_label = {i+1: lab for i, lab in enumerate(labels)}  # 1..400 -> label

print(id_to_label)