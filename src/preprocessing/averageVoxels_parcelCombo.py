import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore

# ---------- helpers ----------
def _first_existing_path(candidates):
    """Return the first path that exists; otherwise the first candidate (let it error naturally on load)."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def _normalize_analysis_type(analysis_type, atlas):
    """
    Keep your original names if they already include a suffix (e.g., 'smallGap_Schaefer').
    Otherwise:
      - Schaefer -> append '_Schaefer'
      - Glasser  -> keep as-is (matches your original Glasser naming)
    """
    if "Schaefer" in analysis_type or "Glasser" in analysis_type:
        return analysis_type
    return f"{analysis_type}_Schaefer" if atlas.lower() == "schaefer" else f"{analysis_type}_Glasser"

def _layer_binning(layer_data, analysis_type, layer_01=True, eight_layers=False):
    """Return integer layer labels (0 = ignore) according to your rules."""
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)
    if layer_01:
        at = analysis_type.lower()
        if "smallgap" in at:
            layer_binary[(layer_data > 0)   & (layer_data <= 0.3)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.7) & (layer_data <  1.0)] = 3
        elif "largegap" in at:
            layer_binary[(layer_data > 0)   & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.8) & (layer_data <  1.0)] = 3
        else:
            raise ValueError(f"Unknown analysis_type for 0-1 layers: {analysis_type}")
    else:
        # integer layers version as in your code
        layer_binary[(layer_data > 1)  & (layer_data <= 3)]  = 1
        layer_binary[(layer_data > 5)  & (layer_data <= 7)]  = 2
        layer_binary[(layer_data > 9)  & (layer_data <= 11)] = 3

    return layer_binary

# ---------- unified function ----------
def averageVoxels_parcels(
    subject: str,
    runNum: str,
    analysis_type: str,
    atlas: str = "schaefer",     # "schaefer" or "glasser"
    layer_01: bool = True
):
    """
    Compute parcel-average z-scored time series for each cortical layer, ensuring
    exactly 400 (Schaefer) or 360 (Glasser) rows per layer (zeros if a parcel is missing).
    Saves one .npy per layer plus a concatenated 'all layers' file.

    Paths:
      - Atlas filenames follow: '{atlas}_L_in-func.nii' and '{atlas}_R_in-func.nii'.
    """
    atlas = atlas.lower()
    if atlas not in ("schaefer", "glasser"):
        raise ValueError("atlas must be 'schaefer' or 'glasser'")

    base_candidates = [
        "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives",
    ]

    # Data files (candidate lists)
    BOLD_candidates = [
        f"{base}/func/{subject}/merged_residuals_{runNum}.nii" for base in base_candidates
    ]
    layer_candidates = [
        f"{base}/ref_anat/{subject}/ln_depths_equivol.nii" for base in base_candidates
    ]
    atlasR_candidates = [
        f"{base}/ref_anat/{subject}/{atlas}_R_in-func.nii" for base in base_candidates
    ]
    atlasL_candidates = [
        f"{base}/ref_anat/{subject}/{atlas}_L_in-func.nii" for base in base_candidates
    ]

    BOLD_path  = _first_existing_path(BOLD_candidates)
    layer_path = _first_existing_path(layer_candidates)
    atlasR_path = _first_existing_path(atlasR_candidates)
    atlasL_path = _first_existing_path(atlasL_candidates)

    # Output path (include atlas to prevent collisions)
    # Preserve your original naming for analysis_type (see normalizer).
    out_label = _normalize_analysis_type(analysis_type, atlas)
    output_path_candidates = [
        f"{base}/correlations/{out_label}/{subject}/" for base in base_candidates
    ]
    output_path = output_path_candidates[0]
    os.makedirs(output_path, exist_ok=True)

    # ----- load -----
    layer_img = nib.load(layer_path)
    atlasR_img = nib.load(atlasR_path)
    atlasL_img = nib.load(atlasL_path)
    bold_img  = nib.load(BOLD_path)

    layer_data = layer_img.get_fdata()
    atlasR = atlasR_img.get_fdata()
    atlasL = atlasL_img.get_fdata()
    bold = bold_img.get_fdata()

    # ----- atlas-specific preprocessing & expected parcels -----
    if atlas == "schaefer":
        atlas_data = atlasR + atlasL
        expected_n = 400
        index_to_code = np.arange(1, expected_n + 1, dtype=int)

    elif atlas == "glasser":
        atlasL_off = np.where(atlasL != 0, atlasL + 1000, atlasL)
        atlasR_off = np.where(atlasR != 0, atlasR + 2000, atlasR)
        atlas_data = atlasL_off + atlasR_off
        expected_n = 360 

        index_to_code = np.concatenate([
            np.arange(1001, 1181, dtype=int),  # left 1..180 → 1001..1180
            np.arange(2001, 2181, dtype=int),  # right 1..180 → 2001..2180
        ])

    # ----- checks -----
    if layer_data.shape != atlas_data.shape or layer_data.shape != bold.shape[:-1]:
        raise ValueError(
            f"Dimension mismatch: layer {layer_data.shape}, atlas {atlas_data.shape}, bold {bold.shape[:-1]}"
        )

    # ----- layer binning -----
    eight_layers = "eightLayers" in analysis_type or "eightlayers" in analysis_type
    layer_binary = _layer_binning(layer_data, analysis_type, layer_01, eight_layers=eight_layers)
    unique_layers = np.unique(layer_binary[layer_binary > 0])

    # ----- iterate layers and parcels -----
    n_tp = bold.shape[-1]
    layer_parcels_flat = []

    # Precompute which codes actually exist in the atlas (for quick skip/printing)
    present_codes = set(np.unique(atlas_data.astype(int)))

    for layer in unique_layers:
        print(f"[{subject} | {atlas} | {out_label}] Processing layer {int(layer)}")
        layer_mask = (layer_binary == layer)

        # Fixed-size output per layer
        out = np.zeros((expected_n, n_tp), dtype=float)

        for i, code in enumerate(index_to_code):
            if code in present_codes:
                parcel_mask = (atlas_data == code)
                mask = layer_mask & parcel_mask

                if np.any(mask):
                    vox_ts = bold[mask]                 # shape: [n_voxels_in_mask, n_tp]
                    mean_ts = np.nanmean(vox_ts, axis=0)
                    out[i, :] = zscore(mean_ts, axis=0)
                # else: keep zeros (parcel missing in this layer for this subject)
            # else: keep zeros (parcel not present in the atlas for this subject)

        # Safety
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  has_nan={np.isnan(out).any()} | layer rows={out.shape[0]} (expected {expected_n})")

        # Save per-layer
        fname = f"Layer_{runNum}_{int(layer):d}.npy"
        np.save(os.path.join(output_path, fname), out)
        print(f"  → wrote {os.path.join(output_path, fname)}")

        # Accumulate for "all layers" file (time x parcels per layer → later concat along feature dim)
        layer_parcels_flat.append(out.T)     # shape: [n_tp, n_parcels]

    # Concatenate all layers along features (columns): [n_tp, n_parcels * n_layers]
    parcel_time_all = np.concatenate(layer_parcels_flat, axis=1)
    np.save(os.path.join(output_path, f"Layer_{runNum}_parcels_all_layers.npy"), parcel_time_all)
    print(f"All layers shape: {parcel_time_all.shape}  → saved Layer_{runNum}_parcels_all_layers.npy")
    print("All done.\n")

# ---------- batch: call for all subjects ----------
subjects = [
    "sub-LAM001","sub-LAM002","sub-LAM003","sub-LAM004","sub-LAM005","sub-LAM006",
    "sub-LAM007","sub-LAM009","sub-LAM010","sub-LAM011","sub-LAM012","sub-LAM013",
    "sub-LAM014","sub-LAM015","sub-LAM016","sub-LAM017","sub-LAM018","sub-LAM019",
    "sub-LAM021","sub-LAM022","sub-LAM023","sub-LAM024","sub-LAM025","sub-LAM026",
    "sub-LAM027","sub-LAM028","sub-LAM029","sub-LAM030","sub-LAM031"
]


for s in subjects:
    averageVoxels_parcels(subject=s, runNum="run1", analysis_type="smallGap", atlas="schaefer", layer_01=True)