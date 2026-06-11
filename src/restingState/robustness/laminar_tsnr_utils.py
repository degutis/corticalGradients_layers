
#!/usr/bin/env python3
"""
Utilities for laminar quality metrics on Schaefer parcels.

We compute FOUR voxelwise metrics:

  1) tSNR_orig   = mean(orig) / std(orig)       (classical tSNR)
  2) tSNR_resid  = mean(orig) / std(residual)   (hybrid post-regression)
  3) R2          = 1 - var(resid) / var(orig)   (variance explained by regressors)
  4) inv_noise   = 1 / std(residual)           (inverse residual noise)

Then we aggregate them per subject, per layer, per parcel.

Conventions for smallGap + layer_01=True:
  layer_binary == 1 -> Deep
  layer_binary == 2 -> Middle
  layer_binary == 3 -> Superficial
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple, Dict, List

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import matplotlib as mpl


# --------------------------------------------------------------------
# Layer binning
# --------------------------------------------------------------------

def make_layer_binary(
    layer_data: np.ndarray,
    analysis_type: str,
    layer_01: bool = True,
) -> np.ndarray:
    """
    Recreates the layer_binary logic from averageVoxels_parcels_with_fs.

    For analysis_type == "smallGap" and layer_01 == True, the mapping is:
        1 -> Deep
        2 -> Middle
        3 -> Superficial
    """
    layer_binary = np.zeros_like(layer_data, dtype=np.uint8)

    if layer_01:
        if analysis_type == "smallGap":
            # 1 = Deep, 2 = Middle, 3 = Superficial
            layer_binary[(layer_data > 0)   & (layer_data <= 0.3)] = 1  # Deep
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2  # Middle
            layer_binary[(layer_data > 0.7) & (layer_data <  1.0)] = 3  # Superficial

        elif analysis_type == "noGap":
            layer_binary[(layer_data > 0)   & (layer_data <= 0.4)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.6) & (layer_data <  1.0)] = 3

        elif analysis_type == "largeGap":
            layer_binary[(layer_data > 0)   & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.4) & (layer_data <= 0.6)] = 2
            layer_binary[(layer_data > 0.8) & (layer_data <  1.0)] = 3

        elif analysis_type == "eightLayers":
            layer_binary[(layer_data > 0.1) & (layer_data <= 0.2)] = 1
            layer_binary[(layer_data > 0.2) & (layer_data <  0.3)] = 2
            layer_binary[(layer_data > 0.3) & (layer_data <  0.4)] = 3
            layer_binary[(layer_data > 0.4) & (layer_data <  0.5)] = 4
            layer_binary[(layer_data > 0.5) & (layer_data <  0.6)] = 5
            layer_binary[(layer_data > 0.6) & (layer_data <  0.7)] = 6
            layer_binary[(layer_data > 0.7) & (layer_data <  0.8)] = 7
            layer_binary[(layer_data > 0.8) & (layer_data <  0.9)] = 8
    else:
        # alt binning (integer depths)
        layer_binary[(layer_data > 1)  & (layer_data <= 3)]  = 1
        layer_binary[(layer_data > 5)  & (layer_data <= 7)]  = 2
        layer_binary[(layer_data > 9)  & (layer_data <= 11)] = 3

    return layer_binary


# --------------------------------------------------------------------
# Voxelwise metrics
# --------------------------------------------------------------------

def _clip_positive_outliers(vol: np.ndarray, n_sd: float = 10.0) -> np.ndarray:
    """
    Clip extreme positive outliers (similar to your 10 SD rule).

    Only applied to vol[vol > 0].
    """
    vol = np.asarray(vol)
    out = vol.copy()

    pos = out[out > 0]
    if pos.size == 0:
        return out

    mu = float(np.mean(pos))
    sd = float(np.std(pos))
    thr = mu + n_sd * sd
    out[out > thr] = 0.0
    return out


def compute_tsnr_4d(data_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical tSNR = mean(signal over time) / std(signal over time).
    Used when no residuals are available.
    """
    mean_signal = np.mean(data_4d, axis=-1)
    std_signal  = np.std(data_4d, axis=-1)

    tsnr = np.where(std_signal > 0, mean_signal / std_signal, 0.0)
    tsnr = _clip_positive_outliers(tsnr, n_sd=10.0)
    return tsnr, mean_signal


def compute_residual_metrics(
    orig_4d: np.ndarray,
    resid_4d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all four voxelwise metrics given original and residual time-series.

    Returns
    -------
    tsnr_orig   : (X,Y,Z)
    tsnr_resid  : (X,Y,Z)
    r2          : (X,Y,Z)
    inv_noise   : (X,Y,Z)
    mean_orig   : (X,Y,Z)
    """
    orig_4d = np.asarray(orig_4d)
    resid_4d = np.asarray(resid_4d)

    if orig_4d.shape != resid_4d.shape:
        raise ValueError(
            f"orig_4d and resid_4d must have same shape, got "
            f"{orig_4d.shape} vs {resid_4d.shape}"
        )

    # First: classical tSNR on original (with clipping)
    tsnr_orig, mean_orig = compute_tsnr_4d(orig_4d)

    # Residual-based quantities
    std_resid = np.std(resid_4d, axis=-1)
    var_resid = std_resid ** 2

    # tSNR_resid: same numerator, residual sigma in denominator
    tsnr_resid = np.where(std_resid > 0, mean_orig / std_resid, 0.0)
    tsnr_resid = _clip_positive_outliers(tsnr_resid, n_sd=10.0)

    # inv_noise = 1 / std(resid)
    inv_noise = np.where(std_resid > 0, 1.0 / std_resid, 0.0)
    inv_noise = _clip_positive_outliers(inv_noise, n_sd=10.0)

    # R² = 1 - var(resid) / var(orig)
    std_orig = np.std(orig_4d, axis=-1)
    var_orig = std_orig ** 2

    r2 = np.zeros_like(var_orig, dtype=float)
    mask = var_orig > 0
    r2[mask] = 1.0 - (var_resid[mask] / var_orig[mask])

    return tsnr_orig, tsnr_resid, r2, inv_noise, mean_orig


def compute_tsnr_hybrid(orig_4d: np.ndarray, resid_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper for older scripts:
      returns (tSNR_resid, mean_orig)
    """
    _, tsnr_resid, _, _, mean_orig = compute_residual_metrics(orig_4d, resid_4d)
    return tsnr_resid, mean_orig


# --------------------------------------------------------------------
# Group aggregation: ALL FOUR metrics per subject / layer / parcel
# --------------------------------------------------------------------

def aggregate_group_layer_parcel_metrics(
    subjects: Sequence[str],
    runNums: Sequence[str],
    FUNC_BASE: str,
    ANAT_BASE: str,
    analysis_type: str,
    layer_01: bool = True,
    n_parcels: int = 400,
    target_layers: Sequence[int] = (1, 2, 3),
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    For each subject:
      - load layer + Schaefer atlases
      - for each run:
          * load original BOLD (NORDIC_mc if present, else mc)
          * load residuals 'merged_residuals_runX.nii' from FUNC_BASE/subject
          * compute:
              tsnr_orig (classical)
              tsnr_resid (hybrid)
              r2
              inv_noise
              mean (mean original signal)
          * average EACH metric per (layer, parcel)
      - average across runs

    Returns
    -------
    all_metrics : dict[str, np.ndarray]
        {
          "tsnr_orig"  : (Nsubj, 3, n_parcels),
          "tsnr_resid" : (Nsubj, 3, n_parcels),
          "r2"         : (Nsubj, 3, n_parcels),
          "inv_noise"  : (Nsubj, 3, n_parcels),
          "mean"       : (Nsubj, 3, n_parcels),
        }
        Layers are (1=Deep, 2=Middle, 3=Superficial) for smallGap+layer_01.
    subject_ids_used : list of str
        Subjects with valid data.
    """
    metric_names = ["tsnr_orig", "tsnr_resid", "r2", "inv_noise", "mean"]

    all_metrics_lists: Dict[str, List[np.ndarray]] = {
        m: [] for m in metric_names
    }
    subject_ids_used: List[str] = []

    full_parcel_list = np.arange(1, n_parcels + 1, dtype=int)

    for subj in subjects:
        print(f"==== Subject {subj} ====")

        layer_path = os.path.join(ANAT_BASE, subj, "ln_depths_equivol.nii")
        atlas_path_right = os.path.join(ANAT_BASE, subj, "schaefer_R_in-func.nii")
        atlas_path_left  = os.path.join(ANAT_BASE, subj, "schaefer_L_in-func.nii")

        if (not os.path.exists(layer_path) or
            not os.path.exists(atlas_path_right) or
            not os.path.exists(atlas_path_left)):
            print(f"[WARN] Missing atlas/layer for {subj}, skipping.")
            continue

        # --- Load layer + atlas once ---
        layer_img = nib.load(layer_path)
        layer_data = layer_img.get_fdata()

        atlas_img_right = nib.load(atlas_path_right)
        atlas_img_left  = nib.load(atlas_path_left)
        atlas_data_right = atlas_img_right.get_fdata()
        atlas_data_left  = atlas_img_left.get_fdata()
        atlas_data = atlas_data_right + atlas_data_left  # parcel IDs 1..400

        fs_parcels = np.unique(atlas_data)
        fs_parcels = fs_parcels[(fs_parcels > 0) & (fs_parcels <= n_parcels)]
        fs_parcels = np.sort(fs_parcels).astype(int)

        if fs_parcels.size != n_parcels:
            print(
                f"[INFO] {subj}: expected {n_parcels} parcels but got "
                f"{fs_parcels.size}. Missing parcels will be NaN."
            )

        # Layer mask: for smallGap + layer_01, 1=Deep, 2=Middle, 3=Superficial
        layer_binary = make_layer_binary(layer_data, analysis_type, layer_01)
        unique_layers = np.unique(layer_binary[layer_binary > 0])
        target_layers = list(target_layers)

        func_dir = os.path.join(FUNC_BASE, subj)

        # per-run metric maps (metric -> list of (L,P))
        subj_run_metric_layer_parcel: Dict[str, List[np.ndarray]] = {
            m: [] for m in metric_names
        }

        for runNum in runNums:
            print(f"  [RUN] {runNum}")

            # 1) Original BOLD (NORDIC_mc if present, else mc)
            bold_path_nordic = os.path.join(
                func_dir, f"{subj}_task-rest_bold_NORDIC_mc.nii"
            )
            bold_path_mc = os.path.join(
                func_dir, f"{subj}_task-rest_bold_mc.nii"
            )

            if os.path.exists(bold_path_nordic):
                bold_path = bold_path_nordic
            elif os.path.exists(bold_path_mc):
                bold_path = bold_path_mc
            else:
                print(f"[WARN] No BOLD file found for {subj} ({runNum}), skipping run.")
                continue

            bold_img = nib.load(bold_path)
            bold_data = bold_img.get_fdata()  # 4D (X,Y,Z,T)

            # 2) Residuals
            resid_path = os.path.join(func_dir, f"merged_residuals_{runNum}.nii")

            if os.path.exists(resid_path):
                print(f"    Using residual metrics: {os.path.basename(resid_path)}")
                resid_img = nib.load(resid_path)
                resid_data = resid_img.get_fdata()

                if resid_data.shape != bold_data.shape:
                    print(
                        f"[WARN] Residuals shape {resid_data.shape} "
                        f"!= BOLD shape {bold_data.shape}; "
                        "falling back to tSNR_orig + mean only."
                    )
                    tsnr_orig_vol, mean_orig_vol = compute_tsnr_4d(bold_data)
                    tsnr_resid_vol = np.full_like(tsnr_orig_vol, np.nan)
                    r2_vol         = np.full_like(tsnr_orig_vol, np.nan)
                    inv_noise_vol  = np.full_like(tsnr_orig_vol, np.nan)
                else:
                    tsnr_orig_vol, tsnr_resid_vol, r2_vol, inv_noise_vol, mean_orig_vol = (
                        compute_residual_metrics(bold_data, resid_data)
                    )
            else:
                print(
                    f"    [WARN] Residuals file not found "
                    f"({os.path.basename(resid_path)}); using tSNR_orig + mean only."
                )
                tsnr_orig_vol, mean_orig_vol = compute_tsnr_4d(bold_data)
                tsnr_resid_vol = np.full_like(tsnr_orig_vol, np.nan)
                r2_vol         = np.full_like(tsnr_orig_vol, np.nan)
                inv_noise_vol  = np.full_like(tsnr_orig_vol, np.nan)

            # shape check
            if (layer_data.shape != atlas_data.shape or
                layer_data.shape != tsnr_orig_vol.shape):
                print(
                    f"[ERROR] Dim mismatch in {subj} {runNum}, skipping run.\n"
                    f"  layer_data: {layer_data.shape}, "
                    f"atlas_data: {atlas_data.shape}, "
                    f"tsnr_orig_vol: {tsnr_orig_vol.shape}"
                )
                continue

            # 3) layer × parcel mean for EACH metric
            run_metric_layer_parcel: Dict[str, np.ndarray] = {}
            for mname in metric_names:
                run_metric_layer_parcel[mname] = np.full(
                    (len(target_layers), n_parcels), np.nan, dtype=float
                )

            metric_vols = {
                "tsnr_orig": tsnr_orig_vol,
                "tsnr_resid": tsnr_resid_vol,
                "r2": r2_vol,
                "inv_noise": inv_noise_vol,
                "mean": mean_orig_vol,
            }

            for li, layer_id in enumerate(target_layers):
                if layer_id not in unique_layers:
                    continue

                layer_mask = (layer_binary == layer_id)

                for parcel_id in full_parcel_list:
                    parcel_mask = (atlas_data == parcel_id)
                    combo_mask = layer_mask & parcel_mask

                    if not np.any(combo_mask):
                        continue

                    for mname, vol in metric_vols.items():
                        vals = vol[combo_mask]
                        if vals.size > 0:
                            run_metric_layer_parcel[mname][li, parcel_id - 1] = (
                                np.nanmean(vals)
                            )

            # store this run
            for mname in metric_names:
                subj_run_metric_layer_parcel[mname].append(
                    run_metric_layer_parcel[mname]
                )

        # average across runs
        if any(len(v) == 0 for v in subj_run_metric_layer_parcel.values()):
            have_any_runs = any(len(v) > 0 for v in subj_run_metric_layer_parcel.values())
            if not have_any_runs:
                print(f"[WARN] {subj}: no valid runs, skipping subject.")
                continue

        subj_metric_layer_parcel_mean: Dict[str, np.ndarray] = {}
        for mname in metric_names:
            runs_list = subj_run_metric_layer_parcel[mname]
            if len(runs_list) == 0:
                continue
            runs_arr = np.stack(runs_list, axis=0)  # (nRuns, L, n_parcels)
            subj_metric_layer_parcel_mean[mname] = np.nanmean(runs_arr, axis=0)

        # store per subject
        for mname in metric_names:
            all_metrics_lists[mname].append(subj_metric_layer_parcel_mean[mname])
        subject_ids_used.append(subj)

    if len(subject_ids_used) == 0:
        raise RuntimeError("No valid subjects found. Check paths / subject list.")

    # stack across subjects
    all_metrics: Dict[str, np.ndarray] = {}
    for mname in metric_names:
        all_metrics[mname] = np.stack(all_metrics_lists[mname], axis=0)
        # shape (Nsubj, 3, n_parcels)

    return all_metrics, subject_ids_used

# --------------------------------------------------------------------
# Group-level saving, ANOVA, and plotting
# --------------------------------------------------------------------

def save_group_parcel_vectors(
    group_mean_layer_parcel: np.ndarray,
    out_dir: str,
    prefix: str = "group_layer",
) -> None:
    """
    Save group-mean parcel vectors for each layer as separate .npy files.

    Parameters
    ----------
    group_mean_layer_parcel : np.ndarray
        Shape (L, n_parcels).
    out_dir : str
        Directory to save files into.
    prefix : str
        Filename prefix. E.g. "all_tsnr_orig_layer" ->
          all_tsnr_orig_layer1_tSNR_byParcel.npy, etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_layers = group_mean_layer_parcel.shape[0]
    for li in range(n_layers):
        vec = group_mean_layer_parcel[li, :]
        fname = os.path.join(out_dir, f"{prefix}{li+1}_tSNR_byParcel.npy")
        np.save(fname, vec)
        print(f"Saved {fname}")


def compute_layer_subject_means(
    all_subject_layer_parcel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse from (Nsubj, 3, n_parcels) to 3 vectors of length Nsubj:
    one mean metric per layer per subject.

    By convention (for smallGap, layer_01=True):
        index 0 -> Deep
        index 1 -> Middle
        index 2 -> Superficial
    """
    if all_subject_layer_parcel.ndim != 3:
        raise ValueError(
            f"Expected shape (Nsubj, 3, n_parcels), got {all_subject_layer_parcel.shape}"
        )
    if all_subject_layer_parcel.shape[1] != 3:
        raise ValueError("This helper assumes exactly 3 layers in axis=1.")

    layer1_subj_means = np.nanmean(all_subject_layer_parcel[:, 0, :], axis=1)
    layer2_subj_means = np.nanmean(all_subject_layer_parcel[:, 1, :], axis=1)
    layer3_subj_means = np.nanmean(all_subject_layer_parcel[:, 2, :], axis=1)
    return layer1_subj_means, layer2_subj_means, layer3_subj_means


def run_layer_anova(
    layer1_subj_means: np.ndarray,
    layer2_subj_means: np.ndarray,
    layer3_subj_means: np.ndarray,
) -> Tuple[float, float]:
    """
    One-way ANOVA across layers (3 groups of subject-wise means).
    """
    Fval, pval = f_oneway(layer1_subj_means, layer2_subj_means, layer3_subj_means)
    return float(Fval), float(pval)


def plot_layer_tsnr_violin(
    layer1_subj_means: np.ndarray,
    layer2_subj_means: np.ndarray,
    layer3_subj_means: np.ndarray,
    out_path: str,
    Fval: float | None = None,
    pval: float | None = None,
    layer_names: Sequence[str] = ("Deep", "Middle", "Superficial"),
    metric_name: str = "tSNR",
    y_label: str | None = None,
) -> None:
    """
    Violin plot of per-subject mean metric for 3 layers, with jittered points.

    metric_name is used in the title and y-axis label.
    Ensures that, for SVG outputs, text is stored as real text (not paths)
    so it remains editable in Illustrator.
    """
    if len(layer_names) != 3:
        raise ValueError("layer_names must have length 3.")

    if Fval is None or pval is None:
        Fval, pval = run_layer_anova(
            layer1_subj_means, layer2_subj_means, layer3_subj_means
        )

    if y_label is None:
        y_label = f"Mean {metric_name} (per subject, avg over parcels)"

    data_for_plot = [
        np.asarray(layer1_subj_means),
        np.asarray(layer2_subj_means),
        np.asarray(layer3_subj_means),
    ]

    # --- Configure SVG text behaviour if saving SVG ---
    ext = os.path.splitext(out_path)[1].lower()
    svg_fonttype_old = mpl.rcParams.get("svg.fonttype")
    text_usetex_old = mpl.rcParams.get("text.usetex")

    try:
        if ext == ".svg":
            # Keep text as text objects in the SVG
            mpl.rcParams["svg.fonttype"] = "none"
            # Disable LaTeX text rendering (forces text as regular SVG <text>)
            mpl.rcParams["text.usetex"] = False

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.violinplot(
            data_for_plot,
            showmeans=True,
            showmedians=False,
            showextrema=True,
        )
        ax.set_ylim(8,22)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(list(layer_names))
        ax.set_ylabel(y_label)
        ax.set_title(
            f"Laminar {metric_name} across subjects\nANOVA: F={Fval:.2f}, p={pval:.2e}"
        )

        # overlay individual subject points with jitter
        x_jitter = 0.08
        for i, layer_vals in enumerate(data_for_plot, start=1):
            xs = np.random.uniform(
                low=i - x_jitter, high=i + x_jitter, size=len(layer_vals)
            )
            ax.scatter(xs, layer_vals, alpha=0.6, edgecolor="k", linewidth=0.5)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved violin plot to {out_path}")

    finally:
        # Restore previous rcParams so we don't globally affect other plots
        if svg_fonttype_old is not None:
            mpl.rcParams["svg.fonttype"] = svg_fonttype_old
        if text_usetex_old is not None:
            mpl.rcParams["text.usetex"] = text_usetex_old

def save_subject_summary(
    layer1_subj_means: np.ndarray,
    layer2_subj_means: np.ndarray,
    layer3_subj_means: np.ndarray,
    out_path: str,
) -> None:
    """
    Save per-subject summary matrix (Nsubj × 3) as .npy.
    """
    summary_mat = np.column_stack(
        [layer1_subj_means, layer2_subj_means, layer3_subj_means]
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, summary_mat)
    print(f"Saved per-subject layer means to {out_path}")