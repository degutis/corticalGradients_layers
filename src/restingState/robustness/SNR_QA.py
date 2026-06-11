#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path

import laminar_tsnr_utils as ltu
from laminar_rs.surface_maps import plotSurfaceMap

# ----------------- Parameters -----------------

DATA_SET = "huppi"   # or "kd"

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19,
            21, 22]

# group definitions (integer subject codes)
SUBJECTS_withNORDIC = [9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22]
SUBJECTS_noNORDIC   = [1, 2, 3, 4, 5, 6, 7, 8, 17]

# laminar binning rule
analysis_type = "smallGap"   # "smallGap", "noGap", "largeGap", "eightLayers", ...
layer_01 = True              # same meaning as before

# Base paths depend on dataset; atlas ignored here
if DATA_SET == "huppi":
    BASE = Path("/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives")
    subject_ids = [f"sub-LAM{s:03d}" for s in SUBJECTS]
    subject_ids_withNORDIC = [f"sub-LAM{s:03d}" for s in SUBJECTS_withNORDIC]
    subject_ids_noNORDIC   = [f"sub-LAM{s:03d}" for s in SUBJECTS_noNORDIC]
elif DATA_SET == "kd":
    BASE = Path("/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives")
    subject_ids = [f"sub-{s:02d}" for s in SUBJECTS]
    # if you later define groups for KD, add formatted IDs here:
    subject_ids_withNORDIC = []
    subject_ids_noNORDIC   = []
else:
    raise ValueError(f"Unknown DATA_SET {DATA_SET!r}")

FUNC_BASE = str(BASE / "func")
ANAT_BASE = str(BASE / "ref_anat")

# output dir for group results / figs
GROUP_OUT = str(BASE / "tSNR")
os.makedirs(GROUP_OUT, exist_ok=True)

# ------------- cache paths for group-level arrays -------------
# encode analysis_type & layer_01 in filenames so different settings don't clash
layer_flag = f"layer01-{int(layer_01)}"
cache_metrics_path = Path(GROUP_OUT) / f"all_subject_layer_parcel_metrics_{analysis_type}_{layer_flag}.npz"
cache_subj_path    = Path(GROUP_OUT) / f"subject_ids_used_{analysis_type}_{layer_flag}.txt"

# ----------------- MAIN -----------------

# 1) Load all metrics if cached, otherwise compute & cache
need_recompute = True

if cache_metrics_path.is_file() and cache_subj_path.is_file():
    print(f"[INFO] Loading cached group-level metrics from {cache_metrics_path}")
    npz = np.load(cache_metrics_path)
    all_metrics = {k: npz[k] for k in npz.files}

    with open(cache_subj_path, "r") as f:
        subject_ids_used = [line.strip() for line in f if line.strip()]

    # check if all expected metrics are present (including 'mean')
    expected_keys = {"tsnr_orig", "tsnr_resid", "r2", "inv_noise", "mean"}
    missing = expected_keys.difference(all_metrics.keys())
    if len(missing) == 0:
        need_recompute = False
    else:
        print(f"[INFO] Cache is missing metrics {missing}; will recompute from NIfTI.")
else:
    print("[INFO] No cached group-level metrics found, computing from NIfTI...")

if need_recompute:
    all_metrics, subject_ids_used = ltu.aggregate_group_layer_parcel_metrics(
        subjects=subject_ids,
        runNums=["run1"],
        FUNC_BASE=FUNC_BASE,
        ANAT_BASE=ANAT_BASE,
        analysis_type=analysis_type,
        layer_01=layer_01,
        n_parcels=400,
        target_layers=(1, 2, 3),   # 1=Deep, 2=Middle, 3=Superficial for smallGap
    )
    # cache results for future runs
    np.savez(cache_metrics_path, **all_metrics)
    with open(cache_subj_path, "w") as f:
        for sid in subject_ids_used:
            f.write(f"{sid}\n")

    print(f"[INFO] Cached group-level metrics to {cache_metrics_path}")
    print(f"[INFO] Cached subject IDs to {cache_subj_path}")

# Pick one metric to infer Nsubj and shape
any_metric = next(iter(all_metrics.values()))
Nsubj = any_metric.shape[0]

print(
    f"Final dataset shapes (per metric): "
    f"{ {k: v.shape for k, v in all_metrics.items()} }"
)
print(f"Subjects used ({Nsubj}): {subject_ids_used}")

# 2) Build index mapping from subject_id -> row index
id_to_idx = {sid: i for i, sid in enumerate(subject_ids_used)}

def indices_for_group(group_ids):
    """Return indices into arrays for a given list of subject IDs."""
    return np.array(
        [id_to_idx[sid] for sid in group_ids if sid in id_to_idx],
        dtype=int,
    )

# Indices for each group
idx_all        = np.arange(Nsubj, dtype=int)
idx_withNORDIC = indices_for_group(subject_ids_withNORDIC)
idx_noNORDIC   = indices_for_group(subject_ids_noNORDIC)

print(f"indices all:        {idx_all}")
print(f"indices withNORDIC: {idx_withNORDIC}")
print(f"indices noNORDIC:   {idx_noNORDIC}")

METRIC_INFO = {
    "tsnr_orig": {
        "label": "tSNR (original)",
        "short": "tSNRorig",
        "do_surface": True,
        "vmin": 5,
        "vmax": 28,
    },
    "tsnr_resid": {
        "label": "tSNR (residual std)",
        "short": "tSNRresid",
        "do_surface": True,
        "vmin": 5,
        "vmax": 175,
    },
    "r2": {
        "label": "R² (variance explained)",
        "short": "R2",
        "do_surface": False,
    },
    "inv_noise": {
        "label": "inv_noise (1/noise_resid)",
        "short": "invNoise",
        "do_surface": True,
        "vmin": None,         # auto-scale from data
        "vmax": None,
    },
    "mean": {
        "label": "Mean signal",
        "short": "Mean",
        "do_surface": True,   # <- we want surface maps for mean
        "vmin": None,         # auto-scale from data
        "vmax": None,
    },
}

# 3) Helper to run summaries + plots for any subset and any metric
def run_group_summary(group_name: str, idx: np.ndarray) -> None:
    """
    For each metric, compute and plot summaries for a subset of subjects (given indices).
    Produces, per metric:
      - group_mean_layer_parcel (saved)
      - violin plot
      - subject summary matrix
      - surface maps for Deep/Middle/Superficial (tSNR metrics only)
    """
    if idx.size == 0:
        print(f"[WARN] Group {group_name} has no subjects, skipping.")
        return

    group_out_dir = os.path.join(GROUP_OUT, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    print(f"[{group_name}] subjects: {idx.size}")

    for metric_key, info in METRIC_INFO.items():
        if metric_key not in all_metrics:
            print(f"[{group_name}] [WARN] metric {metric_key} missing, skipping.")
            continue

        metric_label = info["label"]
        metric_short = info["short"]
        do_surface   = info["do_surface"]

        metric_data = all_metrics[metric_key]  # (Nsubj, 3, 400)
        group_data  = metric_data[idx, :, :]   # (Nsubj_group, 3, 400)

        print(f"[{group_name}] Metric {metric_key} data shape: {group_data.shape}")

        # 3.1 Group-average parcel vectors per layer
        group_mean_layer_parcel = np.nanmean(group_data, axis=0)  # (3, 400)

        # prefix includes group + metric to avoid overwriting
        ltu.save_group_parcel_vectors(
            group_mean_layer_parcel,
            group_out_dir,
            prefix=f"{group_name}_{metric_short}_layer",
        )

        # 3.2 Per-subject layer means (averaged across parcels)
        layer1_subj_means, layer2_subj_means, layer3_subj_means = (
            ltu.compute_layer_subject_means(group_data)
        )

        # 3.3 ANOVA across layers
        Fval, pval = ltu.run_layer_anova(
            layer1_subj_means, layer2_subj_means, layer3_subj_means
        )
        print(
            f"[{group_name}] {metric_key} ANOVA across layers: "
            f"F = {Fval:.4f}, p = {pval:.4e}"
        )

        # 3.4 Violin plot (Deep/Middle/Superficial)
        fig_path = os.path.join(
            group_out_dir, f"violin_{metric_short}_layers_{group_name}.svg"
        )
        y_label = f"Mean {metric_label} (per subject, avg over parcels)"
        ltu.plot_layer_tsnr_violin(
            layer1_subj_means,
            layer2_subj_means,
            layer3_subj_means,
            out_path=fig_path,
            Fval=Fval,
            pval=pval,
            layer_names=("Deep", "Middle", "Superficial"),
            metric_name=metric_label,
            y_label=y_label,
        )

        # 3.5 Per-subject summary matrix (Nsubj_grp × 3)
        summary_path = os.path.join(
            group_out_dir, f"subject_layer_mean_{metric_short}_{group_name}.npy"
        )
        ltu.save_subject_summary(
            layer1_subj_means,
            layer2_subj_means,
            layer3_subj_means,
            out_path=summary_path,
        )

        # 3.6 Surface maps of parcel-wise mean (Deep/Middle/Superficial)
        if do_surface:
            deep_vec = group_mean_layer_parcel[0, :]
            mid_vec  = group_mean_layer_parcel[1, :]
            sup_vec  = group_mean_layer_parcel[2, :]

            # Get preferred vmin/vmax from METRIC_INFO, or compute from data
            vmin = info.get("vmin", None)
            vmax = info.get("vmax", None)

            if vmin is None or vmax is None:
                combined = np.concatenate([deep_vec, mid_vec, sup_vec])
                finite = combined[np.isfinite(combined)]
                if finite.size == 0:
                    vmin, vmax = 0.0, 1.0
                else:
                    # a bit robust to outliers
                    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)

            plotSurfaceMap(
                deep_vec,
                group_out_dir,
                f"SurfaceMap_{group_name}_{metric_short}_Deep.png",
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            plotSurfaceMap(
                mid_vec,
                group_out_dir,
                f"SurfaceMap_{group_name}_{metric_short}_Mid.png",
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            plotSurfaceMap(
                sup_vec,
                group_out_dir,
                f"SurfaceMap_{group_name}_{metric_short}_Sup.png",
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )

# 4) Run summaries for each group
run_group_summary("all",        idx_all)
run_group_summary("withNORDIC", idx_withNORDIC)
run_group_summary("noNORDIC",   idx_noNORDIC)

print("Done.")