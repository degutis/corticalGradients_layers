# laminar_rs/reliability.py
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import LaminarConfig
from .io_utils import group_files_by_layer, group_files_by_run, load_and_concat_layer


def compute_reliability(cfg: LaminarConfig,
                        TR: float = 3.2,
                        min_minutes: int = 1,
                        n_iterations: int = 500) -> Dict:
    """
    Functional version of plotReliability (excluding plotting).
    Returns dictionary: layer_num -> [(tmin, mean_corr), ...], plus 'all'.
    """
    volumes_per_minute = int(round(60.0 / TR))
    layer_groups = group_files_by_layer(cfg)
    reliability_results = {}

    for layer_num in sorted(layer_groups):
        files = layer_groups[layer_num]
        print(f"\nLayer {layer_num}: {len(files)} run(s)")
        data = load_and_concat_layer(cfg, files)
        n_parcels, total_vols = data.shape
        total_mins = total_vols // volumes_per_minute
        print(f"  total_vols={total_vols}, total_mins={total_mins}")

        if total_mins < 2 * min_minutes:
            print(f"  skip (need ≥{2*min_minutes} min total)")
            continue

        iu = np.triu_indices(n_parcels, k=1)
        layer_curve = []

        for tmin in tqdm(range(min_minutes, total_mins // 2 + 1),
                         desc=f"Layer {layer_num}"):
            win_vols = tmin * volumes_per_minute
            corrs = []

            for _ in range(n_iterations):
                start1 = np.random.randint(0, total_vols - win_vols + 1)
                seg1 = data[:, start1:start1 + win_vols]
                start2 = np.random.randint(0, total_vols - win_vols + 1)
                seg2 = data[:, start2:start2 + win_vols]
                if seg1.shape[1] != seg2.shape[1]:
                    seg2 = resample(seg2, seg1.shape[1], axis=1)

                fc1 = np.corrcoef(seg1)
                fc2 = np.corrcoef(seg2)
                v1, v2 = fc1[iu], fc2[iu]
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if valid.sum() < 2:
                    continue
                r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                corrs.append(r)

            layer_curve.append((tmin, np.nan if not corrs else np.mean(corrs)))
        reliability_results[layer_num] = layer_curve

    run_groups = group_files_by_run(layer_groups)
    data_runs = []
    for run_id, fps in run_groups.items():
        fps_sorted = sorted(fps, key=lambda x: int(x.name.split("_")[-1].replace(".npy", "")))
        arrs = [np.load(str(fp)) for fp in fps_sorted]
        T0 = arrs[0].shape[1]
        if any(a.shape[1] != T0 for a in arrs):
            raise ValueError(f"Run {run_id} has mismatched timepoints across layers")
        data_runs.append(np.vstack(arrs))

    data_all = np.concatenate(data_runs, axis=1)
    n_all, total_vols_all = data_all.shape
    total_mins_all = total_vols_all // volumes_per_minute
    print(f"\nALL-LAYERS: parcels={n_all}, total_vols={total_vols_all}, total_mins={total_mins_all}")

    if total_mins_all >= 2 * min_minutes:
        iu_all = np.triu_indices(n_all, k=1)
        all_curve = []
        for tmin in tqdm(range(min_minutes, total_mins_all // 2 + 1),
                         desc="ALL-LAYERS"):
            win_vols = tmin * volumes_per_minute
            corrs = []

            for _ in range(n_iterations):
                s1 = np.random.randint(0, total_vols_all - win_vols + 1)
                seg1 = data_all[:, s1:s1 + win_vols]
                s2 = np.random.randint(0, total_vols_all - win_vols + 1)
                seg2 = data_all[:, s2:s2 + win_vols]
                if seg1.shape[1] != seg2.shape[1]:
                    seg2 = resample(seg2, seg1.shape[1], axis=1)

                fc1 = np.corrcoef(seg1)
                fc2 = np.corrcoef(seg2)
                v1, v2 = fc1[iu_all], fc2[iu_all]
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if valid.sum() < 2:
                    continue
                r = np.corrcoef(v1[valid], v2[valid])[0, 1]
                corrs.append(r)

            all_curve.append((tmin, np.nan if not corrs else np.mean(corrs)))
        reliability_results["all"] = all_curve

    return reliability_results


def plot_reliability_curves(reliability_results: dict,
                            out_dir,
                            name: str = "Reliability_FC_matched.png"):
    """
    Plot result of compute_reliability().
    """
    import os
    out_dir = str(out_dir)
    plt.figure(figsize=(10, 6))
    for key, curve in sorted(reliability_results.items(), key=lambda x: str(x[0])):
        mins, rs = zip(*curve)
        label = "All layers" if key == "all" else f"Layer {key}"
        plt.plot(mins, rs, marker="o", label=label)

    plt.ylim(0, 1)
    plt.xlabel("Window length (minutes)")
    plt.ylabel("Mean FC-matrix corr")
    plt.title("Within-subject FC reliability vs. data amount (matched windows)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(out_dir, name)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot to {outpath}")