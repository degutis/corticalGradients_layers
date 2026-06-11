#!/usr/bin/env python3
"""
Centroid geometry similarity via pairwise Euclidean distances (label-matched):

Given two centroid CSVs (from plot_scatter_centroids), this script:
1) Loads centroids and matches rows by (layer, network)
2) Computes within-dataset pairwise Euclidean distances between all centroids:
      DA(i,j) = ||Ai - Aj||,  DB(i,j) = ||Bi - Bj||
3) Vectorizes the upper triangle of DA and DB and correlates them (Pearson + Spearman)
4) Computes a null distribution by permuting labels (rows) of dataset B and recomputing the correlation

Saves:
- pairwise_distance_vectors__A_vs_B.npz (dA_vec, dB_vec, labels, r_pearson, r_spearman)
- null_corr_pearson.npy
- null_corr_spearman.npy
- centroid_pairwise_distance_corr_null.svg
- summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


def load_centroids_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"layer", "network", "x_centroid", "y_centroid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["layer"] = df["layer"].astype(int)
    df["network"] = df["network"].astype(int)
    df["x_centroid"] = df["x_centroid"].astype(float)
    df["y_centroid"] = df["y_centroid"].astype(float)

    if df.duplicated(subset=["layer", "network"]).any():
        dups = df[df.duplicated(subset=["layer", "network"], keep=False)][["layer", "network"]]
        raise ValueError(f"Duplicate (layer, network) rows in {path}:\n{dups}")

    return df


def match_centroids(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    m = dfA.merge(dfB, on=["layer", "network"], suffixes=("_A", "_B"), how="inner")

    if len(m) != len(dfA) or len(m) != len(dfB):
        Aset = set(map(tuple, dfA[["layer", "network"]].to_numpy()))
        Bset = set(map(tuple, dfB[["layer", "network"]].to_numpy()))
        raise ValueError(
            "Label mismatch between A and B.\n"
            f"  In A not B: {sorted(Aset - Bset)}\n"
            f"  In B not A: {sorted(Bset - Aset)}"
        )

    return m.sort_values(["layer", "network"]).reset_index(drop=True)


def pairwise_distance_vector(xy: np.ndarray) -> np.ndarray:
    """Upper-triangle vector of Euclidean distances (same order as scipy pdist)."""
    return pdist(xy, metric="euclidean")


def corr_ignore_nan(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan
    if method == "pearson":
        return pearsonr(x[mask], y[mask])[0]
    if method == "spearman":
        return spearmanr(x[mask], y[mask])[0]
    raise ValueError("method must be 'pearson' or 'spearman'")


def run_pairwise_distance_correlation(
    csv_A,
    csv_B,
    out_dir=None,
    n_perm=10000,
    seed=0,
):
    csv_A = Path(csv_A)
    csv_B = Path(csv_B)

    if out_dir is None:
        out_dir = csv_A.parent / "centroid_pairwise_distance_corr"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfA = load_centroids_csv(csv_A)
    dfB = load_centroids_csv(csv_B)
    m = match_centroids(dfA, dfB)

    # coordinates in matched label order
    A = m[["x_centroid_A", "y_centroid_A"]].to_numpy()
    B = m[["x_centroid_B", "y_centroid_B"]].to_numpy()
    labels = m[["layer", "network"]].to_numpy()

    dA = pairwise_distance_vector(A)
    dB = pairwise_distance_vector(B)

    rP = corr_ignore_nan(dA, dB, method="pearson")
    rS = corr_ignore_nan(dA, dB, method="spearman")

    # --- null: permute B labels (rows), recompute dB and correlation ---
    rng = np.random.default_rng(seed)
    nullP = np.empty(n_perm, dtype=float)
    nullS = np.empty(n_perm, dtype=float)

    n = B.shape[0]
    for k in range(n_perm):
        perm = rng.permutation(n)
        dB_perm = pairwise_distance_vector(B[perm])
        nullP[k] = corr_ignore_nan(dA, dB_perm, method="pearson")
        nullS[k] = corr_ignore_nan(dA, dB_perm, method="spearman")

    # p-values: "as large or larger than observed" (similar geometry => high correlation)
    pP = (np.sum(nullP >= rP) + 1) / (n_perm + 1)
    pS = (np.sum(nullS >= rS) + 1) / (n_perm + 1)

    # --- save vectors + stats ---
    npz_path = out_dir / f"pairwise_distance_vectors__{csv_A.stem}__VS__{csv_B.stem}.npz"
    np.savez(
        npz_path,
        dA_vec=dA,
        dB_vec=dB,
        labels=labels,
        r_pearson=rP,
        r_spearman=rS,
        p_pearson=pP,
        p_spearman=pS,
        n_perm=n_perm,
        seed=seed,
        csv_A=str(csv_A),
        csv_B=str(csv_B),
    )

    np.save(out_dir / f"null_corr_pearson__{csv_A.stem}__VS__{csv_B.stem}.npy", nullP)
    np.save(out_dir / f"null_corr_spearman__{csv_A.stem}__VS__{csv_B.stem}.npy", nullS)

    # --- plot nulls + observed ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    axes[0].hist(nullP, bins=50, alpha=0.8)
    axes[0].axvline(rP, linewidth=2)
    axes[0].set_title(f"Pearson r (obs={rP:.3f}, p={pP:.3g})")
    axes[0].set_xlabel("Null correlation")
    axes[0].set_ylabel("Count")

    axes[1].hist(nullS, bins=50, alpha=0.8)
    axes[1].axvline(rS, linewidth=2)
    axes[1].set_title(f"Spearman ρ (obs={rS:.3f}, p={pS:.3g})")
    axes[1].set_xlabel("Null correlation")

    fig.suptitle("Pairwise centroid-distance correlation (label permutation null)", y=1.02)
    fig.tight_layout()

    fig_path = out_dir / f"centroid_pairwise_distance_corr_null__{csv_A.stem}__VS__{csv_B.stem}.svg"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- summary ---
    summary_path = out_dir / f"summary__{csv_A.stem}__VS__{csv_B.stem}.txt"
    with open(summary_path, "w") as f:
        f.write(f"A: {csv_A}\n")
        f.write(f"B: {csv_B}\n")
        f.write(f"n_centroids: {n}\n")
        f.write(f"n_pairs: {dA.size}\n")
        f.write("\nObserved correlation of pairwise distances:\n")
        f.write(f"  Pearson r:  {rP:.6g}   (perm p={pP:.6g})\n")
        f.write(f"  Spearman ρ: {rS:.6g}   (perm p={pS:.6g})\n")
        f.write("\nNull generation:\n")
        f.write("  Permuted labels of dataset B (row permutation), recomputed DB and correlation.\n")
        f.write(f"  n_perm: {n_perm}\n")
        f.write(f"  seed: {seed}\n")

    print("[OK] Saved:")
    print("  ", npz_path)
    print("  ", fig_path)
    print("  ", summary_path)


# =============================
# Main script
# =============================

if __name__ == "__main__":

    run_pairwise_distance_correlation(
        csv_A="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/WithinLayer_gradients_kernelCOS_API_interSpecific/dissimilarityGradient/Scatter/Scatter2D_NetCentroids/Scatter_SupDeep_centroid_centroids.csv",
        csv_B="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Glasser/WithinLayer_gradients_kernelCOS_API_interSpecific/dissimilarityGradient/Scatter/Scatter2D_NetCentroids/Scatter_SupDeep_centroid_centroids.csv",
        out_dir="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations/smallGap_Schaefer/WithinLayer_gradients_kernelCOS_API_interSpecific/robustness_centroids"
    )