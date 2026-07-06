# laminar_rs/embedding.py
from __future__ import annotations
from typing import Tuple, Dict

import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt

def run_laplacian_embedding(M: np.ndarray,
                            out_dir,
                            name: str,
                            num_components: int = 10,
                            epsilon: float = 1e-10,
                            convert_to_binary: bool = True,
                            full: bool = False,
                            add_name: str = "",
                            v_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Your original runLaplacianEmbedding, turned into a pure function.

    Parameters
    ----------
    M : adjacency / similarity matrix
    out_dir : base directory for saving plots
    name : subfolder name
    """
    import os
    out_dir = str(out_dir)
    M = M.copy()

    if convert_to_binary:
        M[M != 0] = 1
    else:
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        M[M < 0] = 0.0

    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(M, cmap="viridis", vmin=0, vmax=v_max)
    plt.colorbar(label="Correlation")
    plt.title(f"{name} Block Matrix")
    plt.savefig(f"{out_dir}/{name}/Block_matrix{add_name}.png", bbox_inches="tight")
    plt.close()

    degree = np.sum(M, axis=1)
    D = np.diag(degree)
    L = D - M
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + epsilon))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    if full:
        eigvals, eigvecs = scipy.linalg.eigh(L_norm)
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(L_norm, k=num_components, which="SM")

    return eigvals, eigvecs


def plot_scree(eigvals: np.ndarray, out_dir, name: str, sort: bool = False) -> None:
    """Original plotScree as a standalone function."""
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    if sort:
        eigvals_sorted = np.sort(eigvals)[::-1]
    else:
        eigvals_sorted = eigvals

    eigvals_cumsum = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted) * 100
    num_components = eigvals.size

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, num_components + 1), eigvals_sorted,
             marker="o", linestyle="-", color="b", label="Eigenvalues")
    ax1.set_xlabel("Component Number")
    ax1.set_ylabel("Eigenvalue", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(range(1, num_components + 1), eigvals_cumsum,
             marker="s", linestyle="--", color="r", label="Cumulative Sum")
    ax2.set_ylabel("Cumulative Sum (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("Scree Plot with Cumulative Sum")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.savefig(f"{out_dir}/{name}/screePlot.png", bbox_inches="tight")
    plt.close(fig)


def _detect_local_outliers(vals: np.ndarray,
                           k_neighbors: int = 1,
                           method: str = "zscore",
                           thresh: float = 2.0):
    """
    Identify indices i where vals[i] deviates from the average of its
    k_neighbors on each side by more than thresh (either in SD units or
    absolute units).
    """
    N = len(vals)
    diffs = np.empty(N, dtype=float)
    neigh_mean = np.empty(N, dtype=float)

    for i in range(N):
        lo = max(0, i - k_neighbors)
        hi = min(N, i + k_neighbors + 1)
        nbrs = [j for j in range(lo, hi) if j != i]
        if not nbrs:
            neigh_mean[i] = 0.0
            diffs[i] = 0.0
        else:
            m = vals[nbrs].mean()
            neigh_mean[i] = m
            diffs[i] = vals[i] - m

    if method == "zscore":
        mu, sigma = diffs.mean(), diffs.std(ddof=0)
        z = (diffs - mu) / sigma
        outlier_idxs = np.where(np.abs(z) > thresh)[0]
    elif method == "abs":
        outlier_idxs = np.where(np.abs(diffs) > thresh)[0]
    else:
        raise ValueError("method must be 'zscore' or 'abs'")

    return outlier_idxs, diffs, neigh_mean


def run_plot_zero_crossings(W: np.ndarray,
                            U: np.ndarray,
                            out_dir,
                            name: str) -> np.ndarray:
    """
    Zero-crossings for Laplacian eigenvectors (your original run_plot_zeroCrossings).
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    n_ROI = U.shape[0]
    wZC = np.zeros(U.shape[1])
    for u in range(U.shape[1]):
        summ = 0
        for i in range(n_ROI - 1):
            for j in range(i + 1, n_ROI):
                if U[i, u] * U[j, u] < 0:
                    summ += (W[i, j] >= 1)
        wZC[u] = summ

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(wZC) + 1), wZC, marker="o", linestyle="-", color="b")
    plt.xlabel("Eigenvector")
    plt.ylabel("Zero Crossings")
    plt.title("Zero Crossings for Laplacian Eigenvectors")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{out_dir}/{name}/Crossings.png", bbox_inches="tight")
    plt.close()

    return wZC


def run_plot_FstatComp(eigvecs: np.ndarray,
                       out_dir,
                       name: str,
                       thresh: float = 2.5,
                       target: float = 1.0,
                       k_neighbors: int = 10):
    """
    Your original run_plot_FstatComp, as a function.

    Writes DifferenceInEigvecs.png and OutlierEigenvecs.txt.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    n_rows, n_cols = eigvecs.shape
    if n_rows != 1080:
        raise ValueError(f"Expected 1080 rows; got {n_rows}")

    avg_corrs = np.empty(n_cols)
    dissimilar = np.empty(n_cols)

    for i in range(n_cols):
        col = eigvecs[:, i]
        segs = [col[j * 360:(j + 1) * 360] for j in range(3)]

        r01 = np.corrcoef(segs[0], segs[1])[0, 1]
        r02 = np.corrcoef(segs[0], segs[2])[0, 1]
        r12 = np.corrcoef(segs[1], segs[2])[0, 1]

        zs = np.arctanh([r01, r02, r12])
        z_bar = zs.mean()
        r_bar = np.tanh(z_bar)
        avg_corrs[i] = r_bar
        dissimilar[i] = 1 - r_bar

    outliers, diffs, neigh_mean = _detect_local_outliers(
        dissimilar, k_neighbors=k_neighbors, method="zscore", thresh=thresh
    )

    x = np.arange(1, len(avg_corrs) + 1)
    fig, ax1 = plt.subplots()
    ax1.plot(x, dissimilar, marker="o", label="Avg Pearson r")
    ax1.set_xlabel("Eigenvector Number")
    ax1.set_ylabel("Average Dissimilarity (1-r)")
    fig.suptitle("Difference Metrics per Eigenvector")
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}/DifferenceInEigvecs.png", bbox_inches="tight")
    plt.close(fig)

    txt_path = f"{out_dir}/{name}/OutlierEigenvecs.txt"
    with open(txt_path, "w") as f:
        if len(outliers) == 0:
            f.write("No outliers detected.\n")
            return
        f.write("Index\tValue\tNeighborMean\tDiff\n")
        for i in outliers:
            f.write(f"{i+1}\t{dissimilar[i]:.6f}\t{neigh_mean[i]:.6f}\t{diffs[i]:.6f}\n")


def plot_eigenvector_correlation(eigvecs_orig: np.ndarray,
                                 out_dir,
                                 name: str,
                                 limit: int = 40,
                                 end_num: int = 40):
    """
    Your original plotEigenvectorCorrelation as a function.
    """
    import os
    out_dir = str(out_dir)
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    orig_X = eigvecs_orig.shape[1]
    end = orig_X - end_num
    eigvecs = np.hstack([
        eigvecs_orig[:, :limit],
        eigvecs_orig[:, end:]
    ])
    n_rows, n_cols = eigvecs.shape
    if n_rows != 1080:
        raise ValueError(f"Expected 1080 rows; got {n_rows}")

    layers = [eigvecs[i * 360:(i + 1) * 360, :] for i in range(3)]
    corr_mats: Dict[tuple, np.ndarray] = {}

    for i in range(3):
        for j in range(i, 3):
            A = layers[i]
            B = layers[j]

            if i == j:
                C = np.corrcoef(A, rowvar=False)
            else:
                M = np.concatenate([A, B], axis=1)
                bigC = np.corrcoef(M, rowvar=False)
                C = bigC[:n_cols, n_cols:2 * n_cols]

            corr_mats[(i, j)] = C

    pairs = [(0, 0), (1, 1), (2, 2),
             (0, 1), (0, 2),
             (1, 2)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (i, j) in zip(axes.flat, pairs):
        C = corr_mats[(i, j)]
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="cividis")
        ax.set_title(f"Layer {i} vs Layer {j}")
        ax.set_xlabel("Eigenvector index")
        ax.set_ylabel("Eigenvector index")

    fig.suptitle(f"{name} Layer Correlations", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Pearson r")
    plt.savefig(f"{out_dir}/{name}/CorrelationEigVecsMatrices_First{limit}_Last{end_num}.png",
                bbox_inches="tight")
    plt.close()