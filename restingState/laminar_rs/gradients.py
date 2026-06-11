# laminar_rs/gradients.py
"""
Gradient-based analyses and laminar (dis)similarity metrics.

This module provides:
- BrainSpace-based gradient estimation from connectivity / affinity matrices
- Inter-areal and intra-areal laminar dissimilarity measures
- Clustering of eigenvectors with sign-invariant distances
- Cosine-similarity utilities and visualisation across thresholds
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainspace.gradient import GradientMaps
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _split_layers(G: np.ndarray, N: int = 360) -> List[np.ndarray]:
    """
    Split a (L*N × k) array into L layers of shape (N × k).

    Parameters
    ----------
    G : np.ndarray
        Gradient or embedding array of shape (L*N, k).
    N : int
        Number of parcels per layer.

    Returns
    -------
    layers : list of (N × k) arrays
    """
    if G.ndim != 2:
        raise ValueError(f"Expected 2D array; got {G.ndim}D")
    L, rem = divmod(G.shape[0], N)
    if rem != 0 or L < 1:
        raise ValueError(f"Expected shape (L*{N}, k); got {G.shape}")
    return [G[i * N:(i + 1) * N, :] for i in range(L)]


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalisation with safe handling of zero rows.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Y = X / norms
    zero_rows = np.isclose(np.linalg.norm(X, axis=1), 0.0)
    if np.any(zero_rows):
        Y[zero_rows, :] = 0.0
    return Y


def _equidistant_layer_indices(L: int) -> Tuple[int, int, int]:
    """
    Return three approximately equidistant layer indices (0-based) spanning [0, L-1].

    By default (for L >= 3):
        idx = ceil([L/4, L/2, 3L/4]) - 1

    For small L where this creates duplicates, fall back to linspace over [0, L-1].

    Returns
    -------
    (sup_idx, mid_idx, deep_idx) : tuple of int
        Sorted indices interpreted as (superficial, middle, deep).
    """
    if L < 3:
        raise ValueError("Need at least 3 layers to choose superficial/middle/deep.")
    idx = np.ceil(np.array([1, 2, 3]) * L / 4.0).astype(int) - 1
    idx = np.clip(idx, 0, L - 1)
    if len(np.unique(idx)) < 3:
        idx = np.rint(np.linspace(0, L - 1, 3)).astype(int)
    return tuple(np.sort(idx).tolist())


# ---------------------------------------------------------------------
# Gradient estimation
# ---------------------------------------------------------------------


def run_gradient_analysis_auto(
    conn_matrix: np.ndarray,
    outputDir: str,
    n_components: int | None = None,
    max_components: int = 20,
    var_threshold: float | None = None,
    kernel: str = "cosine",
    approach: str = "dm",
    random_state: int = 0,
    sparsity: float | None = 0.9,
    scree_basename: str = "GradientScree",
) -> Tuple[
    np.ndarray,  # gradients (N × n_keep)
    np.ndarray,  # lambdas (n_keep,)
    np.ndarray,  # all_lambdas (max_components,)
    np.ndarray,  # frac_explained (max_components,)
    np.ndarray,  # cum_explained (max_components,)
    int,         # n_keep
]:
    """
    Run BrainSpace GradientMaps with automatic component selection and scree plot.

    The function first fits up to `max_components` gradients, computes a
    normalised eigenvalue spectrum (fraction explained, cumulative), saves
    a scree plot, and then selects how many gradients to KEEP:

    - If `n_components` is not None: hard override (clipped to max_components).
    - Else if `var_threshold` is not None: keep the minimum number of
      components whose cumulative fraction >= var_threshold.
    - Else: keep all `max_components`.

    Parameters
    ----------
    conn_matrix : np.ndarray
        (N × N) connectivity / similarity matrix (e.g. supra-adjacency).
    outputDir : str
        Directory where the scree plot SVG is saved.
    n_components : int or None
        Manual override for how many gradients to RETURN (n_keep).
    max_components : int
        Number of components to FIT for the scree plot (upper bound).
    var_threshold : float or None
        Target cumulative fraction of the normalised spectrum, in (0, 1].
        Ignored if `n_components` is provided.
    kernel : {"cosine", "pearson", "normalized_angle", ...}
        Kernel type passed to BrainSpace.
    approach : {"dm", "le", "pca"}
        Diffusion map ("dm"), Laplacian eigenmaps ("le"), or PCA.
    random_state : int
        RNG seed for reproducibility.
    sparsity : float or None
        Proportion of strongest affinities to retain. Pass None to disable.
    scree_basename : str
        Base name for the scree SVG file (without extension).

    Returns
    -------
    gradients : np.ndarray
        (N × n_keep) gradient coordinates.
    lambdas : np.ndarray
        (n_keep,) eigenvalues for the kept gradients.
    all_lambdas : np.ndarray
        (max_components,) eigenvalues for all fitted gradients.
    frac_explained : np.ndarray
        (max_components,) normalised “variance-like” contribution per component.
    cum_explained : np.ndarray
        (max_components,) cumulative fraction explained.
    n_keep : int
        Number of components actually returned.
    """
    os.makedirs(outputDir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(conn_matrix, cmap="PRGn")
    plt.title("Adjacency matrix")
    plt.savefig(
        os.path.join(outputDir, "Adjacency_matrix.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    N = conn_matrix.shape[0]
    max_components = int(min(max_components, N - 1))

    # ------------------------------------------------------------------
    # 1) Fit GradientMaps with a generous number of components
    # ------------------------------------------------------------------
    gm = GradientMaps(
        kernel=kernel,
        approach=approach,
        n_components=max_components,
        random_state=random_state,
    )
    gm.fit(conn_matrix, sparsity=sparsity)

    all_lambdas = np.asarray(gm.lambdas_, dtype=float).reshape(-1)
    # Just in case BrainSpace returns fewer than requested
    max_components = all_lambdas.size

    # ------------------------------------------------------------------
    # 2) Turn eigenvalues into a normalised "variance-like" spectrum
    # ------------------------------------------------------------------
    appr = approach.lower()

    if appr == "pca":
        # PCA: lambdas_ already correspond to variance explained
        scores = all_lambdas.copy()
    elif appr.startswith("le"):
        # Laplacian eigenmaps: smaller eigenvalues are more important
        # Use inverse as an importance measure
        eps = 1e-12
        scores = 1.0 / (np.abs(all_lambdas) + eps)
    else:
        # Diffusion maps (dm) and other: use |lambda| as importance
        scores = np.abs(all_lambdas)

    total_score = scores.sum()
    if total_score <= 0:
        raise ValueError(
            "Sum of eigenvalue scores is non-positive; cannot compute "
            "fraction explained."
        )

    frac_explained = scores / total_score
    cum_explained = np.cumsum(frac_explained)

    # ------------------------------------------------------------------
    # 3) Decide how many components to keep
    # ------------------------------------------------------------------
    if n_components is not None:
        # Manual override
        n_keep = int(np.clip(n_components, 1, max_components))
    elif var_threshold is not None:
        if not (0 < var_threshold <= 1):
            raise ValueError("var_threshold must be in (0, 1].")
        # First index where cumulative fraction >= threshold
        n_keep = int(np.searchsorted(cum_explained, var_threshold) + 1)
    else:
        # Default: keep everything we fitted
        n_keep = max_components

    # ------------------------------------------------------------------
    # 4) Scree plot (SVG, same style as your other QC plots)
    # ------------------------------------------------------------------
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    x = np.arange(1, max_components + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, frac_explained * 100.0, "o-", label="Component")
    ax.set_xlabel("Component")
    ax.set_ylabel("Normalised eigenvalue [%]")
    ax.set_title("Gradient scree plot")

    ax2 = ax.twinx()
    ax2.plot(x, cum_explained * 100.0, "s--", color="tab:orange",
             label="Cumulative")
    ax2.set_ylabel("Cumulative [%]")

    # Vertical line for chosen n_keep
    ax.axvline(n_keep, color="r", linestyle="--")
    ax.text(
        n_keep + 0.1,
        ax.get_ylim()[1] * 0.9,
        f"n_keep = {n_keep}\n({cum_explained[n_keep-1]*100:.1f}%)",
        color="r",
    )

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    scree_path = os.path.join(outputDir, f"{scree_basename}.svg")
    fig.savefig(scree_path, bbox_inches="tight", format="svg")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) Truncate gradients and eigenvalues to n_keep
    # ------------------------------------------------------------------
    gradients = gm.gradients_[:, :n_keep]
    lambdas = all_lambdas[:n_keep]

    return gradients, lambdas, all_lambdas, frac_explained, cum_explained, n_keep



def run_gradient_analysis(
    conn_matrix: np.ndarray,
    n_components: int = 10,
    kernel: str = "cosine",
    approach: str = "dm",
    random_state: int = 0,
    sparsity: float | None = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BrainSpace GradientMaps on a connectivity matrix.

    Parameters
    ----------
    conn_matrix : np.ndarray
        (N × N) connectivity / similarity matrix (e.g., supra-adjacency).
    n_components : int
        Number of gradient components.
    kernel : {"cosine", "pearson", "normalized_angle", ...}
        Kernel type passed to BrainSpace.
    approach : {"dm", "le"}
        Diffusion map ("dm") or Laplacian eigenmaps ("le").
    random_state : int
        RNG seed for reproducibility.
    sparsity : float or None
        Proportion of strongest affinities to retain. Pass None to disable.

    Returns
    -------
    gradients : np.ndarray
        (N × n_components) gradient coordinates.
    lambdas : np.ndarray
        Eigenvalues associated with each gradient.
    """
    gm = GradientMaps(
        kernel=kernel,
        approach=approach,
        n_components=n_components,
        random_state=random_state,
    )
    gm.fit(conn_matrix, sparsity=sparsity)
    return gm.gradients_, gm.lambdas_


# ---------------------------------------------------------------------
# Inter-areal & intra-areal laminar dissimilarity
# ---------------------------------------------------------------------

def inter_areal_dissimilarity(
    G_all: np.ndarray,
    outputDir: str,
    N: int = 400,
    zscore_within_layer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inter-areal laminar dissimilarity.

    For each parcel, we build a laminar profile by concatenating that parcel's
    gradient coordinates across layers, normalised within each layer. We then
    measure the mean cosine distance between a parcel's laminar profile and
    all other parcels' profiles.

    Parameters
    ----------
    G_all : np.ndarray
        (L*N × k) matrix of gradients, stacked by layer: [layer1; layer2; ...].
    outputDir : str
        Directory to write QC SVGs into.
    N : int
        Number of parcels per layer.
    zscore_within_layer : bool
        If True, z-score gradients within each layer before normalisation.

    Returns
    -------
    distanceSum : (N,) np.ndarray
        Overall mean cosine distance across concatenated laminar profiles.
    distanceSum_deep : (N,) np.ndarray
    distanceSum_mid : (N,) np.ndarray
    distanceSum_sup : (N,) np.ndarray
        Layer-specific mean cosine distances from equidistant indices
        interpreted as (deep, mid, superficial).
    """
    os.makedirs(outputDir, exist_ok=True)

    layers = _split_layers(G_all, N=N)  # list length L; each (N × k)
    L = len(layers)

    if zscore_within_layer:
        layers = [
            (X - X.mean(axis=0, keepdims=True))
            / (X.std(axis=0, keepdims=True) + 1e-12)
            for X in layers
        ]

    # Unit-row embeddings per layer
    U_layers = [_l2_normalize_rows(X) for X in layers]

    # (N × (L*k)) laminar profile per parcel
    P = np.concatenate(U_layers, axis=1)

    # Use text in SVGs instead of paths
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    # Visual QC: concatenated profile matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(P, cmap="PRGn")
    plt.title("ConcatMatrix P - inter areal dis")
    plt.colorbar(label="InterMatrix")
    plt.savefig(
        os.path.join(outputDir, "ConcatMatrixP_inter.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    # Cosine distance across concatenated profiles.
    # Each layer block is unit-norm, so a concatenated profile has norm sqrt(L);
    # dividing the dot product by L recovers the cosine similarity of the full
    # laminar profile, giving a distance bounded in [0, 2].

    S = (P @ P.T) / L  # cosine similarity
    D = 1.0 - S        # cosine distance
    np.fill_diagonal(D, 0.0)

    # Mean distance per parcel (excluding self)
    distanceSum = D.sum(axis=1) / (N - 1)

    # Per-layer distances & means
    D_layers = [1.0 - (U @ U.T) for U in U_layers]
    for Dl in D_layers:
        np.fill_diagonal(Dl, 0.0)
    distanceSum_layers = np.vstack(
        [Dl.sum(axis=1) / (N - 1) for Dl in D_layers]
    )  # (L, N)

    i_deep, i_mid, i_sup = _equidistant_layer_indices(L)
    distanceSum_sup = distanceSum_layers[i_sup]
    distanceSum_mid = distanceSum_layers[i_mid]
    distanceSum_deep = distanceSum_layers[i_deep]

    # Visual QC: distance matrix & overall mean-distance column
    plt.figure(figsize=(6, 6))
    plt.imshow(D, cmap="viridis")
    plt.title("Distance matrix - inter areal dis")
    plt.colorbar(label="Distance")
    plt.savefig(
        os.path.join(outputDir, "Matrix_interArealDis.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(D_layers[i_deep], cmap="viridis")
    plt.title("Inter areal distance matrix - deep")
    plt.colorbar(label="Distance")
    plt.savefig(
        os.path.join(outputDir, "Matrix_interArealDeep.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(D_layers[i_mid], cmap="viridis")
    plt.title("Inter areal distance matrix - middle")
    plt.colorbar(label="Distance")
    plt.savefig(
        os.path.join(outputDir, "Matrix_interArealMid.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(D_layers[i_sup], cmap="viridis")
    plt.title("Inter areal distance matrix - superficial")
    plt.colorbar(label="Distance")
    plt.savefig(
        os.path.join(outputDir, "Matrix_interArealSup.svg"),
        bbox_inches="tight",
        format="svg",
    )
    plt.close()

    return distanceSum, distanceSum_deep, distanceSum_mid, distanceSum_sup, D

def intra_areal_dissimilarity(
    G_all: np.ndarray,
    outputDir: str,
    N: int = 400,
    zscore_within_layer: bool = True,
    mode: str = "to_mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """
    Intra-areal laminar dissimilarity.

    Quantifies laminar heterogeneity *within* each parcel.

    Parameters
    ----------
    G_all : np.ndarray
        (L*N × k) gradients stacked by layer.
    outputDir : str
        Directory for QC SVGs.
    N : int
        Number of parcels per layer.
    zscore_within_layer : bool
        If True, z-score gradients within each layer before normalisation.
    mode : {"to_mean", "pairwise"}
        - "to_mean": per-layer cosine distance to parcel's mean direction.
        - "pairwise": mean pairwise cosine distance among all layers.

    Returns
    -------
    If mode == "to_mean":
        d_intraMean : (N,)
        d_superficial : (N,)
        d_middle : (N,)
        d_deep : (N,)
    If mode == "pairwise":
        d_pairwise : (N,)
    """
    os.makedirs(outputDir, exist_ok=True)

    layers = _split_layers(G_all, N=N)
    L = len(layers)

    if zscore_within_layer:
        layers = [
            (X - X.mean(axis=0, keepdims=True))
            / (X.std(axis=0, keepdims=True) + 1e-12)
            for X in layers
        ]

    U_layers = [_l2_normalize_rows(X) for X in layers]

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    if mode == "to_mean":
        # Mean raw vector across layers per parcel, then normalise
        Ubar = _l2_normalize_rows(sum(layers) / float(L))

        # Debug plot: Ubar
        plt.figure(figsize=(6, 6))
        plt.imshow(Ubar, cmap="PRGn")
        plt.title("Ubar - intra areal dis")
        plt.savefig(
            os.path.join(outputDir, "Matrix_intraArealDis.svg"),
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

        # Per-layer cosine distance to Ubar
        d_layers = [
            1.0 - np.einsum("ij,ij->i", U, Ubar) for U in U_layers
        ]  # list of (N,)
        D_layers = np.vstack(d_layers)  # (L, N)

        d_intraMean = D_layers.mean(axis=0)

        # Overview heatmap: layers × parcels
        plt.figure(figsize=(8, 6))
        plt.imshow(D_layers, aspect="auto", cmap="cividis")
        plt.title("Per-layer distances to mean direction (layers × parcels)")
        plt.ylabel("Layer")
        plt.xlabel("Parcel")
        plt.savefig(
            os.path.join(outputDir, "Matrix_intraArealDis_layers.svg"),
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

        # Column view
        plt.figure(figsize=(10, 10))
        plt.imshow(d_intraMean[:, np.newaxis], cmap="cividis")
        plt.title("D_intraMean - intra areal dis")
        plt.savefig(
            os.path.join(outputDir, "Matrix_mean_intraArealDis.svg"),
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

        # Equidistant layers interpreted as sup/mid/deep
        i_deep, i_mid, i_sup = _equidistant_layer_indices(L)
        d_superficial = D_layers[i_sup]
        d_middle = D_layers[i_mid]
        d_deep = D_layers[i_deep]


        # Optional: save each selected layer
        for label, vec in [
            ("sup", d_superficial),
            ("mid", d_middle),
            ("deep", d_deep),
        ]:
            plt.figure(figsize=(10, 10))
            plt.imshow(vec[:, np.newaxis], cmap="cividis")
            plt.title(f"D_{label} - intra areal dis")
            plt.savefig(
                os.path.join(outputDir, f"Matrix_{label}_intraArealDis.svg"),
                bbox_inches="tight",
                format="svg",
            )
            plt.close()

        return d_intraMean, d_deep, d_middle, d_superficial

    elif mode == "pairwise":
        if L < 2:
            raise ValueError("pairwise mode requires at least 2 layers.")

        # Stack to (L, N, k)
        Ustack = np.stack(U_layers, axis=0)

        # Layer-layer similarities per parcel: S[a,b,i] = dot(U[a,i,:], U[b,i,:])
        S = np.einsum("aik,bik->abi", Ustack, Ustack)  # (L, L, N)

        # Mean over unique off-diagonal pairs a<b
        iu = np.triu_indices(L, k=1)
        mean_S_pairs = S[iu[0], iu[1], :].mean(axis=0)  # (N,)
        d_pairwise = 1.0 - mean_S_pairs
        return d_pairwise

    else:
        raise ValueError("mode must be 'to_mean' or 'pairwise'")


__all__ = [
    "run_gradient_analysis_auto",
    "run_gradient_analysis",
    "inter_areal_dissimilarity",
    "intra_areal_dissimilarity",
]