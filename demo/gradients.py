"""
gradients.py
============
Gradient-based analysis and inter-areal laminar dissimilarity.

Simplified demo version for the Schaefer-400 parcellation. Contains only the
two functions used by the demo notebook:

    - run_gradient_analysis_auto : BrainSpace gradient estimation with
      automatic component selection and a scree plot.
    - inter_areal_dissimilarity  : mean cosine distance between parcels'
      concatenated laminar gradient profiles.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainspace.gradient import GradientMaps

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _split_layers(G: np.ndarray, N: int = 400) -> List[np.ndarray]:
    """Split a (L*N x k) array into L layers of shape (N x k)."""
    if G.ndim != 2:
        raise ValueError(f"Expected 2D array; got {G.ndim}D")
    L, rem = divmod(G.shape[0], N)
    if rem != 0 or L < 1:
        raise ValueError(f"Expected shape (L*{N}, k); got {G.shape}")
    return [G[i * N:(i + 1) * N, :] for i in range(L)]


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation with safe handling of zero rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Y = X / norms
    zero_rows = np.isclose(np.linalg.norm(X, axis=1), 0.0)
    if np.any(zero_rows):
        Y[zero_rows, :] = 0.0
    return Y


def _equidistant_layer_indices(L: int) -> Tuple[int, int, int]:
    """
    Return three approximately equidistant layer indices (0-based) spanning
    [0, L-1], interpreted as (superficial, middle, deep).
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run BrainSpace GradientMaps with automatic component selection.

    Fits up to ``max_components`` gradients, computes a normalised eigenvalue
    spectrum (fraction explained, cumulative), saves a scree plot, and selects
    how many gradients to keep:

      - ``n_components`` given  -> hard override (clipped to max_components).
      - ``var_threshold`` given -> fewest components with cumulative >= threshold.
      - otherwise               -> keep all ``max_components``.

    Returns
    -------
    gradients      : (N x n_keep) gradient coordinates.
    lambdas        : (n_keep,) eigenvalues for the kept gradients.
    all_lambdas    : (max_components,) eigenvalues for all fitted gradients.
    frac_explained : (max_components,) normalised contribution per component.
    cum_explained  : (max_components,) cumulative fraction explained.
    n_keep         : number of components returned.
    """
    os.makedirs(outputDir, exist_ok=True)

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    # QC: adjacency matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(conn_matrix, cmap="PRGn")
    plt.title("Adjacency matrix")
    plt.savefig(
        os.path.join(outputDir, "Adjacency_matrix.svg"),
        bbox_inches="tight", format="svg",
    )
    plt.close()

    N = conn_matrix.shape[0]
    max_components = int(min(max_components, N - 1))

    # 1) Fit GradientMaps
    gm = GradientMaps(
        kernel=kernel,
        approach=approach,
        n_components=max_components,
        random_state=random_state,
    )
    gm.fit(conn_matrix, sparsity=sparsity)

    all_lambdas = np.asarray(gm.lambdas_, dtype=float).reshape(-1)
    max_components = all_lambdas.size

    # 2) Normalised "variance-like" spectrum
    appr = approach.lower()
    if appr == "pca":
        scores = all_lambdas.copy()
    elif appr.startswith("le"):
        scores = 1.0 / (np.abs(all_lambdas) + 1e-12)
    else:  # diffusion maps and others
        scores = np.abs(all_lambdas)

    total_score = scores.sum()
    if total_score <= 0:
        raise ValueError("Sum of eigenvalue scores is non-positive.")

    frac_explained = scores / total_score
    cum_explained = np.cumsum(frac_explained)

    # 3) Decide how many components to keep
    if n_components is not None:
        n_keep = int(np.clip(n_components, 1, max_components))
    elif var_threshold is not None:
        if not (0 < var_threshold <= 1):
            raise ValueError("var_threshold must be in (0, 1].")
        n_keep = int(np.searchsorted(cum_explained, var_threshold) + 1)
    else:
        n_keep = max_components

    # 4) Scree plot
    x = np.arange(1, max_components + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, frac_explained * 100.0, "o-", label="Component")
    ax.set_xlabel("Component")
    ax.set_ylabel("Normalised eigenvalue [%]")
    ax.set_title("Gradient scree plot")

    ax2 = ax.twinx()
    ax2.plot(x, cum_explained * 100.0, "s--", color="tab:orange", label="Cumulative")
    ax2.set_ylabel("Cumulative [%]")

    ax.axvline(n_keep, color="r", linestyle="--")
    ax.text(
        n_keep + 0.1,
        ax.get_ylim()[1] * 0.9,
        f"n_keep = {n_keep}\n({cum_explained[n_keep-1]*100:.1f}%)",
        color="r",
    )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.savefig(
        os.path.join(outputDir, f"{scree_basename}.svg"),
        bbox_inches="tight", format="svg",
    )
    plt.close(fig)

    # 5) Truncate to n_keep
    gradients = gm.gradients_[:, :n_keep]
    lambdas = all_lambdas[:n_keep]

    return gradients, lambdas, all_lambdas, frac_explained, cum_explained, n_keep


# ---------------------------------------------------------------------
# Inter-areal laminar dissimilarity
# ---------------------------------------------------------------------


def inter_areal_dissimilarity(
    G_all: np.ndarray,
    outputDir: str,
    N: int = 400,
    zscore_within_layer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inter-areal laminar dissimilarity.

    For each parcel we build a laminar profile by concatenating that parcel's
    gradient coordinates across layers (normalised within each layer), then
    measure the mean cosine distance to all other parcels' profiles.

    Returns
    -------
    distanceSum      : (N,) overall mean cosine distance.
    distanceSum_deep : (N,) deep-layer mean cosine distance.
    distanceSum_mid  : (N,) middle-layer mean cosine distance.
    distanceSum_sup  : (N,) superficial-layer mean cosine distance.
    D                : (N x N) overall cosine-distance matrix.
    """
    os.makedirs(outputDir, exist_ok=True)

    layers = _split_layers(G_all, N=N)  # list length L; each (N x k)
    L = len(layers)

    if zscore_within_layer:
        layers = [
            (X - X.mean(axis=0, keepdims=True))
            / (X.std(axis=0, keepdims=True) + 1e-12)
            for X in layers
        ]

    # Unit-row embeddings per layer
    U_layers = [_l2_normalize_rows(X) for X in layers]

    # (N x (L*k)) laminar profile per parcel
    P = np.concatenate(U_layers, axis=1)

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    # QC: concatenated profile matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(P, cmap="PRGn")
    plt.title("ConcatMatrix P - inter areal dis")
    plt.colorbar(label="InterMatrix")
    plt.savefig(
        os.path.join(outputDir, "ConcatMatrixP_inter.svg"),
        bbox_inches="tight", format="svg",
    )
    plt.close()

    # Cosine distance across concatenated profiles. Each layer block is
    # unit-norm, so dividing the dot product by L recovers the cosine
    # similarity of the full laminar profile (distance in [0, 2]).
    S = (P @ P.T) / L
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)

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

    # QC: overall distance matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(D, cmap="viridis")
    plt.title("Distance matrix - inter areal dis")
    plt.colorbar(label="Distance")
    plt.savefig(
        os.path.join(outputDir, "Matrix_interArealDis.svg"),
        bbox_inches="tight", format="svg",
    )
    plt.close()

    return distanceSum, distanceSum_deep, distanceSum_mid, distanceSum_sup, D
