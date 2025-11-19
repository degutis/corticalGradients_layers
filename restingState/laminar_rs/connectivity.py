from __future__ import annotations
from typing import Tuple

import numpy as np
import networkx as nx  # (still unused here, but kept in case other code relies on it)

from .config import LaminarConfig
from .io_utils import group_files_by_layer, load_and_concat_layer


def thresh_and_binarize(
    adj: np.ndarray,
    set_thresh: float = 0.0,
    binarize: bool = False,
) -> np.ndarray:
    """
    Row-wise percentile threshold + optional binarisation + Fisher z.

    Parameters
    ----------
    adj : ndarray, shape (N, N, L)
        Per-layer adjacency/correlation matrix (typically in [-1, 1] or Fisher z).
    set_thresh : float
        Percentage (0–100) of weakest absolute edges per row to discard.
    binarize : bool, optional
        If True, return a 0/1 mask.
        If False, apply Fisher z (atanh) to the masked values and zero the diagonal.

    Returns
    -------
    A : ndarray, shape (N, N, L)
        Thresholded (and optionally z-transformed) matrix.
    """
    N, _, num_layers = adj.shape
    A = np.empty((N, N, num_layers), dtype=float)
    percent_thresh = set_thresh / 100.0

    for layer in range(num_layers):
        mag = np.abs(adj[:, :, layer])

        # Sort indices per row by magnitude
        sorted_idx = np.argsort(mag, axis=1)  # (N, N)
        mask = np.ones_like(mag, dtype=bool)
        rows = np.arange(N)[:, None]

        # Number of weakest edges per row to drop
        n_drop = int(np.floor(percent_thresh * N))
        if n_drop > 0:
            mask[rows, sorted_idx[:, :n_drop]] = False

        if binarize:
            A[:, :, layer] = mask.astype(float)
        else:
            # apply mask to original signed values
            corr_masked = adj[:, :, layer] * mask
            eps = 1.0 - 1e-6
            corr_masked = np.clip(corr_masked, -eps, eps)
            z_transformed = np.arctanh(corr_masked)
            np.fill_diagonal(z_transformed, 0.0)
            A[:, :, layer] = z_transformed

    return A


def fisher_z_to_r(z: np.ndarray) -> np.ndarray:
    """
    Inverse Fisher z-transform.

    Parameters
    ----------
    z : array-like
        Fisher z-transformed correlations.

    Returns
    -------
    r : ndarray
        Pearson correlation coefficients in [-1, 1].
    """
    z_clipped = np.clip(z, -5.0, 5.0)  # avoid overflow
    return np.tanh(z_clipped)


def build_multiplex_adjacency(per_layer_matrix: np.ndarray,
                              interlayer_weight: float = 1.0) -> np.ndarray:
    """
    Construct a multiplex adjacency matrix from per-layer connectivity.

    Parameters
    ----------
    per_layer_matrix : ndarray, shape (N, N, L)
        Connectivity for each of L layers.
    interlayer_weight : float, optional
        Weight of inter-layer edges between the same node across layers.
        Default is 1.0.

    Returns
    -------
    M : ndarray, shape (N*L, N*L)
        Multiplex adjacency matrix with zero diagonal.
    """
    A = np.asarray(per_layer_matrix)
    if A.ndim != 3 or A.shape[0] != A.shape[1]:
        raise ValueError("per_layer_matrix must have shape (N, N, L)")

    N, _, L = A.shape
    dtype = np.result_type(A.dtype, float)

    # Block-diagonal intra-layer connectivity
    M = np.zeros((L * N, L * N), dtype=dtype)
    for l in range(L):
        M[l * N:(l + 1) * N, l * N:(l + 1) * N] = A[:, :, l]

    # Uniform inter-layer coupling between corresponding nodes
    layer_coupling = np.ones((L, L), dtype=dtype) - np.eye(L, dtype=dtype)
    M += interlayer_weight * np.kron(layer_coupling, np.eye(N, dtype=dtype))

    # No self-loops
    np.fill_diagonal(M, 0.0)
    return M


# ---------- FC helpers ----------

def zero_lag_fc(X: np.ndarray) -> np.ndarray:
    """Pearson FC of X (n_parcels×T) at lag 0."""
    Xz = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return (Xz @ Xz.T) / (X.shape[1] - 1)


def lagged_corr(X: np.ndarray, Y: np.ndarray, t: int) -> np.ndarray:
    """
    Pearson corr of X(t) with Y(t+t).

    X, Y shape = (n_parcels, T)
    t > 0: X leads Y
    t < 0: Y leads X
    """
    n, T = X.shape
    if t >= 0:
        Xtr, Ytr = X[:, : T - t], Y[:, t:]
    else:
        Xtr, Ytr = X[:, -t:], Y[:, : T + t]

    Xz = (Xtr - Xtr.mean(axis=1, keepdims=True)) / Xtr.std(axis=1, keepdims=True)
    Yz = (Ytr - Ytr.mean(axis=1, keepdims=True)) / Ytr.std(axis=1, keepdims=True)
    return (Xz @ Yz.T) / (Xtr.shape[1] - 1)


# ---------- adjacency matrices ----------

def within_layer_single_run(cfg: LaminarConfig) -> np.ndarray:
    """
    Single-run version of the within-layer block matrix.

    Assumes files are ordered by layer 0..L-1.
    """
    npy_files = cfg.npy_files()
    if len(npy_files) < cfg.num_layers:
        raise ValueError("Not enough .npy files for num_layers")

    adj_layer = np.empty((cfg.N, cfg.N, cfg.num_layers))

    for i, fp in enumerate(npy_files[:cfg.num_layers]):
        ts = np.load(str(fp))
        corr = np.corrcoef(ts)
        adj_layer[:, :, i] = _threshold_corr(corr, cfg.set_thresh, diag_value=1.0)

    I = np.eye(cfg.N)
    blocks = []
    for i in range(cfg.num_layers):
        row = []
        for j in range(cfg.num_layers):
            if i == j:
                row.append(adj_layer[:, :, i])
            else:
                row.append(I)
        blocks.append(row)

    return np.block(blocks)


def within_layer_block_matrix(cfg: LaminarConfig,
                              subtract_average: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-run, within-layer adjacency + 'raw' (Fisher z) per-layer matrices.

    Returns
    -------
    adj_full : (N*L, N*L) block adjacency
    per_layer_corr_z : (N, N, L) Fisher-z per layer (optionally demeaned across layers)
    """
    layer_groups = group_files_by_layer(cfg)
    sorted_layers = sorted(layer_groups.items())
    L = cfg.num_layers
    if len(sorted_layers) != L:
        raise ValueError(f"Expected {L} layers, found {len(sorted_layers)}")

    adj_layer = np.empty((cfg.N, cfg.N, L))
    corr_layer_z = np.empty((cfg.N, cfg.N, L))

    for i, (layer_idx, files) in enumerate(sorted_layers):
        concatenated = load_and_concat_layer(cfg, files)
        corr = np.corrcoef(concatenated)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 0.0)
        adj_layer[:, :, i] = corr

    if subtract_average:
        avg = corr_layer_z.mean(axis=2)
        corr_layer_z -= avg[..., None]

    I = np.eye(cfg.N)
    blocks = []
    for i in range(L):
        row = []
        for j in range(L):
            if i == j:
                row.append(adj_layer[:, :, i])
            else:
                row.append(I)
        blocks.append(row)

    adj_full = np.block(blocks)
    return adj_full, adj_layer


def full_adjacency_single_run(cfg: LaminarConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate all layers along rows and compute full adjacency for a single session.

    Returns
    -------
    adj_full : |corr| thresholded matrix
    all_series : (N*num_layers, T_total) concatenated time series
    """
    all_series = []

    for fp in cfg.npy_files():
        print("Working on file:", fp.name)
        ts = np.load(str(fp))
        all_series.append(ts)

    all_series_array = np.concatenate(all_series, axis=0)  # (N*L, T)
    full_corr = np.corrcoef(all_series_array)
    full_corr = np.nan_to_num(full_corr, nan=0.0)
    np.fill_diagonal(full_corr, 1.0)
    adj_full = _threshold_corr(full_corr, cfg.set_thresh, diag_value=1.0)
    return adj_full, all_series_array


def full_adjacency_multirun(cfg: LaminarConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate runs within each layer, then across layers, and compute full adjacency.

    Returns
    -------
    adj_full : |corr| thresholded matrix
    all_series : (N*L, T_total) concatenated time series
    full_corr : raw correlation matrix (N*L, N*L)
    """
    layer_groups = group_files_by_layer(cfg)
    sorted_layers = sorted(layer_groups.items())
    concatenated_layers = []

    for layer_idx, files in sorted_layers:
        print(f"Processing Layer {layer_idx} with {len(files)} run(s)")
        concatenated = load_and_concat_layer(cfg, files)
        concatenated_layers.append(concatenated)
        print(f"Concatenated shape: {concatenated.shape}")

    all_series = np.concatenate(concatenated_layers, axis=0)  # (N*L, T)
    full_corr = np.corrcoef(all_series)
    np.fill_diagonal(full_corr, 0.0)
    full_corr = np.nan_to_num(full_corr, nan=0.0)
    adj_full = _threshold_corr(full_corr, cfg.set_thresh, diag_value=0.0)
    return adj_full, all_series, full_corr


def adjacency_single_layer(cfg: LaminarConfig, file_index: int) -> np.ndarray:
    """
    Thresholded adjacency for a single layer file by index in cfg.npy_files().
    """
    npy_files = cfg.npy_files()
    if file_index < 0 or file_index >= len(npy_files):
        raise IndexError("file_index out of range")
    fp = npy_files[file_index]
    print("Working on file:", fp.name)
    ts = np.load(str(fp))
    corr = np.corrcoef(ts)
    adj = _threshold_corr(corr, cfg.set_thresh, diag_value=1.0)
    return adj