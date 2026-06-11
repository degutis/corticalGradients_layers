from __future__ import annotations
from typing import Tuple

import numpy as np
import networkx as nx  # (still unused here, but kept in case other code relies on it)
from typing import List, Optional, Tuple

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



# ---------- adjacency matrices ----------

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

def within_layer_block_matrix_allLayerCombos(
    cfg: LaminarConfig,
    diagonal_pairs: Optional[List[Tuple[int, int]]] = None,
    subtract_average: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Multi-run block adjacency with selectable diagonal-block layer pairs.

    By default, diagonal blocks are within-layer correlations:
    [(0,0), (1,1), ..., (L-1, L-1)].

    Pass `diagonal_pairs` to override. For L=3 with index 0=deep, 1=mid, 2=sup:
        diagonal_pairs=[(0,2), (1,1), (2,2)]   # deep-sup, mid-mid, sup-sup
        diagonal_pairs=[(1,0), (1,2), (0,2)]   # mid-deep, mid-sup, deep-sup

    Off-diagonal blocks are identity matrices.

    Returns
    -------
    adj_full : (N*L, N*L) block adjacency.
    diag_blocks : (N, N, L) the N×N matrices placed on the diagonal blocks,
                  in the order given by `diagonal_pairs`.
    """
    layer_groups = group_files_by_layer(cfg)
    sorted_layers = sorted(layer_groups.items())
    L = cfg.num_layers
    if len(sorted_layers) != L:
        raise ValueError(f"Expected {L} layers, found {len(sorted_layers)}")

    if diagonal_pairs is None:
        diagonal_pairs = [(i, i) for i in range(L)]
    if len(diagonal_pairs) != L:
        raise ValueError(
            f"diagonal_pairs must have length {L}, got {len(diagonal_pairs)}"
        )
    for (i, j) in diagonal_pairs:
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"layer index out of range in pair ({i},{j}); L={L}")

    # Per-layer time series (concatenated across runs)
    layer_ts = [load_and_concat_layer(cfg, files) for _, files in sorted_layers]

    def pair_corr(i: int, j: int) -> np.ndarray:
        if i == j:
            corr = np.corrcoef(layer_ts[i])
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0.0)
            return corr
        stacked = np.vstack([layer_ts[i], layer_ts[j]])      # (2N, T)
        full = np.corrcoef(stacked)
        full = np.nan_to_num(full, nan=0.0)
        cross = full[:cfg.N, cfg.N:]                          # (N, N)
        return 0.5 * (cross + cross.T)                        # symmetrize

    diag_blocks = np.empty((cfg.N, cfg.N, L))
    for k, (i, j) in enumerate(diagonal_pairs):
        diag_blocks[:, :, k] = pair_corr(i, j)

    if subtract_average:
        avg = diag_blocks.mean(axis=2)
        diag_blocks -= avg[..., None]

    I = np.eye(cfg.N)
    blocks = [
        [diag_blocks[:, :, i] if i == j else I for j in range(L)]
        for i in range(L)
    ]
    adj_full = np.block(blocks)

    return adj_full, diag_blocks

def full_adjacency_multirun(cfg: LaminarConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-run, full adjacency: concatenate runs within each layer, then
    concatenate layers along rows and compute the full (N*L, N*L) correlation.

    Returns
    -------
    adj_full : (N*L, N*L) thresholded adjacency
    full_corr : (N*L, N*L) raw correlation matrix (with 0 diagonal)
    """
    layer_groups = group_files_by_layer(cfg)
    sorted_layers = sorted(layer_groups.items())
    L = cfg.num_layers
    if len(sorted_layers) != L:
        raise ValueError(f"Expected {L} layers, found {len(sorted_layers)}")

    layer_series = []
    for layer_idx, files in sorted_layers:
        concatenated = load_and_concat_layer(cfg, files)   # (N, T_layer)
        layer_series.append(concatenated)

    all_series = np.concatenate(layer_series, axis=0)      # (N*L, T)
    full_corr = np.corrcoef(all_series)
    full_corr = np.nan_to_num(full_corr, nan=0.0)
    np.fill_diagonal(full_corr, 0.0)

    return full_corr, full_corr