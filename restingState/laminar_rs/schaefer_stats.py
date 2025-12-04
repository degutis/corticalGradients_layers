# laminar_rs/schaefer_stats.py
"""
Schaefer-400 / RSN-7 network statistics with spatial null models.

This module provides:
- Bookkeeping for Schaefer-400 parcels on fs_LR 32k
- Safe parcel ↔ vertex mapping helpers
- Network-wise ANOVAs (per layer) with spin nulls
- Network × layer interaction tests with spin nulls
- Spin-based tests for (partial) correlations using ENIGMA's API
- CSV savers for tidy outputs
"""

from __future__ import annotations

import csv
from typing import Dict, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.datasets import load_conte69
from brainspace.null_models import SpinPermutations
from enigmatoolbox.permutation_testing import (
    rotate_parcellation,
    perm_sphere_p,
)
from scipy.stats import f_oneway

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

RSN7_NAMES: List[str] = [
    "Visual",
    "Somatomotor",
    "Dorsal Attn",
    "Ventral/Salience",
    "Limbic",
    "Control",
    "Default",
]

# Defaults for fs_LR Schaefer-400 label GIFTIs
_DEFAULT_SCHAEFER_L = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii"
_DEFAULT_SCHAEFER_R = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii"

# Caches for centroids and permutations (so we don’t recompute every time)
_SCHAEFER400_CENTROIDS_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
_SCHAEFER400_PERM_CACHE: Dict[Tuple[str, str, int, int | None], np.ndarray] = {}


# ---------------------------------------------------------------------
# Label helpers (fs_LR 32k Schaefer-400)
# ---------------------------------------------------------------------


def _load_label_gii(path: str) -> Tuple[np.ndarray, Mapping[int, str]]:
    """
    Load a GIFTI label file (fs_LR 32k) and return labels + key->name mapping.
    """
    g = nib.load(path)
    labs = np.asarray(g.agg_data(), dtype=int).squeeze()
    lt = g.labeltable
    key_to_name = {lab.key: lab.label for lab in lt.labels}
    return labs, key_to_name


def _schaefer7_from_name(name: str) -> int:
    """
    Map Schaefer-400 region name to RSN-7 index (0..6).
    """
    n = name.lower()
    if "vis" in n:
        return 0
    if "som" in n or "sommot" in n:
        return 1
    if "dorsattn" in n or ("dors" in n and "attn" in n):
        return 2
    if (
        "ventattn" in n
        or "salventattn" in n
        or ("vent" in n and "attn" in n)
        or "sal" in n
    ):
        return 3
    if "limbic" in n:
        return 4
    if (
        "cont" in n
        or "control" in n
        or "frontoparietal" in n
        or "fp" in n
    ):
        return 5
    if "default" in n:
        return 6
    raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")


def build_schaefer400_bookkeeping(
    schaefer_label_L: str = _DEFAULT_SCHAEFER_L,
    schaefer_label_R: str = _DEFAULT_SCHAEFER_R,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """
    Build Schaefer-400 bookkeeping for fs_LR 32k.

    Parameters
    ----------
    schaefer_label_L, schaefer_label_R : str
        Paths to the left and right GIFTI label files.

    Returns
    -------
    uL, uR : np.ndarray
        Sorted unique parcel IDs (excluding 0) for LH and RH.
    networks0 : np.ndarray, shape (400,)
        RSN-7 index (0..6) for each parcel (LH first, then RH).
    parcel_verts_L, parcel_verts_R : list of np.ndarray
        Indices of vertices belonging to each parcel in uL / uR order.
    mw_L, mw_R : np.ndarray (bool)
        Medial wall masks per hemisphere (True = medial wall).
    """
    L_lab, L_map = _load_label_gii(schaefer_label_L)
    R_lab, R_map = _load_label_gii(schaefer_label_R)

    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    assert len(L_lab) == sphere_lh.n_points and len(R_lab) == sphere_rh.n_points, (
        "Label files must be fs_LR 32k and aligned to Conte69 spheres."
    )

    uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
    uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

    nets0: List[int] = []
    for k in uL:
        nets0.append(_schaefer7_from_name(L_map[k]))
    for k in uR:
        nets0.append(_schaefer7_from_name(R_map[k]))
    networks0 = np.asarray(nets0, dtype=int)  # length 400

    parcel_verts_L = [np.where(L_lab == k)[0] for k in uL]
    parcel_verts_R = [np.where(R_lab == k)[0] for k in uR]

    mw_L = L_lab == 0
    mw_R = R_lab == 0
    return uL, uR, networks0, parcel_verts_L, parcel_verts_R, mw_L, mw_R


# ---------------------------------------------------------------------
# Vertexization / de-vertexization (mask-safe)
# ---------------------------------------------------------------------


def parcel_to_vertices(
    D400: np.ndarray,
    uL: np.ndarray,
    uR: np.ndarray,
    parcel_verts_L: Sequence[np.ndarray],
    parcel_verts_R: Sequence[np.ndarray],
    nL: int,
    nR: int,
    mw_L: np.ndarray | None = None,
    mw_R: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broadcast parcel values (length 400) to vertices; medial wall set to NaN.

    Parameters
    ----------
    D400 : np.ndarray
        Parcel values with length len(uL) + len(uR).
    uL, uR : np.ndarray
        Unique parcel IDs for LH / RH.
    parcel_verts_L, parcel_verts_R : list of np.ndarray
        Indices of vertices per parcel in uL/uR order.
    nL, nR : int
        Number of vertices per hemisphere.
    mw_L, mw_R : np.ndarray or None
        Medial wall masks; if provided, set those vertices to NaN.

    Returns
    -------
    vL, vR : np.ndarray
        Per-vertex values (NaN on medial wall).
    """
    D400 = np.asarray(D400, float).squeeze()
    if D400.shape[0] != (len(uL) + len(uR)):
        raise ValueError("D400 has wrong length for Schaefer-400.")

    vL = np.full(nL, np.nan, float)
    vR = np.full(nR, np.nan, float)

    for i, idxs in enumerate(parcel_verts_L):
        vL[idxs] = float(D400[i])
    for j, idxs in enumerate(parcel_verts_R):
        vR[idxs] = float(D400[len(uL) + j])

    if mw_L is not None:
        vL[mw_L] = np.nan
    if mw_R is not None:
        vR[mw_R] = np.nan

    return vL, vR


def vertices_to_parcels(
    v_full: np.ndarray,
    nL: int,
    parcel_verts_L: Sequence[np.ndarray],
    parcel_verts_R: Sequence[np.ndarray],
    n_parc_L: int,
) -> np.ndarray:
    """
    Average vertices back to parcels; returns NaN if a parcel has no finite verts.

    Parameters
    ----------
    v_full : np.ndarray
        Concatenated LH+RH vertex values (length nL + nR).
    nL : int
        Number of LH vertices.
    parcel_verts_L, parcel_verts_R : list of np.ndarray
        Per-parcel vertex indices per hemisphere.
    n_parc_L : int
        Number of left-hemisphere parcels.

    Returns
    -------
    out : np.ndarray
        Parcel values (length n_parc_L + len(parcel_verts_R)).
    """
    v_full = np.asarray(v_full, float).squeeze()
    vL = v_full[:nL]
    vR = v_full[nL:]

    out = np.empty(n_parc_L + len(parcel_verts_R), float)

    for i, idxs in enumerate(parcel_verts_L):
        vals = vL[idxs]
        good = np.isfinite(vals)
        out[i] = vals[good].mean() if np.any(good) else np.nan

    for j, idxs in enumerate(parcel_verts_R):
        vals = vR[idxs]
        good = np.isfinite(vals)
        out[n_parc_L + j] = vals[good].mean() if np.any(good) else np.nan

    return out


# ---------------------------------------------------------------------
# Stats helpers (ANOVAs)
# ---------------------------------------------------------------------


def anova_F_by_network(D: np.ndarray, networks0: np.ndarray) -> float:
    """
    One-way ANOVA F across 7 RSNs on a parcel vector D (NaNs must be masked).

    Parameters
    ----------
    D : np.ndarray
        Parcel data (length 400).
    networks0 : np.ndarray
        RSN-7 labels 0..6 per parcel.

    Returns
    -------
    F : float
        One-way ANOVA F statistic.
    """
    D = np.asarray(D, float).squeeze()
    groups = [D[networks0 == k] for k in range(7)]
    F, _ = f_oneway(*groups)
    return float(F)


def _F_interaction_general(
    Y_layers: Sequence[np.ndarray],
    networks0: np.ndarray,
) -> Tuple[float, int, int]:
    """
    Partial F for Network×Layer interaction.

    Compares:
      Reduced: Intercept + Layer + Network
      Full   : Reduced + Layer×Network

    Parameters
    ----------
    Y_layers : list of L arrays
        Layer-wise parcel data (same length N).
    networks0 : np.ndarray
        Network labels 0..6 per parcel.

    Returns
    -------
    F : float
    df1, df2 : int
        Numerator and denominator degrees of freedom.
    """
    Y_layers = [np.asarray(Y, float).squeeze() for Y in Y_layers]
    L = len(Y_layers)
    n_parc = Y_layers[0].size

    y = np.concatenate(Y_layers, axis=0)  # length L*N
    net = np.tile(networks0, L)  # 0..6
    layer = np.repeat(np.arange(L), n_parc)  # 0..L-1

    # Reduced model: intercept + layer + network
    X0 = np.ones((y.size, 1))
    X_layer = (
        np.column_stack([(layer == l).astype(float) for l in range(1, L)])
        if L > 1
        else np.empty((y.size, 0))
    )
    X_net = np.column_stack([(net == k).astype(float) for k in range(1, 7)])
    X_red = np.column_stack(
        [X0] + ([X_layer] if L > 1 else []) + [X_net]
    )

    # Interaction terms: (L-1)*(7-1) where l>0, k>0
    X_int_cols: List[np.ndarray] = []
    if L > 1:
        for l in range(1, L):
            for k in range(1, 7):
                X_int_cols.append(
                    ((layer == l) & (net == k)).astype(float)
                )
    X_int = (
        np.column_stack(X_int_cols)
        if len(X_int_cols)
        else np.empty((y.size, 0))
    )
    X_full = np.column_stack([X_red, X_int])

    beta_f, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    rss_f = float(np.sum((y - X_full @ beta_f) ** 2))

    beta_r, *_ = np.linalg.lstsq(X_red, y, rcond=None)
    rss_r = float(np.sum((y - X_red @ beta_r) ** 2))

    df1 = (L - 1) * 6
    df2 = y.size - X_full.shape[1]
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return float(F), int(df1), int(df2)


# ---------------------------------------------------------------------
# Layer-wise one-way ANOVAs (2–4 inputs), SPIN nulls (BrainSpace)
# ---------------------------------------------------------------------


def layerwise_network_anova(
    D_layers: Sequence[np.ndarray],
    layer_names: Sequence[str] | None = None,
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    n_perm: int = 10_000,
    random_state: int = 0,
) -> Tuple[Dict[str, Mapping[str, object]], List[str]]:
    """
    Layer-wise one-way ANOVAs (7 RSNs) with ENIGMA-style spin nulls.

    Parameters
    ----------
    D_layers : sequence of arrays
        2..4 parcel maps, each length 400 in [uL,uR] order.
    layer_names : sequence of str or None
        Names for the layers; default: "Layer1", "Layer2", ...
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs (for bookkeeping + centroids).
    n_perm : int
        Number of permutations / rotations.
    random_state : int
        RNG seed passed to ENIGMA rotate_parcellation helper.

    Returns
    -------
    results : dict
        {layer_name -> {F_obs, p_perm, net_means(7,), net_ns(7,)}}.
        p_perm is ENIGMA-style p_spin for the ANOVA F.
    rsn_names : list of str
        RSN-7 network names in order.
    """
    if not (2 <= len(D_layers) <= 4):
        raise ValueError("D_layers must have 2, 3, or 4 maps for the layer-wise ANOVAs.")

    # Atlas bookkeeping (RSN labels etc.)
    (
        uL,
        uR,
        networks0,
        parcel_verts_L,
        parcel_verts_R,
        mw_L,
        mw_R,
    ) = build_schaefer400_bookkeeping(schaefer_label_L, schaefer_label_R)
    n_parc = len(uL) + len(uR)

    # Layer names
    Lnames = (
        list(layer_names)
        if layer_names is not None
        else [f"Layer{i+1}" for i in range(len(D_layers))]
    )
    if len(Lnames) != len(D_layers):
        raise ValueError("layer_names length must match D_layers length.")

    # Sanitize layer data, precompute masks, F_obs, and network means
    D_layers_arr: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    F_obs: List[float] = []
    net_means: List[np.ndarray] = []

    for i, (name, D) in enumerate(zip(Lnames, D_layers)):
        D = np.asarray(D, float).squeeze()
        if D.shape[0] != n_parc:
            raise ValueError(
                f"{name}: expected length {n_parc} in [uL,uR] order, got {D.size}."
            )
        D_layers_arr.append(D)

        mask = np.isfinite(D)
        if mask.sum() < 5:
            raise ValueError(f"{name}: fewer than 5 finite parcels after masking.")
        masks.append(mask)

        F_obs_i = anova_F_by_network(D[mask], networks0[mask])
        F_obs.append(F_obs_i)

        net_means_i = np.array(
            [np.nanmean(D[networks0 == k]) for k in range(7)],
            dtype=float,
        )
        net_means.append(net_means_i)

    # Network sample sizes (same for all layers)
    net_ns = np.array(
        [int(np.sum(networks0 == k)) for k in range(7)],
        dtype=int,
    )

    # ENIGMA-style spin permutations at parcel level
    perm_id = _get_schaefer400_perm_id(
        n_perm=n_perm,
        schaefer_label_L=schaefer_label_L,
        schaefer_label_R=schaefer_label_R,
        random_state=random_state,
    )
    # perm_id shape: (n_parc, n_perm), int indices into 400-length vectors

    counts = np.zeros(len(D_layers_arr), dtype=int)

    for j in range(n_perm):
        idx = perm_id[:, j].astype(int)
        for li, (D, mask) in enumerate(zip(D_layers_arr, masks)):
            D_perm = D[idx][mask]
            F_perm = anova_F_by_network(D_perm, networks0[mask])
            if F_perm >= F_obs[li]:
                counts[li] += 1

    results: Dict[str, Mapping[str, object]] = {}
    for li, name in enumerate(Lnames):
        p = (counts[li] + 1) / (n_perm + 1)
        results[name] = dict(
            F_obs=float(F_obs[li]),
            p_perm=float(p),
            net_means=net_means[li],
            net_ns=net_ns,
        )

    return results, RSN7_NAMES

# ---------------------------------------------------------------------
# Network × Layer interaction (2 or 3 layers), SPIN (BrainSpace)
# ---------------------------------------------------------------------


def network_layer_interaction_general(
    D_layers: Sequence[np.ndarray],
    layer_names: Sequence[str] | None = None,
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    n_perm: int = 10_000,
    random_state: int = 0,
) -> Mapping[str, object]:
    """
    Network × layer interaction with ENIGMA-style spin nulls.

    Parameters
    ----------
    D_layers : sequence of arrays
        2 or 3 parcel maps (length 400).
    layer_names : sequence of str or None
        Layer names; defaults to "Layer1", "Layer2", ...
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs.
    n_perm : int
        Number of rotations.
    random_state : int
        RNG seed for ENIGMA rotate_parcellation helper.

    Returns
    -------
    res : dict
        Keys:
            F_int_obs, p_int_spin, df1, df2,
            cell_means (7 × L),
            net_ns (7,),
            layer_names (list of str)
    """
    if not isinstance(D_layers, (list, tuple)) or len(D_layers) not in (2, 3):
        raise ValueError("D_layers must be a list/tuple of length 2 or 3 for interaction.")

    L = len(D_layers)
    layer_names = (
        list(layer_names)
        if layer_names is not None
        else [f"Layer{i+1}" for i in range(L)]
    )

    # Atlas bookkeeping (RSN labels etc.)
    (
        uL,
        uR,
        networks0,
        parcel_verts_L,
        parcel_verts_R,
        mw_L,
        mw_R,
    ) = build_schaefer400_bookkeeping(schaefer_label_L, schaefer_label_R)
    n_parc = len(uL) + len(uR)

    # Sanitize data
    Y: List[np.ndarray] = []
    for idx, D in enumerate(D_layers):
        D = np.asarray(D, float).squeeze()
        if D.shape[0] != n_parc:
            raise ValueError(f"Layer {idx}: expected length {n_parc} with [uL,uR] order.")
        Y.append(D)

    # Global mask: parcels must be finite in all layers
    mask = np.ones(n_parc, dtype=bool)
    for D in Y:
        mask &= np.isfinite(D)
    if mask.sum() < 10:
        raise ValueError("Fewer than 10 parcels with finite data across all layers.")

    networks_eff = networks0[mask]
    Y_eff = [D[mask] for D in Y]

    # Observed interaction F
    F_obs, df1, df2 = _F_interaction_general(Y_eff, networks_eff)

    # ENIGMA-style spin permutations (same spins applied to all layers)
    perm_id = _get_schaefer400_perm_id(
        n_perm=n_perm,
        schaefer_label_L=schaefer_label_L,
        schaefer_label_R=schaefer_label_R,
        random_state=random_state,
    )
    # perm_id shape: (n_parc, n_perm)

    exceed = 0
    for j in range(n_perm):
        idx = perm_id[:, j].astype(int)
        Y_perm_eff: List[np.ndarray] = []
        for D in Y:
            D_perm = D[idx][mask]
            Y_perm_eff.append(D_perm)

        F_perm, _, _ = _F_interaction_general(Y_perm_eff, networks_eff)
        if F_perm >= F_obs:
            exceed += 1

    p_perm = (exceed + 1) / (n_perm + 1)

    # Cell means and counts (using original, unmasked maps with NaN-safe means)
    means_by_net_by_layer = np.zeros((7, L), float)
    for l, D in enumerate(Y):
        for k in range(7):
            means_by_net_by_layer[k, l] = float(np.nanmean(D[networks0 == k]))
    net_ns = np.array(
        [int(np.sum(networks0 == k)) for k in range(7)],
        dtype=int,
    )

    return dict(
        F_int_obs=float(F_obs),
        p_int_spin=float(p_perm),
        df1=int(df1),
        df2=int(df2),
        cell_means=means_by_net_by_layer,
        net_ns=net_ns,
        layer_names=list(layer_names),
    )

# ---------------------------------------------------------------------
# ENIGMA-style SPIN for (partial) correlations on Schaefer-400
# ---------------------------------------------------------------------


def _get_schaefer400_centroids(
    schaefer_label_L: str = _DEFAULT_SCHAEFER_L,
    schaefer_label_R: str = _DEFAULT_SCHAEFER_R,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (and cache) Schaefer-400 parcel centroids on the Conte69 sphere.

    Returns
    -------
    coord_l, coord_r : np.ndarray, shape (n_parc_L, 3), (n_parc_R, 3)
    """
    key = (schaefer_label_L, schaefer_label_R)
    if key in _SCHAEFER400_CENTROIDS_CACHE:
        return _SCHAEFER400_CENTROIDS_CACHE[key]

    (
        uL,
        uR,
        _networks0,
        parcel_verts_L,
        parcel_verts_R,
        _mw_L,
        _mw_R,
    ) = build_schaefer400_bookkeeping(
        schaefer_label_L, schaefer_label_R
    )
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)

    # BrainSpace surfaces are BSPolyData; coordinates are in .Points
    coords_l = []
    for idxs in parcel_verts_L:
        coords_l.append(np.mean(sphere_lh.Points[idxs, :], axis=0))
    coords_r = []
    for idxs in parcel_verts_R:
        coords_r.append(np.mean(sphere_rh.Points[idxs, :], axis=0))

    coord_l = np.asarray(coords_l, float)
    coord_r = np.asarray(coords_r, float)

    _SCHAEFER400_CENTROIDS_CACHE[key] = (coord_l, coord_r)
    return coord_l, coord_r


def _get_schaefer400_perm_id(
    n_perm: int,
    schaefer_label_L: str = _DEFAULT_SCHAEFER_L,
    schaefer_label_R: str = _DEFAULT_SCHAEFER_R,
    random_state: int | None = 0,
) -> np.ndarray:
    """
    Get (and cache) ENIGMA-style spin permutations for Schaefer-400.

    Uses enigmatoolbox.permutation_testing.rotate_parcellation under a
    controlled NumPy RNG seed to make results reproducible.
    """
    key = (schaefer_label_L, schaefer_label_R, n_perm, random_state)
    if key in _SCHAEFER400_PERM_CACHE:
        return _SCHAEFER400_PERM_CACHE[key]

    coord_l, coord_r = _get_schaefer400_centroids(
        schaefer_label_L, schaefer_label_R
    )

    # Control numpy RNG so permutations depend only on `random_state`
    if random_state is not None:
        state = np.random.get_state()
        np.random.seed(random_state)
        try:
            perm_id = rotate_parcellation(coord_l, coord_r, nrot=n_perm)
        finally:
            np.random.set_state(state)
    else:
        perm_id = rotate_parcellation(coord_l, coord_r, nrot=n_perm)

    perm_id = np.asarray(perm_id, int)
    _SCHAEFER400_PERM_CACHE[key] = perm_id
    return perm_id


def p_spin_corr_schaefer400(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10_000,
    corr_type: str = "pearson",
    schaefer_label_L: str = _DEFAULT_SCHAEFER_L,
    schaefer_label_R: str = _DEFAULT_SCHAEFER_R,
    random_state: int | None = 0,
) -> Tuple[float, float]:
    """
    ENIGMA-style spin permutation test for correlation on Schaefer-400.

    This is a thin wrapper around:
      - rotate_parcellation (to get `perm_id`)
      - perm_sphere_p (to get the spin p-value)

    Parameters
    ----------
    x, y : array-like, shape (400,)
        Schaefer-400 parcel data in [LH parcels, RH parcels] order.
        Must be finite (no NaNs).
    n_perm : int
        Number of spin rotations.
    corr_type : {'pearson', 'spearman'}
        Correlation type for empirical and null correlations.
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs.
    random_state : int or None
        RNG seed for rotations. If None, use global NumPy RNG.

    Returns
    -------
    r_emp : float
        Empirical correlation between x and y.
    p_spin : float
        Spin-based p-value from ENIGMA's perm_sphere_p.
    """
    x = np.asarray(x, float).squeeze()
    y = np.asarray(y, float).squeeze()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D vectors.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite (no NaNs or infs).")

    # Empirical correlation
    r_emp = float(np.corrcoef(x, y)[0, 1])

    # Spin permutations via ENIGMA
    perm_id = _get_schaefer400_perm_id(
        n_perm,
        schaefer_label_L=schaefer_label_L,
        schaefer_label_R=schaefer_label_R,
        random_state=random_state,
    )

    p_spin = float(
        perm_sphere_p(x, y, perm_id, corr_type=corr_type, null_dist=False)
    )
    return r_emp, p_spin


def p_spin_partial_corr_schaefer400(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    n_perm: int = 10_000,
    corr_type: str = "pearson",
    schaefer_label_L: str = _DEFAULT_SCHAEFER_L,
    schaefer_label_R: str = _DEFAULT_SCHAEFER_R,
    random_state: int | None = 0,
) -> Tuple[float, float]:
    """
    Spin permutation test for partial correlation on Schaefer-400.

    We:
      1. Regress out Z from x and y (multiple regression with intercept)
      2. Compute partial correlation r(x, y | Z) from residuals
      3. Feed residuals into ENIGMA's perm_sphere_p with Schaefer-400 spins

    Parameters
    ----------
    x, y : array-like, shape (400,)
        Target parcel maps (e.g., one laminar dissimilarity vs BigBrain G1).
    Z : array-like, shape (400, p)
        Covariates to control for (e.g., the other 2 laminar layers).
    n_perm : int
        Number of spin rotations.
    corr_type : {'pearson', 'spearman'}
        Correlation type for empirical and null correlations on residuals.
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs.
    random_state : int or None
        RNG seed for rotations.

    Returns
    -------
    r_partial_emp : float
        Empirical partial correlation r(x, y | Z).
    p_spin : float
        Spin-based p-value on the residual-residual correlation.
    """
    x = np.asarray(x, float).squeeze()
    y = np.asarray(y, float).squeeze()
    Z = np.asarray(Z, float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D vectors.")
    if Z.ndim == 1:
        Z = Z[:, None]

    n = x.shape[0]
    if Z.shape[0] != n:
        raise ValueError("Z must have the same number of rows as x and y (400).")

    # Ensure everything is finite
    all_data = np.column_stack([x, y, Z])
    if not np.all(np.isfinite(all_data)):
        raise ValueError("x, y and all columns of Z must be finite (no NaNs/infs).")

    # Design matrix for regression: intercept + Z
    X_design = np.column_stack([np.ones(n), Z])

    # Regress out Z from x and y
    beta_x, *_ = np.linalg.lstsq(X_design, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    res_x = x - X_design @ beta_x
    res_y = y - X_design @ beta_y

    # Empirical partial correlation
    r_partial_emp = float(np.corrcoef(res_x, res_y)[0, 1])

    # Spins via ENIGMA
    perm_id = _get_schaefer400_perm_id(
        n_perm,
        schaefer_label_L=schaefer_label_L,
        schaefer_label_R=schaefer_label_R,
        random_state=random_state,
    )

    p_spin = float(
        perm_sphere_p(res_x, res_y, perm_id, corr_type=corr_type, null_dist=False)
    )
    return r_partial_emp, p_spin


# ---------------------------------------------------------------------
# CSV savers
# ---------------------------------------------------------------------


def save_layerwise_results_csv(
    results: Mapping[str, Mapping[str, object]],
    rsn_names: Sequence[str] = RSN7_NAMES,
    out_csv: str = "layerwise_anova.csv",
) -> None:
    """
    Save layer-wise ANOVA results to CSV.

    Parameters
    ----------
    results : dict
        {layer_name -> {F_obs, p_perm, net_means(7,), net_ns(7,)}}.
    rsn_names : sequence of str
        RSN-7 names (default RSN7_NAMES).
    out_csv : str
        Output CSV path.

    Writes one row per (layer, network).
    """
    rows: List[Mapping[str, object]] = []
    for layer, r in results.items():
        F_obs = float(r["F_obs"])
        pval = float(r["p_perm"])
        means = np.asarray(r["net_means"]).ravel()
        ns = np.asarray(r["net_ns"]).ravel()
        for k, name in enumerate(rsn_names):
            rows.append(
                {
                    "layer": layer,
                    "network": name,
                    "net_mean": float(means[k]),
                    "net_n": int(ns[k]),
                    "F_obs_layer": F_obs,
                    "p_perm_layer": pval,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "layer",
            "network",
            "net_mean",
            "net_n",
            "F_obs_layer",
            "p_perm_layer",
        ],
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to: {out_csv}")


def save_interaction_to_csv(
    res_int: Mapping[str, object],
    rsn_names: Sequence[str] = RSN7_NAMES,
    out_csv: str = "interaction_anova.csv",
) -> None:
    """
    Save Network × Layer interaction (2 or 3 layers) to a tidy CSV.

    One row per (network, layer).
    """
    rows: List[Mapping[str, object]] = []
    means = np.asarray(res_int["cell_means"])
    layer_names = list(res_int["layer_names"])
    net_ns = np.asarray(res_int["net_ns"])

    for k, net_name in enumerate(rsn_names):
        for l_idx, layer_name in enumerate(layer_names):
            rows.append(
                {
                    "network": net_name,
                    "layer": layer_name,
                    "cell_mean": float(means[k, l_idx]),
                    "net_n": int(net_ns[k]),
                    "F_interaction": float(res_int["F_int_obs"]),
                    "df1": int(res_int["df1"]),
                    "df2": int(res_int["df2"]),
                    "p_perm_interaction": float(
                        res_int["p_int_spin"]
                    ),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "network",
            "layer",
            "cell_mean",
            "net_n",
            "F_interaction",
            "df1",
            "df2",
            "p_perm_interaction",
        ],
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to: {out_csv}")


__all__ = [
    "RSN7_NAMES",
    "build_schaefer400_bookkeeping",
    "parcel_to_vertices",
    "vertices_to_parcels",
    "anova_F_by_network",
    "layerwise_network_anova",
    "network_layer_interaction_general",
    "p_spin_corr_schaefer400",
    "p_spin_partial_corr_schaefer400",
    "save_layerwise_results_csv",
    "save_interaction_to_csv",
]