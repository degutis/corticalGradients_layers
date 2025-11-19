# laminar_rs/schaefer_stats.py
"""
Schaefer-400 / RSN-7 network statistics with spatial null models.

This module provides:
- Bookkeeping for Schaefer-400 parcels on fs_LR 32k
- Safe parcel ↔ vertex mapping helpers
- Network-wise ANOVAs (per layer)
- Network × layer interaction tests
- Spin and Moran Spectral Randomization (MSR) nulls
- CSV savers for tidy outputs
"""

from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np
from brainspace.datasets import load_conte69
from brainspace.mesh import mesh_elements as me
from brainspace.null_models import MoranRandomization, SpinPermutations
from scipy.stats import f_oneway

try:
    import pandas as pd  # optional
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


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
    schaefer_label_L: str,
    schaefer_label_R: str,
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
# Stats helpers
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
# Layer-wise one-way ANOVAs (2–4 inputs), with SPIN or MSR nulls
# ---------------------------------------------------------------------


def layerwise_network_anova(
    D_layers: Sequence[np.ndarray],
    layer_names: Sequence[str] | None = None,
    method: str = "spin",  # 'spin' or 'msr'
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    n_perm: int = 10_000,
    random_state: int = 0,
    batch_size: int = 50,
    spin_unique: bool = False,
) -> Tuple[Dict[str, Mapping[str, object]], List[str]]:
    """
    Layer-wise one-way ANOVAs (7 RSNs) with spatial nulls.

    Parameters
    ----------
    D_layers : sequence of arrays
        2..4 parcel maps, each length 400 in [uL,uR] order.
    layer_names : sequence of str or None
        Names for the layers; default: "Layer1", "Layer2", ...
    method : {'spin', 'msr'}
        Spatial null: spherical spin or Moran spectral randomization.
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs.
    n_perm : int
        Number of permutations / surrogates.
    random_state : int
        RNG seed.
    batch_size : int
        Batch size for spin permutations.
    spin_unique : bool
        Whether to enforce unique spins (SpinPermutations).

    Returns
    -------
    results : dict
        {layer_name -> {F_obs, p_perm, net_means(7,), net_ns(7,)}}.
        p_perm is p_spin (method='spin') or p_msr (method='msr').
    rsn_names : list of str
        RSN-7 network names in order.
    """
    if not (2 <= len(D_layers) <= 4):
        raise ValueError(
            "D_layers must have 2, 3, or 4 maps for the layer-wise ANOVAs."
        )

    # Atlas bookkeeping + spheres
    (
        uL,
        uR,
        networks0,
        parcel_verts_L,
        parcel_verts_R,
        mw_L,
        mw_R,
    ) = build_schaefer400_bookkeeping(
        schaefer_label_L, schaefer_label_R
    )
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    nL, nR = sphere_lh.n_points, sphere_rh.n_points

    # Sanitize inputs
    Lnames = (
        list(layer_names)
        if layer_names is not None
        else [f"Layer{i+1}" for i in range(len(D_layers))]
    )
    if len(Lnames) != len(D_layers):
        raise ValueError(
            "layer_names length must match D_layers length."
        )

    D_layers_arr: List[np.ndarray] = []
    for i, D in enumerate(D_layers):
        D = np.asarray(D, float).squeeze()
        if D.shape[0] != (len(uL) + len(uR)):
            raise ValueError(
                f"{Lnames[i]} length {D.size} != 400; ensure [uL,uR] order."
            )
        D_layers_arr.append(D)

    # Precompute per-layer vertex maps + observed F
    vLs: List[np.ndarray] = []
    vRs: List[np.ndarray] = []
    F_obs: List[float] = []
    net_means: List[np.ndarray] = []

    for D in D_layers_arr:
        vL, vR = parcel_to_vertices(
            D,
            uL,
            uR,
            parcel_verts_L,
            parcel_verts_R,
            nL,
            nR,
            mw_L,
            mw_R,
        )
        vLs.append(vL)
        vRs.append(vR)

        F_obs.append(anova_F_by_network(D, networks0))
        net_means.append(
            np.array([D[networks0 == k].mean() for k in range(7)])
        )

    net_ns = np.array(
        [int(np.sum(networks0 == k)) for k in range(7)], dtype=int
    )
    results: Dict[str, Mapping[str, object]] = {}

    method_l = method.lower()
    if method_l == "spin":
        rng = np.random.RandomState(random_state)
        counts = np.zeros(len(D_layers_arr), dtype=int)
        effN = np.zeros(len(D_layers_arr), dtype=int)

        filled = 0
        while filled < n_perm:
            m = min(batch_size, n_perm - filled)
            sp = SpinPermutations(
                n_rep=m,
                random_state=int(rng.randint(1e9)),
                unique=spin_unique,
            )
            sp.fit(sphere_lh, points_rh=sphere_rh)

            # For each layer, compute m spins
            rotL_list: List[np.ndarray] = []
            rotR_list: List[np.ndarray] = []
            for vL, vR in zip(vLs, vRs):
                rL, rR = sp.randomize(vL, vR)  # (m,nL), (m,nR)
                rotL_list.append(rL)
                rotR_list.append(rR)

            for i in range(m):
                for li, (D, rL, rR) in enumerate(
                    zip(D_layers_arr, rotL_list, rotR_list)
                ):
                    v_full = np.concatenate([rL[i], rR[i]])
                    Dp = vertices_to_parcels(
                        v_full,
                        nL,
                        parcel_verts_L,
                        parcel_verts_R,
                        len(uL),
                    )
                    mask = np.isfinite(D) & np.isfinite(Dp)
                    if mask.sum() < 5:
                        continue
                    Fp = anova_F_by_network(Dp[mask], networks0[mask])
                    Fobs = anova_F_by_network(
                        D[mask], networks0[mask]
                    )
                    counts[li] += int(Fp >= Fobs)
                    effN[li] += 1
            filled += m

        for li, name in enumerate(Lnames):
            p = (counts[li] + 1) / (effN[li] + 1)
            results[name] = dict(
                F_obs=float(F_obs[li]),
                p_perm=float(p),
                net_means=net_means[li],
                net_ns=net_ns,
            )

    elif method_l == "msr":
        # Build MSR weights
        wL = me.get_ring_distance(sphere_lh, n_ring=1)
        wL.data **= -1
        wR = me.get_ring_distance(sphere_rh, n_ring=1)
        wR.data **= -1

        msrL = MoranRandomization(
            n_rep=n_perm,
            joint=False,
            tol=1e-6,
            random_state=random_state,
        )
        msrR = MoranRandomization(
            n_rep=n_perm,
            joint=False,
            tol=1e-6,
            random_state=random_state,
        )
        msrL.fit(wL)
        msrR.fit(wR)

        for li, (name, vL, vR, D) in enumerate(
            zip(Lnames, vLs, vRs, D_layers_arr)
        ):
            rotL = msrL.randomize(vL)  # (n_perm,nL)
            rotR = msrR.randomize(vR)  # (n_perm,nR)
            count = 0
            for i in range(n_perm):
                v_full = np.concatenate([rotL[i], rotR[i]])
                Dp = vertices_to_parcels(
                    v_full,
                    nL,
                    parcel_verts_L,
                    parcel_verts_R,
                    len(uL),
                )
                mask = np.isfinite(D) & np.isfinite(Dp)
                if mask.sum() < 5:
                    continue
                Fp = anova_F_by_network(Dp[mask], networks0[mask])
                Fobs = anova_F_by_network(
                    D[mask], networks0[mask]
                )
                count += int(Fp >= Fobs)
            p = (count + 1) / (n_perm + 1)
            results[name] = dict(
                F_obs=float(F_obs[li]),
                p_perm=float(p),
                net_means=net_means[li],
                net_ns=net_ns,
            )

    else:
        raise ValueError("method must be 'spin' or 'msr'.")

    return results, RSN7_NAMES


# ---------------------------------------------------------------------
# Network × Layer interaction (2 or 3 layers), SPIN or MSR
# ---------------------------------------------------------------------


def network_layer_interaction_general(
    D_layers: Sequence[np.ndarray],
    layer_names: Sequence[str] | None = None,
    method: str = "spin",  # 'spin' or 'msr'
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    n_perm: int = 10_000,
    random_state: int = 0,
    batch_size: int = 50,
    spin_unique: bool = False,
) -> Mapping[str, object]:
    """
    Network × layer interaction with spatial nulls.

    Parameters
    ----------
    D_layers : sequence of arrays
        2 or 3 parcel maps (length 400).
    layer_names : sequence of str or None
        Layer names; defaults to "Layer1", "Layer2", ...
    method : {'spin', 'msr'}
        Spatial null (spins or Moran SR).
    schaefer_label_L, schaefer_label_R : str
        Paths to Schaefer-400 fs_LR label GIFTIs.
    n_perm : int
        Number of permutations / surrogates.
    random_state : int
        RNG seed.
    batch_size : int
        Batch size for spins.
    spin_unique : bool
        Enforce unique spins.

    Returns
    -------
    res : dict
        Keys:
            F_int_obs, p_int_spin, df1, df2,
            cell_means (7 × L),
            net_ns (7,),
            layer_names (list of str)
    """
    if not isinstance(D_layers, (list, tuple)) or len(D_layers) not in (
        2,
        3,
    ):
        raise ValueError(
            "D_layers must be a list/tuple of length 2 or 3 for interaction."
        )

    L = len(D_layers)
    layer_names = (
        list(layer_names)
        if layer_names is not None
        else [f"Layer{i+1}" for i in range(L)]
    )

    # Atlas + spheres
    (
        uL,
        uR,
        networks0,
        parcel_verts_L,
        parcel_verts_R,
        mw_L,
        mw_R,
    ) = build_schaefer400_bookkeeping(
        schaefer_label_L, schaefer_label_R
    )
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    nL, nR = sphere_lh.n_points, sphere_rh.n_points

    # Sanitize and vertexize
    Y: List[np.ndarray] = []
    vLs: List[np.ndarray] = []
    vRs: List[np.ndarray] = []

    for idx, D in enumerate(D_layers):
        D = np.asarray(D, float).squeeze()
        if D.shape[0] != (len(uL) + len(uR)):
            raise ValueError(
                f"Layer {idx}: expected length 400 with [uL,uR] order."
            )
        Y.append(D)
        vL, vR = parcel_to_vertices(
            D,
            uL,
            uR,
            parcel_verts_L,
            parcel_verts_R,
            nL,
            nR,
            mw_L,
            mw_R,
        )
        vLs.append(vL)
        vRs.append(vR)

    F_obs, df1, df2 = _F_interaction_general(Y, networks0)

    method_l = method.lower()
    if method_l == "spin":
        rng = np.random.RandomState(random_state)
        exceed = 0
        effN = 0
        filled = 0

        while filled < n_perm:
            m = min(batch_size, n_perm - filled)
            sp = SpinPermutations(
                n_rep=m,
                random_state=int(rng.randint(1e9)),
                unique=spin_unique,
            )
            sp.fit(sphere_lh, points_rh=sphere_rh)

            rotL_list: List[np.ndarray] = []
            rotR_list: List[np.ndarray] = []
            for vL, vR in zip(vLs, vRs):
                rL, rR = sp.randomize(vL, vR)
                rotL_list.append(rL)
                rotR_list.append(rR)

            for i in range(m):
                Dperm_layers: List[np.ndarray] = []
                for rL, rR in zip(rotL_list, rotR_list):
                    v_full = np.concatenate([rL[i], rR[i]])
                    Dp = vertices_to_parcels(
                        v_full,
                        nL,
                        parcel_verts_L,
                        parcel_verts_R,
                        len(uL),
                    )
                    Dperm_layers.append(Dp)

                # Mask fairness: drop parcels NaN in ANY observed or perm layer
                mask = np.ones(Y[0].shape[0], dtype=bool)
                for D in Y:
                    mask &= np.isfinite(D)
                for Dp in Dperm_layers:
                    mask &= np.isfinite(Dp)
                if mask.sum() < 10:
                    continue

                Fp, _, _ = _F_interaction_general(
                    [Dp[mask] for Dp in Dperm_layers], networks0[mask]
                )
                Fobs_masked, _, _ = _F_interaction_general(
                    [D[mask] for D in Y], networks0[mask]
                )
                exceed += int(Fp >= Fobs_masked)
                effN += 1
            filled += m

        p_perm = (exceed + 1) / (effN + 1)

    elif method_l == "msr":
        # Build MSR weights
        wL = me.get_ring_distance(sphere_lh, n_ring=1)
        wL.data **= -1
        wR = me.get_ring_distance(sphere_rh, n_ring=1)
        wR.data **= -1

        # Joint randomization across layers preserves cross-layer dependence
        msrL = MoranRandomization(
            n_rep=n_perm,
            joint=True,
            tol=1e-6,
            random_state=random_state,
        )
        msrR = MoranRandomization(
            n_rep=n_perm,
            joint=True,
            tol=1e-6,
            random_state=random_state,
        )
        msrL.fit(wL)
        msrR.fit(wR)

        VL = np.column_stack(vLs)  # (nL, L)
        VR = np.column_stack(vRs)  # (nR, L)
        rotL = msrL.randomize(VL)  # (n_perm, nL, L)
        rotR = msrR.randomize(VR)  # (n_perm, nR, L)

        exceed = 0
        for i in range(n_perm):
            Dperm_layers: List[np.ndarray] = []
            for l in range(L):
                v_full = np.concatenate(
                    [rotL[i, :, l], rotR[i, :, l]]
                )
                Dp = vertices_to_parcels(
                    v_full,
                    nL,
                    parcel_verts_L,
                    parcel_verts_R,
                    len(uL),
                )
                Dperm_layers.append(Dp)

            mask = np.ones(Y[0].shape[0], dtype=bool)
            for D in Y:
                mask &= np.isfinite(D)
            for Dp in Dperm_layers:
                mask &= np.isfinite(Dp)
            if mask.sum() < 10:
                continue

            Fp, _, _ = _F_interaction_general(
                [Dp[mask] for Dp in Dperm_layers], networks0[mask]
            )
            Fobs_m, _, _ = _F_interaction_general(
                [D[mask] for D in Y], networks0[mask]
            )
            exceed += int(Fp >= Fobs_m)
        p_perm = (exceed + 1) / (n_perm + 1)

    else:
        raise ValueError("method must be 'spin' or 'msr'.")

    # Cell means and counts
    means_by_net_by_layer = np.zeros((7, L), float)
    for l, D in enumerate(Y):
        for k in range(7):
            means_by_net_by_layer[k, l] = float(
                np.nanmean(D[networks0 == k])
            )
    net_ns = np.array(
        [int(np.sum(networks0 == k)) for k in range(7)], dtype=int
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
# CSV savers (SPIN or MSR)
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

    if pd is not None:
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
    else:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "layer",
                    "network",
                    "net_mean",
                    "net_n",
                    "F_obs_layer",
                    "p_perm_layer",
                ],
            )
            w.writeheader()
            w.writerows(rows)
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

    if pd is not None:
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
    else:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
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
            w.writeheader()
            w.writerows(rows)
    print(f"Saved CSV to: {out_csv}")


__all__ = [
    "RSN7_NAMES",
    "build_schaefer400_bookkeeping",
    "parcel_to_vertices",
    "vertices_to_parcels",
    "anova_F_by_network",
    "layerwise_network_anova",
    "network_layer_interaction_general",
    "save_layerwise_results_csv",
    "save_interaction_to_csv",
]