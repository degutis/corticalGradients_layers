# laminar_rs/plots_embedding.py
#
# RSN-based embedding plots, supporting both Yeo-7 and Yeo-17 partitions
# for the Schaefer-400 and Glasser/HCP-MMP1.0 atlases.

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import nibabel as nib
from nilearn import plotting
from scipy.stats import pearsonr, t as t_dist
from scipy.stats import f_oneway, gaussian_kde

import hcp_utils as hcp


# ---------- Yeo network metadata ----------

_YEO7_LABELS = [
    "Visual", "Somatomotor", "Dorsal Attn",
    "Ventral/Salience", "Limbic", "Control", "Default"
]
_YEO7_ABBR = ["VIS", "SOM", "DAN", "VAN", "LIM", "CON", "DMN"]

# Canonical 17-network ordering used by the Yeo / Schaefer atlases.
# Note: the Schaefer 17Networks parcellation distinguishes:
#   VisCent / VisPeri, SomMotA / SomMotB, DorsAttnA / DorsAttnB,
#   SalVentAttnA / SalVentAttnB, LimbicA (TempPole) / LimbicB (OFC),
#   ContA / ContB / ContC, DefaultA / DefaultB / DefaultC, TempPar.
_YEO17_LABELS = [
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",
    "LimbicA", "LimbicB",
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "TempPar",
]
_YEO17_ABBR = [
    "VisC", "VisP",
    "SMA", "SMB",
    "DAN-A", "DAN-B",
    "VAN-A", "VAN-B",
    "LimA", "LimB",
    "ConA", "ConB", "ConC",
    "DMN-A", "DMN-B", "DMN-C",
    "TPar",
]


def _yeo_default_labels(yeo_n: int) -> List[str]:
    if yeo_n == 7:
        return list(_YEO7_LABELS)
    if yeo_n == 17:
        return list(_YEO17_LABELS)
    raise ValueError(f"yeo_n must be 7 or 17, got {yeo_n}")


def _yeo_default_abbr(yeo_n: int) -> List[str]:
    if yeo_n == 7:
        return list(_YEO7_ABBR)
    if yeo_n == 17:
        return list(_YEO17_ABBR)
    raise ValueError(f"yeo_n must be 7 or 17, got {yeo_n}")


# ---------- Schaefer / Glasser helpers ----------

def _load_label_gii(path: str):
    g = nib.load(path)
    labs = np.asarray(g.agg_data(), dtype=int).squeeze()
    lt = g.labeltable
    key_to_name = {lab.key: lab.label for lab in lt.labels}
    return labs, key_to_name


def _schaefer7_from_name(name: str) -> int:
    """Map a Schaefer 7Networks label name to a 0-based Yeo-7 index."""
    n = name.lower()
    if "vis" in n:
        return 0
    if "som" in n or "sommot" in n:
        return 1
    if "dorsattn" in n or ("dors" in n and "attn" in n):
        return 2
    if ("ventattn" in n or "salventattn" in n or
            ("vent" in n and "attn" in n) or "sal" in n):
        return 3
    if "limbic" in n:
        return 4
    if ("cont" in n or "control" in n or
            "frontoparietal" in n or "fp" in n):
        return 5
    if "default" in n:
        return 6
    raise ValueError(f"Unrecognized Schaefer-7 network in label name: {name}")


def _schaefer17_from_name(name: str) -> int:
    """
    Map a Schaefer 17Networks label name (e.g.
    '17Networks_LH_VisCent_ExStr_1', '17Networks_RH_DefaultA_IPL_1') to a
    0-based Yeo-17 index in the canonical order:

        0:  VisCent (VisualA)
        1:  VisPeri (VisualB)
        2:  SomMotA
        3:  SomMotB
        4:  DorsAttnA
        5:  DorsAttnB
        6:  SalVentAttnA
        7:  SalVentAttnB
        8:  LimbicA  (TempPole)
        9:  LimbicB  (OFC)
        10: ContA
        11: ContB
        12: ContC
        13: DefaultA
        14: DefaultB
        15: DefaultC
        16: TempPar
    """
    n = name.lower()

    # Order matters: more specific tokens before their substrings.
    if "viscent" in n:
        return 0
    if "visperi" in n:
        return 1
    if "sommota" in n or "sommot_a" in n:
        return 2
    if "sommotb" in n or "sommot_b" in n:
        return 3
    if "dorsattna" in n or "dorsattn_a" in n:
        return 4
    if "dorsattnb" in n or "dorsattn_b" in n:
        return 5
    if "salventattna" in n or "salventattn_a" in n:
        return 6
    if "salventattnb" in n or "salventattn_b" in n:
        return 7
    if "limbica" in n or "limbic_a" in n or "temppole" in n:
        return 8
    if "limbicb" in n or "limbic_b" in n or "limbic_ofc" in n:
        return 9
    if "conta" in n or "cont_a" in n:
        return 10
    if "contb" in n or "cont_b" in n:
        return 11
    if "contc" in n or "cont_c" in n:
        return 12
    if "defaulta" in n or "default_a" in n:
        return 13
    if "defaultb" in n or "default_b" in n:
        return 14
    if "defaultc" in n or "default_c" in n:
        return 15
    if "temppar" in n:
        return 16

    raise ValueError(f"Unrecognized Schaefer-17 network in label name: {name}")


def _glasser7_from_name(name: str) -> int:
    """Map a Glasser/HCP-MMP1.0 Yeo-7 label name to a 0-based Yeo-7 index."""
    n = name.lower().strip()

    if "visual" in n:
        return 0
    if "somato" in n or "somatomotor" in n or "motor" in n:
        return 1
    if "dorsal attention" in n or ("dorsal" in n and "attention" in n) or "dorsattn" in n:
        return 2
    if ("ventral attention" in n or ("ventral" in n and "attention" in n)
            or "salience" in n or "salventattn" in n):
        return 3
    if "limbic" in n:
        return 4
    if "frontoparietal" in n or "control" in n:
        return 5
    if "default" in n:
        return 6

    raise ValueError(f"Unrecognized Glasser-7/Yeo7 network in label name: {name}")


def _glasser17_from_name(name: str) -> int:
    """
    Map a Glasser/HCP-MMP1.0 Yeo-17 label name to a 0-based Yeo-17 index.

    Accepts either the bare Yeo-17 network name or the full line from the
    cortex_parcel_network_assignments_Yeo17.txt file. The network token is
    expected as the trailing _-separated component (or anywhere in the name)
    and is matched case-insensitively against the canonical Yeo-17 names.
    """
    n = name.lower().strip()

    if "viscent" in n or "visualcent" in n or "visual_a" in n or "visa" in n:
        return 0
    if "visperi" in n or "visualperi" in n or "visual_b" in n or "visb" in n:
        return 1
    if "sommota" in n or "somatomotor_a" in n or "sommot_a" in n or "motor_a" in n:
        return 2
    if "sommotb" in n or "somatomotor_b" in n or "sommot_b" in n or "motor_b" in n:
        return 3
    if "dorsattna" in n or "dorsattn_a" in n or "dorsalattn_a" in n:
        return 4
    if "dorsattnb" in n or "dorsattn_b" in n or "dorsalattn_b" in n:
        return 5
    if "salventattna" in n or "salventattn_a" in n or "salience_a" in n:
        return 6
    if "salventattnb" in n or "salventattn_b" in n or "salience_b" in n:
        return 7
    if "limbica" in n or "limbic_a" in n or "temppole" in n:
        return 8
    if "limbicb" in n or "limbic_b" in n or "limbic_ofc" in n or " ofc" in n:
        return 9
    if "conta" in n or "cont_a" in n or "control_a" in n:
        return 10
    if "contb" in n or "cont_b" in n or "control_b" in n:
        return 11
    if "contc" in n or "cont_c" in n or "control_c" in n:
        return 12
    if "defaulta" in n or "default_a" in n:
        return 13
    if "defaultb" in n or "default_b" in n:
        return 14
    if "defaultc" in n or "default_c" in n:
        return 15
    if "temppar" in n or "temporoparietal" in n or "tempparoccip" in n:
        return 16

    raise ValueError(f"Unrecognized Glasser-17/Yeo17 network in label name: {name}")


def _net_from_name(name: str, yeo_n: int, atlas_kind: str) -> int:
    """Dispatch helper: name -> 0-based Yeo network index."""
    if atlas_kind == "schaefer":
        if yeo_n == 7:
            return _schaefer7_from_name(name)
        if yeo_n == 17:
            return _schaefer17_from_name(name)
    elif atlas_kind == "glasser":
        if yeo_n == 7:
            return _glasser7_from_name(name)
        if yeo_n == 17:
            return _glasser17_from_name(name)
    raise ValueError(f"Bad atlas_kind/yeo_n: {atlas_kind}/{yeo_n}")


def _load_glasser_yeo7_assignments(
        path: str = "cortex_parcel_network_assignments_Yeo7.txt"
) -> np.ndarray:
    """
    Load Yeo-7 assignments for the 360 Glasser/HCP-MMP parcels from a file
    where each line is e.g.

        1_R_V1_ROI_1_Visual
        10_R_FEF_ROI_3_Dorsal Attention
        ...

    The trailing "_"-token is the network name.
    """
    networks: list[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("_")
            net_label = parts[-1]
            net_idx = _glasser7_from_name(net_label)
            networks.append(net_idx)

    networks0 = np.asarray(networks, dtype=int)
    if networks0.min() < 0 or networks0.max() > 6:
        raise ValueError(
            f"Yeo-7 assignments in {path} must be in [0..6] after mapping, "
            f"found range [{networks0.min()}, {networks0.max()}]."
        )
    return networks0


def _load_glasser_yeo17_assignments(
        path: str = "cortex_parcel_network_assignments_Yeo17.txt"
) -> np.ndarray:
    """
    Load Yeo-17 assignments for the 360 Glasser/HCP-MMP parcels from a file
    with the same layout as the Yeo-7 version, e.g.

        1_R_V1_ROI_1_VisCent
        10_R_FEF_ROI_3_DorsAttnB
        ...

    The trailing "_"-token is the Yeo-17 network name. See top-of-file
    notes for how to generate this file from the Yeo-17 partition.

    Returns
    -------
    networks0 : (N,) int array
        0-based Yeo-17 network index per parcel (1..360 for Glasser).
    """
    networks: list[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("_")
            net_label = parts[-1]
            net_idx = _glasser17_from_name(net_label)
            networks.append(net_idx)

    networks0 = np.asarray(networks, dtype=int)
    if networks0.min() < 0 or networks0.max() > 16:
        raise ValueError(
            f"Yeo-17 assignments in {path} must be in [0..16] after mapping, "
            f"found range [{networks0.min()}, {networks0.max()}]."
        )
    return networks0


def _get_yeo_assignments(
        atlas: str,
        yeo_n: int = 7,
        schaefer_label_L: Optional[str] = None,
        schaefer_label_R: Optional[str] = None,
        glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
        glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
) -> Tuple[np.ndarray, List[str]]:
    """
    Unified helper: returns (networks0, default_labels) for the requested
    atlas + Yeo partition.

    For Schaefer this reads Yeo membership directly from the .label.gii
    label names, so make sure the *.label.gii files you pass come from the
    matching Schaefer 7Networks vs 17Networks parcellation:

        yeo_n=7  -> Schaefer2018_400Parcels_7Networks_order
        yeo_n=17 -> Schaefer2018_400Parcels_17Networks_order

    For Glasser, this reads from the corresponding text file:

        yeo_n=7  -> cortex_parcel_network_assignments_Yeo7.txt
        yeo_n=17 -> cortex_parcel_network_assignments_Yeo17.txt
    """
    if yeo_n not in (7, 17):
        raise ValueError(f"yeo_n must be 7 or 17, got {yeo_n}")

    atlas_lower = atlas.lower()
    labels = _yeo_default_labels(yeo_n)

    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

        nets0 = []
        for k in uL:
            nets0.append(_net_from_name(L_map[k], yeo_n, "schaefer"))
        for k in uR:
            nets0.append(_net_from_name(R_map[k], yeo_n, "schaefer"))
        networks0 = np.asarray(nets0, dtype=int)

    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        if yeo_n == 7:
            networks0 = _load_glasser_yeo7_assignments(glasser_yeo7_path)
        else:
            networks0 = _load_glasser_yeo17_assignments(glasser_yeo17_path)
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

    return networks0, labels


def _network_colours(yeo_n: int, network_cmap: str = "tab20") -> List:
    """
    Return a list of colours, one per Yeo network.

    For Yeo-7 we use a 7-shade selection from `network_cmap` and reorder
    them so that adjacent networks aren't visually similar (matches the
    original behaviour of the file). For Yeo-17 we use 17 distinct shades
    from `network_cmap` (typically tab20).
    """
    try:
        seq_cmap = plt.get_cmap(network_cmap)
    except Exception:
        seq_cmap = plt.get_cmap("tab20")

    if yeo_n == 7:
        shades = [seq_cmap(x_) for x_ in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]
        net_colours = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours[net_idx] = shades[rank]
        return net_colours

    if yeo_n == 17:
        # Prefer discrete tab20-style colormaps when possible.
        if network_cmap in ("tab20", "tab20b", "tab20c"):
            return [seq_cmap(i % 20) for i in range(17)]
        return [seq_cmap(i / 16.0) for i in range(17)]

    raise ValueError(f"yeo_n must be 7 or 17, got {yeo_n}")


def _legend_order(yeo_n: int, present: set) -> List[int]:
    """Return a nice legend ordering for the present Yeo networks."""
    if yeo_n == 7:
        order = [1, 0, 3, 2, 6, 5, 4]
    else:
        order = list(range(17))
    return [i for i in order if i in present]


# ---------- 2D scatter with regression & marginals ----------

def plot_scatter_with_global_correlation(
        eigvecs: np.ndarray,
        out_dir,
        name: str = "Scatter2D",
        eigvecs_to_plot: Tuple[int, int] = (0, 1),
        layer_labels: Optional[List[str]] = None,
        network_labels: Optional[List[str]] = None,
        x_label: str = "Emb1",
        y_label: str = "Emb2",
        fname: Optional[str] = None,
        network_cmap: str = "tab20",
        dot_size: int = 40,
        atlas: str = "schaefer",
        yeo_n: int = 7,
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.R.label.gii",
        glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
        glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
        show_marginal_hists: bool = True,
        hist_bins: int = 30,
        hist_size: float = 0.1,
        hist_pad: float = 0.02,
        hist_alpha: float = 0.6,
        show_ci_band: bool = True,
        ci_level: float = 0.95,
        band_kind: str = "confidence",
        ci_band_alpha: float = 0.2,
        return_stats: bool = False,
):
    """
    2D scatter with regression line and global Pearson r, coloured by Yeo RSNs.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    networks0, default_labels = _get_yeo_assignments(
        atlas=atlas, yeo_n=yeo_n,
        schaefer_label_L=schaefer_label_L, schaefer_label_R=schaefer_label_R,
        glasser_yeo7_path=glasser_yeo7_path, glasser_yeo17_path=glasser_yeo17_path,
    )
    N = networks0.size
    if network_labels is None:
        network_labels = default_labels

    # ----- rows vs layers -----
    nrows, ndims = eigvecs.shape
    if nrows == 3 * N:
        mode = "multilayer"
    elif nrows == N:
        mode = "single"
    elif nrows % 3 == 0 and (nrows // 3) == N:
        mode = "multilayer"
    else:
        raise ValueError(f"eigvecs has {nrows} rows, atlas implies N={N} or 3N={3*N}.")

    x_dim, y_dim = eigvecs_to_plot
    if x_dim >= ndims or y_dim >= ndims:
        x_dim, y_dim = x_dim - 1, y_dim - 1
    if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
        raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

    if mode == "multilayer":
        nets = np.tile(networks0, 3)
        layers = np.repeat([0, 1, 2], N)
        shapes = ["o", "s", "^"]
        if layer_labels is None:
            layer_labels = ["Deep", "Middle", "Superficial"]
    else:
        nets = networks0
        layers = np.zeros(N, dtype=int)
        shapes = ["o"]
        if layer_labels is None:
            layer_labels = ["AcrossLayers"]

    net_colours = _network_colours(yeo_n, network_cmap)

    # ----- layout -----
    if show_marginal_hists:
        fig = plt.figure(figsize=(7.5, 7.5))
        left, bottom = 0.10, 0.10
        width, height = 0.64, 0.64
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + hist_pad, width, hist_size]
        rect_histy = [left + width + hist_pad, bottom, hist_size, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax_histx = ax_histy = None

    # ----- scatter -----
    for lyr in np.unique(layers):
        for net in np.unique(nets):
            m = (layers == lyr) & (nets == net)
            if not m.any():
                continue
            ax.scatter(
                eigvecs[m, x_dim], eigvecs[m, y_dim],
                s=dot_size,
                marker=shapes[int(lyr if mode == "multilayer" else 0)],
                facecolor=net_colours[int(net)],
                edgecolor="k", linewidths=0.25, alpha=0.8
            )

    # ----- regression & stats -----
    x = eigvecs[:, x_dim].astype(float)
    y = eigvecs[:, y_dim].astype(float)
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for regression.")

    r, p = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat

    x_bar = x.mean()
    Sxx = np.sum((x - x_bar) ** 2)
    if Sxx <= 0:
        raise ValueError("All x-values identical; cannot fit line.")
    df = n - 2
    sigma_hat = np.sqrt(np.sum(resid ** 2) / df)
    tcrit = t_dist.ppf(1 - (1 - ci_level) / 2, df)

    slope_se = sigma_hat / np.sqrt(Sxx)
    intercept_se = sigma_hat * np.sqrt(1 / n + (x_bar ** 2) / Sxx)
    slope_ci = (slope - tcrit * slope_se, slope + tcrit * slope_se)
    intercept_ci = (intercept - tcrit * intercept_se, intercept + tcrit * intercept_se)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Embedding coloured by Yeo-{yeo_n} RSN" +
                 (" (layers as shapes)" if mode == "multilayer" else ""))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
    line_y = slope * xs + intercept
    ax.plot(xs, line_y, color="k", ls="--", lw=1, zorder=5)

    if show_ci_band:
        mult = 1.0 if band_kind.lower().startswith("pred") else 0.0
        se_line = sigma_hat * np.sqrt(mult + (1 / n) + ((xs - x_bar) ** 2) / Sxx)
        upper = line_y + tcrit * se_line
        lower = line_y - tcrit * se_line
        ax.fill_between(xs, lower, upper, alpha=ci_band_alpha,
                        edgecolor="none", facecolor="gray", zorder=4)

    txt = (
        f"r = {r:.3f}\np = {p:.3g}\n"
        f"slope = {slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]\n"
        f"intercept = {intercept:.3f} [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]"
    )
    bbox_props = dict(boxstyle="round,pad=0.25", fc="w", ec="k", lw=0.4)
    ax.text(-0.05, 1.06, txt, ha="right", va="bottom",
            transform=ax.transAxes, fontsize=5, bbox=bbox_props)

    # ----- marginals -----
    if show_marginal_hists:
        ax_histx.hist(x, bins=hist_bins, range=(0, 1), edgecolor="k", alpha=hist_alpha)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", labelleft=False)
        for spine in ("right", "top"):
            ax_histx.spines[spine].set_visible(False)

        ax_histy.hist(y, bins=hist_bins, range=(0, 1),
                      orientation="horizontal", edgecolor="k", alpha=hist_alpha)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelbottom=False)
        for spine in ("right", "top"):
            ax_histy.spines[spine].set_visible(False)

    # ----- legends -----
    n_nets_present = int(nets.max()) + 1
    if len(network_labels) < n_nets_present:
        network_labels = list(network_labels) + [
            f"Net{i}" for i in range(len(network_labels), n_nets_present)
        ]

    net_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=net_colours[int(i)],
               markeredgecolor="k",
               markersize=8,
               label=network_labels[int(i)])
        for i in np.unique(nets)
    ]
    ax.legend(handles=net_handles, title="RSN",
              bbox_to_anchor=(1.32, 1), loc="upper left")

    if mode == "multilayer":
        lyr_handles = [
            Line2D([0], [0], marker=shapes[i], color="w",
                   markeredgecolor="k", markersize=8,
                   label=layer_labels[i])
            for i in np.unique(layers)
        ]
        ax.add_artist(ax.legend(handles=lyr_handles, title="Layer", loc="upper right"))

    if fname is None:
        fname = f"ScatterCorr_yeo{yeo_n}_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}.png"
    fig.tight_layout()
    outpath = out_dir / fname
    fig.savefig(outpath, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)

    if return_stats:
        return {
            "n": int(n),
            "df": int(df),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "slope": float(slope),
            "intercept": float(intercept),
            "sigma_hat": float(sigma_hat),
            "slope_se": float(slope_se),
            "intercept_se": float(intercept_se),
            "slope_ci": tuple(map(float, slope_ci)),
            "intercept_ci": tuple(map(float, intercept_ci)),
            "ci_level": float(ci_level),
            "band_kind": band_kind.lower(),
        }


# ---------- 3D scatter with plane & marginals ----------

def plot_scatter3D_with_plane(
        X: np.ndarray,
        out_dir,
        name: str = "Scatter3D",
        Y: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        dims_to_plot: Tuple[int, int, int] = (0, 1, 2),
        layer_labels: Optional[List[str]] = None,
        network_labels: Optional[List[str]] = None,
        x_label: str = "Emb1",
        y_label: str = "Emb2",
        z_label: str = "Emb3",
        fname: Optional[str] = None,
        network_cmap: str = "tab20",
        dot_size: int = 30,
        show_plane: bool = False,
        equalize_axes: bool = True,
        cube_pad: float = 0.06,
        proj_type: str = "ortho",
        plane_alpha: float = 0.18,
        atlas: str = "schaefer",
        yeo_n: int = 7,
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.R.label.gii",
        glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
        glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
        show_marginals: bool = True,
        hist_bins: int = 20,
):
    """
    3D embedding coloured by Yeo RSNs, optional best-fit plane and marginals.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    networks0, default_labels = _get_yeo_assignments(
        atlas=atlas, yeo_n=yeo_n,
        schaefer_label_L=schaefer_label_L, schaefer_label_R=schaefer_label_R,
        glasser_yeo7_path=glasser_yeo7_path, glasser_yeo17_path=glasser_yeo17_path,
    )
    N = networks0.size
    if network_labels is None:
        network_labels = default_labels

    def _from_matrix(M, dims):
        nrows, ndims = M.shape
        i, j, k = dims
        if i >= ndims or j >= ndims or k >= ndims:
            i, j, k = i - 1, j - 1, k - 1
        if not (0 <= i < ndims and 0 <= j < ndims and 0 <= k < ndims):
            raise ValueError(f"dims_to_plot {dims} not in [0..{ndims-1}]")
        return M[:, i].astype(float), M[:, j].astype(float), M[:, k].astype(float), nrows

    if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
        x, y, z, nrows = _from_matrix(X, dims_to_plot)
    else:
        if Y is None or Z is None:
            raise ValueError("Provide either matrix X and dims_to_plot, or explicit X,Y,Z.")
        x = np.asarray(X, float).squeeze()
        y = np.asarray(Y, float).squeeze()
        z = np.asarray(Z, float).squeeze()
        nrows = x.size

    if nrows == 3 * N:
        mode = "multilayer"
    elif nrows == N:
        mode = "single"
    elif nrows % 3 == 0 and nrows // 3 == N:
        mode = "multilayer"
    else:
        raise ValueError(f"Data has {nrows} rows, but atlas implies N={N} or 3N={3*N}.")

    if mode == "multilayer":
        nets = np.tile(networks0, 3)
        layers = np.repeat([0, 1, 2], N)
        shapes = ["o", "s", "^"]
        if layer_labels is None:
            layer_labels = ["Deep", "Middle", "Superficial"]
    else:
        nets = networks0
        layers = np.zeros(N, int)
        shapes = ["o"]
        if layer_labels is None:
            layer_labels = ["AcrossLayers"]

    net_colours = _network_colours(yeo_n, network_cmap)

    fig = plt.figure(figsize=(8.8, 8.0))

    if show_marginals:
        ax = fig.add_axes([0.08, 0.08, 0.6, 0.7], projection="3d")
        ax_histx = fig.add_axes([0.08, 0.80, 0.6, 0.16])
        ax_histy = fig.add_axes([0.70, 0.08, 0.16, 0.7])
        ax_histz = fig.add_axes([0.70, 0.80, 0.16, 0.16])
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax_histx = ax_histy = ax_histz = None

    try:
        ax.set_proj_type(proj_type)
    except Exception:
        pass

    # ----- scatter -----
    for lyr in np.unique(layers):
        for net in np.unique(nets):
            m = (layers == lyr) & (nets == net)
            if not np.any(m):
                continue
            ax.scatter(x[m], y[m], z[m],
                       s=dot_size,
                       marker=shapes[int(lyr if mode == "multilayer" else 0)],
                       facecolor=net_colours[int(net)],
                       edgecolor="k", linewidths=0.25, alpha=0.9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(f"3D Embedding coloured by Yeo-{yeo_n} RSN" +
                 (" (layers as shapes)" if mode == "multilayer" else ""))

    def _safe_range(lo, hi):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return -1.0, 1.0
        if hi == lo:
            return lo - 0.5, hi + 0.5
        return lo, hi

    xlo, xhi = _safe_range(np.nanmin(x), np.nanmax(x))
    ylo, yhi = _safe_range(np.nanmin(y), np.nanmax(y))
    zlo, zhi = _safe_range(np.nanmin(z), np.nanmax(z))

    if equalize_axes:
        xm, ym, zm = (xlo + xhi)/2.0, (ylo + yhi)/2.0, (zlo + zhi)/2.0
        r = max(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
        r *= (1.0 + float(cube_pad))
        xl, yl, zl = xm - r, ym - r, zm - r
        xh, yh, zh = xm + r, ym + r, zm + r
    else:
        xl, xh = xlo, xhi
        yl, yh = ylo, yhi
        zl, zh = zlo, zhi

    r_xy, p_xy = pearsonr(x, y)
    r_xz, p_xz = pearsonr(x, z)
    r_yz, p_yz = pearsonr(y, z)
    r2 = np.nan

    # ----- plane fit -----
    if show_plane:
        Xmat = np.column_stack([x, y, np.ones_like(x)])
        a, b, c = np.linalg.lstsq(Xmat, z, rcond=None)[0]
        z_hat = a * x + b * y + c
        ss_res = np.sum((z - z_hat) ** 2)
        ss_tot = np.sum((z - np.mean(z)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        xx = np.linspace(xl, xh, 45)
        yy = np.linspace(yl, yh, 45)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = a * XX + b * YY + c
        ax.plot_surface(XX, YY, ZZ, alpha=plane_alpha, linewidth=0, antialiased=True)

    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_zlim(zl, zh)
    if equalize_axes:
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    ax.view_init(elev=22, azim=38)

    # ----- marginals -----
    if show_marginals:
        bins_x = np.linspace(xl, xh, hist_bins + 1)
        bins_y = np.linspace(yl, yh, hist_bins + 1)
        bins_z = np.linspace(zl, zh, hist_bins + 1)
        uniq_nets = np.unique(nets)

        for net in uniq_nets:
            m = (nets == net)
            ax_histx.hist(x[m], bins=bins_x,
                          color=net_colours[int(net)], alpha=0.4)
        ax_histx.set_xlim(xl, xh)
        ax_histx.set_xticklabels([])
        ax_histx.set_ylabel("count", fontsize=7)
        ax_histx.tick_params(axis="y", labelsize=7)

        for net in uniq_nets:
            m = (nets == net)
            ax_histy.hist(y[m], bins=bins_y,
                          orientation="horizontal",
                          color=net_colours[int(net)], alpha=0.4)
        ax_histy.set_ylim(yl, yh)
        ax_histy.set_yticklabels([])
        ax_histy.set_xlabel("count", fontsize=7)
        ax_histy.tick_params(axis="x", labelsize=7)

        for net in uniq_nets:
            m = (nets == net)
            ax_histz.hist(z[m], bins=bins_z,
                          color=net_colours[int(net)], alpha=0.4)
        ax_histz.set_xlabel(z_label, fontsize=7)
        ax_histz.set_ylabel("count", fontsize=7)
        ax_histz.tick_params(axis="both", labelsize=7)

    stats_txt = (f"r_xy={r_xy:.3f} (p={p_xy:.2g})\n"
                 f"r_xz={r_xz:.3f} (p={p_xz:.2g})\n"
                 f"r_yz={r_yz:.3f} (p={p_yz:.2g})")
    if show_plane and np.isfinite(r2):
        stats_txt += f"\nPlane $R^2$={r2:.3f}"
    bbox_props = dict(boxstyle="round,pad=0.25", fc="w", ec="k", lw=0.4)
    ax.text2D(0.02, 0.98, stats_txt, transform=ax.transAxes, ha="left", va="top",
              fontsize=7, bbox=bbox_props)

    # ----- legends -----
    present = set(int(i) for i in np.unique(nets))
    legend_order = _legend_order(yeo_n, present)

    net_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=net_colours[i], markeredgecolor="k",
                          markersize=8, label=network_labels[i])
                   for i in legend_order]
    leg1 = ax.legend(handles=net_handles, title="RSN",
                     bbox_to_anchor=(1.32, 1), loc="upper left")
    ax.add_artist(leg1)

    if mode == "multilayer":
        lyr_handles = [Line2D([0], [0], marker=shapes[i], color="w",
                              markeredgecolor="k", markersize=8,
                              label=layer_labels[i])
                       for i in np.unique(layers)]
        ax.legend(handles=lyr_handles, title="Layer", loc="upper right")

    if fname is None:
        if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
            i, j, k = dims_to_plot
            fname = f"Scatter3D_yeo{yeo_n}_d{i+1}{j+1}{k+1}_{'multi' if mode=='multilayer' else 'single'}.png"
        else:
            fname = f"Scatter3D_yeo{yeo_n}_{'multi' if mode=='multilayer' else 'single'}.png"

    if not show_marginals:
        fig.tight_layout()
    outpath = out_dir / fname
    fig.savefig(outpath, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)



# ---------- 2D centroids (RSN×layer) ----------

def plot_scatter_centroids(
        eigvecs: np.ndarray,
        out_dir,
        name: str = "Scatter2D_NetCentroids",
        eigvecs_to_plot: Tuple[int, int] = (0, 1),
        layer_labels: Optional[List[str]] = None,
        network_labels: Optional[List[str]] = None,
        x_label: str = "Emb1",
        y_label: str = "Emb2",
        fname: Optional[str] = None,
        network_cmap: str = "tab20",
        dot_size: int = 60,
        annotate: bool = False,
        atlas: str = "schaefer",
        yeo_n: int = 7,
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.R.label.gii",
        glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
        glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
):
    """
    2D centroids (RSN × layer) for Yeo-7 / Yeo-17 networks.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    networks0, default_labels = _get_yeo_assignments(
        atlas=atlas, yeo_n=yeo_n,
        schaefer_label_L=schaefer_label_L, schaefer_label_R=schaefer_label_R,
        glasser_yeo7_path=glasser_yeo7_path, glasser_yeo17_path=glasser_yeo17_path,
    )
    N = networks0.size
    if network_labels is None:
        network_labels = default_labels

    nrows, ndims = eigvecs.shape
    if nrows == 3 * N:
        mode = "multilayer"
    elif nrows == N:
        mode = "single"
    elif nrows % 3 == 0 and (nrows // 3) == N:
        mode = "multilayer"
    else:
        raise ValueError(f"eigvecs has {nrows} rows, atlas implies N={N} or 3N.")

    x_dim, y_dim = eigvecs_to_plot
    if x_dim >= ndims or y_dim >= ndims:
        x_dim, y_dim = x_dim - 1, y_dim - 1
    if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
        raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

    if mode == "multilayer":
        nets = np.tile(networks0, 3)
        layers = np.repeat([0, 1, 2], N)
        shapes = ["o", "s", "^"]
        if layer_labels is None:
            layer_labels = ["Deep", "Middle", "Superficial"]
    else:
        nets = networks0
        layers = np.zeros(N, dtype=int)
        shapes = ["o"]
        if layer_labels is None:
            layer_labels = ["AcrossLayers"]

    net_colours = _network_colours(yeo_n, network_cmap)

    uniq_layers = np.unique(layers)
    uniq_nets = np.unique(nets)
    centroids = []

    for lyr in uniq_layers:
        for net in uniq_nets:
            m = (layers == lyr) & (nets == net)
            if not np.any(m):
                continue
            xy = eigvecs[m][:, [x_dim, y_dim]]
            if xy.size == 0:
                continue
            centroids.append((int(lyr), int(net),
                              float(np.mean(xy[:, 0])),
                              float(np.mean(xy[:, 1])),
                              int(xy.shape[0])))

    stem = Path(fname).stem if fname else f"centroids_yeo{yeo_n}"
    pd.DataFrame(centroids, columns=["layer", "network", "x_centroid", "y_centroid", "n_parcels"])\
    .to_csv(out_dir / f"{stem}_centroids.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    xm = ym = 0.0
    for lyr, net, xm, ym, cnt in centroids:
        ax.scatter(xm, ym,
                   s=dot_size,
                   marker=shapes[int(lyr if mode == "multilayer" else 0)],
                   facecolor=net_colours[int(net)],
                   edgecolor="k", linewidths=0.6, alpha=0.95)
        if annotate:
            ax.text(xm, ym, f" {network_labels[net]}",
                    va="center", ha="left", fontsize=7, alpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(alpha=0.2, linestyle=":")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Centroids by Yeo-{yeo_n} RSN" +
                 (" (layers as shapes)" if mode == "multilayer" else ""))

    net_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=net_colours[i], markeredgecolor="k",
                          markersize=8, label=network_labels[i])
                   for i in np.unique(nets)]
    ax.legend(handles=net_handles, title="RSN",
              bbox_to_anchor=(1.32, 1), loc="upper left")

    if mode == "multilayer":
        lyr_handles = [Line2D([0], [0], marker=shapes[i], color="w",
                              markeredgecolor="k", markersize=8,
                              label=layer_labels[i])
                       for i in np.unique(layers)]
        ax.add_artist(ax.legend(handles=lyr_handles, title="Layer", loc="upper right"))

    if fname is None:
        fname = f"ScatterCentroids_yeo{yeo_n}_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}_unitaxes.png"
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    return xm, ym


# ---------- Network-centroid 3D cycle ----------

def plot_network_centroids3D(
        X: np.ndarray,
        out_dir,
        name: str = "Scatter3D_NetCentroids",
        Y: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        dims_to_plot: Tuple[int, int, int] = (0, 1, 2),
        x_label: str = "Emb1",
        y_label: str = "Emb2",
        z_label: str = "Emb3",
        network_labels: Optional[List[str]] = None,
        fname: Optional[str] = None,
        network_cmap: str = "tab20",
        centroid_size: int = 200,
        line_alpha: float = 0.6,
        equalize_axes: bool = True,
        cube_pad: float = 0.08,
        proj_type: str = "ortho",
        annotate: bool = True,
        write_coords: bool = True,
        coords_fname: Optional[str] = None,
        print_coords: bool = True,
        return_coords: bool = True,
        atlas: str = "schaefer",
        yeo_n: int = 7,
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.R.label.gii",
        glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
        glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
):
    """
    3D Yeo network centroids (single point per RSN), joined by a non-crossing cycle.
    """
    import csv

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    networks0, default_labels = _get_yeo_assignments(
        atlas=atlas, yeo_n=yeo_n,
        schaefer_label_L=schaefer_label_L, schaefer_label_R=schaefer_label_R,
        glasser_yeo7_path=glasser_yeo7_path, glasser_yeo17_path=glasser_yeo17_path,
    )
    N = networks0.size
    if network_labels is None:
        network_labels = default_labels
    net_abbr = _yeo_default_abbr(yeo_n)

    def _from_matrix(M, dims):
        nrows, ndims = M.shape
        i, j, k = dims
        if i >= ndims or j >= ndims or k >= ndims:
            i, j, k = i - 1, j - 1, k - 1
        if not (0 <= i < ndims and 0 <= j < ndims and 0 <= k < ndims):
            raise ValueError(f"dims_to_plot {dims} not in [0..{ndims-1}]")
        return M[:, i].astype(float), M[:, j].astype(float), M[:, k].astype(float), nrows

    if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
        x, y, z, nrows = _from_matrix(X, dims_to_plot)
    else:
        if Y is None or Z is None:
            raise ValueError("Provide matrix X and dims_to_plot, or explicit X,Y,Z.")
        x = np.asarray(X, float).squeeze()
        y = np.asarray(Y, float).squeeze()
        z = np.asarray(Z, float).squeeze()
        nrows = x.size

    if nrows == 3 * N:
        mode = "multilayer"
    elif nrows == N:
        mode = "single"
    elif nrows % 3 == 0 and nrows // 3 == N:
        mode = "multilayer"
    else:
        raise ValueError(f"Data has {nrows} rows, but atlas implies N={N} or 3N={3*N}.")

    if mode == "multilayer":
        x_par = (x[0:N] + x[N:2*N] + x[2*N:3*N]) / 3.0
        y_par = (y[0:N] + y[N:2*N] + y[2*N:3*N]) / 3.0
        z_par = (z[0:N] + z[N:2*N] + z[2*N:3*N]) / 3.0
    else:
        x_par, y_par, z_par = x, y, z

    kvals = np.array(sorted(np.unique(networks0)))
    K = len(kvals)
    cx = np.zeros(K)
    cy = np.zeros(K)
    cz = np.zeros(K)
    n_parcels = np.zeros(K, dtype=int)
    labels_out = []
    for idx, k in enumerate(kvals):
        sel = (networks0 == k)
        n_parcels[idx] = int(sel.sum())
        if n_parcels[idx] == 0:
            cx[idx] = cy[idx] = cz[idx] = np.nan
        else:
            cx[idx] = float(np.nanmean(x_par[sel]))
            cy[idx] = float(np.nanmean(y_par[sel]))
            cz[idx] = float(np.nanmean(z_par[sel]))
        labels_out.append(network_labels[int(k)] if 0 <= k < len(network_labels) else f"Net {int(k)}")

    finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
    idx_map = np.where(finite)[0]
    if idx_map.size < 3:
        raise ValueError("Fewer than 3 valid centroids — cannot form cycle.")
    CX, CY, CZ = cx[finite], cy[finite], cz[finite]
    Kf = idx_map.size

    # ----- project to 2D for non-crossing test -----
    P = np.vstack([CX, CY, CZ]).T
    P -= P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    B = U[:, :2] * S[:2]

    def _pairwise_dists_3d(Q):
        d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
        return d

    D3 = _pairwise_dists_3d(P)

    def _intersect_2d(a, b, c, d, eps=1e-12):
        def orient(p, q, r):
            return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

        def on_seg(p, q, r):
            return (min(p[0], r[0])-eps <= q[0] <= max(p[0], r[0])+eps and
                    min(p[1], r[1])-eps <= q[1] <= max(p[1], r[1])+eps)

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        if (o1*o2 < 0) and (o3*o4 < 0):
            return True
        if abs(o1) < eps and on_seg(a, c, b):
            return True
        if abs(o2) < eps and on_seg(a, d, b):
            return True
        if abs(o3) < eps and on_seg(c, a, d):
            return True
        if abs(o4) < eps and on_seg(c, b, d):
            return True
        return False

    parent = list(range(Kf))
    rank = [0]*Kf

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    cand = []
    for i in range(Kf):
        for j in range(i+1, Kf):
            cand.append((D3[i, j], i, j))
    cand.sort(key=lambda t: t[0])

    deg = np.zeros(Kf, dtype=int)
    edges: list[Tuple[int, int]] = []

    def _crosses_any(i, j):
        p1, p2 = B[i], B[j]
        for (a, b) in edges:
            if len({i, j, a, b}) < 4:
                continue
            if _intersect_2d(p1, p2, B[a], B[b]):
                return True
        return False

    for w, i, j in cand:
        if deg[i] == 2 or deg[j] == 2:
            continue
        if _crosses_any(i, j):
            continue
        same_comp = (find(i) == find(j))
        if same_comp and len(edges) != Kf - 1:
            continue
        if same_comp and _crosses_any(i, j):
            continue
        edges.append((i, j))
        union(i, j)
        deg[i] += 1
        deg[j] += 1
        if len(edges) == Kf:
            break

    if not (len(edges) == Kf and np.all(deg == 2)):
        edges = []
        deg[:] = 0
        ctr = B.mean(axis=0)
        ang = np.arctan2(B[:, 1]-ctr[1], B[:, 0]-ctr[0])
        order = np.argsort(ang)
        for u, v in zip(order, np.roll(order, -1)):
            edges.append((u, v))
        deg += 2

    # ----- plotting -----
    fig = plt.figure(figsize=(8.4, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_proj_type(proj_type)
    except Exception:
        pass

    net_colours = _network_colours(yeo_n, network_cmap)

    for local_idx, g_idx in enumerate(idx_map):
        net_idx = int(kvals[g_idx])
        if 0 <= net_idx < len(net_colours):
            col = net_colours[net_idx]
        else:
            col = net_colours[net_idx % len(net_colours)]
        ax.scatter(CX[local_idx], CY[local_idx], CZ[local_idx],
                   s=centroid_size, marker="o",
                   facecolor=col, edgecolor="k",
                   linewidths=0.6, alpha=0.95)

    if annotate:
        for local_idx, g_idx in enumerate(idx_map):
            kidx = int(kvals[g_idx])
            if 0 <= kidx < len(net_abbr):
                short = net_abbr[kidx]
            else:
                short = labels_out[g_idx][:3].upper()
            ax.text(CX[local_idx], CY[local_idx], CZ[local_idx],
                    f"  {short}", fontsize=8, va="center")

    for (i, j) in edges:
        ax.plot([CX[i], CX[j]], [CY[i], CY[j]], [CZ[i], CZ[j]],
                color="gray", alpha=line_alpha, lw=2.0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(f"Network centroids (Yeo-{yeo_n}, non-crossing 2-regular cycle)")

    def _cube_limits(xs, ys, zs):
        xlo, xhi = np.nanmin(xs), np.nanmax(xs)
        ylo, yhi = np.nanmin(ys), np.nanmax(ys)
        zlo, zhi = np.nanmin(zs), np.nanmax(zs)
        xm, ym, zm = (xlo + xhi)/2.0, (ylo + yhi)/2.0, (zlo + zhi)/2.0
        r = max(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
        r = r if np.isfinite(r) and r > 0 else 1.0
        r *= (1.0 + float(cube_pad))
        return (xm - r, xm + r), (ym - r, ym + r), (zm - r, zm + r)

    if equalize_axes:
        (xl, xh), (yl, yh), (zl, zh) = _cube_limits(CX, CY, CZ)
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_zlim(zl, zh)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    ax.view_init(elev=22, azim=38)

    present = set(int(kvals[g_idx]) for g_idx in idx_map)
    legend_order = _legend_order(yeo_n, present)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=(net_colours[k] if 0 <= k < len(net_colours)
                                       else net_colours[k % len(net_colours)]),
                      markeredgecolor="k", markersize=8,
                      label=(network_labels[k] if 0 <= k < len(network_labels) else f"Net {k}"))
               for k in legend_order]
    ax.legend(handles=handles, title="RSN", loc="upper right")

    if fname is None:
        if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
            i, j, k = dims_to_plot
            fname = f"NetCentroids3D_yeo{yeo_n}_d{i+1}{j+1}{k+1}.png"
        else:
            fname = f"NetCentroids3D_yeo{yeo_n}.png"
    fig.tight_layout()
    fpath = out_dir / fname
    fig.savefig(fpath, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fpath)

    rows = []
    for idx, k in enumerate(kvals):
        rows.append({
            "network_index": int(k),
            "network_name": labels_out[idx],
            "x": float(cx[idx]),
            "y": float(cy[idx]),
            "z": float(cz[idx]),
            "n_parcels": int(n_parcels[idx]),
        })

    if write_coords:
        if coords_fname is None:
            if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
                i, j, k = dims_to_plot
                coords_fname = f"NetCentroids3D_yeo{yeo_n}_d{i+1}{j+1}{k+1}_coords.csv"
            else:
                coords_fname = f"NetCentroids3D_yeo{yeo_n}_coords.csv"
        cpath = out_dir / coords_fname
        with open(cpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["network_index", "network_name", "x", "y", "z", "n_parcels"])
            for r in rows:
                w.writerow([
                    r["network_index"], r["network_name"],
                    f"{r['x']:.6f}", f"{r['y']:.6f}",
                    f"{r['z']:.6f}", r["n_parcels"]
                ])
        print("Saved centroid coordinates:", cpath)

    if print_coords:
        print("Centroid coordinates (network_index, name, x, y, z, n_parcels):")
        for r in rows:
            print(f"{r['network_index']:>2}  {r['network_name']:<16}  "
                  f"{r['x']: .6f}  {r['y']: .6f}  {r['z']: .6f}   {r['n_parcels']}")

    return rows if return_coords else None


# ---------- plot violin or bar plots ----------

def plot_rsn_distributions_by_network(
    arrays: Sequence[np.ndarray],
    out_dir: str | Path,
    name: str = "RSN_netwise",
    *,
    atlas: str = "schaefer",
    yeo_n: int = 7,
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400_7N.R.label.gii",
    glasser_yeo7_path: str = "cortex_parcel_network_assignments_Yeo7.txt",
    glasser_yeo17_path: str = "cortex_parcel_network_assignments_Yeo17.txt",
    network_labels: Optional[List[str]] = None,
    array_labels: Optional[List[str]] = None,
    kind: Literal["violin", "bar", "raincloud"] = "raincloud",
    array_cmap: str = "tab10",
    y_label: str = "Value",
    fname: Optional[str] = None,
    fdr_alpha: float = 0.05,
    group_gap: float = 1.0,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RSN-wise distributions in a SINGLE plot: networks along the x-axis, with
    one cluster of `n_arrays` (e.g. layers) per network and a gap after each
    triplet. Legend distinguishes the arrays (layers). One-way ANOVA per RSN
    (across arrays) and BH-FDR across RSNs.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    networks0, default_labels = _get_yeo_assignments(
        atlas=atlas, yeo_n=yeo_n,
        schaefer_label_L=schaefer_label_L, schaefer_label_R=schaefer_label_R,
        glasser_yeo7_path=glasser_yeo7_path, glasser_yeo17_path=glasser_yeo17_path,
    )
    N = networks0.size
    if network_labels is None:
        network_labels = default_labels

    arrays = [np.asarray(a).reshape(-1) for a in arrays]
    for i, a in enumerate(arrays):
        if a.size != N:
            raise ValueError(
                f"Array {i} has length {a.size}, but atlas implies N={N} parcels."
            )

    n_arrays = len(arrays)
    if array_labels is None:
        array_labels = [f"Index {i+1}" for i in range(n_arrays)]
    elif len(array_labels) != n_arrays:
        raise ValueError("array_labels must have same length as arrays.")

    uniq_nets = np.array(sorted(np.unique(networks0)))
    n_nets = len(uniq_nets)

    cmap = plt.get_cmap(array_cmap, n_arrays)
    array_colors = [cmap(i) for i in range(n_arrays)]

    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if all_vals.size == 0:
        raise ValueError("All values are NaN; cannot set y-axis.")

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    if y_min == y_max:
        pad = 0.5 if y_min == 0 else 0.05 * abs(y_min)
        y_min -= pad
        y_max += pad
    # headroom for the per-network stats text
    y_range = y_max - y_min
    y_text = y_max + 0.04 * y_range
    y_top = y_max + 0.14 * y_range

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    # ---- single-axis layout -------------------------------------------------
    step = n_arrays + group_gap

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * n_nets), 4.5))

    F_vals = np.full(n_nets, np.nan, float)
    p_raw = np.full(n_nets, np.nan, float)

    group_centers = np.zeros(n_nets, float)

    max_width = 0.4
    jitter = 0.08
    alpha_kde = 0.6
    alpha_pts = 0.7

    for idx_net, net in enumerate(uniq_nets):
        base = idx_net * step
        xpos = base + np.arange(n_arrays)
        group_centers[idx_net] = xpos.mean()

        groups = []
        for arr in arrays:
            vals = arr[networks0 == net]
            vals = vals[np.isfinite(vals)]
            groups.append(vals)

        try:
            F, p = f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        F_vals[idx_net] = F
        p_raw[idx_net] = p

        if kind == "violin":
            v = ax.violinplot(
                groups,
                positions=xpos,
                widths=0.8,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body, color in zip(v["bodies"], array_colors):
                body.set_facecolor(color)
                body.set_edgecolor("k")
                body.set_alpha(0.8)
            if "cmedians" in v:
                v["cmedians"].set_color("k")
                v["cmedians"].set_linewidth(1.2)

        elif kind == "bar":
            means = [np.nanmean(g) if g.size > 0 else np.nan for g in groups]
            stds = [np.nanstd(g) if g.size > 1 else 0.0 for g in groups]
            ax.bar(
                xpos,
                means,
                width=0.8,
                yerr=stds,
                color=array_colors,
                edgecolor="k",
                linewidth=0.7,
                alpha=0.9,
                capsize=3,
            )

        elif kind == "raincloud":
            for j, (g_vals, color) in enumerate(zip(groups, array_colors)):
                xc = base + j  # this array's slot within the network cluster
                if g_vals.size == 0:
                    continue

                if g_vals.size > 1 and np.nanstd(g_vals) > 0:
                    try:
                        kde = gaussian_kde(g_vals)
                        y_grid = np.linspace(
                            max(y_min, np.nanmin(g_vals)),
                            min(y_max, np.nanmax(g_vals)),
                            100,
                        )
                        density = kde(y_grid)
                        if np.max(density) > 0:
                            density = density / np.max(density) * max_width
                            x_left = xc - density
                            x_right = np.full_like(y_grid, xc)
                            x_poly = np.concatenate([x_left, x_right[::-1]])
                            y_poly = np.concatenate([y_grid, y_grid[::-1]])
                            ax.fill(
                                x_poly,
                                y_poly,
                                color=color,
                                alpha=alpha_kde,
                                edgecolor="k",
                                linewidth=0.5,
                            )
                    except Exception:
                        pass

                x_jitter = xc + (np.random.rand(g_vals.size) - 0.5) * 2 * jitter
                ax.scatter(
                    x_jitter,
                    g_vals,
                    color=color,
                    alpha=alpha_pts,
                    edgecolor="k",
                    linewidth=0.3,
                    s=10,
                )

                med = np.nanmedian(g_vals)
                ax.scatter(
                    xc + max_width * 0.6,
                    med,
                    color="k",
                    marker="_",
                    s=80,
                    linewidths=1.2,
                    zorder=3,
                )
        else:
            raise ValueError("kind must be 'violin', 'bar', or 'raincloud'.")

    # ---- FDR across networks ------------------------------------------------
    p_fdr, sig_mask = _fdr_bh(p_raw, alpha=fdr_alpha)

    # ---- per-network stats annotation above each cluster --------------------
    for idx_net, net in enumerate(uniq_nets):
        F = F_vals[idx_net]
        p = p_raw[idx_net]
        q = p_fdr[idx_net]
        star = " *" if sig_mask[idx_net] else ""
        txt = f"F={F:.2f}\np={p:.2g}\nq={q:.2g}{star}"
        ax.text(
            group_centers[idx_net],
            y_text,
            txt,
            ha="center",
            va="bottom",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=0.4),
        )

    # ---- axes cosmetics -----------------------------------------------------
    ax.set_xlim(-0.5, (n_nets - 1) * step + (n_arrays - 1) + 0.5)
    ax.set_ylim(y_min, y_top)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [network_labels[int(net)] for net in uniq_nets],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel(y_label)

    # legend distinguishes the arrays (layers)
    handles = [
        Line2D(
            [0], [0],
            marker="s", linestyle="none",
            markerfacecolor=array_colors[i],
            markeredgecolor="k",
            markersize=8,
            label=array_labels[i],
        )
        for i in range(n_arrays)
    ]
    ax.legend(
        handles=handles,
        title="Layer",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=min(n_arrays, 4),
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if fname is None:
        fname = f"{name}_yeo{yeo_n}_{kind}_single.svg"
    outpath = out_dir / fname
    fig.savefig(outpath, bbox_inches="tight", format="svg")
    plt.close(fig)

    print("Saved:", outpath)
    return str(outpath), F_vals, p_raw, p_fdr, sig_mask