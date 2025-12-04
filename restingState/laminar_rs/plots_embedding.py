# laminar_rs/plots_embedding.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, List, Literal, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import nibabel as nib
from nilearn import plotting
from scipy.stats import pearsonr, t as t_dist
from scipy.stats import f_oneway, gaussian_kde

import hcp_utils as hcp


# ---------- KMeans on embedding ----------

def run_kmeans(eigvecs: np.ndarray,
               out_dir,
               name: str,
               num_clusters: int = 3,
               random_state: int = 99,
               eigvecs_to_plot: Tuple[int, int] = (1, 2)) -> None:
    """
    Your original runKMeans method.
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / name
    layer_dir.mkdir(parents=True, exist_ok=True)

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(eigvecs)
    eigvecs_str = "".join(map(str, eigvecs_to_plot))

    plt.figure(figsize=(8, 6))
    plt.scatter(eigvecs[:, eigvecs_to_plot[0]],
                eigvecs[:, eigvecs_to_plot[1]],
                c=labels, cmap="viridis", edgecolor="k", s=50)
    plt.xlabel(f"Eigenvector {eigvecs_to_plot[0]+1}")
    plt.ylabel(f"Eigenvector {eigvecs_to_plot[1]+1}")
    plt.title("KMeans Clustering")
    plt.colorbar(label="Cluster")
    plt.savefig(layer_dir / f"KMeans_laplacian_embedding_{eigvecs_str}.png",
                bbox_inches="tight")
    plt.close()


# ---------- Schaefer helpers ----------

def _load_label_gii(path: str):
    g = nib.load(path)
    labs = np.asarray(g.agg_data(), dtype=int).squeeze()
    lt = g.labeltable
    key_to_name = {lab.key: lab.label for lab in lt.labels}
    return labs, key_to_name


def _schaefer7_from_name(name: str) -> int:
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


def _glasser7_from_name(name: str) -> int:
    """
    Map a Glasser/HCP-MMP1.0 network label (e.g. 'Visual', 'Somatomotor',
    'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal',
    'Default') to a 0-based Yeo-7 index:

        0: Visual
        1: Somatomotor
        2: Dorsal Attention
        3: Ventral Attention / Salience
        4: Limbic
        5: Frontoparietal / Control
        6: Default

    Accepts either the bare network name or the full line from the
    cortex_parcel_network_assignments_Yeo7.txt file.
    """
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


def _load_glasser_yeo7_assignments(
        path: str = "cortex_parcel_network_assignments_Yeo7.txt"
) -> np.ndarray:
    """
    Load Yeo-7 assignments for the 360 Glasser/HCP-MMP parcels from the
    text file cortex_parcel_network_assignments_Yeo7.txt.

    File format (one parcel per line), e.g.:

        1_R_V1_ROI_1_Visual
        10_R_FEF_ROI_3_Dorsal Attention
        ...

    Returns
    -------
    networks0 : (N,) int array
        0-based Yeo-7 network index per parcel (N should be 360).
        Order is the line order in the file (1..360).
    """
    networks: list[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("_")
            net_label = parts[-1]  # e.g. 'Visual', 'Somatomotor', 'Dorsal Attention'
            net_idx = _glasser7_from_name(net_label)
            networks.append(net_idx)

    networks0 = np.asarray(networks, dtype=int)
    if networks0.min() < 0 or networks0.max() > 6:
        raise ValueError(
            f"Yeo-7 assignments in {path} must be in [0..6] after mapping, "
            f"found range [{networks0.min()}, {networks0.max()}]."
        )
    return networks0

# ---------- 2D embedding colored by RSNs & layers ----------

def plot_two_dim_embedding(
        eigvecs: np.ndarray,
        out_dir,
        name: str,
        eigvecs_to_plot: Tuple[int, int] = (0, 1),
        layer_labels: Optional[List[str]] = None,
        network_labels: Optional[List[str]] = None,
        x_label: str = "Emb1",
        y_label: str = "Emb2",
        network_cmap: str = "tab20",
        atlas: str = "schaefer",
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
) -> None:
    """
    2D embedding coloured by RSNs (Schaefer-7 or Glasser/Yeo-7) and optionally by layer.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_lower = atlas.lower()

    # ----- RSN assignments per parcel -----
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("Provide schaefer_label_L and schaefer_label_R (.label.gii).")

        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

        networks0 = []
        for k in uL:
            networks0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            networks0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(networks0, dtype=int)
        N = networks0.size

        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Control", "Default"
            ]

    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size
        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Frontoparietal", "Default"
            ]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

    # ----- single vs multilayer -----
    nrows, ndims = eigvecs.shape
    if nrows == 3 * N:
        mode = "multilayer"
    elif nrows == N:
        mode = "single"
    elif nrows % 3 == 0 and nrows // 3 == N:
        mode = "multilayer"
    else:
        raise ValueError(f"eigvecs has {nrows} rows, but atlas implies N={N} or 3N={3*N}.")

    x_dim, y_dim = eigvecs_to_plot
    if x_dim >= ndims or y_dim >= ndims:
        x_dim = max(0, x_dim - 1)
        y_dim = max(0, y_dim - 1)
    if not (0 <= x_dim < ndims and 0 <= y_dim < ndims):
        raise ValueError(f"Requested dims {eigvecs_to_plot} not in [0..{ndims-1}]")

    if mode == "multilayer":
        networks = np.tile(networks0, 3)
        layers = np.repeat([0, 1, 2], N)
    else:
        networks = networks0
        layers = np.zeros(N, dtype=int)

    if isinstance(layer_labels, str):
        layer_labels = [layer_labels]
    if layer_labels is None:
        layer_labels = ["Superficial", "Middle", "Deep"] if mode == "multilayer" else ["AcrossLayers"]
    elif mode == "single" and len(layer_labels) != 1:
        layer_labels = [layer_labels[0]]

    base_cmap = plt.get_cmap(network_cmap, len(network_labels))
    network_colors = [base_cmap(i) for i in range(len(network_labels))]
    shapes = ["o", "s", "^"] if mode == "multilayer" else ["o"]

    fig, ax = plt.subplots(figsize=(7, 7))
    unique_layers = np.unique(layers)
    unique_nets = np.unique(networks)

    for lyr in unique_layers:
        for net in unique_nets:
            mask = (layers == lyr) & (networks == net)
            if not np.any(mask):
                continue
            ax.scatter(
                eigvecs[mask, x_dim],
                eigvecs[mask, y_dim],
                s=10,
                marker=shapes[int(lyr if mode == "multilayer" else 0)],
                facecolor=network_colors[int(net)],
                edgecolor="k",
                linewidths=0.2,
                alpha=0.85
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Embedding coloured by Yeo-7 RSN; shapes = layers"
                 if mode == "multilayer"
                 else "Embedding coloured by Yeo-7 RSN")
    ax.set_aspect("equal", adjustable="box")

    # Layer legend
    if mode == "multilayer":
        layer_handles = [
            Line2D([0], [0], marker=shapes[i], color="w", markeredgecolor="k",
                   markersize=9, label=layer_labels[int(lyr)])
            for i, lyr in enumerate(unique_layers)
        ]
        leg1 = ax.legend(handles=layer_handles, title="Layer", loc="upper right")
        ax.add_artist(leg1)

    # RSN legend
    network_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=network_colors[int(i)], markeredgecolor="k",
               markersize=9, label=network_labels[int(i)])
        for i in unique_nets
    ]
    ax.legend(handles=network_handles, title="RSN",
              bbox_to_anchor=(1.32, 1), loc="upper left")

    plt.tight_layout()
    eigstr = f"{x_dim}{y_dim}"
    suffix = "_multi" if mode == "multilayer" else "_single"
    outpath = out_dir / f"Embedding_withNetworks_{eigstr}{suffix}.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=500)
    plt.close(fig)
    print("Saved embedding plot to:", outpath)



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
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
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
    2D scatter with regression line and global Pearson r, coloured by Yeo-7 RSNs.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_lower = atlas.lower()

    # ----- RSN assignments -----
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

        networks0 = []
        for k in uL:
            networks0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            networks0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(networks0, dtype=int)
        N = networks0.size
        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Control", "Default"
            ]

    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size
        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Frontoparietal", "Default"
            ]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

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

    # ----- colours per RSN -----
    seq_cmap = matplotlib.cm.get_cmap(network_cmap)
    n_nets = int(nets.max()) + 1
    if len(network_labels) == 7:
        shades = [seq_cmap(x) for x in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]  # just for nicer separation
        net_colours = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours[net_idx] = shades[rank]
    else:
        if n_nets < 1:
            raise ValueError("No networks found.")
        net_colours = [seq_cmap(i / max(n_nets - 1, 1)) for i in range(n_nets)]

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
    ax.set_title("Embedding coloured by Yeo-7 RSN" +
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
    if len(network_labels) < n_nets:
        network_labels = list(network_labels) + [
            f"Net{i}" for i in range(len(network_labels), n_nets)
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
        fname = f"ScatterCorr_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}.png"
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
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
        show_marginals: bool = True,
        hist_bins: int = 20,
):
    """
    3D embedding coloured by Yeo-7 RSNs, optional best-fit plane and marginals.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- RSN assignments -----
    atlas_lower = atlas.lower()
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))
        nets0 = []
        for k in uL:
            nets0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            nets0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(nets0, int)
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Control", "Default"]
    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Frontoparietal", "Default"]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

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

    # ----- colours -----
    if len(network_labels) == 7:
        try:
            seq_cmap = plt.get_cmap(network_cmap)
        except Exception:
            seq_cmap = plt.get_cmap("tab20")
        shades = [seq_cmap(x_) for x_ in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]
        net_colours = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours[net_idx] = shades[rank]
    else:
        cmap = plt.get_cmap(network_cmap, len(network_labels))
        net_colours = [cmap(i) for i in range(len(network_labels))]

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
    ax.set_title("3D Embedding coloured by Yeo-7 RSN" +
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
    if len(network_labels) == 7:
        present = set(int(i) for i in np.unique(nets))
        legend_order = [i for i in [1, 0, 3, 2, 6, 5, 4] if i in present]
    else:
        legend_order = list(int(i) for i in np.unique(nets))

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
            fname = f"Scatter3D_d{i+1}{j+1}{k+1}_{'multi' if mode=='multilayer' else 'single'}.png"
        else:
            fname = f"Scatter3D_{'multi' if mode=='multilayer' else 'single'}.png"

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
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
):
    """
    2D centroids (RSN × layer) for Yeo-7 networks.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_lower = atlas.lower()

    # ----- per-parcel RSN vector -----
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))
        networks0 = []
        for k in uL:
            networks0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            networks0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(networks0, dtype=int)
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Control", "Default"]
    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Frontoparietal", "Default"]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

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

    # ----- colours -----
    seq_cmap = matplotlib.cm.get_cmap(network_cmap)
    n_nets = int(nets.max()) + 1
    if len(network_labels) == 7:
        if n_nets > 7:
            raise ValueError(f"Expected at most 7 networks, found {n_nets}")
        shades = [seq_cmap(x) for x in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]
        net_colours = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours[net_idx] = shades[rank]
    else:
        if n_nets < 1:
            raise ValueError("No networks found.")
        net_colours = [seq_cmap(i / max(n_nets - 1, 1)) for i in range(n_nets)]

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

    fig, ax = plt.subplots(figsize=(7, 7))
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
    ax.set_title("Centroids by Yeo-7 RSN" +
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
        fname = f"ScatterCentroids_d{x_dim+1}{y_dim+1}_{'multi' if mode=='multilayer' else 'single'}_unitaxes.png"
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

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
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
):
    """
    3D Yeo-7 network centroids (single point per RSN), joined by a non-crossing cycle.
    """
    import csv
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- RSN assignments -----
    atlas_lower = atlas.lower()
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))
        nets0 = []
        for k in uL:
            nets0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            nets0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(nets0, int)
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Control", "Default"]
        net_abbr = ["VIS", "SOM", "DAN", "VAN", "LIM", "CON", "DMN"]
    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size
        if network_labels is None:
            network_labels = ["Visual", "Somatomotor", "Dorsal Attn",
                              "Ventral/Salience", "Limbic", "Frontoparietal", "Default"]
        net_abbr = ["VIS", "SOM", "DAN", "VAN", "LIM", "FPN", "DMN"]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

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

    kvals = np.array(sorted(np.unique(networks0)))  # 0..6 subset
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

    # ----- project to 2D -----
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

    # fallback: simple angular cycle if MST-like attempt failed
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

    # colours
    if len(network_labels) == 7:
        try:
            seq_cmap = plt.get_cmap(network_cmap)
        except Exception:
            seq_cmap = plt.get_cmap("tab20")
        shades = [seq_cmap(x_) for x_ in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]
        net_colours = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours[net_idx] = shades[rank]
    else:
        cmap = plt.get_cmap(network_cmap, len(network_labels))
        net_colours = [cmap(i) for i in range(len(network_labels))]

    # plot centroids
    for local_idx, g_idx in enumerate(idx_map):
        net_idx = int(kvals[g_idx])  # 0..6
        if 0 <= net_idx < len(net_abbr):
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
            # safe abbreviation lookup
            if 0 <= kidx < len(net_abbr):
                short = net_abbr[kidx]
            else:
                short = labels_out[g_idx][:3].upper()
            ax.text(CX[local_idx], CY[local_idx], CZ[local_idx],
                    f"  {short}", fontsize=8, va="center")

    # draw cycle edges
    for (i, j) in edges:
        ax.plot([CX[i], CX[j]], [CY[i], CY[j]], [CZ[i], CZ[j]],
                color="gray", alpha=line_alpha, lw=2.0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title("Network centroids (Yeo-7, non-crossing 2-regular cycle)")

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

    # Legend
    present = set(int(kvals[g_idx]) for g_idx in idx_map)
    if len(network_labels) == 7:
        legend_order = [i for i in [1, 0, 3, 2, 6, 5, 4] if i in present]
    else:
        legend_order = [k for k in range(len(network_labels)) if k in present]

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=(net_colours[k] if len(network_labels) == 7
                                       else net_colours[k % len(net_colours)]),
                      markeredgecolor="k", markersize=8,
                      label=(network_labels[k] if 0 <= k < len(network_labels) else f"Net {k}"))
               for k in legend_order]
    ax.legend(handles=handles, title="RSN", loc="upper right")

    # ----- save figure -----
    if fname is None:
        if Y is None and Z is None and getattr(X, "ndim", 1) == 2:
            i, j, k = dims_to_plot
            fname = f"NetCentroids3D_d{i+1}{j+1}{k+1}.png"
        else:
            fname = "NetCentroids3D.png"
    fig.tight_layout()
    fpath = out_dir / fname
    fig.savefig(fpath, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fpath)

    # ----- export coordinates -----
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
                coords_fname = f"NetCentroids3D_d{i+1}{j+1}{k+1}_coords.csv"
            else:
                coords_fname = "NetCentroids3D_coords.csv"
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



# ---------- eigvecs -> NIfTI & surface ----------

def plot_on_mmhcp_surface_multipleLayers(
        Xp: np.ndarray,
        out_dir,
        eigValue,
        vmin=None,
        vmax=None,
        cm: str = "cividis",
        noSubcortical: bool = True,
        titles: List[str] = None,
        folder_name: str = "surface_map",
        atlas: str = "schaefer",
        schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
        schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
):

    out_dir = Path(out_dir)
    os.makedirs(out_dir / folder_name, exist_ok=True)

    if titles is None:
        titles = ["Deep", "Middle", "Superficial", "Average"]

    def _build_rank_map(keys):
        u = np.unique(keys[keys > 0])
        return {k: i for i, k in enumerate(sorted(u))}, len(u)

    def _map_parcels_to_vertices_schaefer(vals_lr, L_lab, R_lab, L_rank, R_rank, n_hemi):
        left = np.full(L_lab.shape, np.nan, float)
        mL = L_lab > 0
        if np.any(mL):
            idxL = np.array([L_rank[k] for k in L_lab[mL]])
            left[mL] = vals_lr[idxL]
        right = np.full(R_lab.shape, np.nan, float)
        mR = R_lab > 0
        if np.any(mR):
            idxR = np.array([R_rank[k] for k in R_lab[mR]])
            right[mR] = vals_lr[n_hemi + idxR]
        return left, right

    if atlas.lower() == "mmp":
        mmp_labels = hcp.mmp.labels
        n_parcels_target = len(mmp_labels)

        def map_layer(vals_lr):
            vtx_both = hcp.cortex_data(hcp.unparcellate(vals_lr, hcp.mmp))
            nL = len(vtx_both) // 2
            return vtx_both[:nL], vtx_both[nL:]
        
    elif atlas.lower() == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")
        L_lab = nib.load(schaefer_label_L).agg_data().squeeze().astype(int)
        R_lab = nib.load(schaefer_label_R).agg_data().squeeze().astype(int)
        L_rank, nL_parcels = _build_rank_map(L_lab)
        R_rank, nR_parcels = _build_rank_map(R_lab)
        assert nL_parcels == nR_parcels, f"Unequal parcels per hemi: L={nL_parcels}, R={nR_parcels}"
        n_parcels_target = 2 * nL_parcels

        def map_layer(vals_lr):
            return _map_parcels_to_vertices_schaefer(vals_lr, L_lab, R_lab, L_rank, R_rank, nL_parcels)
    else:
        raise ValueError("atlas must be 'mmp' or 'schaefer'.")

    Xp = np.asarray(Xp)
    if Xp.ndim != 2:
        raise ValueError("Xp must be 2D (n_parcels, n_layers).")
    current_length = Xp.shape[0]
    if noSubcortical:
        zeros_to_add = n_parcels_target - current_length
        if zeros_to_add > 0:
            Xp = np.concatenate((Xp, np.zeros((zeros_to_add, Xp.shape[1]))), axis=0)
        elif zeros_to_add < 0:
            raise ValueError(f"Xp has {current_length} rows but atlas expects {n_parcels_target}.")
    else:
        if current_length != n_parcels_target:
            raise ValueError(f"Xp has {current_length} rows but atlas expects {n_parcels_target}.")

    left_right_layers = []
    for i in range(Xp.shape[1]):
        left_i, right_i = map_layer(Xp[:, i])
        left_right_layers.append((left_i, right_i))

    all_data = np.hstack([np.concatenate((L, R)) for (L, R) in left_right_layers])
    if vmin is None or vmax is None:
        finite = np.isfinite(all_data)
        if not np.any(finite):
            raise ValueError("All mapped values NaN.")
        vmin, vmax = np.nanpercentile(all_data[finite], [2, 98])
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6

    orientations = ["lateral", "medial", "medial", "lateral"]
    fig, axes = plt.subplots(
        Xp.shape[1], len(orientations),
        figsize=(20, 5 * Xp.shape[1]),
        subplot_kw={"projection": "3d"}
    )

    for i in range(Xp.shape[1]):
        left_i, right_i = left_right_layers[i]
        row_title = titles[i] if (titles is not None and i < len(titles)) else f"Layer {i+1}"

        for j, view in enumerate(orientations):
            try:
                ax = axes[i, j]
            except Exception:
                ax = axes[j]

            if j in (0, 1):
                plotting.plot_surf_stat_map(
                    hcp.mesh.inflated_left,
                    left_i,
                    view=view,
                    colorbar=False,
                    bg_map=hcp.mesh.sulc_left,
                    bg_on_data=True,
                    darkness=0.3,
                    axes=ax,
                    figure=fig,
                    cmap=cm,
                    vmin=vmin,
                    vmax=vmax,
                    symmetric_cbar=False,
                )
            else:
                plotting.plot_surf_stat_map(
                    hcp.mesh.inflated_right,
                    right_i,
                    view=view,
                    colorbar=False,
                    bg_map=hcp.mesh.sulc_right,
                    bg_on_data=True,
                    darkness=0.3,
                    axes=ax,
                    figure=fig,
                    cmap=cm,
                    vmin=vmin,
                    vmax=vmax,
                    symmetric_cbar=False,
                )
            ax.set_title(f"{row_title} - {orientations[j].capitalize()}", fontsize=14)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cm), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.suptitle(f"Surface plot", fontsize=16)
    out_path = out_dir / folder_name / f"Surface_{eigValue}_twoHem.png"
    print(out_path)
    plt.savefig(out_path, facecolor="white", dpi=300)
    plt.close()
    return out_path


def eigvecs_to_nifti(
        eigvecs: np.ndarray,
        out_dir,
        name: str,
        N: int,
        num_layers: int,
        atlas_path,
        hcp_atlas: bool = True,
        force_run: bool = True,
        scaleEigVecs: bool = False,
        saveNifti: bool = False,
        add_name: str = "",
):
    """
    Functional version of eigvecs_to_nifti (calls plot_on_mmhcp_surface_multipleLayers).
    """
    out_dir = Path(out_dir)
    atlas_path = Path(atlas_path)

    if scaleEigVecs:
        M = np.max(np.abs(eigvecs), axis=0)
        eigvecs_scaled = eigvecs / M
        M_max = np.max(np.abs(eigvecs_scaled))
        eigvecs = eigvecs_scaled * M_max

    parcel_atlas_img = nib.load(str(atlas_path))
    parcel_atlas = parcel_atlas_img.get_fdata()
    unique_parcels = np.unique(parcel_atlas)

    if hcp_atlas:
        import warnings
        warnings.warn("Selecting cortex parcels of the HCP-MMP1.0 atlas.")
        unique_parcels = unique_parcels[
            (unique_parcels >= 1001) & (unique_parcels <= 3000) & (unique_parcels != 2000)
        ]
    else:
        unique_parcels = unique_parcels[(unique_parcels > 0)]

    print(f"Unique parcels: {len(unique_parcels)}")

    total_regions = eigvecs.shape[0]
    num_components = eigvecs.shape[1]
    threshold = 40

    if num_components > threshold:
        indices = list(range(20)) + list(range(num_components - 20, num_components))
    else:
        indices = list(range(num_components))

    if total_regions % num_layers != 0:
        raise ValueError("Total regions must be evenly divisible by number of layers.")

    if N * num_layers != total_regions:
        print(f"[warn] Provided N={N} and num_layers={num_layers} but N*L={N*num_layers} "
              f"!= rows {total_regions}. Using inferred N={total_regions//num_layers}")
        N = total_regions // num_layers

    print(f"Mapping {total_regions} nodes into {num_layers} layers of {N} regions each.")

    eig_layers = np.split(eigvecs, num_layers, axis=0)

    for i in indices:
        if force_run or not (out_dir / name / "eigenvector_layers").exists():
            try:
                (out_dir / name / f"eigenvector_layers{add_name}").mkdir(parents=True, exist_ok=True)
                folder = out_dir / name / f"eigenvector_layers{add_name}"
            except Exception:
                folder = out_dir / name / "eigenvector_layers"
                folder.mkdir(parents=True, exist_ok=True)

            layer_imgs = []
            for layer_idx, layer_data in enumerate(eig_layers):
                map_3D = np.zeros_like(parcel_atlas)
                for roi_idx, parcel in enumerate(unique_parcels):
                    parcel_mask = np.zeros(parcel_atlas.shape)
                    parcel_mask[parcel_atlas == parcel] = 1
                    parcel_mask = np.array(parcel_mask, dtype=bool)
                    final_mask = parcel_mask
                    map_3D[final_mask] = layer_data[roi_idx, i]

                layer_img = nib.Nifti1Image(map_3D, affine=parcel_atlas_img.affine)
                if saveNifti:
                    nib.save(layer_img,
                             folder / f"eigenvector_{i+1}_layer_{layer_idx+1}.nii.gz")
                layer_imgs.append(layer_img)

        Xp_layers = []
        for layer_idx in range(num_layers):
            Xp_layers.append(eig_layers[layer_idx][:, i])
        Xp_layers = np.array(Xp_layers)

        if hcp_atlas:
            plot_on_mmhcp_surface_multipleLayers(
                Xp_layers.T, out_dir, name,
                eigValue=i+1,
                folder_name="eigenvector_layers"
            )
        else:
            # fallback: just save volume NIfTIs (already done)
            pass

    print("All brain maps saved successfully!")


# ---------- per-network 2D plots by network ----------

def plot_two_dim_embedding_byNetwork(
        eigvecs: np.ndarray,
        out_dir,
        name: str,
        eigvecs_to_plot: Tuple[int, int] = (1, 2),
        layer_labels: Optional[List[str]] = None,
        network_labels: Optional[List[str]] = None,
        network_cmap: str = "tab20",
):
    """
    Functional version of plotTwoDimEmbedding_byNetwork.
    """
    out_dir = Path(out_dir)
    out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    x_dim, y_dim = eigvecs_to_plot
    P3, _ = eigvecs.shape
    P = P3 // 3

    cats0 = np.loadtxt("cortex_parcel_network_assignments.txt", dtype=int)
    networks0 = cats0 - 1
    networks = np.tile(networks0, 3)
    layers = np.repeat([0, 1, 2], P)

    if layer_labels is None:
        layer_labels = ["Superficial", "Middle", "Deep"]
    if network_labels is None:
        network_labels = [
            "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
            "Dorsal-Attentional", "Language", "Frontoparietal", "Auditory",
            "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"
        ]

    base_cmap = plt.get_cmap(network_cmap, len(network_labels))
    network_colors = [base_cmap(i) for i in range(len(network_labels))]
    coords2d = eigvecs[:, [x_dim, y_dim]]

    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    shapes = ["o", "s", "^"]
    for net in range(len(network_labels)):
        ax = axes[net]
        mask_net = (networks == net)
        net_coords = coords2d[mask_net]
        net_layers = layers[mask_net]

        if len(np.unique(net_layers)) > 1:
            sil_score = silhouette_score(net_coords, net_layers)
        else:
            sil_score = np.nan

        for lyr in [0, 1, 2]:
            mask = mask_net & (layers == lyr)
            if not mask.any():
                continue
            ax.scatter(
                eigvecs[mask, x_dim],
                eigvecs[mask, y_dim],
                marker=shapes[lyr],
                facecolor=network_colors[net],
                edgecolor="k",
                alpha=0.7,
                label=layer_labels[lyr]
            )
        title = f"{network_labels[net]}\nSilhouette = {sil_score:.2f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"EV {x_dim+1}")
        ax.set_ylabel(f"EV {y_dim+1}")
        if net == 0:
            ax.legend(title="Layer", loc="best")

    for ax in axes[len(network_labels):]:
        ax.axis("off")

    plt.tight_layout()
    eigstr = f"{x_dim}{y_dim}"
    outpath = out_dir / f"Laplacian_embedding_byNetwork_{eigstr}.png"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)



# ---------- plot violin or bar plots ----------


def plot_rsn_distributions(
    arrays: Sequence[np.ndarray],
    out_dir: str | Path,
    name: str = "RSN_distributions",
    *,
    atlas: str = "schaefer",
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    network_labels: Optional[List[str]] = None,
    array_labels: Optional[List[str]] = None,
    kind: Literal["violin", "bar"] = "violin",
    network_cmap: str = "tab20",
    y_label: str = "Value",
    fname: Optional[str] = None,
    share_yaxis: bool = True,
) -> str:
    """
    Plot distributions of parcel-wise values per Yeo-7 RSN for one or more arrays.
    """
    from matplotlib.lines import Line2D  # for legend handles

    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_lower = atlas.lower()

    # ----- RSN assignments -----
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")

        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

        nets0 = []
        for k in uL:
            nets0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            nets0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(nets0, int)
        N = networks0.size

        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Control", "Default"
            ]
    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        networks0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        N = networks0.size

        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Frontoparietal", "Default",
            ]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

    # ----- check arrays -----
    arrays = [np.asarray(a).reshape(-1) for a in arrays]
    for idx, a in enumerate(arrays):
        if a.size != N:
            raise ValueError(
                f"Array {idx} has length {a.size}, expected N={N} from atlas."
            )

    n_arrays = len(arrays)
    if array_labels is None:
        array_labels = [f"Index {i+1}" for i in range(n_arrays)]
    elif len(array_labels) != n_arrays:
        raise ValueError("array_labels must have same length as arrays.")

    uniq_nets = np.array(sorted(np.unique(networks0)))
    n_nets = len(uniq_nets)

    # ----- RSN colours -----
    if len(network_labels) == 7:
        try:
            seq_cmap = plt.get_cmap(network_cmap)
        except Exception:
            seq_cmap = plt.get_cmap("tab20")

        shades = [seq_cmap(x_) for x_ in np.linspace(0.2, 0.9, 7)]
        order_idx = [1, 0, 3, 2, 6, 5, 4]
        net_colours_all = [None] * 7
        for rank, net_idx in enumerate(order_idx):
            net_colours_all[net_idx] = shades[rank]
    else:
        cmap = plt.get_cmap(network_cmap, len(network_labels))
        net_colours_all = [cmap(i) for i in range(len(network_labels))]

    present_colours = [net_colours_all[int(n)] for n in uniq_nets]
    present_labels = [network_labels[int(n)] for n in uniq_nets]

    # ----- plotting -----
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    fig, axes = plt.subplots(
        1, n_arrays, figsize=(4.0 * n_arrays, 4.0), sharey=share_yaxis
    )
    if n_arrays == 1:
        axes = [axes]

    for idx, (arr, ax, title) in enumerate(zip(arrays, axes, array_labels)):
        data_by_net = [arr[networks0 == net] for net in uniq_nets]

        if kind == "violin":
            v = ax.violinplot(
                data_by_net,
                positions=np.arange(n_nets) + 1,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body, c in zip(v["bodies"], present_colours):
                body.set_facecolor(c)
                body.set_edgecolor("k")
                body.set_alpha(0.8)
            if "cmedians" in v:
                v["cmedians"].set_color("k")
                v["cmedians"].set_linewidth(1.2)

        elif kind == "bar":
            means = [np.nanmean(vals) for vals in data_by_net]
            stds = [np.nanstd(vals) for vals in data_by_net]
            xpos = np.arange(n_nets) + 1
            ax.bar(
                xpos,
                means,
                yerr=stds,
                color=present_colours,
                edgecolor="k",
                linewidth=0.7,
                alpha=0.9,
                capsize=3,
            )
        else:
            raise ValueError("kind must be 'violin' or 'bar'.")

        ax.set_title(title)
        ax.set_xticks(np.arange(n_nets) + 1)
        ax.set_xticklabels(present_labels, rotation=45, ha="right", fontsize=8)
        if share_yaxis:
            if idx == 0:
                ax.set_ylabel(y_label)
        else:
            ax.set_ylabel(y_label)

    # RSN legend
    handles = [
        Line2D(
            [0], [0], marker="s", linestyle="none",
            markerfacecolor=c, markeredgecolor="k",
            markersize=8, label=lab,
        )
        for c, lab in zip(present_colours, present_labels)
    ]
    axes[-1].legend(
        handles=handles,
        title="RSN",
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )

    fig.tight_layout()

    if fname is None:
        fname = f"{name}_{kind}.svg"
    outpath = out_dir / fname
    fig.savefig(outpath, bbox_inches="tight", format="svg")
    plt.close(fig)

    print("Saved:", outpath)
    return str(outpath)


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.
    """
    p = np.asarray(pvals, float).ravel()
    m = p.size
    if m == 0:
        return p.copy(), np.zeros_like(p, dtype=bool)

    order = np.argsort(p)
    ranked_p = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    q = ranked_p * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    p_fdr = np.empty_like(q)
    p_fdr[order] = q

    significant = p_fdr < alpha
    return p_fdr, significant



def plot_rsn_distributions_by_network(
    arrays: Sequence[np.ndarray],
    out_dir: str | Path,
    name: str = "RSN_netwise",
    *,
    atlas: str = "schaefer",
    schaefer_label_L: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.L.label.gii",
    schaefer_label_R: str = "/home/degutis/repos/SchaeferAtlas/Schaefer400.R.label.gii",
    network_labels: Optional[List[str]] = None,
    array_labels: Optional[List[str]] = None,
    kind: Literal["violin", "bar", "raincloud"] = "raincloud",
    array_cmap: str = "tab10",
    y_label: str = "Value",
    fname: Optional[str] = None,
    fdr_alpha: float = 0.05,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RSN-wise distributions (one subplot per Yeo-7 network) for several arrays,
    plus one-way ANOVA per RSN and BH-FDR across RSNs.
    """
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_lower = atlas.lower()

    # ----- RSN assignments & labels -----
    if atlas_lower == "schaefer":
        if schaefer_label_L is None or schaefer_label_R is None:
            raise ValueError("For atlas='schaefer', provide Schaefer label.gii paths.")

        L_lab, L_map = _load_label_gii(schaefer_label_L)
        R_lab, R_map = _load_label_gii(schaefer_label_R)
        uL = np.array(sorted(np.unique(L_lab[L_lab > 0])))
        uR = np.array(sorted(np.unique(R_lab[R_lab > 0])))

        nets0 = []
        for k in uL:
            nets0.append(_schaefer7_from_name(L_map[k]))
        for k in uR:
            nets0.append(_schaefer7_from_name(R_map[k]))
        networks0 = np.asarray(nets0, int)
        N = networks0.size

        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Control", "Default",
            ]
    elif atlas_lower in ("hcp", "glasser", "glasser360", "mmp"):
        nets0 = _load_glasser_yeo7_assignments(
            "cortex_parcel_network_assignments_Yeo7.txt"
        )
        networks0 = nets0
        N = networks0.size

        if network_labels is None:
            network_labels = [
                "Visual", "Somatomotor", "Dorsal Attn",
                "Ventral/Salience", "Limbic", "Frontoparietal", "Default",
            ]
    else:
        raise ValueError(f"Unknown atlas '{atlas}'. Use 'schaefer' or 'glasser'/'hcp'/'mmp'.")

    # ----- arrays & labels -----
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

    # ----- colours for arrays -----
    cmap = plt.get_cmap(array_cmap, n_arrays)
    array_colors = [cmap(i) for i in range(n_arrays)]

    # ----- global y-limits -----
    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if all_vals.size == 0:
        raise ValueError("All values are NaN; cannot set y-axis.")

    y_min = 0
    y_max = float(np.nanmax(all_vals))
    if y_min == y_max:
        pad = 0.5 if y_min == 0 else 0.05 * abs(y_min)
        y_min -= pad
        y_max += pad

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    # grid
    n_cols = min(4, n_nets)
    n_rows = int(np.ceil(n_nets / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    F_vals = np.full(n_nets, np.nan, float)
    p_raw = np.full(n_nets, np.nan, float)

    xpos = np.arange(n_arrays) + 1
    x_min, x_max = 0.5, n_arrays + 0.5

    # ----- per-network loop -----
    for idx_net, net in enumerate(uniq_nets):
        r = idx_net // n_cols
        c = idx_net % n_cols
        ax = axes[r, c]

        groups = []
        for arr in arrays:
            vals = arr[networks0 == net]
            vals = vals[np.isfinite(vals)]
            groups.append(vals)

        # ANOVA
        try:
            F, p = f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        F_vals[idx_net] = F
        p_raw[idx_net] = p

        # plotting
        if kind == "violin":
            v = ax.violinplot(
                groups,
                positions=xpos,
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
                yerr=stds,
                color=array_colors,
                edgecolor="k",
                linewidth=0.7,
                alpha=0.9,
                capsize=3,
            )

        elif kind == "raincloud":
            max_width = 0.4
            jitter = 0.08
            alpha_kde = 0.6
            alpha_pts = 0.7

            for j, (g_vals, color) in enumerate(zip(groups, array_colors), start=1):
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
                            x_left = j - density
                            x_right = np.full_like(y_grid, j)
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

                x_jitter = j + (np.random.rand(g_vals.size) - 0.5) * 2 * jitter
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
                    j + max_width * 0.6,
                    med,
                    color="k",
                    marker="_",
                    s=80,
                    linewidths=1.2,
                    zorder=3,
                )
        else:
            raise ValueError("kind must be 'violin', 'bar', or 'raincloud'.")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(xpos)
        ax.set_xticklabels(array_labels, rotation=45, ha="right", fontsize=7)

        if c == 0:
            ax.set_ylabel(y_label)

        net_label = network_labels[int(net)]
        ax.set_title(net_label, fontsize=9)

    # hide unused subplots
    for idx in range(n_nets, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    # ----- FDR -----
    p_fdr, sig_mask = _fdr_bh(p_raw, alpha=fdr_alpha)

    for idx_net, net in enumerate(uniq_nets):
        r = idx_net // n_cols
        c = idx_net % n_cols
        ax = axes[r, c]

        F = F_vals[idx_net]
        p = p_raw[idx_net]
        q = p_fdr[idx_net]
        star = " *" if sig_mask[idx_net] else ""
        txt = f"F={F:.2f}, p={p:.2g}\nq={q:.2g}{star}"
        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=0.4),
        )

    # legend for arrays
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
    fig.legend(
        handles=handles,
        title="Index",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(n_arrays, 4),
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if fname is None:
        fname = f"{name}_{kind}_byNetwork.svg"
    outpath = out_dir / fname
    fig.savefig(outpath, bbox_inches="tight", format="svg")
    plt.close(fig)

    print("Saved:", outpath)
    return str(outpath), F_vals, p_raw, p_fdr, sig_mask