# laminar_rs/flatmaps.py
"""
Flatmap visualisations and simple matrix plotting utilities.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np


def plotMatrix(M: np.ndarray, outputDir: str, name: str) -> None:
    """
    Simple helper to plot a matrix as an SVG.

    Parameters
    ----------
    M : np.ndarray
        2D matrix to visualise.
    outputDir : str
        Directory to write the figure into.
    name : str
        Output filename (e.g. 'Adjacency.svg').
    """
    os.makedirs(outputDir, exist_ok=True)

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["text.usetex"] = False

    plt.figure(figsize=(6, 6))
    plt.imshow(M, cmap="viridis")
    plt.title("Adjacency matrix")
    outpath = os.path.join(outputDir, name)
    plt.savefig(outpath, bbox_inches="tight", format="svg")
    plt.close()


def plotFlatMap(
    M: np.ndarray,
    outdir: str,
    outname: str,
    inputdir_1: str = "",
    inputdir_2: str = "/home/degutis/repos/HumanCorticalParcellations",
    cmap: str = "viridis",
    HCP: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    rasterize: bool = False,
) -> str:
    """
    Plot a parcelwise scalar as a 2D cortical flatmap.

    Parameters
    ----------
    M : array-like
        (2 * n_hemi,) array of parcel values [LH parcels..., RH parcels...].
        For HCP-Glasser: length 360.
        For Schaefer400: length 400 (200 per hemisphere).
    outdir : str
        Output directory.
    outname : str
        Output filename (e.g. 'D_interFlatMap.svg' or .png).
    inputdir_1 : str
        Directory containing label files (automatically overridden for HCP/Schaefer).
    inputdir_2 : str
        Directory containing flat surface .surf.gii files.
    cmap : str
        Matplotlib colormap name.
    HCP : bool
        If True, use HCP Glasser atlas; otherwise use Schaefer400.
    vmin, vmax : float or None
        Colour scale limits. If None, inferred from data.
    symmetric : bool
        If True, make the colour scale symmetric around zero.
    rasterize : bool
        If True, rasterise the trisurfaces inside vector output to keep file
        sizes manageable.

    Returns
    -------
    outpath : str
        Path to the saved figure.
    """
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    vals = np.asarray(M).reshape(-1)
    if vals.size % 2 != 0:
        raise ValueError(
            f"Expected even number of parcel values (LH+RH), got {vals.shape}"
        )

    # ---- Map parcel values to per-vertex metrics
    if HCP:
        # Glasser HCP atlas
        inputdir_1 = "/home/degutis/repos/HCP_WB_parcels"
        L_lab = (
            nib.load(
                os.path.join(
                    inputdir_1, "GlasserAtlas.L.32k_fs_LR.label.gii"
                )
            )
            .agg_data()
            .astype(int)
            .squeeze()
        )
        R_lab = (
            nib.load(
                os.path.join(
                    inputdir_1, "GlasserAtlas.R.32k_fs_LR.label.gii"
                )
            )
            .agg_data()
            .astype(int)
            .squeeze()
        )

        metric_L = np.full(L_lab.shape, np.nan, float)
        metric_R = np.full(R_lab.shape, np.nan, float)
        mL = L_lab > 0
        mR = R_lab > 0
        metric_L[mL] = vals[L_lab[mL] - 1]
        metric_R[mR] = vals[180 + L_lab[mR] - 1]

    else:
        # Schaefer 400 parcels (200 per hemisphere)
        inputdir_1 = "/home/degutis/repos/SchaeferAtlas"
        L_lab = (
            nib.load(os.path.join(inputdir_1, "Schaefer400.L.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )
        R_lab = (
            nib.load(os.path.join(inputdir_1, "Schaefer400.R.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )

        n_total = vals.size
        n_hemi = n_total // 2
        if n_total % 2 != 0:
            raise ValueError("M should be [LH parcels..., RH parcels...]")
        uL = np.unique(L_lab[L_lab > 0])
        uR = np.unique(R_lab[R_lab > 0])
        if len(uL) != n_hemi or len(uR) != n_hemi:
            raise ValueError(
                f"Expected {n_hemi} parcels per hemi; got {len(uL)} (L), {len(uR)} (R)."
            )

        L_rank = {k: i for i, k in enumerate(sorted(uL))}
        R_rank = {k: i for i, k in enumerate(sorted(uR))}

        metric_L = np.full(L_lab.shape, np.nan, float)
        metric_R = np.full(R_lab.shape, np.nan, float)
        mL = L_lab > 0
        mR = R_lab > 0
        metric_L[mL] = vals[[L_rank[k] for k in L_lab[mL]]]
        metric_R[mR] = vals[
            n_hemi + np.array([R_rank[k] for k in R_lab[mR]])
        ]

    # ---- Load flat surfaces (coords, faces)
    def load_surf_xy(path: str):
        g = nib.load(path)
        coords = np.asarray(g.darrays[0].data, dtype=float)
        faces = np.asarray(g.darrays[1].data, dtype=int)
        return coords[:, 0], coords[:, 1], faces

    xL, yL, fL = load_surf_xy(
        os.path.join(inputdir_2, "S1200.L.flat.32k_fs_LR.surf.gii")
    )
    xR, yR, fR = load_surf_xy(
        os.path.join(inputdir_2, "S1200.R.flat.32k_fs_LR.surf.gii")
    )

    if metric_L.size != xL.size or metric_R.size != xR.size:
        raise ValueError("Vertex count mismatch (labels vs surface).")

    # ---- Colour scale
    data_all = np.concatenate(
        [metric_L[np.isfinite(metric_L)], metric_R[np.isfinite(metric_R)]]
    )
    if data_all.size == 0:
        raise ValueError("All metric values are NaN.")

    if symmetric:
        m = np.nanmax(np.abs(data_all))
        vmin = -m
        vmax = m
    else:
        if vmin is None:
            vmin = float(np.nanmin(data_all))
        if vmax is None:
            vmax = float(np.nanmax(data_all))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6

    triL = mtri.Triangulation(xL, yL, fL)
    triR = mtri.Triangulation(xR, yR, fR)

    maskL = np.any(np.isnan(metric_L)[fL], axis=1)
    maskR = np.any(np.isnan(metric_R)[fR], axis=1)
    triL.set_mask(maskL)
    triR.set_mask(maskR)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")

    imL = axes[0].tripcolor(triL, metric_L, **kw)
    if rasterize:
        imL.set_rasterized(True)
    axes[0].set_title("Left hemisphere")

    imR = axes[1].tripcolor(triR, metric_R, **kw)
    if rasterize:
        imR.set_rasterized(True)
    axes[1].set_title("Right hemisphere")

    cbar = fig.colorbar(imR, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Value")

    if outpath.lower().endswith(".png"):
        fig.savefig(outpath, dpi=300)
    else:
        fig.savefig(outpath)
    plt.close(fig)

    return outpath


__all__ = ["plotMatrix", "plotFlatMap"]