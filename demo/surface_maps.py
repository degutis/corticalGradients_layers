"""
surface_maps.py
===============
Plot a parcelwise scalar on the cortical surface (Schaefer-400).

The function maps 400 parcel values onto the fs_LR 32k midthickness
surfaces and renders lateral (and optionally medial) views, returning the
Matplotlib figure so it can be shown inline in a notebook as well as saved.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import plotting, surface


def plotSurfaceMap(
    M: np.ndarray,
    outdir: str,
    outname: str,
    schaefer_dir: str = "/SchaeferAtlas",
    surf_dir: str = "/SchaeferAtlas",
    lh_surf_file: str = "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
    rh_surf_file: str = "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = None,
    main_view: Literal[
        "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
    ] = "lateral",
    show_medial: bool = True,
):
    """
    Plot a Schaefer-400 parcelwise scalar on cortical surfaces.

    Parameters
    ----------
    M : array-like
        Parcelwise values, shape (400,), ordered [LH 200..., RH 200...].
    outdir, outname : str
        Output directory and filename for the saved figure.
    schaefer_dir : str
        Directory containing the Schaefer400 L/R .label.gii files.
    surf_dir : str
        Directory containing the fs_LR 32k midthickness .surf.gii files.
    lh_surf_file, rh_surf_file : str
        Names of the LH/RH midthickness surfaces (relative to surf_dir).
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Colour scale limits. If None, inferred from data.
    symmetric : bool
        If True, colour scale is symmetric around zero.
    main_view : str
        View for the top row (e.g. 'lateral' or 'dorsal').
    show_medial : bool
        If True, add a second row of medial views.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure (also saved to ``outdir/outname``).
    """
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    vals = np.asarray(M).reshape(-1)
    if vals.size != 400:
        raise ValueError(f"Expected 400 parcel values (200 LH + 200 RH), got {vals.size}")

    # ------------------------------------------------------------
    # 1. Map parcels -> per-vertex metrics
    # ------------------------------------------------------------
    L_lab = (
        nib.load(os.path.join(schaefer_dir, "Schaefer400.L.label.gii"))
        .agg_data().astype(int).squeeze()
    )
    R_lab = (
        nib.load(os.path.join(schaefer_dir, "Schaefer400.R.label.gii"))
        .agg_data().astype(int).squeeze()
    )

    n_hemi = vals.size // 2

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
    metric_R[mR] = vals[n_hemi + np.array([R_rank[k] for k in R_lab[mR]])]

    # ------------------------------------------------------------
    # 2. Load surfaces
    # ------------------------------------------------------------
    coords_L, faces_L = surface.load_surf_mesh(os.path.join(surf_dir, lh_surf_file))
    coords_R, faces_R = surface.load_surf_mesh(os.path.join(surf_dir, rh_surf_file))

    if metric_L.size != coords_L.shape[0] or metric_R.size != coords_R.shape[0]:
        raise ValueError("Vertex count mismatch (labels vs surface).")

    # ------------------------------------------------------------
    # 3. Colour scale
    # ------------------------------------------------------------
    data_all = np.concatenate(
        [metric_L[np.isfinite(metric_L)], metric_R[np.isfinite(metric_R)]]
    )
    if data_all.size == 0:
        raise ValueError("All metric values are NaN.")

    if symmetric:
        m = float(np.nanmax(np.abs(data_all)))
        vmin, vmax = -m, m
    else:
        if vmin is None:
            vmin = float(np.nanmin(data_all))
        if vmax is None:
            vmax = float(np.nanmax(data_all))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6

    # ------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------
    n_rows = 2 if show_medial else 1
    fig, axes = plt.subplots(
        n_rows, 2,
        subplot_kw={"projection": "3d"},
        figsize=(10, 4.5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axL, axR = axes[0], axes[1]
        all_axes = [axL, axR]
    else:
        axL, axR = axes[0, 0], axes[0, 1]
        axL_med, axR_med = axes[1, 0], axes[1, 1]
        all_axes = [axL, axR, axL_med, axR_med]

    # Top row: main_view
    plotting.plot_surf(
        surf_mesh=(coords_L, faces_L), surf_map=metric_L, hemi="left",
        view=main_view, engine="matplotlib", cmap=cmap,
        symmetric_cmap=symmetric, colorbar=False, vmin=vmin, vmax=vmax,
        bg_on_data=False, axes=axL, figure=fig,
    )
    axL.set_title(f"Left hemisphere ({main_view})")

    plotting.plot_surf(
        surf_mesh=(coords_R, faces_R), surf_map=metric_R, hemi="right",
        view=main_view, engine="matplotlib", cmap=cmap,
        symmetric_cmap=symmetric, colorbar=False, vmin=vmin, vmax=vmax,
        bg_on_data=False, axes=axR, figure=fig,
    )
    axR.set_title(f"Right hemisphere ({main_view})")

    # Second row: medial view
    if show_medial:
        plotting.plot_surf(
            surf_mesh=(coords_L, faces_L), surf_map=metric_L, hemi="left",
            view="medial", engine="matplotlib", cmap=cmap,
            symmetric_cmap=symmetric, colorbar=False, vmin=vmin, vmax=vmax,
            bg_on_data=False, axes=axL_med, figure=fig,
        )
        axL_med.set_title("Left hemisphere (medial)")

        plotting.plot_surf(
            surf_mesh=(coords_R, faces_R), surf_map=metric_R, hemi="right",
            view="medial", engine="matplotlib", cmap=cmap,
            symmetric_cmap=symmetric, colorbar=False, vmin=vmin, vmax=vmax,
            bg_on_data=False, axes=axR_med, figure=fig,
        )
        axR_med.set_title("Right hemisphere (medial)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=all_axes, shrink=0.8, pad=0.03)
    cbar.set_label("Value")

    # Save (also return the figure for inline display)
    if outpath.lower().endswith(".png"):
        fig.savefig(outpath, dpi=300)
    else:
        fig.savefig(outpath)

    return fig
