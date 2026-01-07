import os
from typing import Literal, Optional, Sequence

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import plotting, surface


def plotSurfaceMap(
    M: np.ndarray,
    outdir: str,
    outname: str,
    HCP: bool = False,
    # paths mirroring your flatmap code
    schaefer_dir: str = "/home/degutis/repos/SchaeferAtlas",
    glasser_dir: str = "/home/degutis/repos/HCP_WB_parcels",
    surf_dir: str = "/home/degutis/repos/HumanCorticalParcellations",
    lh_surf_file: str = "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
    rh_surf_file: str = "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    rasterize: bool = False,  # kept for API symmetry; nilearn ignores this
    main_view: Literal[
        "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
    ] = "lateral",
    show_medial: bool = True,
) -> str:
    """
    Plot a parcelwise scalar on cortical surfaces (lateral, and optionally medial)
    using Nilearn.

    Parameters
    ----------
    M : array-like
        Parcelwise values.
        - If HCP=True : shape (360,), [LH 180..., RH 180...] (Glasser).
        - If HCP=False: shape (400,), [LH 200..., RH 200...] (Schaefer400).
    outdir : str
        Output directory.
    outname : str
        Output filename.
    HCP : bool
        If True, use Glasser360; else Schaefer400.
    schaefer_dir, glasser_dir : str
        Directories with label .gii files.
    surf_dir : str
        Directory with HCP fs_LR 32k surface .surf.gii files.
    lh_surf_file, rh_surf_file : str
        Names of LH/RH midthickness surfaces (relative to surf_dir).
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Colour scale limits. If None, inferred from data.
    symmetric : bool
        If True, colour scale symmetric around zero.
    rasterize : bool
        Present for compatibility; not used by nilearn.
    main_view : str
        View for the top row (e.g., 'lateral' or 'dorsal').
    show_medial : bool
        If True, also plot medial views in a second row.

    Returns
    -------
    outpath : str
        Path to saved figure.
    """
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    vals = np.asarray(M).reshape(-1)
    if vals.size % 2 != 0:
        raise ValueError("Expected even number of parcel values (LH+RH).")

    # ------------------------------------------------------------
    # 1. Map parcels -> per-vertex metrics (same logic as flatmap)
    # ------------------------------------------------------------
    if HCP:
        # Glasser360: 180 per hemi
        if vals.size != 360:
            raise ValueError(
                f"HCP=True expects 360 values (180 LH + 180 RH), got {vals.size}"
            )
        n_hemi = 180

        L_lab = (
            nib.load(
                os.path.join(glasser_dir, "GlasserAtlas.L.32k_fs_LR.label.gii")
            )
            .agg_data()
            .astype(int)
            .squeeze()
        )
        R_lab = (
            nib.load(
                os.path.join(glasser_dir, "GlasserAtlas.R.32k_fs_LR.label.gii")
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
        metric_R[mR] = vals[n_hemi + R_lab[mR] - 1]

    else:
        # Schaefer400: 200 per hemi
        if vals.size != 400:
            raise ValueError(
                f"HCP=False expects 400 values (200 LH + 200 RH), got {vals.size}"
            )

        L_lab = (
            nib.load(os.path.join(schaefer_dir, "Schaefer400.L.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )
        R_lab = (
            nib.load(os.path.join(schaefer_dir, "Schaefer400.R.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )

        n_total = vals.size
        n_hemi = n_total // 2

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
    # 2. Load surfaces via nilearn
    # ------------------------------------------------------------
    lh_surf_path = os.path.join(surf_dir, lh_surf_file)
    rh_surf_path = os.path.join(surf_dir, rh_surf_file)

    coords_L, faces_L = surface.load_surf_mesh(lh_surf_path)
    coords_R, faces_R = surface.load_surf_mesh(rh_surf_path)

    if metric_L.size != coords_L.shape[0] or metric_R.size != coords_R.shape[0]:
        raise ValueError("Vertex count mismatch (labels vs surface).")

    # ------------------------------------------------------------
    # 3. Colour scale (same semantics as your flatmap)
    # ------------------------------------------------------------
    data_all = np.concatenate(
        [metric_L[np.isfinite(metric_L)], metric_R[np.isfinite(metric_R)]]
    )
    if data_all.size == 0:
        raise ValueError("All metric values are NaN.")

    if symmetric:
        m = float(np.nanmax(np.abs(data_all)))
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

    # ------------------------------------------------------------
    # 4. Plot with nilearn.plot_surf
    # ------------------------------------------------------------
    n_rows = 2 if show_medial else 1
    fig, axes = plt.subplots(
        n_rows,
        2,
        subplot_kw={"projection": "3d"},
        figsize=(10, 4.5 * n_rows),
        constrained_layout=True,
    )
    # axes handling convenience
    if n_rows == 1:
        axL, axR = axes[0], axes[1]
        all_axes = [axL, axR]
    else:
        axL = axes[0, 0]
        axR = axes[0, 1]
        axL_med = axes[1, 0]
        axR_med = axes[1, 1]
        all_axes = [axL, axR, axL_med, axR_med]

    # Top row: main_view
    plotting.plot_surf(
        surf_mesh=(coords_L, faces_L),
        surf_map=metric_L,
        hemi="left",
        view=main_view,
        engine="matplotlib",
        cmap=cmap,
        symmetric_cmap=symmetric,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        bg_on_data=False,
        axes=axL,
        figure=fig,
    )
    axL.set_title(f"Left hemisphere ({main_view})")

    plotting.plot_surf(
        surf_mesh=(coords_R, faces_R),
        surf_map=metric_R,
        hemi="right",
        view=main_view,
        engine="matplotlib",
        cmap=cmap,
        symmetric_cmap=symmetric,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        bg_on_data=False,
        axes=axR,
        figure=fig,
    )
    axR.set_title(f"Right hemisphere ({main_view})")

    # Second row: medial view (if requested)
    if show_medial:
        plotting.plot_surf(
            surf_mesh=(coords_L, faces_L),
            surf_map=metric_L,
            hemi="left",
            view="medial",
            engine="matplotlib",
            cmap=cmap,
            symmetric_cmap=symmetric,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            bg_on_data=False,
            axes=axL_med,
            figure=fig,
        )
        axL_med.set_title("Left hemisphere (medial)")

        plotting.plot_surf(
            surf_mesh=(coords_R, faces_R),
            surf_map=metric_R,
            hemi="right",
            view="medial",
            engine="matplotlib",
            cmap=cmap,
            symmetric_cmap=symmetric,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            bg_on_data=False,
            axes=axR_med,
            figure=fig,
        )
        axR_med.set_title("Right hemisphere (medial)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=all_axes, shrink=0.8, pad=0.03)
    cbar.set_label("Value")

    # Save
    if outpath.lower().endswith(".png"):
        fig.savefig(outpath, dpi=300)
    else:
        fig.savefig(outpath)
    plt.close(fig)

    return outpath



def plotSurfaceMap_LH_gradients(
    M: np.ndarray,
    outdir: str,
    outname: str,
    HCP: bool = False,
    # atlas / surface paths
    schaefer_dir: str = "/home/degutis/repos/SchaeferAtlas",
    glasser_dir: str = "/home/degutis/repos/HCP_WB_parcels",
    surf_dir: str = "/home/degutis/repos/HumanCorticalParcellations",
    lh_surf_file: str = "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    rasterize: bool = False,   # kept for API symmetry; nilearn ignores this
    view: Literal[
        "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
    ] = "lateral",
    grad_labels: Optional[Sequence[str]] = None,
    layer_labels: Optional[Sequence[str]] = None,
    same_vmin_vmax_across_layers: bool = False,
) -> str:


    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D (parcels × gradients), got shape {M.shape}.")

    # ------------------------------------------------------------
    # 0. Orient M so that: shape = (L * N_per_layer, n_grad)
    # ------------------------------------------------------------
    if HCP:
        N_per_layer = 360
    else:
        N_per_layer = 400

    n0, n1 = M.shape
    if n0 % N_per_layer == 0:
        L = n0 // N_per_layer
        M_oriented = M
        n_grad = n1
    elif n1 % N_per_layer == 0:
        L = n1 // N_per_layer
        M_oriented = M.T
        n_grad = n0
    else:
        raise ValueError(
            f"Neither dimension is a multiple of {N_per_layer}; "
            f"got shape {M.shape}."
        )

    if L != 3:
        raise ValueError(
            f"Expected 3 layers, but got {L} from shape {M.shape} "
            f"and N_per_layer={N_per_layer}."
        )

    M_layers = M_oriented.reshape(L, N_per_layer, n_grad)
    M_layers = M_layers[::-1, :, :]  # [superficial, middle, deep]

    if grad_labels is None:
        grad_labels = [f"Grad {i+1}" for i in range(n_grad)]
    elif len(grad_labels) != n_grad:
        raise ValueError("grad_labels must have same length as number of gradients.")

    if layer_labels is None:
        layer_labels = ["Superficial", "Middle", "Deep"]
    elif len(layer_labels) != L:
        raise ValueError("layer_labels must have length equal to number of layers (3).")

    # ------------------------------------------------------------
    # 1. Build LH parcel->vertex mapping for each layer
    # ------------------------------------------------------------
    if HCP:
        L_lab = (
            nib.load(
                os.path.join(glasser_dir, "GlasserAtlas.L.32k_fs_LR.label.gii")
            )
            .agg_data()
            .astype(int)
            .squeeze()
        )

        mL = L_lab > 0
        label_ids = L_lab[mL] - 1  # 0..179 indices into LH portion

        metric_L_layers = np.full(
            (L, L_lab.shape[0], n_grad), np.nan, dtype=float
        )

        for ell in range(L):
            vals_LH = M_layers[ell, 0:180, :]  # (180, n_grad)
            metric_L_layers[ell, mL, :] = vals_LH[label_ids, :]

    else:
        L_lab = (
            nib.load(os.path.join(schaefer_dir, "Schaefer400.L.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )
        R_lab = (
            nib.load(os.path.join(schaefer_dir, "Schaefer400.R.label.gii"))
            .agg_data()
            .astype(int)
            .squeeze()
        )

        uL = np.unique(L_lab[L_lab > 0])
        uR = np.unique(R_lab[R_lab > 0])
        n_hemi = N_per_layer // 2  # 200
        if len(uL) != n_hemi or len(uR) != n_hemi:
            raise ValueError(
                f"Expected {n_hemi} parcels per hemi; got {len(uL)} (L), {len(uR)} (R)."
            )

        L_rank = {k: i for i, k in enumerate(sorted(uL))}
        mL = L_lab > 0
        label_ids_L = np.array([L_rank[k] for k in L_lab[mL]], int)  # 0..199

        metric_L_layers = np.full(
            (L, L_lab.shape[0], n_grad), np.nan, dtype=float
        )

        for ell in range(L):
            vals_LH = M_layers[ell, 0:n_hemi, :]  # (200, n_grad)
            metric_L_layers[ell, mL, :] = vals_LH[label_ids_L, :]

    # ------------------------------------------------------------
    # 2. Load LH midthickness surface
    # ------------------------------------------------------------
    lh_surf_path = os.path.join(surf_dir, lh_surf_file)
    coords_L, faces_L = surface.load_surf_mesh(lh_surf_path)

    if metric_L_layers.shape[1] != coords_L.shape[0]:
        raise ValueError(
            f"Vertex count mismatch: labels={metric_L_layers.shape[1]}, "
            f"surface={coords_L.shape[0]}."
        )

    # ------------------------------------------------------------
    # 3. Global vs per-gradient / per-subplot colour scale
    # ------------------------------------------------------------
    use_global_scale = (vmin is not None) or (vmax is not None)

    if use_global_scale:
        data_all = metric_L_layers[np.isfinite(metric_L_layers)]
        if data_all.size == 0:
            raise ValueError("All metric values are NaN.")

        if vmin is None:
            vmin = float(np.nanmin(data_all))
        if vmax is None:
            vmax = float(np.nanmax(data_all))
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
    else:
        if same_vmin_vmax_across_layers:
            grad_vmin = np.zeros(n_grad, dtype=float)
            grad_vmax = np.zeros(n_grad, dtype=float)
            for g in range(n_grad):
                d = metric_L_layers[:, :, g]
                d = d[np.isfinite(d)]
                if d.size == 0:
                    grad_vmin[g], grad_vmax[g] = -1e-6, 1e-6
                else:
                    if symmetric:
                        mval = float(np.nanmax(np.abs(d)))
                        grad_vmin[g], grad_vmax[g] = -mval, mval
                    else:
                        mn = float(np.nanmin(d))
                        mx = float(np.nanmax(d))
                        if mn == mx:
                            mn -= 1e-6
                            mx += 1e-6
                        grad_vmin[g], grad_vmax[g] = mn, mx
        else:
            grad_vmin = grad_vmax = None  # per-subplot, computed in loop

    # ------------------------------------------------------------
    # 4. Figure layout: 3 rows (layers) × n_grad columns
    #    -> reduce figsize and tighten spacing
    # ------------------------------------------------------------
    n_rows, n_cols = L, n_grad

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        subplot_kw={"projection": "3d"},
        figsize=(2.5 * n_cols, 2.2 * n_rows),  # smaller panels
        constrained_layout=False,              # we'll control spacing manually
    )

    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    sm_list = []  # to store (ax, ScalarMappable)

    # ------------------------------------------------------------
    # 5. Plot each (layer, gradient) on LH surface
    # ------------------------------------------------------------
    for ell in range(L):
        for g in range(n_grad):
            ax = axes[ell, g]
            data = metric_L_layers[ell, :, g]

            if use_global_scale:
                this_vmin, this_vmax = vmin, vmax
            else:
                if same_vmin_vmax_across_layers:
                    this_vmin, this_vmax = grad_vmin[g], grad_vmax[g]
                else:
                    d = data[np.isfinite(data)]
                    if d.size == 0:
                        this_vmin, this_vmax = -1e-6, 1e-6
                    else:
                        if symmetric:
                            mval = float(np.nanmax(np.abs(d)))
                            this_vmin, this_vmax = -mval, mval
                        else:
                            mn = float(np.nanmin(d))
                            mx = float(np.nanmax(d))
                            if mn == mx:
                                mn -= 1e-6
                                mx += 1e-6
                            this_vmin, this_vmax = mn, mx

            plotting.plot_surf(
                surf_mesh=(coords_L, faces_L),
                surf_map=data,
                hemi="left",
                view=view,
                engine="matplotlib",
                cmap=cmap,
                symmetric_cmap=False,
                colorbar=False,
                vmin=this_vmin,
                vmax=this_vmax,
                bg_on_data=False,
                axes=ax,
                figure=fig,
            )

            if ell == 0:
                ax.set_title(grad_labels[g], fontsize=9)

            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=this_vmin, vmax=this_vmax)
            )
            sm.set_array([])
            sm_list.append((ax, sm))

    # ------------------------------------------------------------
    # 6. Tighten whitespace between panels
    # ------------------------------------------------------------
    # Shrink outer margins and reduce inter-panel spacing
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,   # horizontal spacing between subplots
        hspace=0,   # vertical spacing between subplots
    )

    # ------------------------------------------------------------
    # 7. Colorbars
    # ------------------------------------------------------------
    if use_global_scale:
        sm_global = sm_list[0][1]
        all_axes = axes.ravel().tolist()
        cbar = fig.colorbar(sm_global, ax=all_axes, shrink=0.8, pad=0.02)
        cbar.set_label("Value", fontsize=9)

    if outpath.lower().endswith(".svg"):
        fig.savefig(outpath)
    else:
        fig.savefig(outpath, dpi=300)
    plt.close(fig)

    return outpath