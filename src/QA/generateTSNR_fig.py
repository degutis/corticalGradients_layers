"""
Improved tSNR computation and visualization.

-----------------------------------------
* Multi-slice mosaics (7 cuts per orientation) instead of a single ortho slice.
* Composite report figure: axial + sagittal + coronal + histogram + stats panel.
* Interactive HTML viewer via ``nilearn.plotting.view_img`` — click through
  the volume in a browser, no Python needed to re-open.
* Rough brain mask (from the mean image) so stats and the colour scale
  aren't dragged around by the air outside the head.
* ``inferno`` colormap — perceptually uniform and intuitive for tSNR
  (darker = worse, brighter = better).
* Cleaner structure: compute / plot / save are separate functions, config
  is at the top, everything is driven by ``main()``.

Dependencies: nibabel, numpy, matplotlib, nilearn (>=0.10).
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import plotting

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STUDY_DIR = Path("/media/miplab-nas2/Data/Karolis/tests/Laminar3DTest_2026")
INPUT_DIR = STUDY_DIR / "derivatives" / "func"
OUTPUT_DIR = STUDY_DIR / "derivatives" / "tsnr"
FIGURE_DIR = OUTPUT_DIR / "figures"
HTML_DIR = OUTPUT_DIR / "interactive"

for d in (OUTPUT_DIR, FIGURE_DIR, HTML_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def compute_tsnr(img_4d, sd_clip=10):
    """Return (tsnr_img, mean_img) from a 4D NIfTI.

    Extreme voxels above ``sd_clip`` SDs from the non-zero mean are zeroed
    out to suppress edge / ghost artefacts that blow up the colour scale.
    """
    data = img_4d.get_fdata()
    mean_signal = data.mean(axis=-1)
    std_signal  = data.std(axis=-1)

    tsnr = np.zeros_like(mean_signal, dtype=np.float32)
    np.divide(mean_signal, std_signal, out=tsnr, where=std_signal > 0)

    nz = tsnr[tsnr > 0]
    if nz.size:
        upper = nz.mean() + sd_clip * nz.std()
        tsnr[tsnr > upper] = 0

    tsnr_img = nib.Nifti1Image(tsnr, affine=img_4d.affine, header=img_4d.header)
    mean_img = nib.Nifti1Image(mean_signal.astype(np.float32), affine=img_4d.affine)
    return tsnr_img, mean_img


def in_brain_values(tsnr_img, mean_img, pct=40):
    """tSNR values inside a rough brain mask — used for stats and vmax."""
    mean_data = mean_img.get_fdata()
    thresh = np.percentile(mean_data[mean_data > 0], pct)
    mask = mean_data > thresh
    return tsnr_img.get_fdata()[mask]


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_tsnr_report(tsnr_img, mean_img, title, out_png):
    """Composite report: 3 orthogonal montages + histogram + stats."""
    vals = in_brain_values(tsnr_img, mean_img)
    vmax = float(np.ceil(np.percentile(vals, 98) / 10) * 10) if vals.size else 50

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.1])

    panels = [
        ("z", "Axial",    gs[0, :3]),
        ("x", "Sagittal", gs[1, :3]),
        ("y", "Coronal",  gs[2, :3]),
    ]
    for mode, label, cell in panels:
        ax = fig.add_subplot(cell)
        plotting.plot_stat_map(
            tsnr_img,
            bg_img=mean_img,
            display_mode=mode,
            cut_coords=7,
            vmin=0,
            vmax=vmax,
            cmap="inferno",
            colorbar=(mode == "z"),
            black_bg=False,
            draw_cross=False,
            annotate=True,
            axes=ax,
        )
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left")

    # Histogram
    ax_hist = fig.add_subplot(gs[0, 3])
    ax_hist.hist(vals, bins=80, color="#d94801", edgecolor="white", linewidth=0.4)
    ax_hist.axvline(np.median(vals), color="#2c3e50", ls="--", lw=1.2,
                    label=f"median = {np.median(vals):.1f}")
    ax_hist.set_xlabel("tSNR")
    ax_hist.set_ylabel("voxels")
    ax_hist.set_title("Distribution", fontsize=11, fontweight="bold")
    ax_hist.legend(frameon=False, fontsize=9)
    ax_hist.spines[["top", "right"]].set_visible(False)

    # Stats panel
    ax_stats = fig.add_subplot(gs[1:, 3])
    ax_stats.axis("off")
    stats = {
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std":    float(np.std(vals)),
        "p5":     float(np.percentile(vals, 5)),
        "p95":    float(np.percentile(vals, 95)),
        "voxels": int(vals.size),
    }
    lines = [
        "Summary",
        "─" * 22,
        f"{'mean':<8}{stats['mean']:>10.2f}",
        f"{'median':<8}{stats['median']:>10.2f}",
        f"{'std':<8}{stats['std']:>10.2f}",
        f"{'p5':<8}{stats['p5']:>10.2f}",
        f"{'p95':<8}{stats['p95']:>10.2f}",
        f"{'voxels':<8}{stats['voxels']:>10,}",
        "",
        f"plot vmax = {vmax:.0f}",
    ]
    ax_stats.text(0.0, 1.0, "\n".join(lines),
                  family="monospace", fontsize=10, va="top", ha="left")

    fig.suptitle(f"tSNR report — {title}", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stats, vmax


def save_interactive(tsnr_img, mean_img, title, out_html, vmax):
    """HTML viewer you can open in any browser and scrub through."""
    view = plotting.view_img(
        tsnr_img,
        bg_img=mean_img,
        threshold=0,
        vmin=0,
        vmax=vmax,
        cmap="inferno",
        symmetric_cmap=False,
        opacity=0.8,
        title=f"tSNR — {title}",
    )
    view.save_as_html(str(out_html))


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(("mc.nii", "mc.nii.gz"))
    )
    if not files:
        print(f"No *mc.nii(.gz) files found in {INPUT_DIR}")
        return

    for fname in files:
        print(f"\n→ {fname}")
        img = nib.load(INPUT_DIR / fname)

        tsnr_img, mean_img = compute_tsnr(img)

        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        tsnr_path = OUTPUT_DIR / f"tsnr_{stem}.nii.gz"
        nib.save(tsnr_img, tsnr_path)

        png_path  = FIGURE_DIR / f"tsnr_{stem}.png"
        html_path = HTML_DIR   / f"tsnr_{stem}.html"

        stats, vmax = plot_tsnr_report(tsnr_img, mean_img, stem, png_path)
        save_interactive(tsnr_img, mean_img, stem, html_path, vmax)

        print(f"   NIfTI → {tsnr_path.name}")
        print(f"   PNG   → {png_path.name}  (median={stats['median']:.1f})")
        print(f"   HTML  → {html_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()