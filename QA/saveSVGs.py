import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# -------------- config --------------
studyDataDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
subject = "sub-LAM010"
outputDir = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/GIFs/{subject}"
Path(outputDir).mkdir(parents=True, exist_ok=True)

# reproducible random atlas colors
RANDOM_CMAP_SEED = 42
RANDOM_CMAP_COLORS = 256

# treat anything <= this (after normalization) as background for T1 alpha
T1_ALPHA_EPS = 1e-8
# ------------------------------------


# ---------- utilities ----------
def norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    amin, amax = np.nanmin(a), np.nanmax(a)
    if amax > amin:
        return (a - amin) / (amax - amin)
    return np.zeros_like(a, dtype=np.float32)

def figure_no_axes(w: int, h: int, dpi: int = 100):
    """Borderless canvas sized to image pixels."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

def transparent_cmap(base_cmap):
    """Return cmap with fully transparent 'bad' values."""
    try:
        return base_cmap.with_extremes(bad=(0, 0, 0, 0))
    except AttributeError:
        base_cmap.set_bad((0, 0, 0, 0))
        return base_cmap

def build_random_cmap(n: int = RANDOM_CMAP_COLORS, seed: int | None = RANDOM_CMAP_SEED):
    rng = np.random.default_rng(seed)
    colors = rng.random((n, 3))
    colors[0] = [0, 0, 0]  # index 0 is black (won't show if masked)
    return mcolors.ListedColormap(colors)
# -------------------------------


# ---------- load data ----------
t1 = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/fs_t1_in-func.nii").get_fdata()
func = nib.load(f"{studyDataDir}/derivatives/func/{subject}/{subject}_MEAN.nii").get_fdata()
layers = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/ln_depths_equivol.nii").get_fdata()
atlas_L = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/glasser_L_in-func.nii").get_fdata()
atlas_R = nib.load(f"{studyDataDir}/derivatives/ref_anat/{subject}/glasser_R_in-func.nii").get_fdata()
atlas = atlas_L + atlas_R

# sanity checks
for name, arr in {"func": func, "layers": layers, "atlas": atlas}.items():
    if arr.shape != t1.shape:
        raise ValueError(f"Shape mismatch for {name}: {arr.shape} vs T1 {t1.shape}")

# middle axial slice
k = t1.shape[2] // 2
t1_raw, func_raw = t1[:, :, k], func[:, :, k]
layers_raw, atlas_raw = layers[:, :, k], atlas[:, :, k]

# normalize to [0,1]
t1_slice = norm01(t1_raw)
func_slice = norm01(func_raw)
layers_slice = norm01(layers_raw)
atlas_slice = norm01(atlas_raw)

# masks (zeros fully transparent for func/layers/atlas)
t1_mask     = t1_raw     <= 0
func_mask   = func_raw   <= 0
layers_mask = layers_raw <= 0
atlas_mask  = atlas_raw  <= 0

t1_ma     = np.ma.array(t1_slice,     mask=t1_mask)
func_ma   = np.ma.array(func_slice,   mask=func_mask)
layers_ma = np.ma.array(layers_slice, mask=layers_mask)
atlas_ma  = np.ma.array(atlas_slice,  mask=atlas_mask)

H, W = t1_slice.shape

# colormaps per your spec:
# - func: gray
# - layers: red→yellow
# - atlas: random
gray_cmap   = transparent_cmap(plt.get_cmap("gray"))
layers_cmap = transparent_cmap(plt.get_cmap("autumn"))
atlas_cmap  = transparent_cmap(build_random_cmap())


# ---------- save helpers ----------
def save_single(img_ma, cmap, path):
    fig, ax = figure_no_axes(W, H, dpi=100)
    ax.imshow(img_ma, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_overlay(base_ma, base_cmap, over_ma, over_cmap, path):
    fig, ax = figure_no_axes(W, H, dpi=100)
    ax.imshow(base_ma, cmap=base_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(over_ma, cmap=over_cmap, vmin=0, vmax=1, interpolation="nearest")
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_t1_single_with_alpha(t1_norm_slice, path, eps=T1_ALPHA_EPS):
    """
    Render T1 as RGBA and make near-black pixels fully transparent.
    """
    fig, ax = figure_no_axes(W, H, dpi=100)
    rgba = plt.get_cmap("gray")(t1_norm_slice)  # (H,W,4), floats in [0,1]
    rgba[..., 3] = (t1_norm_slice > eps).astype(np.float32)  # alpha: 0 where ~black
    ax.imshow(rgba, interpolation="nearest")
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------- exports ----------
# T1 alone (black→transparent via RGBA alpha)
save_t1_single_with_alpha(t1_slice, f"{outputDir}/T1_slice.svg")

# func (gray), layers (redyellow), atlas (random) — singles
save_single(func_ma,   gray_cmap,   f"{outputDir}/func_slice_gray.svg")
save_single(layers_ma, layers_cmap, f"{outputDir}/layers_slice_redyellow.svg")
save_single(atlas_ma,  atlas_cmap,  f"{outputDir}/atlas_slice_random.svg")

# overlays on T1 (these were already fine)
save_overlay(t1_slice, gray_cmap, func_ma,   gray_cmap,   f"{outputDir}/T1_over_func.svg")
save_overlay(t1_slice, gray_cmap, layers_ma, layers_cmap, f"{outputDir}/T1_over_layers.svg")
save_overlay(t1_slice, gray_cmap, atlas_ma,  atlas_cmap,  f"{outputDir}/T1_over_atlas.svg")