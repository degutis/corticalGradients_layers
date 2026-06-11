# laminar_rs/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class LaminarConfig:
    """
    Configuration for a single subject / dataset.

    Parameters
    ----------
    data_dir : Path-like
        Directory containing the subject's data (e.g. *.npy or correlation files).
    N : int
        Number of parcels per layer.
    set_thresh : float
        Percentage of weakest edges per row to discard (0–100) before
        Fisher z-transform / binarisation, matching `threshold_and_z_transform`.
    num_layers : int
        Number of cortical layers (usually 3).
    atlas_path : Path-like
        Nifti atlas for LaminarRestingState / surface projections.
    """
    data_dir: Path
    N: int
    set_thresh: float = 0.0
    num_layers: int = 3
    # atlas_path: Path = Path("HCP-MMP1_in-func.nii")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        # self.atlas_path = Path(self.atlas_path)

    def npy_files(self) -> List[Path]:
        """Return sorted list of *.npy time-series files (if relevant)."""
        return sorted(self.data_dir.glob("*.npy"))