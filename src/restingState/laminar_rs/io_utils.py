# laminar_rs/io_utils.py
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np

from .config import LaminarConfig


def group_files_by_layer(cfg: LaminarConfig) -> Dict[int, List[Path]]:
    """
    Group .npy time-series files by layer number parsed from filename suffix.

    Expected filename convention: *_<layerNum>.npy
    """
    layer_groups: Dict[int, List[Path]] = defaultdict(list)

    for f in cfg.npy_files():
        name = f.name
        try:
            layer_str = name.split("_")[-1].replace(".npy", "")
            layer_idx = int(layer_str)
            layer_groups[layer_idx].append(f)
        except Exception as e:
            # warnings.warn(
            #     f"Could not extract layer number from filename {name!r}: {e}",
            #     RuntimeWarning,
            # )
            # print(f"Could not extract layer number from filename: {f}")
            continue

    return layer_groups


def group_files_by_run(layer_groups: Dict[int, List[Path]]) -> Dict[str, List[Path]]:
    """
    Group files by 'run' id (assumed to be 2nd token in filename).

    Example filename: sub-01_run2_layer1.npy
    """
    run_groups: Dict[str, List[Path]] = defaultdict(list)
    for files in layer_groups.values():
        for fp in files:
            parts = fp.name.split("_")
            if len(parts) < 2:
                raise ValueError(f"Cannot parse run id from {fp.name}")
            run_id = parts[1]  # e.g. 'run2'
            run_groups[run_id].append(fp)
    return run_groups


def load_and_concat_layer(cfg: LaminarConfig,
                          file_list: List[Path]) -> np.ndarray:
    """
    Load all time-series in file_list and concatenate along time axis.

    Returns
    -------
    data : (N, T_total)
    """
    series = [np.load(str(fp)) for fp in file_list]
    data = np.concatenate(series, axis=1)
    if data.shape[0] != cfg.N:
        raise ValueError(f"Expected {cfg.N} parcels, got {data.shape[0]} in {file_list}")
    return data


def load_all_layers_concat_by_layer(cfg: LaminarConfig) -> Tuple[np.ndarray, Dict[int, List[Path]]]:
    """
    Returns
    -------
    data_layers : (N, num_layers, T_total_per_layer)
        Concatenated runs per layer.
    layer_groups : dict
        Mapping layer_num -> list[Path]
    """
    layer_groups = group_files_by_layer(cfg)
    sorted_layers = sorted(layer_groups.items())
    all_concat = []

    for layer_idx, files in sorted_layers:
        arr = load_and_concat_layer(cfg, files)
        all_concat.append(arr)

    data_layers = np.stack(all_concat, axis=1)  # (N, L, T)
    return data_layers, layer_groups