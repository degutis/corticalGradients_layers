# --- add near your imports (reuse helpers if you already pasted them) ---
import xml.etree.ElementTree as ET
import json

def _present_keys(img, lo=1, hi=400):
    vals = np.unique(img.get_fdata().astype(int))
    return vals[(vals >= lo) & (vals <= hi)]

def _extract_wb_label_table_from_header(img):
    name_map = {}
    for ext in getattr(img.header, "extensions", []):
        payload = ext.get_content()
        txt = payload.decode("utf-8", "ignore") if isinstance(payload, (bytes, bytearray)) else str(payload)
        if "<Label" not in txt:
            continue
        try:
            root = ET.fromstring(txt)
        except ET.ParseError:
            continue
        for lab in root.iter("Label"):
            key_attr = (lab.get("Key") or lab.get("Index") or lab.get("Number") or lab.get("key"))
            if key_attr is None:
                continue
            key = int(key_attr)
            name = (lab.text or "").strip() or lab.get("Name") or f"Label_{key}"
            name_map[key] = name
    return name_map

def _parse_wb_label_table_txt(txt_path):
    if not os.path.exists(txt_path):
        return {}
    mapping = {}
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines) - 1:
        name = lines[i]
        parts = lines[i + 1].split()
        try:
            key = int(parts[0]); mapping[key] = name
        except Exception:
            pass
        i += 2
    return mapping

def _load_schaefer_labelmap_for_subject(ref_dir):
    mapping = {}
    for fname in ("schaefer_L_in-func.nii", "schaefer_R_in-func.nii"):
        path = os.path.join(ref_dir, fname)
        if os.path.exists(path):
            try:
                img = nib.load(path)
                mapping.update(_extract_wb_label_table_from_header(img))
            except Exception:
                pass
    if mapping:
        return mapping
    surf_table = os.path.join(ref_dir, "surf_table.txt")
    mapping = _parse_wb_label_table_txt(surf_table)
    if mapping:
        return mapping
    L_tab = os.path.join(ref_dir, "schaefer_L_vol_table.txt")
    R_tab = os.path.join(ref_dir, "schaefer_R_vol_table.txt")
    mapping = {}
    mapping.update(_parse_wb_label_table_txt(L_tab))
    mapping.update(_parse_wb_label_table_txt(R_tab))
    return mapping  # may be empty


# ---------- NEW: parcels×layers×time×subjects extractor ----------
def extract_vis12_PLTS_all_subjects(
    subjects,
    runNum="run1",
    analysis_type="largeGap_Schaefer",
    layer_01=True,
    LAYERS_TO_USE=(1, 3),  # deep & superficial. For whole ribbon: (None,)
    OUT_BASE="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations"
):
    """
    Returns and saves an array shaped (parcels=4, layers=len(LAYERS_TO_USE), time, subjects).
    Parcel order: [1, 2, 201, 202] = LH_Vis_1, LH_Vis_2, RH_Vis_1, RH_Vis_2.
    """
    VIS_KEYS = [1, 2, 201, 202]
    per_subj = []
    t_common = None

    for subj in subjects:
        print(f"\n=== {subj} ===")
        bold_path  = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/{subj}/merged_residuals_{runNum}.nii"
        layer_path = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subj}/ln_depths_equivol.nii"
        R_path     = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subj}/schaefer_R_in-func.nii"
        L_path     = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subj}/schaefer_L_in-func.nii"

        # load
        layer_img = nib.load(layer_path)
        R_img = nib.load(R_path)
        L_img = nib.load(L_path)
        bold_img = nib.load(bold_path)

        layer = layer_img.get_fdata()
        R = R_img.get_fdata()
        L = L_img.get_fdata()
        bold = bold_img.get_fdata()

        # sanity + merge
        if np.any((L > 0) & (R > 0)):
            raise RuntimeError(f"[{subj}] LH and RH Schaefer volumes overlap.")
        atlas = np.where(L > 0, L, 0) + np.where(R > 0, R, 0)

        if (layer.shape != atlas.shape) or (layer.shape != bold.shape[:-1]):
            raise ValueError(f"[{subj}] shape mismatch. layer:{layer.shape}, atlas:{atlas.shape}, bold:{bold.shape}")

        # build layer masks
        if LAYERS_TO_USE == (None,):
            layers = (None,)
            layer_masks = {None: (layer > 0) & (layer < 1.0)}  # whole ribbon
        else:
            layers = tuple(LAYERS_TO_USE)
            layer_bin = np.zeros_like(layer, dtype=np.uint8)
            if layer_01:
                if analysis_type == "smallGap_Schaefer":
                    layer_bin[(layer > 0)   & (layer <= 0.3)] = 1
                    layer_bin[(layer > 0.4) & (layer <= 0.6)] = 2
                    layer_bin[(layer > 0.7) & (layer <  1.0)] = 3
                elif analysis_type == "noGap":
                    layer_bin[(layer > 0)   & (layer <= 0.4)] = 1
                    layer_bin[(layer > 0.4) & (layer <= 0.6)] = 2
                    layer_bin[(layer > 0.6) & (layer <  1.0)] = 3
                elif analysis_type == "largeGap_Schaefer":
                    layer_bin[(layer > 0) & (layer <= 0.2)] = 1
                    layer_bin[(layer > 0.4) & (layer <= 0.6)] = 2
                    layer_bin[(layer > 0.8) & (layer < 1)] = 3
            else:
                layer_bin[(layer > 1)  & (layer <= 3)]  = 1
                layer_bin[(layer > 5)  & (layer <= 7)]  = 2
                layer_bin[(layer > 9)  & (layer <= 11)] = 3
            layer_masks = {lay: (layer_bin == lay) for lay in layers}

        # per subject matrix: (parcels=4, layers=len(layers), time)
        nT = bold.shape[-1]
        subj_mat = np.zeros((len(VIS_KEYS), len(layers), nT), dtype=float)

        for p_idx, key in enumerate(VIS_KEYS):
            parcel_mask = (atlas == key)
            for l_idx, lay in enumerate(layers):
                mask = parcel_mask if lay is None else (parcel_mask & layer_masks[lay])
                if np.any(mask):
                    vox_ts = bold[mask]        # (n_vox, time)
                    mean_ts = np.nanmean(vox_ts, axis=0)
                    subj_mat[p_idx, l_idx, :] = zscore(mean_ts, axis=0)
                else:
                    subj_mat[p_idx, l_idx, :] = 0.0

        per_subj.append(subj_mat)
        t_common = nT if t_common is None else min(t_common, nT)

    # truncate to shortest time across subjects (or change to strict check if you prefer)
    per_subj = [m[:, :, :t_common] for m in per_subj]

    # stack → (parcels, layers, time, subjects)
    group_mat = np.stack(per_subj, axis=-1)

    # --- save ---
    out_group_dir = os.path.join(OUT_BASE, analysis_type, "group")
    os.makedirs(out_group_dir, exist_ok=True)
    layer_tag = "layers{}_{}".format(*LAYERS_TO_USE) if LAYERS_TO_USE != (None,) else "allribbon"
    npy_path = os.path.join(out_group_dir, f"VIS12_PLTS_{layer_tag}_{runNum}_4xLxTxS.npy")
    np.save(npy_path, group_mat)
    print(f"\nSaved → {npy_path} | shape={group_mat.shape} (parcels, layers, time, subjects)")

    # metadata: parcel names & layer ids
    # get names from any subject’s ref dir (assume consistent across subjects)
    ref_dir0 = f"/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/{subjects[0]}"
    name_map = _load_schaefer_labelmap_for_subject(ref_dir0)
    parcel_labels = [name_map.get(k, f"Label_{k}") for k in VIS_KEYS]
    layer_list = ["allribbon"] if LAYERS_TO_USE == (None,) else [int(x) for x in layers]

    order_tsv = os.path.join(out_group_dir, f"VIS12_PLTS_{layer_tag}_{runNum}_order.tsv")
    with open(order_tsv, "w") as f:
        f.write("parcel_idx\tparcel_key\tparcel_label\tlayer_idx\tlayer_id\n")
        for p_idx, key in enumerate(VIS_KEYS):
            for l_idx, lay in enumerate(layer_list):
                f.write(f"{p_idx}\t{key}\t{parcel_labels[p_idx]}\t{l_idx}\t{lay}\n")

    meta_json = os.path.join(out_group_dir, f"VIS12_PLTS_{layer_tag}_{runNum}_meta.json")
    with open(meta_json, "w") as f:
        json.dump({
            "subjects": subjects,
            "shape": {
                "parcels": group_mat.shape[0],
                "layers": group_mat.shape[1],
                "time": group_mat.shape[2],
                "subjects": group_mat.shape[3],
            },
            "parcel_keys": VIS_KEYS,
            "parcel_labels": parcel_labels,
            "layers": layer_list,
            "run": runNum,
            "analysis_type": analysis_type,
        }, f, indent=2)

    print(f"Saved order → {order_tsv}")
    print(f"Saved metadata → {meta_json}")
    return group_mat, npy_path, order_tsv, meta_json


# -------- example call (your subject list) --------
subjects = ["sub-LAM001","sub-LAM002","sub-LAM003","sub-LAM004",
            "sub-LAM005","sub-LAM006","sub-LAM009","sub-LAM011"]

# parcels×layers×time×subjects; deep & superficial only
_ = extract_vis12_PLTS_all_subjects(
        subjects,
        runNum="run1",
        analysis_type="largeGap_Schaefer",
        layer_01=True,
        LAYERS_TO_USE=(1,3),
        OUT_BASE="/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/correlations"
    )