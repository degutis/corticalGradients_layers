#!/bin/bash
set -euo pipefail

# --- bases ---
BASE="${BASE:-$HOME/repos/HCP_WB_parcels}"   # keep using your HCP_WB_parcels folder
ALT="${ALT:-$BASE}"                          # optional alternate folder
cd "$BASE"

# --- choose Schaefer variant ---
NPARC="${NPARC:-400}"        # 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
NETWORKS="${NETWORKS:-7}"    # 7 or 17

# Expected filename (fs_LR 32k)
start="$BASE/Schaefer2018_${NPARC}Parcels_${NETWORKS}Networks_order.dlabel.nii"

# Fallbacks (put your file anywhere and this will try ALT too)
if [[ ! -f "$start" ]]; then
  if [[ -f "$ALT/Schaefer2018_${NPARC}Parcels_${NETWORKS}Networks_order.dlabel.nii" ]]; then
    start="$ALT/Schaefer2018_${NPARC}Parcels_${NETWORKS}Networks_order.dlabel.nii"
  else
    echo "Couldn't find Schaefer dlabel (expected Schaefer2018_${NPARC}Parcels_${NETWORKS}Networks_order.dlabel.nii) in $BASE or $ALT" >&2
    exit 1
  fi
fi

# --- spheres & midthickness you already have ---
L_SPH_32="$BASE/Q1-Q6_R440.L.sphere.32k_fs_LR.surf.gii"
R_SPH_32="$BASE/Q1-Q6_R440.R.sphere.32k_fs_LR.surf.gii"
L_SPH_164="$BASE/Q1-Q6_R440.L.sphere.164k_fs_LR.surf.gii"
R_SPH_164="$BASE/Q1-Q6_R440.R.sphere.164k_fs_LR.surf.gii"

L_MID_32="$BASE/Q1-Q6_R440.L.midthickness.32k_fs_LR.surf.gii"
R_MID_32="$BASE/Q1-Q6_R440.R.midthickness.32k_fs_LR.surf.gii"
L_MID_164="$BASE/Q1-Q6_R440.L.midthickness.164k_fs_LR.surf.gii"
R_MID_164="$BASE/Q1-Q6_R440.R.midthickness.164k_fs_LR.surf.gii"

# --- sanity checks ---
for f in "$start" "$L_SPH_32" "$R_SPH_32" "$L_SPH_164" "$R_SPH_164" \
         "$L_MID_32" "$R_MID_32" "$L_MID_164" "$R_MID_164"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

# --- 1) split Schaefer dlabel -> hemisphere .label.gii (32k) ---
outL32="$BASE/Schaefer${NPARC}_${NETWORKS}net.L.32k_fs_LR.label.gii"
outR32="$BASE/Schaefer${NPARC}_${NETWORKS}net.R.32k_fs_LR.label.gii"
wb_command -cifti-separate "$start" COLUMN -label CORTEX_LEFT  "$outL32"
wb_command -cifti-separate "$start" COLUMN -label CORTEX_RIGHT "$outR32"

# --- 2) resample 32k -> 164k with area correction ---
outL164="$BASE/Schaefer${NPARC}_${NETWORKS}net.L.164k_fs_LR.label.gii"
outR164="$BASE/Schaefer${NPARC}_${NETWORKS}net.R.164k_fs_LR.label.gii"

wb_command -label-resample "$outL32" "$L_SPH_32" "$L_SPH_164" ADAP_BARY_AREA "$outL164" \
  -area-surfs "$L_MID_32" "$L_MID_164"

wb_command -label-resample "$outR32" "$R_SPH_32" "$R_SPH_164" ADAP_BARY_AREA "$outR164" \
  -area-surfs "$R_MID_32" "$R_MID_164"

echo "Done. Wrote:"
ls -1 "$outL32" "$outR32" "$outL164" "$outR164"