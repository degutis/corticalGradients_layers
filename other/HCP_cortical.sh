#!/bin/bash
set -euo pipefail

# --- bases ---
BASE="$HOME/repos/HCP_WB_parcels"
cd "$BASE"

start_left="$BASE/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii"
start_right="$BASE/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii"

if [[ ! -f "$start_left" ]]; then
  # alternate common filenames (Validation vs Parcellation)
  if [[ -f "$ALT/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii" ]]; then
    start_left="$ALT/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii"
  elif [[ -f "$ALT/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii" ]]; then
    start_left="$ALT/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
  else
    echo "Couldn't find LEFT 32k Glasser dlabel. Put it in $BASE or $ALT."; exit 1
  fi
fi

if [[ ! -f "$start_right" ]]; then
  if [[ -f "$ALT/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii" ]]; then
    start_right="$ALT/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii"
  elif [[ -f "$ALT/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii" ]]; then
    start_right="$ALT/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
  else
    echo "Couldn't find RIGHT 32k Glasser dlabel. Put it in $BASE or $ALT."; exit 1
  fi
fi

# --- spheres & midthickness from your HCP_WB_parcels folder ---
L_SPH_32="$BASE/Q1-Q6_R440.L.sphere.32k_fs_LR.surf.gii"
R_SPH_32="$BASE/Q1-Q6_R440.R.sphere.32k_fs_LR.surf.gii"
L_SPH_164="$BASE/Q1-Q6_R440.L.sphere.164k_fs_LR.surf.gii"
R_SPH_164="$BASE/Q1-Q6_R440.R.sphere.164k_fs_LR.surf.gii"

L_MID_32="$BASE/Q1-Q6_R440.L.midthickness.32k_fs_LR.surf.gii"
R_MID_32="$BASE/Q1-Q6_R440.R.midthickness.32k_fs_LR.surf.gii"
L_MID_164="$BASE/Q1-Q6_R440.L.midthickness.164k_fs_LR.surf.gii"
R_MID_164="$BASE/Q1-Q6_R440.R.midthickness.164k_fs_LR.surf.gii"

# --- sanity checks ---
for f in "$start_left" "$start_right" \
         "$L_SPH_32" "$R_SPH_32" "$L_SPH_164" "$R_SPH_164" \
         "$L_MID_32" "$R_MID_32" "$L_MID_164" "$R_MID_164"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

# --- 1) split the 32k dlabels into hemisphere .label.gii (saved in $BASE) ---
wb_command -cifti-separate "$start_left"  COLUMN -label CORTEX_LEFT  "$BASE/GlasserAtlas.L.32k_fs_LR.label.gii"
wb_command -cifti-separate "$start_right" COLUMN -label CORTEX_RIGHT "$BASE/GlasserAtlas.R.32k_fs_LR.label.gii"

# --- 2) resample 32k -> 164k with area correction using midthickness surfaces ---
wb_command -label-resample "$BASE/GlasserAtlas.L.32k_fs_LR.label.gii" \
  "$L_SPH_32" "$L_SPH_164" ADAP_BARY_AREA "$BASE/GlasserAtlas.L.164k_fs_LR.label.gii" \
  -area-surfs "$L_MID_32" "$L_MID_164"

wb_command -label-resample "$BASE/GlasserAtlas.R.32k_fs_LR.label.gii" \
  "$R_SPH_32" "$R_SPH_164" ADAP_BARY_AREA "$BASE/GlasserAtlas.R.164k_fs_LR.label.gii" \
  -area-surfs "$R_MID_32" "$R_MID_164"

echo "Done. Wrote:"
ls -1 "$BASE"/GlasserAtlas.*.label.gii