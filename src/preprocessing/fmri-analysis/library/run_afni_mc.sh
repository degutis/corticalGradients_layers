#!/bin/bash

# outdir="${3:-.}"
# mkdir -p "$outdir"

# fNameBase=$(remove_ext $1)

# nVols=$(fslinfo ${fNameBase} | grep ^dim4 | awk '{print $2}')

# if [ -z "$2" ]
# then
#     base=$(expr ${nVols} / 2)
# else
#     base=$2
# fi

# 3dvolreg -prefix "$outdir/${fNameBase}_mc.nii" \
#          -Fourier \
#          -float \
#          -base $base \
#          -dfile "$outdir/${fNameBase}_mc.par" \
#          -maxdisp1D "$outdir/${fNameBase}_mc_maxdisp" \
#          ${fNameBase}.n
#         #  -dfile ${fNameBase}_mc.par \
#         #  -maxdisp1D ${fNameBase}_mc_maxdisp \


# awk -F' ' '{s=$2;$1=$3;$2=-$4;$3=s;t=$5;$4=$6;$5=-$7;$6=t;$7=$8;$8=$9;$9=""}1' OFS=' ' \
#     ${fNameBase}_mc.par >  ${fNameBase}_mc_reordered.par
# fsl_tsplot -i ${fNameBase}_mc_reordered.par -t 'AFNI estimated rotations (radians)' \
#            -u 1 --start=1 --finish=3 -a x,y,z -w 640 -h 144 -o "$outdir/${fNameBase}_rot.png" 
# fsl_tsplot -i ${fNameBase}_mc_reordered.par -t 'AFNI estimated translations (mm)' \
#            -u 1 --start=4 --finish=6 -a x,y,z -w 640 -h 144 -o "$outdir/${fNameBase}_trans.png"
# fsl_tsplot -i ${fNameBase}_mc_maxdisp,${fNameBase}_mc_maxdisp_delt -t 'AFNI estimated max displacement (mm)' \
#            -u 1 -w 640 -h 144 -a absolute,relative -o "$outdir/${fNameBase}_disp.png"

# Usage: run_afni_mc.sh <infile> <base_vol> <outdir>
#   e.g. run_afni_mc.sh sub-04_run-01_bold.nii 51 /path/to/outdir

#!/usr/bin/env bash
# run_afni_mc.sh <infile> <base_selector> <outdir>
set -euo pipefail

infile="$1"
base_sel="$2"
outdir="$3"
mkdir -p "$outdir"

fname=$(basename "$infile")
# strip .nii or .nii.gz
fbase=${fname%%.nii*}

echo "Running 3dvolreg on $fname → $outdir/${fbase}_mc.nii (base: $base_sel)"

3dvolreg \
  -prefix "${outdir}/${fbase}_mc.nii" \
  -Fourier -float \
  -base "${base_sel}" \
  -dfile "${outdir}/${fbase}_mc.par" \
  -maxdisp1D "${outdir}/${fbase}_mc_maxdisp.1D" \
  "$infile"

# Re-order columns (AFNI roll/pitch/yaw,dS,dL,dP → FSL style)
awk '
{ s=$2; $1=$3; $2=-$4; $3=s;
  t=$5; $4=$6; $5=-$7; $6=t;
  $7=$8; $8=$9; $9="" }1' OFS=' ' \
  "${outdir}/${fbase}_mc.par" > "${outdir}/${fbase}_mc_reordered.par"

# QC plots
fsl_tsplot -i "${outdir}/${fbase}_mc_reordered.par" \
  -t 'AFNI estimated rotations (radians)' \
  -u 1 --start=1 --finish=3 -a x,y,z \
  -w 640 -h 144 \
  -o "${outdir}/${fbase}_rot.png"

fsl_tsplot -i "${outdir}/${fbase}_mc_reordered.par" \
  -t 'AFNI estimated translations (mm)' \
  -u 1 --start=4 --finish=6 -a x,y,z \
  -w 640 -h 144 \
  -o "${outdir}/${fbase}_trans.png"

fsl_tsplot -i "${outdir}/${fbase}_mc_maxdisp.1D,${outdir}/${fbase}_mc_maxdisp.1D_delt" \
  -t 'AFNI estimated max displacement (mm)' \
  -u 1 -w 640 -h 144 \
  -a absolute,relative \
  -o "${outdir}/${fbase}_disp.png"