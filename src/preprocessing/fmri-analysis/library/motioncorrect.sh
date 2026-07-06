# #!/bin/bash
# #
# # motioncorrect.sh run1.nii run2.nii ...
# #
# # - runs motion correction on a list of runs, registering them all to a common robust volume
# # - uses afni and depends on run_afni_mc.sh
# # - writes output as run1_mc.nii run2_mc.nii ...


# # fileNames="$@"

# # Capture the last argument as the output directory:
# outdir="${!#}"
# mkdir -p "$outdir"

# # Capture everything except the last argument as the list of files:
# fileNames=( "${@:1:$#-1}" )

# echo "Output directory is: $outdir"
# echo "Will process files:"
# printf "  %s\n" "${fileNames[@]}"

# # check how many runs and how many volumes
# fileArray=($fileNames)
# numFiles=${#fileArray[@]}
# fileName=${fileArray[0]}
# fBaseName=$(remove_ext ${fileName})
# nVols=$(3dinfo -nti ${fBaseName}.nii)

# # in case we only have 1 run and less then 5 volumes, we take the last volume to account for signal to reach steady-state;
# # in all other cases, we use the best volume to register to and ignore first 3 volumes
# if [[ $numFiles -eq 1 && $nVols -lt 5 ]]; then
#     echo "***** only 1 run with " $(expr $nVols + 1) "volumes - setting to last volume"
#     minoutrun=$(expr 1)
#     minouttr=$(expr $nVols)  
# else
#     # find best volume to register to 
#     echo "***** finding best volume to register to"
#     nVols=()
#     > all_outcount.1D

#     # for fileName in $fileNames
#     # do
#     #     fBaseName=$(remove_ext ${fileName})
#     #     3dToutcount -automask -fraction -polort 5 -legendre \
#     #                 ${fBaseName}.nii'[3..$]' >> all_outcount.1D
#     #     nVols+=($(expr $(3dinfo -nti ${fBaseName}.nii) + 1 - 3))
#     # done
    
#     for fileName in "${fileNames[@]}"
#     do
#     # pass the *full path* including .nii to the worker script:
#     run_afni_mc.sh \
#         "$fileName" \
#         "$mcbase" \
#         "$outdir"
#     done
    
#     ## get run number and TR index for minimum outlier volume
#     minindex=`3dTstat -argmin -prefix -  all_outcount.1D\'`
#     ovals=(`1d_tool.py -set_run_lengths ${nVols[@]} -index_to_run_tr $minindex`)

#     ## save run and TR indices for extraction of vr_base_min_outlier
#     minoutrun=${ovals[0]}
#     minouttr=$(expr ${ovals[1]} + 2)
# fi
# echo "***** min outlier: run $minoutrun, TR $minouttr" | tee min_outlier.txt

# # run motion correction
# fileNameList=($fileNames)
# mcbase=${fileNameList[$(expr ${minoutrun} - 1)]}\[${minouttr}\]
# echo "************* mcbase *******************"
# echo ${mcbase}
# for fileName in $fileNames
# do
#     fBaseName=$(remove_ext ${fileName})
#     run_afni_mc.sh ${fBaseName} ${mcbase} "$outdir"
# done


#!/usr/bin/env bash
# usage: motioncorrect.sh run1.nii [run2.nii ...] /path/to/outdir
set -euo pipefail

libdir="${@: -2:1}"                 # second to last
outdir="${@: -1}"                   # last
mkdir -p "$outdir"

# all but last
mapfile -t files < <(printf '%s\n' "${@:1:$#-2}")

echo "Output directory: $outdir"
printf '  %s\n' "${files[@]}"

first=${files[0]}
nVols=$(3dinfo -nv "$first")

if (( ${#files[@]} == 1 && nVols < 5 )); then
  echo "***** only 1 run with $nVols volumes → using last volume"
  minoutrun=1
  minouttr=$((nVols-1))
else
  echo "***** finding best volume to register to"
  all_out="$outdir/all_outcount.1D"
  : >"$all_out"
  runLengths=()

  for f in "${files[@]}"; do
    rl=$(3dinfo -nv "$f")
    runLengths+=( "$rl" )
    3dToutcount -automask -fraction -polort 5 -legendre \
      "${f}[3..$]" >>"$all_out"
  done

  minidx=$(awk 'NR==1{min=$1;idx=0;next}{if($1<min){min=$1;idx=NR-1}}END{print idx}' "$all_out")

  read -r minoutrun tr0 <<<"$(
    1d_tool.py -set_run_lengths "${runLengths[@]}" -index_to_run_tr "$minidx"
  )"

  # we skipped first 3 vols → base TR index = tr0 + 3 (not 2)
  minouttr=$(( tr0 + 3 ))
fi

echo "***** min outlier: run $minoutrun, TR $minouttr" | tee "$outdir/min_outlier.txt"

basefile=${files[$((minoutrun-1))]}
mcbase="${basefile}[${minouttr}]"

echo "************* mcbase *******************"
echo "$mcbase"
echo "************* OutDir *******************"
echo "$outdir"

for f in "${files[@]}"; do
  "$libdir/run_afni_mc.sh" "$f" "$mcbase" "$outdir"
done