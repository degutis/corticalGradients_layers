# #!/bin/bash
# set -euo pipefail

# subject=sub-LAM023
# # fs_dir=/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/freesurfer/${subject}/
# fs_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer/${subject}/
# bold_file=${subject}_MEAN.nii
# bold_file_withoutExt=${subject}_MEAN
# func_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/${subject}
# ref_anat_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/${subject}
# # func_dir=/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/func/${subject}
# # ref_anat_dir=/media/miplab-nas2/Data/Karolis/high_res_resting/derivatives/ref_anat/${subject}

# cd ${func_dir}

# if [[ -f "${subject}_MEAN.nii" ]]; then
#     echo "${subject}_MEAN.nii already exists. Skipping computation."
# else
#     echo "Merging all motion-corrected runs for ${subject}…"
#     fslmerge -t all_func_runs.nii.gz *_mc.nii*
#     # fslmerge -t all_func_runs.nii.gz *NORDIC_mc.nii*

#     echo "Computing temporal mean…"
#     fslmaths all_func_runs.nii.gz -Tmean "${subject}_MEAN.nii.gz"

#     echo "Removing intermediate 4D file…"
#     rm all_func_runs.nii.gz

#     gunzip "${subject}_MEAN.nii.gz"  # -k keeps .gz, -f overwrites
#     echo "Done! Output: ${subject}_MEAN.nii.gz"
# fi

# mri_convert ${fs_dir}/mri/brain.mgz fs_T1.nii

# n4bold_file=${bold_file_withoutExt}_n4.nii
# N4BiasFieldCorrection -i ${bold_file} -o ${n4bold_file}
# bold_file=${n4bold_file}

# bold_brain_file=${bold_file_withoutExt}_n4_brain.nii
# # bet ${bold_file} ${bold_brain_file} -f 0.15
# bet ${bold_file} ${bold_brain_file} -f 0.2
# bold_file=${bold_brain_file}
# gunzip ${bold_file}.gz -f

# # bold_brain_file=${bold_file_withoutExt}_n4_brain.nii
# # bold_brain_file_mask=${bold_file_withoutExt}_n4_brain_mask.nii

# # 3dAutomask -clfrac 0.8 -peels 1 \
# #            -prefix       ${bold_brain_file_mask} \
# #            -apply_prefix ${bold_brain_file} \
# #            ${bold_file}



# mask_file=mask_le1500.nii.gz
# fslmaths ${bold_file} -uthr 650 -bin ${mask_file}

# threshold_file=${bold_file_withoutExt}_n4_brain_thresh.nii
# fslmaths ${bold_file} -mas ${mask_file} ${threshold_file}.gz
# gunzip ${threshold_file}.gz -f

# bold_brain_file=${bold_file_withoutExt}_n4_brain2.nii
# bet ${threshold_file} ${bold_brain_file} -f 0.1
# bold_file=${bold_brain_file}
# gunzip ${bold_file}.gz -f


# antsRegistration \
#     --verbose 1 \
#     --dimensionality 3  \
#     --float 0  \
#     --collapse-output-transforms 1  \
#     --interpolation BSpline[5] \
#     --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii]  \
#     --use-histogram-matching 0  \
#     --winsorize-image-intensities [0.005,0.995]  \
#     --transform Rigid[0.1]  \
#     --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
#     --convergence [1000x500x250x100,1e-6,10]  \
#     --shrink-factors 12x8x4x2  \
#     --smoothing-sigmas 4x3x2x1vox  \
#     -x mask.nii \
#     --transform Affine[0.1]  \
#     --metric MI[${bold_file},fs_T1.nii,1,32,Regular,0.25]  \
#     --convergence [1000x500x250x100,1e-6,10]  \
#     --shrink-factors 12x8x4x2  \
#     --smoothing-sigmas 4x3x2x1vox  \
#     -x mask.nii \
#     --transform SyN[0.1,3,0]  \
#     --metric MI[${bold_file},fs_T1.nii,1,64,Regular,0.25]  \
#     --convergence [50x50x70x50x20,1e-6,10]  \
#     --shrink-factors 10x6x4x2x1  \
#     --smoothing-sigmas 5x3x2x1x0vox  \
#     -x mask.nii > antsRegistration.log 2>&1

# cp fs_to_func_Warped.nii fs_t1_in-func.nii
# fslcpgeom ${bold_file} fs_t1_in-func.nii # correct for possible small affine changes

# mkdir -p $ref_anat_dir

# mv fs_to_func_Warped.nii $ref_anat_dir/fs_to_func_Warped.nii
# mv fs_to_func_InverseWarped.nii $ref_anat_dir/fs_to_func_InverseWarped.nii
# mv fs_to_func_0GenericAffine.mat $ref_anat_dir/fs_to_func_0GenericAffine.mat
# mv fs_to_func_1Warp.nii.gz $ref_anat_dir/fs_to_func_1Warp.nii.gz
# mv fs_to_func_1InverseWarp.nii.gz $ref_anat_dir/fs_to_func_1InverseWarp.nii.gz
# mv fs_t1_in-func.nii $ref_anat_dir/fs_t1_in-func.nii

#     # --metric CC[${bold_file},fs_T1.nii,1,4]  \


#!/bin/bash
# ============================================================
#  T2* mean BOLD → FreeSurfer T1 registration
#  Functional resolution: 0.8 mm isotropic
#
#  Pipeline:
#    1. Temporal mean (if not cached)
#    2. FreeSurfer brain.mgz → NIfTI
#    3. N4 bias correction (BOLD + T1)
#    4. [optional] Clip + BET the BOLD
#    5. [optional] bbregister linear init  OR  ANTs Rigid+Affine
#    6. ANTs SyN (or full Rigid+Affine+SyN)
#    7. Organise outputs
#
#  Dependencies: FSL, FreeSurfer, ANTs, Convert3D (c3d_affine_tool)
# ============================================================




set -euo pipefail


# ════════════════════════════════════════════════════════════
#  USER PARAMETERS — edit these
# ════════════════════════════════════════════════════════════

subject=sub-LAM027
fs_subjects_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer
func_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/func/${subject}
ref_anat_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ref_anat/${subject}

# ── Brain extraction ─────────────────────────────────────────
# DO_CLIP_AND_BET=true : clip BOLD at CLIP_THRESHOLD, then BET
#                        → brain-extracted image passed to ANTs
#                        → brain mask used to constrain ANTs cost function
# DO_CLIP_AND_BET=false: use the N4-corrected BOLD as-is (no skull-strip,
#                        no ANTs masking). Useful when BET fails or when
#                        running bbregister which does not require it.
DO_CLIP_AND_BET=true
CLIP_THRESHOLD=1500      # upper intensity threshold (a.u.) for pre-BET clipping

# ── Linear registration strategy ────────────────────────────
# USE_BBR=true : bbregister (boundary-based, designed for T2* EPI)
#                provides a highly accurate rigid initialisation;
#                ANTs then only corrects residual EPI distortions (SyN).
#                Requires: FreeSurfer recon-all completed for this subject,
#                          c3d_affine_tool on PATH.
# USE_BBR=false: ANTs Rigid + Affine + SyN — fully self-contained,
#                no FreeSurfer surface needed at registration time.
USE_BBR=true


# ════════════════════════════════════════════════════════════
#  DERIVED PATHS  (no edits needed below)
# ════════════════════════════════════════════════════════════

fs_dir=${fs_subjects_dir}/${subject}
bold_base=${subject}_MEAN
bold_raw=${bold_base}.nii

export SUBJECTS_DIR=${fs_subjects_dir}  # required by bbregister / tkregister2

cd "${func_dir}"


# ── Step 1: Temporal mean ─────────────────────────────────────────────────────
if [[ -f "${bold_raw}" ]]; then
    echo "[1/7] ${bold_raw} already exists – skipping mean computation."
else
    echo "[1/7] Merging motion-corrected runs and computing temporal mean…"
    fslmerge -t all_func_runs.nii.gz *_mc.nii*
    fslmaths  all_func_runs.nii.gz -Tmean "${bold_base}.nii.gz"
    rm        all_func_runs.nii.gz
    gunzip    "${bold_base}.nii.gz"
    echo "      → ${bold_raw}"
fi


# ── Step 2: FreeSurfer brain → NIfTI ─────────────────────────────────────────
# brain.mgz is already skull-stripped; we use it as the moving T1.
echo "[2/7] mri_convert: brain.mgz → fs_T1.nii"
mri_convert "${fs_dir}/mri/brain.mgz" fs_T1.nii


# ── Step 3: N4 bias-field correction ─────────────────────────────────────────
# Applied to both images so any remaining intensity non-uniformity is handled
# symmetrically before the MI metric evaluates them.
echo "[3/7] N4 bias-field correction (BOLD + T1)…"

bold_n4="${bold_base}_n4.nii"
N4BiasFieldCorrection -d 3 -i "${bold_raw}"  -o "${bold_n4}"
N4BiasFieldCorrection -d 3 -i fs_T1.nii      -o fs_T1_n4.nii


# ── Step 4: Brain extraction (optional) ──────────────────────────────────────
# Sets two variables consumed by all downstream steps:
#   bold_reg_input — image passed to registration as fixed image
#   ants_mask_args — array, either empty or ("--masks" "[bold_mask,T1_mask]")

if [[ "${DO_CLIP_AND_BET}" == "true" ]]; then
    echo "[4/7] Clipping at ${CLIP_THRESHOLD} then brain-extracting BOLD…"

    # Clip to suppress bright non-brain signal (vessels, eyes, residual fat)
    # that would skew BET's intensity model and the ANTs cost function.
    bold_clipped="${bold_base}_n4_clipped.nii.gz"
    fslmaths "${bold_n4}" -uthr "${CLIP_THRESHOLD}" "${bold_clipped}"

    # BET flags:
    #   -f 0.2 : conservative threshold; T2* GRE has low edge contrast
    #   -m     : save binary brain mask (needed for ANTs --masks)
    #   -R     : robust centre-of-mass estimation; important at 0.8 mm where
    #            peripheral high-res signal can displace the initial centre
    bold_brain="${bold_base}_n4_brain"
    bet "${bold_clipped}" "${bold_brain}" -f 0.2 -m -R
    gunzip -f "${bold_brain}.nii.gz"
    gunzip -f "${bold_brain}_mask.nii.gz"

    bold_reg_input="${bold_brain}.nii"
    bold_mask="${bold_brain}_mask.nii"

    # Binarise the skull-stripped T1 for the moving-image mask
    fslmaths fs_T1_n4.nii -bin fs_T1_mask.nii

    ants_mask_args=(--masks "[${bold_mask},fs_T1_mask.nii]")
else
    echo "[4/7] Skipping clip+BET — using N4 BOLD directly."
    bold_reg_input="${bold_n4}"
    ants_mask_args=()   # no masking; ANTs optimises over the whole FOV
fi


# ── Step 5: Linear registration initialisation ────────────────────────────────

if [[ "${USE_BBR}" == "true" ]]; then

    echo "[5/7] bbregister: BOLD → FreeSurfer T1 (boundary-based, T2*)…"
    # --t2      : treat BOLD as T2-like contrast (correct for T2* GRE EPI)
    # --init-coreg: FreeSurfer header-based initialisation; no external FSL
    #              registration needed. Swap for --init-fsl if coreg fails.
    bbregister \
        --s  "${subject}" \
        --mov "${bold_reg_input}" \
        --reg bbr.reg.dat \
        --t2 \
        --init-coreg

    # Export bbr result as an FSL-convention matrix (BOLD → T1 direction)
    tkregister2 \
        --mov    "${bold_reg_input}" \
        --targ   fs_T1_n4.nii \
        --reg    bbr.reg.dat \
        --fslregout bbr_bold_to_T1.mat \
        --noedit

    # Convert to ANTs/ITK format, inverted to T1 → BOLD direction.
    # ANTs convention: fixed=BOLD, moving=T1 → the initial transform must
    # map T1 voxels into BOLD space.  bbregister gives the opposite direction,
    # so -inv is required.
    c3d_affine_tool \
        -ref fs_T1_n4.nii \
        -src "${bold_reg_input}" \
        bbr_bold_to_T1.mat \
        -fsl2ras -inv \
        -oitk bbr_T1_to_bold_init.txt

    ants_init_args=(--initial-moving-transform bbr_T1_to_bold_init.txt)
    echo "      bbregister done — ANTs will correct residual EPI distortion only."

else
    echo "[5/7] No bbregister — ANTs will handle Rigid + Affine + SyN."
    ants_init_args=()
fi


# ── Step 6: ANTs registration ─────────────────────────────────────────────────
#
#  Fixed  (reference) : brain-extracted mean BOLD  (T2*, 0.8 mm)
#  Moving (to warp)   : FreeSurfer skull-stripped T1
#
#  Common settings:
#    --float 1                       ~2× faster than double; sufficient quality
#    --collapse-output-transforms 1  produces the minimal set of output files
#    --winsorize-image-intensities   clips intensity outliers before MI; critical
#                                    for T2* which has signal dropout & bright CSF
#    --random-seed 42                reproducible results
#    BSpline[5]                      high-quality interpolation for the warped output
#    MI / 64 bins / Random 25%       cross-modality metric; Random sampling avoids
#                                    aliasing from regular grids at 0.8 mm
#
#  Shrink factors (Rigid/Affine) 8x4x2x1:
#    Coarsest level = 8 × 0.8 mm = 6.4 mm — captures global anatomy without
#    losing too much detail. Ends at ×1 (native 0.8 mm) unlike the original.
#
#  SyN[0.1, 2, 0]:
#    Gradient step 0.1, update-field Gaussian σ = 2 mm (tighter than 3 mm
#    default to avoid absorbing EPI-specific contrast differences as apparent
#    geometric deformation), total-field regularisation = 0.
#
#  SyN shrink factors 10x6x4x2x1:
#    5-level pyramid captures both large EPI distortions (coarse) and fine
#    sulcal alignment (native resolution).
#
#  NOTE on output file numbering:
#    USE_BBR=false → Rigid+Affine+SyN → *_0GenericAffine.mat + *_1Warp.nii.gz
#    USE_BBR=true  → SyN only          → *_0Warp.nii.gz  (affine from bbr baked
#                                        in via --collapse-output-transforms)
#  Both cases produce *_Warped.nii and *_InverseWarped.nii.
#  See "Step 7" for how to apply transforms downstream.

echo "[6/7] ANTs registration…"

if [[ "${USE_BBR}" == "true" ]]; then
    # ── SyN only (bbregister handles linear alignment) ────────────────────────
    antsRegistration \
        --verbose 1 \
        --dimensionality 3 \
        --float 1 \
        --collapse-output-transforms 1 \
        --interpolation BSpline[5] \
        --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii] \
        --use-histogram-matching 0 \
        --winsorize-image-intensities [0.005,0.995] \
        "${ants_mask_args[@]+"${ants_mask_args[@]}"}" \
        "${ants_init_args[@]}" \
        \
        --transform SyN[0.1,3,0] \
        --metric MI["${bold_reg_input}",fs_T1_n4.nii,1,64,Random,0.25] \
        --convergence [50x50x70x50x20,1e-6,10]  \
        --shrink-factors 10x6x4x2x1 \
        --smoothing-sigmas 5x3x2x1x0vox \
        > antsRegistration.log 2>&1


else
    # ── Full Rigid + Affine + SyN ─────────────────────────────────────────────
    antsRegistration \
        --verbose 1 \
        --dimensionality 3 \
        --float 1 \
        --collapse-output-transforms 1 \
        --interpolation BSpline[5] \
        --output [fs_to_func_,fs_to_func_Warped.nii,fs_to_func_InverseWarped.nii] \
        --use-histogram-matching 0 \
        --winsorize-image-intensities [0.005,0.995] \
        --random-seed 42 \
        "${ants_mask_args[@]+"${ants_mask_args[@]}"}" \
        \
        --transform Rigid[0.1] \
        --metric MI["${bold_reg_input}",fs_T1_n4.nii,1,64,Random,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        \
        --transform Affine[0.1] \
        --metric MI["${bold_reg_input}",fs_T1_n4.nii,1,64,Random,0.25] \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        \
        --transform SyN[0.1,2,0] \
        --metric MI["${bold_reg_input}",fs_T1_n4.nii,1,64,Random,0.25] \
        --convergence [100x70x50x20x10,1e-6,10] \
        --shrink-factors 10x6x4x2x1 \
        --smoothing-sigmas 5x3x2x1x0vox \
        > antsRegistration.log 2>&1
fi

echo "      ANTs finished. Check antsRegistration.log for convergence."


# ── Step 7: Finalise and organise outputs ─────────────────────────────────────
#
#  To apply these transforms to another image in T1/FS space → BOLD space,
#  use antsApplyTransforms with transforms in reverse order:
#
#  USE_BBR=true:
#    antsApplyTransforms -d 3 -i <T1_space_img> -r <bold> \
#        -t fs_to_func_0Warp.nii.gz \
#        -o output.nii
#    (the warp already incorporates the bbr affine via --collapse-output-transforms)
#
#  USE_BBR=false:
#    antsApplyTransforms -d 3 -i <T1_space_img> -r <bold> \
#        -t fs_to_func_1Warp.nii.gz \
#        -t fs_to_func_0GenericAffine.mat \
#        -o output.nii
#
echo "[7/7] Organising outputs → ${ref_anat_dir}"

# fslcpgeom propagates the qform/sform from the BOLD to the warped T1,
# correcting any sub-voxel affine drift introduced during registration.
cp fs_to_func_Warped.nii fs_t1_in-func.nii
fslcpgeom "${bold_reg_input}" fs_t1_in-func.nii

mkdir -p "${ref_anat_dir}"

# Move all standard outputs (names differ by pipeline — use glob+existence check)
mv fs_to_func_Warped.nii          "${ref_anat_dir}/fs_to_func_Warped.nii"
mv fs_to_func_InverseWarped.nii   "${ref_anat_dir}/fs_to_func_InverseWarped.nii"
mv fs_t1_in-func.nii              "${ref_anat_dir}/fs_t1_in-func.nii"

for f in \
    fs_to_func_0GenericAffine.mat \
    fs_to_func_0Warp.nii.gz \
    fs_to_func_0InverseWarp.nii.gz \
    fs_to_func_1Warp.nii.gz \
    fs_to_func_1InverseWarp.nii.gz
do
    [[ -f "${f}" ]] && mv "${f}" "${ref_anat_dir}/${f}"
done

# Save bbregister artefacts alongside ANTs outputs for quality control
if [[ "${USE_BBR}" == "true" ]]; then
    cp bbr.reg.dat             "${ref_anat_dir}/bbr.reg.dat"
    cp bbr_T1_to_bold_init.txt "${ref_anat_dir}/bbr_T1_to_bold_init.txt"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo " Registration complete"
echo "  Pipeline : USE_BBR=${USE_BBR}  DO_CLIP_AND_BET=${DO_CLIP_AND_BET}  CLIP_THRESHOLD=${CLIP_THRESHOLD}"
echo "  Fixed    : ${bold_reg_input}"
echo "  Moving   : fs_T1_n4.nii  (FreeSurfer brain, N4-corrected)"
echo "  Outputs  : ${ref_anat_dir}/"
echo "  ANTs log : ${func_dir}/antsRegistration.log"
echo "════════════════════════════════════════════════════"