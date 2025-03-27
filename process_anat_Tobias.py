#!/usr/bin/env python3

import subprocess
import os
import sys

# Add script paths to sys.path
os.environ["SUBJECTS_DIR"] = "/Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/"
script_dir = "/Users/karolis/Desktop/dynamicConnectivityMovieWatching_2025/fmri-analysis/library"
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

# Add fMRI submodule functions
import layer_analysis as analysis
import voxeldepths_from_surfaces as vdfs


def process_ref_anat_subject(studyDataDir, subject):
    """
    Process reference anatomical data for a given subject.

    Args:
        studyDataDir (str): Path to the study data directory.
        subject (str): Subject identifier.

    Steps:
        1. Register FreeSurfer to EPI.
        2. Transform FreeSurfer surfaces to EPI space.
        3. Transform FreeSurfer ribbon.
        4. Calculate VDFS depths.
        5. Generate functional mid surfaces and transform atlases.
    """
    print(f"Processing subject: {subject} in study directory: {studyDataDir}")

    # Define necessary directories
    ref_anat_dir = os.path.join(studyDataDir, "derivatives", "ref_anat_new", subject)
    fs_dir = os.path.join(studyDataDir, "derivatives", "freesurfer", subject)
    preprocess_vaso_dir = os.path.join(
        studyDataDir, "derivatives", "ref_anat_new", subject
    )
    ciftify_dir = os.path.join(studyDataDir, "derivatives", "ciftify", subject)

    # Ensure reference anatomy directory exists
    os.makedirs(ref_anat_dir, exist_ok=True)

    # Step 1: Register FreeSurfer to EPI
    subprocess.run(
        [
            "register_fs-to-bold_no-manual.sh",
            os.path.join(preprocess_vaso_dir, "func_all.nii"),
            fs_dir,
        ],
        cwd=ref_anat_dir,
    )
    print("Registration complete. Cleaning up temporary files...")
    os.remove(os.path.join(ref_anat_dir, "fs_T1.nii"))
    os.remove(os.path.join(ref_anat_dir, "fs_to_func_Warped.nii"))
    os.remove(os.path.join(ref_anat_dir, "fs_to_func_InverseWarped.nii"))

    # Step 2: Transform FreeSurfer surfaces to EPI space
    fs_to_func_reg = [
        os.path.join(ref_anat_dir, filename)
        for filename in [
            "fs_t1_in-func.nii",
            "fs_to_func_0GenericAffine.mat",
            "fs_to_func_1Warp.nii.gz",
            "fs_to_func_1InverseWarp.nii.gz",
        ]
    ]
    surface_fs_trans_files = analysis.fs_surface_to_func_legacy(
        fs_to_func_reg, fs_dir, ref_anat_dir, force=True
    )
    print("Transformation of surfaces complete. Cleaning up intermediate files...")
    for hemi, hemi_hcp in zip(["lh", "rh"], ["L", "R"]):
        for surf in ["pial", "white"]:
            os.remove(os.path.join(ref_anat_dir, f"{hemi}.{surf}_converted.gii"))
            os.remove(os.path.join(ref_anat_dir, f"{hemi}.{surf}_convertedpoints.csv"))
            os.remove(
                os.path.join(
                    ref_anat_dir, f"{hemi}.{surf}_convertedpoints_transformed.csv"
                )
            )
            os.rename(
                os.path.join(ref_anat_dir, f"{hemi}.{surf}_converted.transformed.gii"),
                os.path.join(ref_anat_dir, f"{hemi_hcp}.{surf}.func.surf.gii"),
            )
            os.rename(
                os.path.join(ref_anat_dir, f"{hemi}.{surf}_func"),
                os.path.join(ref_anat_dir, f"{hemi}.func.{surf}"),
            )

    # Step 3: Transform FreeSurfer ribbon
    fs_rim = analysis.import_fs_ribbon_to_func(fs_dir, ref_anat_dir, force=True)
    os.remove(os.path.join(ref_anat_dir, "fs_ribbon.nii"))

    # Step 4: Calculate VDFS depths
    area_files = {
        ("lh", "white"): os.path.join(fs_dir, "surf", "lh.area"),
        ("rh", "white"): os.path.join(fs_dir, "surf", "rh.area"),
        ("lh", "pial"): os.path.join(fs_dir, "surf", "lh.area.pial"),
        ("rh", "pial"): os.path.join(fs_dir, "surf", "rh.area.pial"),
    }
    for method in ["equivol", "equidist"]:
        print(f"Calculating VDFS and layers using method: {method}...")
        vdfs_depths, vdfs_columns = vdfs.process_dc_voxeldepth_from_surfaces(
            os.path.join(ref_anat_dir, "L.white.func.surf.gii"),
            area_files["lh", "white"],
            os.path.join(ref_anat_dir, "L.pial.func.surf.gii"),
            area_files["lh", "pial"],
            os.path.join(ref_anat_dir, "R.white.func.surf.gii"),
            area_files["rh", "white"],
            os.path.join(ref_anat_dir, "R.pial.func.surf.gii"),
            area_files["rh", "pial"],
            os.path.join(ref_anat_dir, "fs_t1_in-func.nii"),
            os.path.join(ref_anat_dir, f"vdfs_depths_{method}.nii"),
            os.path.join(ref_anat_dir, f"vdfs_columns_{method}.nii"),
            method=method,
            upsample_factor=None,
            n_jobs=8,
            force=True,
        )
        analysis.calc_layers_laynii(
            os.path.join(ref_anat_dir, "rim.nii"),
            method=method,
            force=True,
        )
        os.rename(
            os.path.join(ref_anat_dir, f"rim_metric_{method}.nii"),
            os.path.join(ref_anat_dir, f"ln_depths_{method}.nii"),
        )
        os.remove(os.path.join(ref_anat_dir, f"rim_layers_{method}.nii"))
        os.remove(os.path.join(ref_anat_dir, f"rim_midGM_{method}.nii"))

    # Step 5: Generate functional mid surfaces and transform atlases
    glasser_labels = {
        "L": os.path.join(studyDataDir, "Glasser_group_files", "GlasserAtlas.L.164k_fs_LR.label.gii"),
        "R": os.path.join(studyDataDir, "Glasser_group_files", "GlasserAtlas.R.164k_fs_LR.label.gii"),
    }
    print("Processing complete!")


if __name__ == "__main__":

    studyDir = "/Users/karolis/Desktop/highRes_Resting"
    subject = "sub-01"
    process_ref_anat_subject(studyDir, subject)