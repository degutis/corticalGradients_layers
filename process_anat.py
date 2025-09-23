#! /usr/bin/env python3
import subprocess
import os
import sys

# os.environ["SUBJECTS_DIR"] = "/media/miplab-nas2/Data/Karolis/high_res_resting//derivatives/freesurfer/"
os.environ["SUBJECTS_DIR"] = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting//derivatives/freesurfer/"
script_dir = "/home/degutis/dynamicConnectivityMovieWatching_2025/fmri-analysis/library"
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

import layer_analysis as analysis
# import voxeldepths_from_surfaces as vdfs

def process_ref_anat_subject(studyDataDir, subject):

    ref_anat_dir = os.path.join(studyDataDir, "derivatives", "ref_anat", subject)
    fs_dir = os.path.join(studyDataDir, "derivatives", "freesurfer", subject)
    preprocess_vaso_dir = os.path.join(
        studyDataDir, "derivatives", "ref_anat", subject
    )
    ciftify_dir = os.path.join(studyDataDir, "derivatives", "ciftify", subject)

    #os.makedirs(ref_anat_dir, exist_ok=True)

     # 1. register freesurfer to epi
    #subprocess.run(
    #     [
    #         "register_fs-to-bold_no-manual.sh",
    #         os.path.join(preprocess_vaso_dir, "sub-01_bold_SMSEPI_mc_MEAN.nii"),
    #         fs_dir,
    #     ],
    #     cwd=ref_anat_dir,
    # )

     # clean up
    # os.remove(os.path.join(ref_anat_dir, "fs_T1.nii"))
    # os.remove(os.path.join(ref_anat_dir, "fs_to_func_Warped.nii"))
    # os.remove(os.path.join(ref_anat_dir, "fs_to_func_InverseWarped.nii"))

     # 2. transform freesurfer surface to epi_space
    fs_to_func_reg = [
        os.path.join(ref_anat_dir, filename)
        for filename in [
            "fs_t1_in-func.nii",
            "fs_to_func_0GenericAffine.mat",
            "fs_to_func_1Warp.nii.gz",
            "fs_to_func_1InverseWarp.nii.gz"
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

     # 3. transform freesurfer ribbon
    fs_rim = analysis.import_fs_ribbon_to_func(fs_dir, ref_anat_dir, force=True)
     # clean up
    os.remove(os.path.join(ref_anat_dir, "fs_ribbon.nii"))

    # 4. calculate LAYNII depths
    area_files = {
        ("lh", "white"): os.path.join(fs_dir, "surf", "lh.area"),
        ("rh", "white"): os.path.join(fs_dir, "surf", "rh.area"),
        ("lh", "pial"): os.path.join(fs_dir, "surf", "lh.area.pial"),
        ("rh", "pial"): os.path.join(fs_dir, "surf", "rh.area.pial"),
    }
    for method in ["equivol", "equidist"]:
        print(f"Calculating VDFS and layers using method: {method}...")
        # vdfs_depths, vdfs_columns = vdfs.process_dc_voxeldepth_from_surfaces(
        #     os.path.join(ref_anat_dir, "L.white.func.surf.gii"),
        #     area_files["lh", "white"],
        #     os.path.join(ref_anat_dir, "L.pial.func.surf.gii"),
        #     area_files["lh", "pial"],
        #     os.path.join(ref_anat_dir, "R.white.func.surf.gii"),
        #     area_files["rh", "white"],
        #     os.path.join(ref_anat_dir, "R.pial.func.surf.gii"),
        #     area_files["rh", "pial"],
        #     os.path.join(ref_anat_dir, "fs_t1_in-func.nii"),
        #     os.path.join(ref_anat_dir, f"vdfs_depths_{method}.nii"),
        #     os.path.join(ref_anat_dir, f"vdfs_columns_{method}.nii"),
        #     method=method,
        #     upsample_factor=None,
        #     n_jobs=8,
        #     force=True,
        # )
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



    # 5. generate func mid surf and transform atlasses
    glasser_labels = {
        "L": "/home/degutis/repos/HCP_WB_parcels/GlasserAtlas.L.164k_fs_LR.label.gii",
        "R": "/home/degutis/repos/HCP_WB_parcels/GlasserAtlas.R.164k_fs_LR.label.gii",
    }

    schaefer_labels = {
        "L": "/home/degutis/repos/HCP_WB_parcels/Schaefer400_7net.L.164k_fs_LR.label.gii",
        "R": "/home/degutis/repos/HCP_WB_parcels/Schaefer400_7net.R.164k_fs_LR.label.gii",
    }
    

    for hemi in ["L", "R"]:
        fs_LR_sphere = os.path.join(
            ciftify_dir, "MNINonLinear", f"{subject}.{hemi}.sphere.164k_fs_LR.surf.gii"
        )
        fs_LR_mid_surf = os.path.join(
            ciftify_dir,
            "MNINonLinear",
            f"{subject}.{hemi}.midthickness.164k_fs_LR.surf.gii",
        )
        fs_LR_roi = os.path.join(
            ciftify_dir,
            "MNINonLinear",
            f"{subject}.{hemi}.atlasroi.164k_fs_LR.shape.gii",
        )

        native_sphere = os.path.join(
            ciftify_dir,
            "MNINonLinear",
            "Native",
            f"{subject}.{hemi}.sphere.MSMSulc.native.surf.gii",
        )
        native_pial_surf = os.path.join(ref_anat_dir, f"{hemi}.pial.func.surf.gii")
        native_white_surf = os.path.join(ref_anat_dir, f"{hemi}.white.func.surf.gii")
        native_mid_surf = os.path.join(ref_anat_dir, f"{hemi}.mid.func.surf.gii")
        # generate mid surf
        subprocess.run(
            [
                "wb_command",
                "-surface-average",
                native_mid_surf,
                "-surf",
                native_pial_surf,
                "-surf",
                native_white_surf,
            ]
        )

        for atlas, labels in zip(["glasser", "schaefer"], [glasser_labels, schaefer_labels]):
        # for atlas, labels in zip(["glasser"], [glasser_labels]):
            atlas_fs_LR_space = labels[hemi]
            atlas_native_surf = os.path.join(
                ref_anat_dir, f"{subject}.{atlas}.{hemi}.native.label.gii"
            )
            atlas_native_volume = os.path.join(
                ref_anat_dir, f"{atlas}_{hemi}_in-func.nii"
            )
            subprocess.run(
                [
                    "wb_command",
                    "-label-resample",
                    atlas_fs_LR_space,
                    fs_LR_sphere,
                    native_sphere,
                    "ADAP_BARY_AREA",
                    atlas_native_surf,
                    "-area-surfs",
                    fs_LR_mid_surf,
                    native_mid_surf,
                    "-current-roi",
                    fs_LR_roi,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "wb_command",
                    "-label-to-volume-mapping",
                    atlas_native_surf,
                    native_mid_surf,
                    os.path.join(preprocess_vaso_dir, "fs_t1_in-func.nii"),
                    atlas_native_volume,
                    "-ribbon-constrained",
                    native_white_surf,
                    native_pial_surf,
                ],
                check=True,
            )


if __name__ == "__main__":
    studyDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting"
    subject = "sub-LAM022"
    process_ref_anat_subject(studyDir, subject)