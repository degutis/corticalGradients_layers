#! /usr/bin/env python3
import subprocess
import os
import sys

os.environ["SUBJECTS_DIR"] = "/Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/"
script_dir = "/Users/karolis/Desktop/dynamicConnectivityMovieWatching_2025/fmri-analysis/library"
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

import layer_analysis as analysis
import voxeldepths_from_surfaces as vdfs

def process_ref_anat_subject(studyDataDir, subject):

    ref_anat_dir = os.path.join(studyDataDir, "derivatives", "ref_anat", subject)
    fs_dir = os.path.join(studyDataDir, "derivatives", "freesurfer", subject)
    preprocess_vaso_dir = os.path.join(
        studyDataDir, "derivatives", "ref_anat", subject
    )
    ciftify_dir = os.path.join(studyDataDir, "derivatives", "ciftify", subject)

    os.makedirs(ref_anat_dir, exist_ok=True)

     # clean up
    #os.remove(os.path.join(ref_anat_dir, "fs_T1.nii"))
    #os.remove(os.path.join(ref_anat_dir, "fs_to_func_Warped.nii"))
    #os.remove(os.path.join(ref_anat_dir, "fs_to_func_InverseWarped.nii"))

     # 2. transform freesurfer surface to epi_space
    fs_to_func_reg = [
        os.path.join(ref_anat_dir, filename)
        for filename in [
            "fs_to_func_0GenericAffine.mat",
            "fs_to_func_1Warp.nii.gz",
        ]
    ]
    surface_fs_trans_files = analysis.fs_surface_to_func(
        fs_to_func_reg, fs_dir, ref_anat_dir, force=True
    )
     # clean up


if __name__ == "__main__":
    studyDir = "/Users/karolis/Desktop/highRes_Resting"
    subject = "sub-01"
    process_ref_anat_subject(studyDir, subject)