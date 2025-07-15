import os
import sys
import subprocess


studyDataDir = "/Users/karolis/Desktop/highRes_resting"  # Set your study data directory
subject = "sub-04"
sys.path.append('/Users/karolis/Desktop/dynamicConnectivityMovieWatching_2025/fmri-analysis/library')

inv2_path = f"{studyDataDir}/{subject}/anat/{subject}_INV2_T1w.nii"
uni_path = f"{studyDataDir}/{subject}/anat/{subject}_UNI-DEN_T1w.nii"
fs_dir = f"{studyDataDir}/derivatives/freesurfer/{subject}"

os.environ["SUBJECTS_DIR"] = f"{studyDataDir}/derivatives/freesurfer/"

subprocess.run([
    "python", "fmri-analysis/library/mp2rage_recon-all.py",
    inv2_path,
    uni_path,
    "--fs_dir", fs_dir,
    "--spm_dir", "/Users/karolis/Documents/spm"
], check=True)