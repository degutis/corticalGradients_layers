import os
import sys
import subprocess


studyDataDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"
subject = "sub-LAM022"
sys.path.append('/home/degutis/dynamicConnectivityMovieWatching_2025/fmri-analysis/library')

inv2_path = f"{studyDataDir}/{subject}/anat/{subject}_inv-2_MP2RAGE.nii"

uni_path = f"{studyDataDir}/{subject}/anat/{subject}_acq-denoised_T1w.nii"

fs_dir = f"{studyDataDir}/derivatives/freesurfer/{subject}"

os.environ["SUBJECTS_DIR"] = f"{studyDataDir}/derivatives/freesurfer/"

subprocess.run([
    "python", "fmri-analysis/library/mp2rage_recon-all.py",
    inv2_path,
    uni_path,
    "--fs_dir", fs_dir,
    "--spm_dir", "/home/degutis/repos/spm/"
], check=True)