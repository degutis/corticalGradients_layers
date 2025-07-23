import os
import sys
import subprocess

# studyDataDir = "/media/miplab-nas2/Data/Karolis/high_res_resting/"  # Set your study data directory
studyDataDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"  # Set your study data directory
subject = "sub-LAM006"

lib_dir = '/home/degutis/dynamicConnectivityMovieWatching_2025/fmri-analysis/library'
out_func = f"{studyDataDir}/derivatives/func/{subject}"
os.makedirs(out_func, exist_ok=True)


script_dir = "/home/degutis/dynamicConnectivityMovieWatching_2025/fmri-analysis/library"
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

func_dir = f"{studyDataDir}/{subject}/func/"

# print("Running SMSEPI correction")
# subprocess.run([
#     "bash", "../fmri-analysis/library/motioncorrect.sh",
#     f"{func_dir}/{subject}_run-01_bold_SMSEPI_NORDIC.nii",
#     f"{func_dir}/{subject}_run-02_bold_SMSEPI_NORDIC.nii",
#     f"{func_dir}/{subject}_run-03_bold_SMSEPI_NORDIC.nii",
#     lib_dir,
#     out_func
#     # f"{func_dir}/{subject}_run-04_bold_SMSEPI.nii.gz",
#     # f"{func_dir}/{subject}_run-05_bold_SMSEPI.nii.gz"
# ], check=True)

print("Running motion correction")
subprocess.run([
    "bash", "../fmri-analysis/library/motioncorrect.sh",
    f"{func_dir}/{subject}_task-rest_bold.nii.gz",
    lib_dir,
    out_func
], check=True)