import os
import sys
import subprocess

studyDataDir = "/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/"  # Set your study data directory
subject = "sub-LAM026"

lib_dir = '/home/degutis/dynamicConnectivityMovieWatching_2025/src/preprocessing/fmri-analysis/library'
out_func = f"{studyDataDir}/derivatives/func/{subject}"
os.makedirs(out_func, exist_ok=True)


script_dir = '/home/degutis/dynamicConnectivityMovieWatching_2025/src/preprocessing/fmri-analysis/library'
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

func_dir = f"{studyDataDir}/{subject}/func/"

print("Running motion correction")

nordic_func = f"{func_dir}/{subject}_task-rest_bold_NORDIC.nii"
standard_func = f"{func_dir}/{subject}_task-rest_bold.nii.gz"

if os.path.exists(nordic_func):
    input_func = nordic_func
else:
    input_func = standard_func

subprocess.run([
    "bash", "fmri-analysis/library/motioncorrect.sh",
    input_func,
    lib_dir,
    out_func
], check=True)