import os
import sys
import subprocess

studyDataDir = "/Users/karolis/Desktop/highRes_Resting"  # Set your study data directory
subject = "sub-03"


script_dir = "/Users/karolis/Desktop/dynamicConnectivityMovieWatching_2025/fmri-analysis/library"
os.environ["PATH"] += os.pathsep + script_dir
sys.path.append(script_dir)

func_dir = f"{studyDataDir}/{subject}/func/"

#print("Running Mid correction")
#subprocess.run([
#    "bash", "fmri-analysis/library/motioncorrect.sh",
#    f"{func_dir}/sub-01_bold_GRE_MID.nii.gz"
#], check=True)


#print("Running SE correction")
#subprocess.run([
#    "bash", "fmri-analysis/library/motioncorrect.sh",
#    f"{func_dir}/sub-01_run-01_bold_SE.nii.gz",
#    f"{func_dir}/sub-01_run-02_bold_SE.nii.gz"
#], check=True)


#print("Running GRE correction")
#subprocess.run([
#    "bash", "fmri-analysis/library/motioncorrect.sh",
#    f"{func_dir}/sub-01_run-02_bold_GRE.nii.gz"
#], check=True)

print("Running SMSEPI correction")
subprocess.run([
    "bash", "../fmri-analysis/library/motioncorrect.sh",
    f"{func_dir}/{subject}_run-01_bold_SMSEPI.nii.gz",
    # f"{func_dir}/{subject}_run-02_bold_SMSEPI.nii.gz",
    f"{func_dir}/{subject}_run-03_bold_SMSEPI.nii.gz",
    # f"{func_dir}/{subject}_run-04_bold_SMSEPI.nii.gz",
    # f"{func_dir}/{subject}_run-05_bold_SMSEPI.nii.gz"
], check=True)

# print("Running 3D EPI correction")
# subprocess.run([
#     "bash", "fmri-analysis/library/motioncorrect.sh",
#     f"{func_dir}/{subject}_3depi_bold.nii.gz"
#  ], check=True)
