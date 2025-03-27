import numpy as np
from pydfc import data_loader

# load sub-0001 data from nifti file
BOLD = data_loader.nifti2timeseries(
    nifti_file="sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    n_rois=100,
    Fs=1 / 0.75,
    subj_id="sub-0001",
    confound_strategy="no_motion",  # no_motion, no_motion_no_gsr, or none
    standardize=False,
    TS_name=None,
    session=None,
)
