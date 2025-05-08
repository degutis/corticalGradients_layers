export PATH=$PATH:/Users/karolis/Desktop/repos/MSM

subject=sub-01
fs_dir=/Users/karolis/Desktop/highRes_Resting/derivatives/freesurfer/

ciftify_dir=/Users/karolis/Desktop/highRes_Resting/derivatives/ciftify/
mkdir -p ${ciftify_dir}

MSM_config_file=/opt/homebrew/Caskroom/miniconda/base/envs/analysis/lib/python3.10/site-packages/ciftify/data/hcp_config/MSMSulcStrainFinalconf
ciftify_conf_file=/Users/karolis/Desktop/repos/ciftify_lib/ciftify_expert_settings_0.5mm_changed.yaml    

export TMPDIR=/Users/karolis/Desktop/repos/ciftify/tmp
mkdir -p "${TMPDIR}"


ciftify_recon_all  -v --fs-subjects-dir "${fs_dir}" \
        --ciftify-work-dir "${ciftify_dir}" \
        --MSM-config "${MSM_config_file}" \
        --ciftify-conf "${ciftify_conf_file}" \
        --resample-to-T1w32k --no-symlinks "${subject}"
