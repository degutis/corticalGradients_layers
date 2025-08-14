# export PATH=$PATH:/Users/karolis/Desktop/repos/MSM

subject=sub-LAM004
fs_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/freesurfer

ciftify_dir=/media/miplab-nas2/Data/Karolis/huppi_high_res_resting/derivatives/ciftify/
mkdir -p ${ciftify_dir}

# MSM_config_file=/opt/homebrew/Caskroom/miniconda/base/envs/analysis/lib/python3.10/site-packages/ciftify/data/hcp_config/MSMSulcStrainFinalconf
ciftify_conf_file=/home/degutis/repos/ciftify_lib/ciftify_expert_settings_0.5mm.yaml    


# ciftify_recon_all  -v --fs-subjects-dir "${fs_dir}" \
#         --ciftify-work-dir "${ciftify_dir}" \
#         --MSM-config \
#         --ciftify-conf "${ciftify_conf_file}" \
#         --resample-to-T1w32k --no-symlinks "${subject}"


ciftify_recon_all  -v --fs-subjects-dir "${fs_dir}" \
        --ciftify-work-dir "${ciftify_dir}" \
        --ciftify-conf "${ciftify_conf_file}" \
        --resample-to-T1w32k --no-symlinks "${subject}"
