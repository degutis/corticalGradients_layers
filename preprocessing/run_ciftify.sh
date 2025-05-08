#!/bin/bash

export PATH=$PATH:/Users/karolis/Desktop/repos/MSM
subjectList=(sub-02)
data_dir=/Users/karolis/Desktop/kenshu_dataset/derivatives

for subject in ${subjectList[@]}
do

	echo PROCESSING ... ${subject}

	#prepare FS dir to work - it expects the subject folder in the freesurfer folder
	fs_dir=${data_dir}/freesurfer/
	fs_dir_subject=${fs_dir}/${subject}
	
	if [ ! -d ${fs_dir_subject} ]; then
		echo ....... moving freesurfer files into: ${fs_dir_subject}		
		mkdir -p ${fs_dir_subject}
		mv -t ${fs_dir_subject} ${fs_dir}/trash ${fs_dir}/touch ${fs_dir}/tmp ${fs_dir}/surf ${fs_dir}/stats ${fs_dir}/scripts ${fs_dir}/mri ${fs_dir}/label
	fi

	ciftify_dir=${data_dir}/ciftify/
	MSM_config_file=/opt/homebrew/Caskroom/miniconda/base/envs/analysis/lib/python3.10/site-packages/ciftify/data/hcp_config/MSMSulcStrainFinalconf
	
	if [ ! -d ${ciftify_dir} ]; then
		echo ....... creating directory: ${ciftify_dir}		
		mkdir -p ${ciftify_dir}
	fi

	#ciftify_recon_all --ciftify-work-dir ${ciftify_dir} --fs-subjects-dir ${fs_dir} --resample-to-T1w32k ${subject}
	ciftify_recon_all --ciftify-work-dir ${ciftify_dir} --fs-subjects-dir ${fs_dir} --ciftify-conf /Users/karolis/Desktop/repos/ciftify_lib/ciftify_expert_settings_0.5mm.yaml --resample-to-T1w32k ${subject}
	export CIFTIFY_WORKDIR=${ciftify_dir} 
	#cifti_vis_recon_all snaps ${subject}
done 

#https://github.com/edickie/ciftify/blob/master/ciftify/bin/ciftify_recon_all.py
