clear all
close all
addpath(genpath('/home/degutis/repos/spm'));
spm('Defaults','fmri');  % Initialize SPM
spm_jobman('initcfg');   % Initialize job manager

TR = 3.29;

subject = 'sub-LAM006'

funcDir = fullfile('/media','miplab-nas2','Data','Karolis','huppi_high_res_resting','derivatives','func',subject);

funcFiles = dir(fullfile(funcDir, '*_mc.nii'));
motionFiles = dir(fullfile(funcDir, '*_mc.par'));

atlas_file = fullfile('/media','miplab-nas2','Data','Karolis','huppi_high_res_resting','derivatives','ref_anat',subject,'HCP-MMP1_in-func.nii');
atlas_file

V_a = spm_vol(atlas_file);
Y_a = spm_read_vols(V_a);     


for file = 1:length(funcFiles)
    disp(file)
    timecourse_file = fullfile(fullfile(funcFiles(file).folder, funcFiles(file).name));
    V_t = spm_vol(timecourse_file);
    Y_t = spm_read_vols(V_t);     

    timecourse_dim = size(Y_t,4);

    %% Run preprocessing

    %% Run motion 

    motionFile = fullfile(fullfile(motionFiles(file).folder, motionFiles(file).name));
    motionParams = load(motionFile);
    motionParams = motionParams(:,2:end-2);
    motionDerivatives = diff(motionParams);
    motionDerivatives = [zeros(1,size(motionDerivatives,2)); motionDerivatives];

    %% WM
    whiteMatter = Y_a(Y_a>3000);
    whiteMatterIndex = ismember(Y_a, whiteMatter);
    whiteMatterIndex = repmat(whiteMatterIndex, [1, 1, 1, timecourse_dim]);

    whiteMatterSignal = nan(size(Y_t));
    whiteMatterSignal(whiteMatterIndex) = Y_t(whiteMatterIndex);  % This operation preserves the size
    whiteMatterSignal = squeeze(mean(whiteMatterSignal,[1,2,3],'omitnan'));

    %% Run CSF
    
    CSF_values = [4, 15, 14, 43];
    CSFIndex = ismember(Y_a, CSF_values);
    CSFIndex = repmat(CSFIndex, [1, 1, 1, timecourse_dim]);
    
    CSFSignal = nan(size(Y_t));
    CSFSignal(CSFIndex) = Y_t(CSFIndex);  % This operation preserves the size
    CSFSignal = squeeze(mean(CSFSignal,[1,2,3],'omitnan'));
    
    %% Design matrix with nuisance regressors
    
    baseline = ones(timecourse_dim,1);
    linearTrend = (1:timecourse_dim)';
    quadraticTrend = linearTrend.^2;
    
    % Concatenate nuisance regressors
    nuisanceRegressors = [baseline, linearTrend, quadraticTrend, whiteMatterSignal, CSFSignal, motionParams, motionDerivatives];
    
    % Save the nuisance regressor matrix
    regressorFile = fullfile(funcDir, append('nuisance_regressors_run',num2str(file),'.txt'));
    save(regressorFile, 'nuisanceRegressors', '-ascii');
    
    % Set up SPM batch for multiple regression
    matlabbatch = {};
    matlabbatch{1}.spm.stats.fmri_spec.dir = {funcDir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR; 
    %matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    %matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = {timecourse_file};
    
    % Specify nuisance regressors
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regressorFile};
    
    % Specify high-pass filter cutoff (0.01 Hz)
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 100;
    
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {append(funcDir,'/SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;  % Add this line to save residuals
    
    % Run the batch
    spm_jobman('run', matlabbatch);
    
    residual_files = spm_select('FPList', funcDir, '^Res_.*\.nii$');
    output_file = fullfile(funcDir, append('merged_residuals_run',num2str(file),'.nii'));
    spm_file_merge(cellstr(residual_files), output_file, 0);
    
    for i = 1:size(residual_files, 1)
        delete(strtrim(residual_files(i, :)));
    end

    delete(fullfile(funcDir,'SPM.mat'))
    delete(fullfile(funcDir,'beta*'))
    delete(fullfile(funcDir,'ResMS.nii'))
    delete(fullfile(funcDir,'mask.nii'))
    delete(fullfile(funcDir,'nuisance*.txt'))
    delete(fullfile(funcDir,'RPV.nii'))

end

