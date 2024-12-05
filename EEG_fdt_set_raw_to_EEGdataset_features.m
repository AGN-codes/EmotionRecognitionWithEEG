%% clearing things up
close all;
clear all;
clc;

%% loading raw .set files into ALLEEG_raw
% {'02wxw','03sf','04cmm','06zhy','08zqh', '09hsq', '10mqk', '11cx', '12wlz', '13cxt', '14zyr', '15yyw', '17sh', '19hpy', '20wrq', '22wf', '23wh', '24hc', '25gyz'};
% filenames = {'02wxw','03sf'};
filenames = {'02wxw','03sf','04cmm','06zhy','08zqh', '09hsq', '10mqk', '11cx', '12wlz', '13cxt', '14zyr', '15yyw', '17sh', '19hpy', '20wrq', '22wf', '23wh', '24hc', '25gyz'};
filepath = cd; %['/Users/tit/Desktop/BCH_PROJECT/Codes/EEG_DATA_RAW']

for i =1:length(filenames)
    fname = [filenames{i}, '_epoched.set'];
    ALLEEG_all(i) = pop_loadset(fname, filepath);
end

%% Loading subjects' valence rates
textData = importdata('valence_rates.txt');

%% Making the raw dataset of subjects into EEGdataset

wcount = 1; % counter for the valence_rates.txt
% making the data  set in EEGdataset
for i = 1:length(filenames)
    ALLEEG = ALLEEG_all(i); % dummy variable

    EEGdataset(i).subjectID = filenames{i}; % subject ID
    EEGdataset(i).trialNO = ALLEEG.trials; % subject's number of trials
    EEGdataset(i).sRate = ALLEEG.srate;
    %EEGdataset(i).
    % setting the video types (255 = -, 0 = neutral, 1 = positive)
    for j=1:length(ALLEEG.epoch)
        EEGdataset(i).videoType(j,1) = ALLEEG.epoch(j).eventtype;
    end
    % setting the subject's 
    EEGdataset(i).valenceRate=[];
    for j = 1:ALLEEG.trials
        EEGdataset(i).valenceRate = [EEGdataset(i).valenceRate; textData.data(wcount)];
        wcount = wcount+1;
    end
    % EEG recordings of the subject (59 channels x 5000 x number of trials)
    EEGdataset(i).EEG = ALLEEG.data;
end

%% Making the Matrix for Basic work

dataset_features = [];
dataset_labels = [];
for i = 1:length(EEGdataset)
    for j = 1:EEGdataset(i).trialNO
        if EEGdataset(i).valenceRate(j) ~= 5
            % dataset_labels = [dataset_labels; EEGdataset(i).valenceRate(j)];
            if EEGdataset(i).valenceRate(j) > 5
                dataset_labels = [dataset_labels; 1];
            else
                dataset_labels = [dataset_labels; -1];
            end

            % computing band powers
            trial_Data = EEGdataset(i).EEG(:, 1001:end, j);
            [pxx,f] = pwelch(trial_Data', [], [], [], EEGdataset(i).sRate);
            trial_THETA = bandpower(pxx, f, [4 8], 'psd');
            trial_ALPHA = bandpower(pxx, f, [8 13], 'psd');
            trial_BETA = bandpower(pxx, f, [13 30], 'psd');
            trial_GAMMA = bandpower(pxx, f, [30 49], 'psd');
            trial_TOTAL_POWER = trial_THETA+trial_ALPHA+trial_BETA+trial_BETA;
            trial_bandPowers = [trial_THETA, trial_ALPHA, trial_BETA, trial_GAMMA, trial_TOTAL_POWER];
            trial_bandPowers_row = reshape(trial_bandPowers, 1, []);
            % computing relative band powers
            trial_THETA_relative = trial_THETA./trial_TOTAL_POWER;
            trial_ALPHA_relative = trial_ALPHA./trial_TOTAL_POWER;
            trial_BETA_relative = trial_BETA./trial_TOTAL_POWER;
            trial_GAMMA_relative = trial_GAMMA./trial_TOTAL_POWER;
            trial_relative_bandPowers = [trial_THETA_relative; trial_ALPHA_relative; trial_BETA_relative; trial_GAMMA_relative];
            trial_relative_bandPowers_row = reshape(trial_relative_bandPowers, 1, []);
            % interhemispheric EEG power asymmetry 
            left_ASM = [4,6,9,11,13,15,2,18,20,22,27,29,31,35,37,39,44,46,48,24,33,41,51,53,55,58]; 
            right_ASM = [5,7,10,12,14,16,3,19,21,23,28,30,32,36,38,40,45,47,49,25,34,42,52,54,56,59];
            % rational asymmetry (RASM)
            RASM_features_THETA = trial_THETA_relative(left_ASM)./trial_THETA_relative(right_ASM);
            RASM_features_ALPHA = trial_ALPHA_relative(left_ASM)./trial_ALPHA_relative(right_ASM);
            RASM_features_BETA = trial_BETA_relative(left_ASM)./trial_BETA_relative(right_ASM);
            RASM_features_GAMMA = trial_GAMMA_relative(left_ASM)./trial_GAMMA_relative(right_ASM);
            RASM_features_TOTAL = trial_TOTAL_POWER(left_ASM)./trial_TOTAL_POWER(right_ASM);
            % differential asymmetry (DASM)
            DASM_features_THETA = trial_THETA_relative(left_ASM)-trial_THETA_relative(right_ASM);
            DASM_features_ALPHA = trial_ALPHA_relative(left_ASM)-trial_ALPHA_relative(right_ASM);
            DASM_features_BETA = trial_BETA_relative(left_ASM)-trial_BETA_relative(right_ASM);
            DASM_features_GAMMA = trial_GAMMA_relative(left_ASM)-trial_GAMMA_relative(right_ASM);
            DASM_features_TOTAL = trial_TOTAL_POWER(left_ASM)-trial_TOTAL_POWER(right_ASM);
            % rational asymmetry (RASM) (right/left)
            RASM_features_THETA_inverse = trial_THETA_relative(right_ASM)./trial_THETA_relative(left_ASM);
            RASM_features_ALPHA_inverse = trial_ALPHA_relative(right_ASM)./trial_ALPHA_relative(left_ASM);
            RASM_features_BETA_inverse = trial_BETA_relative(right_ASM)./trial_BETA_relative(left_ASM);
            RASM_features_GAMMA_inverse = trial_GAMMA_relative(right_ASM)./trial_GAMMA_relative(left_ASM);
            RASM_features_TOTAL_inverse = trial_TOTAL_POWER(right_ASM)./trial_TOTAL_POWER(left_ASM);

            trial_features = [trial_bandPowers_row, trial_relative_bandPowers_row, RASM_features_THETA, RASM_features_ALPHA, RASM_features_BETA, RASM_features_GAMMA, RASM_features_TOTAL, DASM_features_THETA, DASM_features_ALPHA, DASM_features_BETA, DASM_features_GAMMA, DASM_features_TOTAL, RASM_features_THETA_inverse, RASM_features_ALPHA_inverse, RASM_features_BETA_inverse, RASM_features_GAMMA_inverse, RASM_features_TOTAL_inverse];
            % trial_features = trial_relative_bandPowers_row;
            dataset_features = [dataset_features; trial_features];
        end
    end
end

%% saving the matrices
% save('dataset_features.mat', 'dataset_features');
% save('dataset_labels.mat', 'dataset_labels');