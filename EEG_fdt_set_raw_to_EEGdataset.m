%% clearing things up
close all;
clear all;
clc;

%% loading raw .set files into ALLEEG_raw
% {'02wxw','03sf','04cmm','06zhy','08zqh', '09hsq', '10mqk', '11cx', '12wlz', '13cxt', '14zyr', '15yyw', '17sh', '19hpy', '20wrq', '22wf', '23wh', '24hc', '25gyz'};
filenames = {'02wxw','03sf','04cmm','06zhy','08zqh', '09hsq', '10mqk', '11cx', '12wlz', '13cxt', '14zyr', '15yyw', '17sh', '19hpy', '20wrq', '22wf', '23wh', '24hc', '25gyz'};
filepath = cd; %['/Users/tit/Desktop/BCH_PROJECT/Codes/EEG_DATA_RAW']

for i =1:length(filenames)
    fname = [filenames{i}, '_epoched.set'];
    ALLEEG_all(i) = pop_loadset(fname, filepath);
end

%% Loading subjects' valence rates
textData = importdata('valence_rates.txt');

%% Making the raw dataset of subjects into 

wcount = 1; % counter for the valence_rates.txt
% making the data  set in EEGdataset
for i = 1:length(filenames)
    ALLEEG = ALLEEG_all(i); % dummy variable

    EEGdataset(i).subjectID = filenames{i}; % subject ID
    EEGdataset(i).trialNO = ALLEEG.trials; % subject's number of trials
    EEGdataset(i).sRate = ALLEEG.srate;
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
            trial_Data = EEGdataset(i).EEG(:, 1001:end, j);
            [pxx,f] = pwelch(trial_Data', [], [], [], EEGdataset(i).sRate);
            trial_THETA = bandpower(pxx, f, [2 4], 'psd');
            trial_ALPHA = bandpower(pxx, f, [8 13], 'psd');
            trial_BETA = bandpower(pxx, f, [13 30], 'psd');
            trial_GAMMA = bandpower(pxx, f, [30 49], 'psd');
            trial_bandPower = [trial_THETA; trial_ALPHA; trial_BETA; trial_GAMMA];
            trial_bandPower = reshape(trial_bandPower, 1, []);
            dataset_features = [dataset_features; trial_bandPower];
        end
    end
end

%% saving the matrices
save('dataset_features.mat', 'dataset_features');
save('dataset_labels.mat', 'dataset_labels');