%% clearing things up
close all;
clear all;
clc;

%% loading files
load('dataset_features.mat');
load('dataset_labels.mat');

%% assembling the dataset
dataset = [dataset_features dataset_labels];

%% shuffling the data
positiveValenceNO = sum(dataset_labels == 1);
negativeValenceNO = sum(dataset_labels == -1);

rndm_idx = randperm(positiveValenceNO+negativeValenceNO);
dataset_shuffle = dataset(rndm_idx, :);

%% partitioning the data according to the labels (positive and negative)
dataset_positive = [];
dataset_negative = [];
for i = 1:positiveValenceNO+negativeValenceNO
    if(dataset(i, end) == 1)
        dataset_positive = [dataset_positive; dataset(i,:)];
    else
        dataset_negative = [dataset_negative; dataset(i,:)];
    end
end

%% shuffling negative and positive labels
rndm_idx_p = randperm(positiveValenceNO);
rndm_idx_n = randperm(negativeValenceNO);
dataset_positive = dataset_positive(rndm_idx_p, :);
dataset_negative = dataset_negative(rndm_idx_n, :);

%% same amount of negative and positive labels
minValenceNO = min(positiveValenceNO, negativeValenceNO);
dataset_positive = dataset_positive(1:minValenceNO, :);
dataset_negative = dataset_negative(1:minValenceNO, :);

%% Folding of the dataset (into kFold parts in kData struct) and shuffling
kFold = 4;
foldSize = minValenceNO/kFold;
for i = 1:kFold
    kData(i).Data = [dataset_positive((i-1)*foldSize+1:(i)*foldSize, :); dataset_negative((i-1)*foldSize+1:(i)*foldSize, :)];
    
    rndm_perm = randperm(foldSize*2);
    kData(i).Data = kData(i).Data(rndm_perm, :);
end

%% making the kFold dataset
for i = 1:kFold
    kFoldData(i).validation = kData(i).Data;
    kFoldData(i).training = [];

    indexFold = 1:kFold;
    indexFold(i) = [];
    for j = indexFold
        kFoldData(i).training = [kFoldData(i).training; kData(j).Data];
    end
end

%% normalizing the data
for i = 1:kFold
    [tNorm, PS] = mapstd((kFoldData(i).training(:, 1:end-1)'));
    kFoldData(i).training(:,1:end-1) = tNorm';

    tNorm = mapstd((kFoldData(i).validation(:, 1:end-1)'), PS);
    kFoldData(i).validation(:, 1:end-1) = tNorm';
end

%% mrmr for feature selection for each fold
for i = 1:kFold
    [kFoldData(i).mrmrIdx,kFoldData(i).mrmrScores] = fscmrmr(kFoldData(i).training(:,1:end-1), kFoldData(i).training(:,end));
end

% save('mrmr.mat');

%% plotting mrmr for each fold
figure;
tiledlayout(2,2);
for i = 1:kFold
    nexttile;
    bar(kFoldData(i).mrmrScores(kFoldData(i).mrmrIdx));
    title(['MRMR results for the fold No. ', int2str(i)]);
    xlabel('Predictor rank');
    ylabel('Predictor importance score');
end

%% feature selection for clustering validation and clustering test datasets (with mrmr results)
clusteringNOfeatures = 100;

for i = 1:kFold
    kFoldData(i).clusteringIdxSelect = kFoldData(i).mrmrIdx(1:clusteringNOfeatures);
    kFoldData(i).clusteringTrainingSelect = [kFoldData(i).training(:, kFoldData(i).clusteringIdxSelect), kFoldData(i).training(:, end)];
    kFoldData(i).clusteringValidationSelect = [kFoldData(i).validation(:, kFoldData(i).clusteringIdxSelect), kFoldData(i).validation(:, end)];
end

%% clustering and testing the hyperparameter
clusteringRange = 12;

for i = 1:kFold
    kFoldData(i).sumOfDistanceHyperparameter = zeros(clusteringRange-1,1);
    kFoldData(i).sumOfDistance2Hyperparameter = zeros(clusteringRange-1,1);
    for k = 1:clusteringRange
        [kFoldData(i).hyperparameterClustering(k).idx, kFoldData(i).hyperparameterClustering(k).C, kFoldData(i).hyperparameterClustering(k).sumd] = kmeans(kFoldData(i).clusteringTrainingSelect(:,1:end-1), k);
        
        kFoldData(i).hyperparameterClustering(k).sumdsum = sum(kFoldData(i).hyperparameterClustering(k).sumd);
        kFoldData(i).hyperparameterClustering(k).sumd2sum = sum((kFoldData(i).hyperparameterClustering(k).sumd).^2);
    
        kFoldData(i).sumOfDistanceHyperparameter(k) = kFoldData(i).hyperparameterClustering(k).sumdsum;
        kFoldData(i).sumOfDistance2Hyperparameter(k) = kFoldData(i).hyperparameterClustering(k).sumd2sum;
    end
    kFoldData(i).variationOfSumOfDistanceHyperparameter = kFoldData(i).sumOfDistanceHyperparameter(2:end) - kFoldData(i).sumOfDistanceHyperparameter(1:end-1);
    kFoldData(i).variationOfSumOfDistance2Hyperparameter = kFoldData(i).sumOfDistance2Hyperparameter(2:end) - kFoldData(i).sumOfDistance2Hyperparameter(1:end-1);
end

%% plotting hyperparameter clustering - variation in sum of distances and distances^2
for i = 1:kFold
    figure;
    tiledlayout(2,1);
    nexttile;
    stem(2:clusteringRange, abs(kFoldData(i).variationOfSumOfDistanceHyperparameter));
    title([int2str(i), ' Variation in sum of distances vs sum of distances']);
    xlabel('Sum of distances');
    ylabel('Variation in sum of distances');
    xlim([1 clusteringRange+1]);
    nexttile;
    stem(2:clusteringRange, abs(kFoldData(i).variationOfSumOfDistance2Hyperparameter));
    title([int2str(i), ' Variation in sum of distances vs sum of distances^2']);
    xlabel('Sum of distances^2');
    ylabel('Variation in sum of distances^2');
    xlim([1 clusteringRange+1]);
end

%% selecting the number of clusters (for each fold)
% same elements as kFold number
ClusterNo = [4 4 4 4];

%% choosing the right model for clustering and then classification (for each fold)
for i = 1:kFold
    kFoldData(i).clusteringModel = kFoldData(i).hyperparameterClustering(ClusterNo(i));
end

%% classification feature selection with mrmr
featureToTrainingDivision = 10;
for i = 1:kFold
    kFoldData(i).classificationIdxSelect = kFoldData(i).mrmrIdx(1:floor(2*(kFold-1)*foldSize/featureToTrainingDivision));
    kFoldData(i).classificationTrainingSelect = [kFoldData(i).training(:, kFoldData(i).classificationIdxSelect), kFoldData(i).training(:, end)];
    kFoldData(i).classificationValidationSelect = [kFoldData(i).validation(:, kFoldData(i).classificationIdxSelect), kFoldData(i).validation(:, end)];
end

%% partitioning the training clusters for each fold
for i = 1:kFold
    for j = 1:ClusterNo(i)
        kFoldData(i).clusterData(j).trainingData = [];
        for q = 1:(foldSize*(kFold-1)*2)
            if(kFoldData(i).clusteringModel.idx(q)==j)
                kFoldData(i).clusterData(j).trainingData = [kFoldData(i).clusterData(j).trainingData; kFoldData(i).classificationTrainingSelect(q, :)];
            end
        end
    end
end

%% learning with SVM (fitcsvm)
% here
for i = 1:kFold
    for j = 1:ClusterNo(i)
        kFoldData(i).clusterData(j).model = fitcsvm(kFoldData(i).clusterData(j).trainingData(:, 1:end-1), kFoldData(i).clusterData(j).trainingData(:, end));
    end
end

% save('cluster.mat');

%% training accuracy (old method)
for i = 1:kFold
    for j = 1:ClusterNo(i)
        kFoldData(i).clusterData(j).trainingAcc = 100*sum(kFoldData(i).clusterData(j).trainingData(:, end) == predict(kFoldData(i).clusterData(j).model, kFoldData(i).clusterData(j).trainingData(:, 1:end-1)))/(size(kFoldData(i).clusterData(j).trainingData, 1));

    end
end

%% training accuracy (new method)
% for i = 1:kFold
%     kFoldData(i).trainingPredictLabels = zeros(2*(kFold-1)*foldSize, ClusterNo(i));
%     kFoldData(i).trainingScoreLabels = zeros(2*(kFold-1)*foldSize, ClusterNo(i));
%     kFoldData(i).trainingPredictLabels = zeros(2*(kFold-1)*foldSize, ClusterNo(i));
% 
%     for j = 1:ClusterNo(i)
%         
% 
%         [jLabel, jScore] = predict(kFoldData(i).clusterData(j).model, kFoldData(i).classificationValidationSelect(q, 1:end-1));
%         jDistance = kFoldData(i).clusteringValidationSelect(q, 1:end-1)-kFoldData(i).clusteringModel.C(j, :);
%         parameter = parameter + (jScore);%/(norm(jDistance));
%     end
%         
%     kFoldData(i).validatioinAcc = 100*sum(kFoldData(i).classificationTrainingPredict==kFoldData(i).classificationValidationSelect(:, end))/(2*foldSize);
% end

%% validation accuracy (old method)
for i = 1:kFold
    kFoldData(i).classificationTrainingPredict = [];

    for q = 1:size(kFoldData(i).classificationValidationSelect, 1)
        parameter = 0;

        for j = 1:ClusterNo(i)
            [jLabel, jScore] = predict(kFoldData(i).clusterData(j).model, kFoldData(i).classificationValidationSelect(q, 1:end-1));
            jDistance = kFoldData(i).clusteringValidationSelect(q, 1:end-1)-kFoldData(i).clusteringModel.C(j, :);
            parameter = parameter + (jScore);%/(norm(jDistance));
        end

        if(parameter > 1)
            kFoldData(i).classificationTrainingPredict = [kFoldData(i).classificationTrainingPredict; 1];
        else
            kFoldData(i).classificationTrainingPredict = [kFoldData(i).classificationTrainingPredict; -1];
        end
    end
    kFoldData(i).validatioinAcc = 100*sum(kFoldData(i).classificationTrainingPredict==kFoldData(i).classificationValidationSelect(:, end))/(2*foldSize);
end

%% %
%
%