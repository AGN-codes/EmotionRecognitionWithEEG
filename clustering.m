%% clearing things up
close all;
clear all;
clc;

%% loading files
load('dataset_features.mat');
load('dataset_labels.mat');
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

%% making of clustering dataset and test dataset (and shuffling) 
partition = 0.75; % percent for clustering

clusteringDataset = [dataset_positive(1:floor(partition*minValenceNO),:); dataset_negative(1:floor(partition*minValenceNO),:)];
rndm_idx_clustering = randperm(size(clusteringDataset, 1));
clusteringDataset = clusteringDataset(rndm_idx_clustering, :);

testingDataset = [dataset_positive(floor(partition*minValenceNO)+1:end,:); dataset_negative(floor(partition*minValenceNO)+1:end,:)];
rndm_idx_testing = randperm(size(testingDataset, 1));
testingDataset = testingDataset(rndm_idx_testing, :);

%% clustering feature selection with mrmr
[clusteringMrmrIdx,clusteringMrmrScores] = fscmrmr(clusteringDataset(:,1:end-1), clusteringDataset(:,end));

%% plotting clustering feature selection with mrmr
% figure;
% bar(clusteringMrmrScores(clusteringMrmrIdx));
% title('MRMR feautre selection results for the whole dataset');
% xlabel('Predictor rank');
% ylabel('Predictor importance score');

%% clustering dataset with mrmr selected features
clusteringNOfeatures = 139;
clusteringDatasetMrmr = [clusteringDataset(:, clusteringMrmrIdx(1:clusteringNOfeatures)), clusteringDataset(:, end)];

%% hyperparameter clustering
clusteringRange = 10;

sumOfDistanceHyperparameter = zeros(clusteringRange-1,1);
sumOfDistance2Hyperparameter = zeros(clusteringRange-1,1);
for k = 1:clusteringRange
    [hyperparameterClustering(k).idx, hyperparameterClustering(k).C, hyperparameterClustering(k).sumd] = kmeans(clusteringDatasetMrmr(:,1:end-1), k);
    
    hyperparameterClustering(k).sumdsum = sum(hyperparameterClustering(k).sumd);
    hyperparameterClustering(k).sumd2sum = sum((hyperparameterClustering(k).sumd).^2);

    sumOfDistanceHyperparameter(k) = hyperparameterClustering(k).sumdsum;
    sumOfDistance2Hyperparameter(k) = hyperparameterClustering(k).sumd2sum;
end

variationOfSumOfDistanceHyperparameter = sumOfDistanceHyperparameter(2:end) - sumOfDistanceHyperparameter(1:end-1);
variationOfSumOfDistance2Hyperparameter = sumOfDistance2Hyperparameter(2:end) - sumOfDistance2Hyperparameter(1:end-1);

%% plotting hyperparameter clustering - variation in sum of distances and distances^2
figure;
tiledlayout(2,1);
nexttile;
stem(2:clusteringRange, abs(variationOfSumOfDistanceHyperparameter));
title('Variation in sum of distances vs sum of distances');
xlabel('Sum of distances');
ylabel('Variation in sum of distances');
xlim([1 clusteringRange+1]);
nexttile;
stem(2:clusteringRange, abs(variationOfSumOfDistance2Hyperparameter));
title('Variation in sum of distances vs sum of distances^2');
xlabel('Sum of distances^2');
ylabel('Variation in sum of distances^2');
xlim([1 clusteringRange+1]);

%% selecting the number of clusters
ClusterNo = 4;

%% 
clusteringModel = hyperparameterClustering(ClusterNo);
%%
save('clustering.mat');
