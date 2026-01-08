%% =========================
%  k-Nearest Neighbours Script (with detailed console output)
%  Data source selectable
%  =========================

clear; clc; close all;

%% -------- USER PARAMETERS --------

dataSource = 'random';     % 'file', 'manual', 'random'
dataFile = 'data.mat';

X_manual = [1 2; 2 3; 3 3; 6 5; 7 7];
y_manual = [0; 0; 0; 1; 1];

N = 200; D = 2; taskType = 'classification';
numClasses = 2; noiseStd = 0.2;

k = 3;
distanceMetric = 'euclidean';   % 'euclidean', 'cityblock', 'cosine', 'chebychev'
weighting = 'uniform';          % 'uniform' or 'distance'

trainRatio = 0.7;
normalizeFeatures = true;
randomSeed = 1;

%% -------- LOAD / CREATE DATA --------

rng(randomSeed);
fprintf('\n=== Loading / Generating Data ===\n');

switch dataSource
    case 'file'
        fprintf('Loading data from file: %s\n', dataFile);
        S = load(dataFile);
        X = S.X;
        y = S.y;
        
    case 'manual'
        fprintf('Using manually defined data\n');
        X = X_manual;
        y = y_manual;
        
    case 'random'
        fprintf('Generating random synthetic data\n');
        if strcmp(taskType,'classification')
            X = randn(N,D);
            y = randi(numClasses,N,1) - 1;
        else
            X = randn(N,D);
            trueW = randn(D,1);
            y = X*trueW + noiseStd*randn(N,1);
        end
    otherwise
        error('Unknown dataSource option');
end

fprintf('Data size: %d samples, %d features\n', size(X,1), size(X,2));

%% -------- TRAIN / TEST SPLIT --------
fprintf('\n=== Splitting Data into Train and Test ===\n');

Ntotal = size(X,1);
idx = randperm(Ntotal);

Ntrain = round(trainRatio * Ntotal);
trainIdx = idx(1:Ntrain);
testIdx  = idx(Ntrain+1:end);

Xtrain = X(trainIdx,:);
ytrain = y(trainIdx);

Xtest = X(testIdx,:);
ytest = y(testIdx);

fprintf('Training samples: %d\n', Ntrain);
fprintf('Testing samples: %d\n', Ntotal - Ntrain);

Xtrain_raw = Xtrain;
Xtest_raw  = Xtest;

%% -------- FEATURE NORMALIZATION --------
if normalizeFeatures
    fprintf('\n=== Normalizing Features ===\n');
    mu = mean(Xtrain,1);
    sigma = std(Xtrain,[],1);
    sigma(sigma == 0) = 1;

    Xtrain = (Xtrain - mu) ./ sigma;
    Xtest  = (Xtest  - mu) ./ sigma;
    
    fprintf('Feature means after normalization (should be ~0):\n');
    disp(mean(Xtrain,1));
    fprintf('Feature std devs after normalization (should be 1):\n');
    disp(std(Xtrain,[],1));
end

%% -------- kNN IMPLEMENTATION --------
fprintf('\n=== kNN Prediction ===\n');
Ntest = size(Xtest,1);
ypred = zeros(Ntest,1);

for i = 1:Ntest
    fprintf('\nPredicting sample %d (true label: %d)\n', i, ytest(i));
    
    d = pdist2(Xtrain, Xtest(i,:), distanceMetric);
    fprintf('Distances to all training samples:\n');
    disp(d');
    
    [dSorted, idxSorted] = sort(d,'ascend');
    kEff = min(k, numel(idxSorted));
    nnIdx = idxSorted(1:kEff);
    nnLabels = ytrain(nnIdx);
    
    fprintf('Indices of %d nearest neighbors: ', kEff); disp(nnIdx');
    fprintf('Labels of nearest neighbors: '); disp(nnLabels');
    
    if strcmp(weighting,'distance')
        w = 1 ./ (dSorted(1:kEff) + eps);
        fprintf('Weights (inverse distance): '); disp(w');
    else
        w = ones(kEff,1);
        fprintf('Uniform weights: '); disp(w');
    end
    
    if strcmp(taskType,'classification')
        classes = unique(ytrain);
        scores = zeros(length(classes),1);

        for c = 1:length(classes)
            scores(c) = sum(w(nnLabels == classes(c)));
            fprintf('Class %d score: %.3f\n', classes(c), scores(c));
        end

        [~,maxIdx] = max(scores);
        ypred(i) = classes(maxIdx);
        fprintf('Predicted label: %d\n', ypred(i));
        
    else
        ypred(i) = sum(w .* nnLabels) / sum(w);
        fprintf('Predicted value (regression): %.3f\n', ypred(i));
    end
end

%% -------- EVALUATION --------
fprintf('\n=== Evaluation ===\n');
if strcmp(taskType,'classification')
    accuracy = mean(ypred == ytest);
    fprintf('Classification accuracy: %.2f %%\n', accuracy * 100);
else
    rmse = sqrt(mean((ypred - ytest).^2));
    fprintf('Regression RMSE: %.4f\n', rmse);
end

%% -------- OPTIONAL VISUALIZATION --------

if size(X,2) == 2 && strcmp(taskType,'classification')

    %% ---- Plot 1: Train (true) + Test (predicted)
    figure; hold on;
    gscatter(Xtrain_raw(:,1), Xtrain_raw(:,2), ytrain, 'br','o',8,'off');
    gscatter(Xtest_raw(:,1), Xtest_raw(:,2), ypred, 'br','x',10,'off');

    legend({'Train class 0','Train class 1', ...
            'Test predicted class 0','Test predicted class 1'}, ...
            'Location','best');

    title(sprintf('kNN Classification Result (k=%d)',k));
    xlabel('Feature 1'); ylabel('Feature 2');
    grid on;

    %% ---- Plot 2: Decision Boundary
    x1range = linspace(min(X(:,1))-1, max(X(:,1))+1, 300);
    x2range = linspace(min(X(:,2))-1, max(X(:,2))+1, 300);
    [x1g,x2g] = meshgrid(x1range,x2range);
    Xgrid_raw = [x1g(:) x2g(:)];

    Xgrid = Xgrid_raw;
    if normalizeFeatures
        Xgrid = (Xgrid - mu) ./ sigma;
    end

    ygrid = zeros(size(Xgrid,1),1);

    for i = 1:size(Xgrid,1)
        d = pdist2(Xtrain, Xgrid(i,:), distanceMetric);
        [dSorted, idxSorted] = sort(d,'ascend');
        kEff = min(k, numel(idxSorted));
        nnIdx = idxSorted(1:kEff);
        nnLabels = ytrain(nnIdx);

        if strcmp(weighting,'distance')
            w = 1 ./ (dSorted(1:kEff) + eps);
        else
            w = ones(kEff,1);
        end

        classes = unique(ytrain);
        scores = zeros(length(classes),1);
        for c = 1:length(classes)
            scores(c) = sum(w(nnLabels == classes(c)));
        end

        [~,maxIdx] = max(scores);
        ygrid(i) = classes(maxIdx);
    end

    ygrid = reshape(ygrid,size(x1g));

    figure; hold on;
    contourf(x1g,x2g,ygrid,[0 0.5 1],'LineColor','none');
    colormap([0.85 0.85 1; 1 0.85 0.85]);

    gscatter(Xtrain_raw(:,1),Xtrain_raw(:,2),ytrain,'br','o',6,'off');
    gscatter(Xtest_raw(:,1),Xtest_raw(:,2),ytest,'br','x',8,'off');

    title(sprintf('kNN Decision Boundary (k=%d)',k));
    xlabel('Feature 1'); ylabel('Feature 2');
    grid on;

    %% ---- Plot 3: Confusion Matrix
    figure;
    confusionchart(ytest, ypred);
    title('Confusion Matrix (Test Data)');

end
