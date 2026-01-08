%% Breast Cancer Classifier: Naive Bayes (Gaussian, KDE) and k-NN
% This script demonstrates three classifiers for the Wisconsin Breast Cancer dataset:
%   1. Gaussian Naive Bayes (parametric)
%   2. Kernel Density Estimation Naive Bayes (nonparametric)
%   3. k-Nearest Neighbor (nonparametric)
%
% The script:
%   - Loads the data
%   - Splits it into training and test sets
%   - Trains and evaluates each classifier
%   - Displays confusion matrices and accuracy
%   - Allows you to experiment with hyperparameters (kernel width h and k in k-NN)

%% Load Data
data = readtable('Breast_Cancer.csv'); % Data must be in this file
rng(1234) % For reproducibility

% Extract features (X) and labels (y)
X = data{:, 3:end};        % Features start from the third column
y = data{:, 2};            % Labels are in the second column

% Convert non-numeric labels (e.g., 'M', 'B') to numeric (0/1)
if ~isnumeric(y)
    y = grp2idx(y) - 1;    % 0: "M" (Malignant), 1: "B" (Benign)
end

%% Split into Training and Test Sets
% Use 70% of data for training, 30% for testing
cv = cvpartition(y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

classes = unique(y_train);
numClasses = length(classes);
[numTrain, numFeatures] = size(X_train);

%% 1. Gaussian Naive Bayes Classifier (Parametric)
% Each class/feature is modeled by a Gaussian

% Calculate prior probability for each class (empirical frequency)
prior = zeros(numClasses, 1);
for c = 1:numClasses
    idx = (y_train == classes(c));
    N_idx = sum(idx);
    prior(c) = N_idx / numTrain; % Probability of each class
end

% Calculate mean and variance for each class/feature
mu = zeros(numClasses, numFeatures);
sigma = zeros(numClasses, numFeatures);
for c = 1:numClasses
    idx = (y_train == classes(c));
    X_c = X_train(idx, :);
    mu(c, :) = mean(X_c, 1);         % Mean per feature
    sigma(c, :) = var(X_c, 1);       % Variance per feature (biased: N)
end

% Predict test set labels using Gaussian NB
numTest = size(X_test, 1);
y_predGaussian = zeros(numTest, 1);
for i = 1:numTest
    posteriors = zeros(numClasses, 1);
    for c = 1:numClasses
        likelihood = 1;
        for j = 1:numFeatures
            sigma_val = sigma(c, j);
            if sigma_val == 0, sigma_val = eps; end % Prevent division by zero
            % Gaussian PDF:
            likelihood = likelihood * (1/sqrt(2*pi*sigma_val)) * ...
                exp(-((X_test(i, j) - mu(c, j))^2) / (2*sigma_val));
        end
        posteriors(c) = likelihood * prior(c);
    end
    % MAP decision rule: class with maximum posterior
    [~, idx] = max(posteriors);
    y_predGaussian(i) = classes(idx);
end

% Evaluate Gaussian NB
cmGaussian = confusionmat(y_test, y_predGaussian);
accuracyGaussian = sum(diag(cmGaussian)) / sum(cmGaussian(:));
fprintf('Gaussian Naive Bayes Accuracy: %.2f%%\n', accuracyGaussian * 100);
figure; confusionchart(cmGaussian);
title('Confusion Matrix - Gaussian Naive Bayes');

%% 2. Kernel Density Estimation (KDE) Naive Bayes Classifier (Nonparametric)
% Here, feature distributions are estimated using kernel density from training data

% Prepare data for each class
X_train_byClass = cell(numClasses, 1);
for c = 1:numClasses
    idx = (y_train == classes(c));
    X_train_byClass{c} = X_train(idx, :);
end

% Set kernel width (bandwidth) - you can change this value
h = 75;   % Try different h: e.g., 5, 10, 25, 100

y_predKernel = zeros(numTest, 1);
for i = 1:numTest
    posteriors = zeros(numClasses, 1);
    for c = 1:numClasses
        likelihood = 1;
        X_class = X_train_byClass{c};
        n_c = size(X_class, 1);
        for j = 1:numFeatures
            diff = X_test(i, j) - X_class(:, j);
            kernel_vals = (1/(h*sqrt(2*pi))) * exp(-(diff.^2) / (2*h^2));
            p_xj = sum(kernel_vals) / n_c;
            likelihood = likelihood * p_xj;
        end
        posteriors(c) = likelihood * prior(c);
    end
    [~, idx] = max(posteriors);
    y_predKernel(i) = classes(idx);
end

% Evaluate KDE NB
cmKernel = confusionmat(y_test, y_predKernel);
accuracyKernel = sum(diag(cmKernel)) / sum(cmKernel(:));
fprintf('Kernel Naive Bayes (h = %.2f) Accuracy: %.2f%%\n', h, accuracyKernel * 100);
figure; confusionchart(cmKernel);
title(sprintf('Confusion Matrix - Kernel Naive Bayes (h = %.2f)', h));

%% 3. k-Nearest Neighbor Classifier (Nonparametric)
% Classifies based on the majority class among the k nearest neighbors

% Set number of neighbors - experiment with this value!
numNeighbors = 3; % Try e.g., 1, 3, 5, 10, 15, 20
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', numNeighbors);

% Predict on test set
y_predKNN = predict(knnModel, X_test);

% Evaluate k-NN
cmKNN = confusionmat(y_test, y_predKNN);
accuracyKNN = sum(diag(cmKNN)) / sum(cmKNN(:));
fprintf('KNN (NumNeighbors = %d) Accuracy: %.2f%%\n', numNeighbors, accuracyKNN * 100);
figure; confusionchart(cmKNN);
title(sprintf('Confusion Matrix - KNN (NumNeighbors = %d)', numNeighbors));

%% -------------------------
% Interpretation/Experimentation Instructions:
%
% 1. Compare accuracy and confusion matrices for all classifiers.
% 2. Notice how Gaussian NB is a *parametric* method (fixed, simple model),
%    while KDE NB and k-NN are *nonparametric* (can become arbitrarily complex with more data).
% 3. Experiment by changing "h" (KDE) and "numNeighbors" (k-NN) above:
%    - Small h: sharp kernels, risk overfitting. Large h: smooth, may underfit.
%    - Small k: more flexible, more noise. Large k: more stable, less flexible.
% 4. Discuss, for each method, the main types of errors visible in the confusion matrix:
%    - Which class is most commonly misclassified? Are there more false positives or negatives?
%    - Which model achieves the best trade-off for your application?
%
% (End of script)
