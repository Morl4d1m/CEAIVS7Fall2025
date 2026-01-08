%% ============================================================
%  PCA DEMONSTRATION SCRIPT (COV or SVD) WITH INTERMEDIATE OUTPUT
% ============================================================

clear; close all; clc;

%% ---------------- USER PARAMETERS ----------------

DATA_SOURCE = 'iris';     % 'file', 'manual', 'random', 'iris'
PCA_METHOD = 'svd';       % 'cov' or 'svd'
STANDARDIZE_DATA = false;  % true = z-score
NUM_COMPONENTS = [];       % [] = all
N_SAMPLES  = 200;
N_FEATURES = 4;
RANDOM_SEED = 1;
DATA_FILE = 'data.csv';
FILE_VARIABLE = 'X';
X_manual = [
    2 4 6;
    3 5 7;
    4 6 8;
    5 7 9
]';
SHOW_3D = true;

%% ============================================================
%  LOAD DATA
% ============================================================

fprintf('--- Loading Data ---\n');
labels = [];
featureNames = [];

switch lower(DATA_SOURCE)
    case 'iris'
        load fisheriris
        X = meas';  % d x N
        labels = species;
        featureNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};
        fprintf('Loaded Iris dataset: %d features, %d samples\n', size(X,1), size(X,2));
    case 'file'
        [~,~,ext] = fileparts(DATA_FILE);
        if strcmp(ext,'.mat')
            S = load(DATA_FILE);
            X = S.(FILE_VARIABLE);
        else
            X = readmatrix(DATA_FILE)';
        end
    case 'manual'
        X = X_manual;
        fprintf('Using manual data: %d features, %d samples\n', size(X,1), size(X,2));
    case 'random'
        rng(RANDOM_SEED);
        X = randn(N_FEATURES, N_SAMPLES) .* (1:N_FEATURES)' + randn(N_FEATURES,1)*3;
        fprintf('Generated random data: %d features, %d samples\n', size(X,1), size(X,2));
    otherwise
        error('Invalid DATA_SOURCE');
end

[d, N] = size(X);
if isempty(NUM_COMPONENTS)
    NUM_COMPONENTS = d;
end

%% ============================================================
%  CENTER / STANDARDIZE
% ============================================================

fprintf('\n--- Centering and Standardizing Data ---\n');
X_bar = mean(X, 2);
Xc = X - X_bar;
fprintf('Mean of each feature (before centering):\n');
disp(X_bar');

if STANDARDIZE_DATA
    sigma = std(Xc,0,2);
    Xc = Xc ./ sigma;
    fprintf('Data standardized by feature standard deviation:\n');
    disp(sigma');
else
    sigma = ones(d,1);
    fprintf('Standardization not applied.\n');
end

%% ============================================================
%  PCA COMPUTATION
% ============================================================

fprintf('\n--- PCA Computation (%s method) ---\n', PCA_METHOD);

switch lower(PCA_METHOD)
    case 'cov'
        C = (Xc * Xc') / (N-1);
        fprintf('Covariance matrix C:\n');
        disp(C);
        [U, L] = eig(C);
        [lambda, idx] = sort(diag(L), 'descend');
        U = U(:,idx);
        fprintf('Eigenvalues (sorted):\n');
        disp(lambda');
        S = diag(sqrt(lambda*(N-1)));
        V = (U' * Xc)' ./ diag(S)';
    case 'svd'
        [U, S, V] = svd(Xc, 'econ');
        lambda = diag(S).^2 / (N-1);
        fprintf('Singular values (S):\n');
        disp(diag(S)');
        fprintf('Eigenvalues from SVD:\n');
        disp(lambda');
    otherwise
        error('Invalid PCA_METHOD');
end

explained = 100 * lambda / sum(lambda);
cumExplained = cumsum(explained);
fprintf('Explained variance per PC (%%):\n');
disp(explained');
fprintf('Cumulative explained variance (%%):\n');
disp(cumExplained');

U_r = U(:,1:NUM_COMPONENTS);
Z = U_r' * Xc;   % PC scores
fprintf('PC scores (first 5 samples):\n');
disp(Z(:,1:min(5,N))');

%% ============================================================
%  FEATURE LOADINGS
% ============================================================

fprintf('\n--- Feature Loadings ---\n');
for k = 1:min(2,d)
    fprintf('PC%d loadings:\n', k);
    disp(U(:,k)');
end

%% ============================================================
%  RECONSTRUCTION & ERROR
% ============================================================

fprintf('\n--- Reconstruction & Error ---\n');
maxK = d;
mse = zeros(maxK,1);

for k = 1:maxK
    Xc_hat = U(:,1:k) * (U(:,1:k)' * Xc);
    mse(k) = mean(sum((Xc - Xc_hat).^2,1));
    fprintf('MSE with %d PCs: %.4f\n', k, mse(k));
end

%% ============================================================
%  FULL RECONSTRUCTION USING SELECTED PCs
% ============================================================

Xc_hat = U_r * Z;
X_hat  = Xc_hat .* sigma + X_bar;
fprintf('\nMean reconstruction error using %d PCs: %.4f\n', NUM_COMPONENTS, mean(sum((X - X_hat).^2,1)));

%% ============================================================
%  PLOTTING (unchanged)
% ============================================================

% Explained variance figure
figure;
bar(explained,'FaceAlpha',0.6); hold on
plot(cumExplained,'r-o','LineWidth',1.5);
hold off
xlabel('Principal Component'); ylabel('Variance Explained (%)');
title('Explained Variance');
legend('Individual','Cumulative','Location','Best'); grid on;

% Score plots (PC1 vs PC2)
if NUM_COMPONENTS >= 2
    figure; hold on;
    if isempty(labels)
        scatter(Z(1,:), Z(2,:), 40, 'filled');
    else
        labs = unique(labels); markers = {'o','s','^','d','v','>'}; colors = lines(numel(labs));
        for c = 1:numel(labs)
            idx = strcmp(labels, labs{c});
            scatter(Z(1,idx), Z(2,idx), 40, colors(c,:), markers{c}, 'filled');
        end
        legend(labs,'Location','Best');
    end
    hold off; xlabel('PC1'); ylabel('PC2'); title('PC1 vs PC2'); grid on;
end

% Score plots (PC1 vs PC2 vs PC3)
if NUM_COMPONENTS >= 3 && SHOW_3D
    figure; hold on;
    if isempty(labels)
        scatter3(Z(1,:), Z(2,:), Z(3,:), 40, 'filled');
    else
        for c = 1:numel(labs)
            idx = strcmp(labels, labs{c});
            scatter3(Z(1,idx), Z(2,idx), Z(3,idx), 40, colors(c,:), markers{c}, 'filled');
        end
        legend(labs,'Location','Best');
    end
    hold off; xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
    title('PC1–PC2–PC3'); grid on; view(-30,10);
end

% Feature loadings figure
figure; bar(U(:,1:min(2,d))); ylabel('Loading'); title('Feature Loadings');
if ~isempty(featureNames)
    set(gca,'XTick',1:d,'XTickLabel',featureNames,'XTickLabelRotation',45);
end
legend('PC1','PC2','Location','Best'); grid on;

% Reconstruction MSE figure
figure; plot(1:maxK, mse,'-o','LineWidth',1.5);
xlabel('Number of PCs'); ylabel('Reconstruction MSE'); title('Reconstruction Error vs Number of PCs'); grid on;
