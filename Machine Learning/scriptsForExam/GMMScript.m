%% Gaussian Mixture Model (GMM) Demo Script
% This script allows you to input or generate data, fit a GMM, and visualize results.

clear; clc; close all;

%% ===================== PARAMETERS =====================
% Data options
dataOption = 'random'; % Options: 'file', 'manual', 'random'

% File options (used if dataOption = 'file')
dataFile = 'data.csv'; % CSV file with numeric columns

% Manual input (used if dataOption = 'manual')
manualData = [1,2; 2,1; 3,5; 5,3]; % Each row is a data point

% Random data generation (used if dataOption = 'random')
numPoints = 300;       % Total number of data points
numClusters = 3;       % Number of Gaussian components
randomSeed = 42;       % Seed for reproducibility

% GMM options
covarianceType = 'full'; % 'full', 'diagonal', 'spherical'
maxIterations = 500;     % Maximum EM iterations
tolerance = 1e-6;        % Convergence tolerance

%% ===================== LOAD / GENERATE DATA =====================
switch lower(dataOption)
    case 'file'
        data = readmatrix(dataFile); % Load CSV
    case 'manual'
        data = manualData;
    case 'random'
        rng(randomSeed);
        % Generate random Gaussian clusters
        data = [];
        mu = [0 0; 5 5; -5 5];  % means
        Sigma(:,:,1) = [1 0.5; 0.5 1];
        Sigma(:,:,2) = [1 -0.3; -0.3 1];
        Sigma(:,:,3) = [0.5 0; 0 0.5];
        pointsPerCluster = floor(numPoints/numClusters);
        for k = 1:numClusters
            data = [data; mvnrnd(mu(k,:), Sigma(:,:,k), pointsPerCluster)];
        end
    otherwise
        error('Invalid data option!');
end

%% ===================== FIT GMM =====================
options = statset('MaxIter', maxIterations, 'TolFun', tolerance);
gmm = fitgmdist(data, numClusters, 'CovarianceType', covarianceType, ...
    'Options', options);

% Display GMM parameters
disp('GMM Means:');
disp(gmm.mu);
disp('GMM Covariances:');
disp(gmm.Sigma);
disp('GMM Component Weights:');
disp(gmm.ComponentProportion);

%% ===================== PLOT DATA =====================
figure;
scatter(data(:,1), data(:,2), 36, 'k', 'filled');
title('Data Points');
xlabel('X1'); ylabel('X2'); grid on;

%% ===================== PLOT CLUSTER ASSIGNMENTS =====================
clusterIdx = cluster(gmm, data);
figure;
gscatter(data(:,1), data(:,2), clusterIdx);
hold on;
plot(gmm.mu(:,1), gmm.mu(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('GMM Cluster Assignments');
xlabel('X1'); ylabel('X2'); grid on;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids');

%% ===================== PLOT GMM PDF CONTOURS =====================
x = linspace(min(data(:,1))-1, max(data(:,1))+1, 100);
y = linspace(min(data(:,2))-1, max(data(:,2))+1, 100);
[X, Y] = meshgrid(x, y);
XY = [X(:) Y(:)];
pdfVals = pdf(gmm, XY);
pdfVals = reshape(pdfVals, size(X));

figure;
contour(X, Y, pdfVals);
hold on;
scatter(data(:,1), data(:,2), 15, clusterIdx, 'filled');
plot(gmm.mu(:,1), gmm.mu(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('GMM PDF Contours');
xlabel('X1'); ylabel('X2'); grid on;

%% ===================== OPTIONAL: 3D SURFACE PLOT =====================
figure;
surf(X, Y, pdfVals, 'EdgeColor', 'none');
hold on;
scatter3(data(:,1), data(:,2), zeros(size(data,1),1), 15, clusterIdx, 'filled');
plot3(gmm.mu(:,1), gmm.mu(:,2), max(pdfVals(:))*ones(numClusters,1), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('GMM PDF Surface Plot');
xlabel('X1'); ylabel('X2'); zlabel('PDF'); grid on;
view(45,30);
colorbar;

disp('GMM fitting and plotting completed!');
