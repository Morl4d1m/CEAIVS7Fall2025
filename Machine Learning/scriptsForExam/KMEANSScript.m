%% =========================================================
%  K-MEANS CLUSTERING DEMO SCRIPT
%  All parameters are set here
%  Updated to show intermediate calculations
% =========================================================

clear; close all; clc;

%% -------------------------
% USER PARAMETERS
% -------------------------

dataSource = 'random';       % 'file' | 'manual' | 'random'
fileName = 'data.csv';       % Used only if dataSource = 'file'
fileVariable = 'X';          % Used only for .mat files

numPoints = 300;
numFeatures = 2;             % Must be 2 for visualization
randomSeed = 1;

X_manual = [ ...
    1 1;
    1.5 2;
    3 4;
    5 7;
    3.5 5;
    4.5 5;
    3.5 4.5 ];

K = 3;                       % Number of clusters
distanceMetric = 'sqeuclidean';
numReplicates = 10;
maxIterations = 300;
displayKmeans = 'iter';      % Show iteration info

plotElbow = true;
maxKElbow = 10;

%% =========================================================
% DATA LOADING / GENERATION
% =========================================================

switch lower(dataSource)
    case 'file'
        [~,~,ext] = fileparts(fileName);
        if strcmp(ext,'.mat')
            S = load(fileName);
            X = S.(fileVariable);
        elseif strcmp(ext,'.csv')
            X = readmatrix(fileName);
        else
            error('Unsupported file format.');
        end
        fprintf('Loaded %d points with %d features from file.\n', size(X,1), size(X,2));
        
    case 'manual'
        X = X_manual;
        fprintf('Using manual data with %d points and %d features.\n', size(X,1), size(X,2));
        
    case 'random'
        rng(randomSeed);
        X = randn(numPoints, numFeatures);
        fprintf('Generated %d random points with %d features.\n', numPoints, numFeatures);
        
    otherwise
        error('Unknown dataSource option.');
end

if size(X,2) < 2
    error('Data must have at least 2 features for visualization.');
end

disp('First 5 data points:');
disp(X(1:min(5,end),:));

%% =========================================================
% RUN K-MEANS WITH EXPLANATIONS
% =========================================================

opts = statset('MaxIter', maxIterations, 'Display', displayKmeans);

fprintf('\nRunning K-means clustering with K = %d...\n', K);
[idx, C, sumd, D] = kmeans(X, K, ...
    'Distance', distanceMetric, ...
    'Replicates', numReplicates, ...
    'Options', opts);

fprintf('\nCluster centroids:\n');
disp(C);

fprintf('Sample distances of first 5 points to centroids:\n');
disp(D(1:min(5,end),:));

totalWCSS = sum(sumd);
fprintf('Total within-cluster sum of squares (WCSS): %.4f\n', totalWCSS);

% Show cluster assignment of first 10 points
fprintf('Cluster assignment for first 10 points:\n');
disp(idx(1:min(10,end)));

%% =========================================================
% PLOT 1: CLUSTERED DATA
% =========================================================

figure('Name','K-Means Clustering');
gscatter(X(:,1), X(:,2), idx);
hold on;
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title(sprintf('K-Means Clustering (K = %d)', K));
xlabel('Feature 1');
ylabel('Feature 2');
legend('Location','best');
grid on;
hold off;

%% =========================================================
% PLOT 2: DISTANCE TO CENTROIDS
% =========================================================

figure('Name','Distances to Centroids');
imagesc(D);
colorbar;
xlabel('Cluster Index');
ylabel('Data Point Index');
title('Distance of Each Point to Each Centroid');

%% =========================================================
% PLOT 3: SILHOUETTE
% =========================================================

figure('Name','Silhouette Plot');
silhouette(X, idx, distanceMetric);
title('Silhouette Analysis');
grid on;

%% =========================================================
% PLOT 4: ELBOW METHOD WITH EXPLANATIONS
% =========================================================

if plotElbow
    WCSS = zeros(maxKElbow,1);
    fprintf('\nComputing WCSS for K = 1:%d for elbow method...\n', maxKElbow);
    
    for k = 1:maxKElbow
        [~,~,sumd_k] = kmeans(X, k, ...
            'Distance', distanceMetric, ...
            'Replicates', numReplicates, ...
            'Options', opts);
        WCSS(k) = sum(sumd_k);
        fprintf('K = %d, WCSS = %.4f\n', k, WCSS(k));
    end
    
    reduction = WCSS(1:end-1) - WCSS(2:end);
    fprintf('Reduction in WCSS for each increase in K:\n');
    for k = 2:maxKElbow
        fprintf('K = %d -> K = %d, Reduction = %.4f\n', k-1, k, reduction(k-1));
    end
    
    % Plot reduction
    figure('Name','Elbow Plot: Reduction in WCSS');
    plot(2:maxKElbow, reduction, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Number of Clusters (K)');
    ylabel('Reduction in WCSS');
    title('Elbow Plot: Reduction in WCSS');
    grid on;
    
    % Plot WCSS
    figure('Name','Elbow Method');
    plot(1:maxKElbow, WCSS, '-o','LineWidth',2);
    xlabel('Number of Clusters (K)');
    ylabel('Within-Cluster Sum of Squares');
    title('Elbow Method for Optimal K');
    grid on;
end

fprintf('\nK-means demo completed.\n');
