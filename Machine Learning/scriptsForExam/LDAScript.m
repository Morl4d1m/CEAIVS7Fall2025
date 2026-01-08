%% ============================================================
%  LINEAR DISCRIMINANT ANALYSIS (LDA) DEMO SCRIPT
%  All parameters are defined here
%  Now displays intermediate calculations and explanations
% ============================================================

clear; clc; close all;

%% ===================== USER PARAMETERS ======================

% Data source: 'random', 'file', or 'manual'
dataSource = 'random';

% Number of features (must be 2 for boundary plotting)
numFeatures = 2;

% Random data parameters (used if dataSource = 'random')
numSamplesClass1 = 100;
numSamplesClass2 = 100;
meanClass1 = [1; 2];
meanClass2 = [4; 5];
covClass1 = [1 0.3; 0.3 1];
covClass2 = [1 -0.2; -0.2 1];

% File loading parameters (used if dataSource = 'file')
% File must contain variables: X (Nx2) and y (Nx1, labels 1 or 2)
dataFile = 'lda_data.mat';

% Manual data (used if dataSource = 'manual')
X_manual = [1 2; 2 3; 3 3; 6 5; 7 6; 8 5];
y_manual = [1; 1; 1; 2; 2; 2];

% Prior probabilities (optional, set empty for equal priors)
prior1 = [];
prior2 = [];

%% ===================== LOAD / GENERATE DATA =================

disp('=== Loading / Generating Data ===');

switch lower(dataSource)
    case 'random'
        fprintf('Generating random data for Class 1 and Class 2...\n');
        X1 = mvnrnd(meanClass1, covClass1, numSamplesClass1);
        X2 = mvnrnd(meanClass2, covClass2, numSamplesClass2);
        X = [X1; X2];
        y = [ones(numSamplesClass1,1); 2*ones(numSamplesClass2,1)];
        fprintf('Data generated with %d samples per class.\n', numSamplesClass1);

    case 'file'
        fprintf('Loading data from file: %s\n', dataFile);
        S = load(dataFile);
        X = S.X;
        y = S.y;

    case 'manual'
        fprintf('Using manual data input.\n');
        X = X_manual;
        y = y_manual;

    otherwise
        error('Invalid data source selection.');
end

%% ===================== SEPARATE CLASSES =====================

X1 = X(y == 1, :);
X2 = X(y == 2, :);

n1 = size(X1,1);
n2 = size(X2,1);

if isempty(prior1)
    prior1 = n1 / (n1 + n2);
    prior2 = n2 / (n1 + n2);
end

fprintf('\nClass priors:\n');
fprintf('  Prior1 = %.3f\n', prior1);
fprintf('  Prior2 = %.3f\n', prior2);

%% ===================== LDA COMPUTATION ======================

disp('=== Computing LDA ===');

% Class means
mu1 = mean(X1)';
mu2 = mean(X2)';
fprintf('\nClass means:\n');
fprintf('  mu1 = [%s]\n', num2str(mu1', ' %.3f'));
fprintf('  mu2 = [%s]\n', num2str(mu2', ' %.3f'));

% Within-class scatter matrix
S1 = cov(X1); 
S2 = cov(X2);
Sw = S1 + S2;
fprintf('\nWithin-class scatter matrices:\n');
fprintf('  S1 = \n'); disp(S1);
fprintf('  S2 = \n'); disp(S2);
fprintf('  Sw (S1 + S2) = \n'); disp(Sw);

% LDA projection vector
w = Sw \ (mu2 - mu1);  % equivalent to inv(Sw)*(mu2 - mu1)
w = w / norm(w);        % normalize
fprintf('\nLDA projection vector w (normalized):\n');
disp(w);

% Project data
z1 = X1 * w;
z2 = X2 * w;

% Projected means
m1 = mean(z1);
m2 = mean(z2);
fprintf('\nProjected class means on LDA axis:\n');
fprintf('  m1 = %.3f\n', m1);
fprintf('  m2 = %.3f\n', m2);

% Decision threshold
threshold = (m1 + m2)/2 + log(prior2/prior1) / (m2 - m1);
fprintf('\nDecision threshold in projected space:\n');
fprintf('  threshold = %.3f\n', threshold);

%% ===================== PLOTS ================================

disp('=== Plotting Data ===');

% -------------------------------------------------------------
% Plot 1: Original data
% -------------------------------------------------------------
figure;
hold on; grid on;
scatter(X1(:,1), X1(:,2), 50, 'b', 'filled');
scatter(X2(:,1), X2(:,2), 50, 'r', 'filled');
plot(mu1(1), mu1(2), 'bx', 'MarkerSize', 12, 'LineWidth', 2);
plot(mu2(1), mu2(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
legend('Class 1','Class 2','Mean 1','Mean 2');
title('Original Data');
xlabel('Feature 1');
ylabel('Feature 2');
axis equal;

% -------------------------------------------------------------
% Plot 2: LDA projection direction
% -------------------------------------------------------------
scale = max(range(X));
quiver(mean(X(:,1)), mean(X(:,2)), ...
       w(1)*scale, w(2)*scale, ...
       'k','LineWidth',2);
text(mean(X(:,1)), mean(X(:,2)), '  LDA direction');

% -------------------------------------------------------------
% Plot 3: Projected data (1D)
% -------------------------------------------------------------
figure;
hold on; grid on;
histogram(z1, 20, 'Normalization','pdf');
histogram(z2, 20, 'Normalization','pdf');
xline(threshold, 'k', 'LineWidth', 2);
legend('Class 1','Class 2','Decision threshold');
title('Projected Data on LDA Axis');
xlabel('Projection value');
ylabel('Probability density');

% -------------------------------------------------------------
% Plot 4: Decision boundary in original space
% -------------------------------------------------------------
xRange = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
yRange = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
[Xg, Yg] = meshgrid(xRange, yRange);
gridPoints = [Xg(:) Yg(:)];
scores = gridPoints * w;

decisionMap = reshape(scores > threshold, size(Xg));

figure;
hold on; grid on;
contourf(Xg, Yg, decisionMap, [0 1], 'FaceAlpha',0.2);
scatter(X1(:,1), X1(:,2), 40, 'b', 'filled');
scatter(X2(:,1), X2(:,2), 40, 'r', 'filled');
title('LDA Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
axis equal;
legend('Decision region','Class 1','Class 2');

disp('=== LDA Completed ===');
