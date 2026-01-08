%% DECISION TREE OVERVIEW SCRIPT
% Fully corrected and version-robust
% Updated to display intermediate math, explanations, and results

clear; clc; close all;

%% ========================= USER PARAMETERS =========================

DATA_SOURCE = 'random'; % 'file', 'manual', 'random'
DATA_FILE = 'data.csv';
LABEL_COLUMN = 'Class';

NUM_SAMPLES  = 300;
NUM_FEATURES = 2;
NUM_CLASSES  = 2;
RANDOM_SEED  = 1;

MAX_DEPTH        = 5;
MIN_LEAF_SIZE    = 5;
SPLIT_CRITERION  = 'gdi';

GRID_RESOLUTION = 200;

%% ========================= DATA LOADING =========================

fprintf('\n--- Loading Data ---\n');

switch lower(DATA_SOURCE)
    case 'file'
        [~,~,ext] = fileparts(DATA_FILE);
        fprintf('Loading data from file: %s\n', DATA_FILE);
        if strcmp(ext,'.mat')
            S = load(DATA_FILE);
            X = S.X;
            Y = S.Y;
        else
            T = readtable(DATA_FILE);
            Y = categorical(T.(LABEL_COLUMN));
            T.(LABEL_COLUMN) = [];
            X = table2array(T);
        end

    case 'manual'
        fprintf('Using manual dataset.\n');
        X = [1 1; 1 2; 2 1; 2 2; 3 3; 3 4; 4 3; 4 4];
        Y = categorical([0 0 0 0 1 1 1 1]);

    case 'random'
        rng(RANDOM_SEED);
        fprintf('Generating random dataset: %d samples, %d features, %d classes.\n', ...
            NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES);
        X = randn(NUM_SAMPLES, NUM_FEATURES);
        Y = categorical(randi(NUM_CLASSES, NUM_SAMPLES, 1));

    otherwise
        error('Unknown DATA_SOURCE');
end

fprintf('Data loaded: %d samples, %d features.\n', size(X,1), size(X,2));

%% ========================= TRAIN DECISION TREE =========================

fprintf('\n--- Training Decision Tree ---\n');
tree = fitctree( ...
    X, Y, ...
    'MaxNumSplits', 2^MAX_DEPTH - 1, ...
    'MinLeafSize', MIN_LEAF_SIZE, ...
    'SplitCriterion', SPLIT_CRITERION);

fprintf('Tree trained with MaxDepth=%d, MinLeafSize=%d, SplitCriterion=%s.\n', ...
    MAX_DEPTH, MIN_LEAF_SIZE, SPLIT_CRITERION);

%% ========================= TREE STRUCTURE =========================

fprintf('\n--- Tree Structure ---\n');
if usejava('jvm')
    figure('Name','Decision Tree Structure');
    view(tree,'Mode','graph');
    fprintf('Tree visualized graphically.\n');
else
    fprintf('Text-based tree view:\n');
    disp(evalc('view(tree)'));
end

%% ========================= FEATURE IMPORTANCE =========================

fprintf('\n--- Feature Importance ---\n');
importance = predictorImportance(tree);
for i = 1:length(importance)
    fprintf('Feature %d importance: %.4f\n', i, importance(i));
end

figure('Name','Feature Importance');
bar(importance);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance');
grid on;

%% ========================= TRAINING DATA PLOT =========================

if size(X,2) == 2
    fprintf('\n--- Plotting Training Data ---\n');
    figure('Name','Training Data');
    gscatter(X(:,1), X(:,2), Y, 'rb', 'ox');
    xlabel('Feature 1'); ylabel('Feature 2');
    title('Training Data');
    grid on;
end

%% ========================= DECISION BOUNDARY =========================

if size(X,2) == 2
    fprintf('\n--- Computing Decision Boundary ---\n');
    x1 = linspace(min(X(:,1)), max(X(:,1)), GRID_RESOLUTION);
    x2 = linspace(min(X(:,2)), max(X(:,2)), GRID_RESOLUTION);
    [X1, X2] = meshgrid(x1, x2);
    Xgrid = [X1(:), X2(:)];

    fprintf('Predicting %d grid points for decision boundary.\n', size(Xgrid,1));
    Ygrid = predict(tree, Xgrid);

    figure('Name','Decision Boundary');
    gscatter(Xgrid(:,1), Xgrid(:,2), Ygrid, [0.85 0.85 1; 1 0.85 0.85], '.', 1);
    hold on;
    gscatter(X(:,1), X(:,2), Y, 'rb', 'ox');
    xlabel('Feature 1'); ylabel('Feature 2');
    title('Decision Boundary');
    grid on;
end

%% ========================= CROSS-VALIDATION =========================

fprintf('\n--- Performing 5-Fold Cross-Validation ---\n');
cvTree = crossval(tree,'KFold',5);
cvErr = kfoldLoss(cvTree);
fprintf('5-fold CV classification error: %.4f\n', cvErr);

%% ========================= CONFUSION MATRIX =========================

fprintf('\n--- Confusion Matrix ---\n');
Yhat = predict(tree,X);
figure('Name','Confusion Matrix');
confusionchart(Y,Yhat);
title('Confusion Matrix');

%% ========================= PRUNING ANALYSIS =========================

fprintf('\n--- Pruning Analysis ---\n');
maxLevel = max(tree.PruneList);
levels = 0:maxLevel;
loss = zeros(numel(levels),1);

for i = 1:numel(levels)
    tPruned = prune(tree,'Level',levels(i));
    cvt = crossval(tPruned,'KFold',5);
    loss(i) = kfoldLoss(cvt);
    fprintf('Pruning level %d => CV error %.4f\n', levels(i), loss(i));
end

[bestLoss, idx] = min(loss);
bestLevel = levels(idx);
fprintf('Best pruning level: %d with CV error %.4f\n', bestLevel, bestLoss);

figure('Name','Pruning Curve');
plot(levels, loss, '-o','LineWidth',2);
xlabel('Pruning Level');
ylabel('Cross-Validated Classification Error');
title('Cost-Complexity Pruning');
grid on;

%% ========================= SUMMARY =========================

fprintf('\n--- Summary ---\n');
fprintf('Observations: %d\n', size(X,1));
fprintf('Features:     %d\n', size(X,2));
treeDepth = computeTreeDepth(tree);
fprintf('Tree depth:   %d\n', treeDepth);
fprintf('Training accuracy: %.2f %%\n', 100 * mean(Yhat == Y));

%% ========================= HELPER FUNCTION =========================

function d = computeTreeDepth(tree)
    children = tree.Children;
    d = recurseDepth(1);

    function depth = recurseDepth(node)
        if children(node,1) == 0
            depth = 1;
        else
            depthLeft  = recurseDepth(children(node,1));
            depthRight = recurseDepth(children(node,2));
            depth = 1 + max(depthLeft, depthRight);
        end
    end
end
