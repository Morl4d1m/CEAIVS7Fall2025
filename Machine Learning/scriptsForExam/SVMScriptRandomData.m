%% ================= USER CONFIGURATION =================
% Choose SVM model type: 'linear hard', 'linear soft', 'polynomial', 'rbf'
% (rbf=radial basis function)
svmModelType = 'linear soft';

% Choose data source: 'random', 'manual', 'file'
dataSource = 'random';

% For random data
numPoints = 50;  % number of points per class

% For manual data (example)
manualDataX = [2,2; 2.5,2; 1.8,2.2; 6,6; 6.2,5.8; 5.9,6.1];
manualDataY = [1;1;1;-1;-1;-1];

% For file data
dataFileName = 'svm_data.csv';  % CSV: col1=x1, col2=x2, col3=label

%% ==================== DATA GENERATION ====================
disp('Preparing dataset...');
switch dataSource
    case 'random'
        disp('Generating random synthetic data...');
        % Class 1: centered at (2,2)
        class1 = [2 + 0.5*randn(numPoints, 1), 2 + 0.5*randn(numPoints, 1)];
        % Class -1: centered at (6,6)
        class2 = [6 + 0.5*randn(numPoints, 1), 6 + 0.5*randn(numPoints, 1)];
        % Combine
        X = [class1; class2];
        y = [ones(numPoints,1); -ones(numPoints,1)];
        
    case 'manual'
        disp('Using manual dataset...');
        X = manualDataX;
        y = manualDataY;
        
    case 'file'
        disp(['Reading data from file: ', dataFileName]);
        T = readtable(dataFileName);
        if size(T,2) < 3
            error('CSV file must have at least 3 columns: x1, x2, label');
        end
        X = table2array(T(:,1:2));
        y = table2array(T(:,3));
        
    otherwise
        error('Invalid dataSource. Choose ''random'', ''manual'', or ''file''.');
end

disp(['Dataset size: ', mat2str(size(X))]);
disp(['First 5 data points:']);
disp(X(1:min(5,end),:));
disp(['Corresponding labels:']);
disp(y(1:min(5,end)));

%% ==================== TRAIN SVM ====================
disp(['Training SVM model: ', svmModelType]);

switch svmModelType
    case 'linear hard'
        SVMModel = fitcsvm(X, y, 'KernelFunction', 'linear', 'Standardize', true, ...
            'ClassNames', [-1,1], 'BoxConstraint', Inf);
    case 'linear soft'
        SVMModel = fitcsvm(X, y, 'KernelFunction', 'linear', 'Standardize', true, ...
            'ClassNames', [-1,1], 'BoxConstraint', 1);
    case 'polynomial'
        SVMModel = fitcsvm(X, y, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, ...
            'Standardize', true, 'ClassNames', [-1,1], 'BoxConstraint', 1);
    case 'rbf'
        SVMModel = fitcsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 1, ...
            'Standardize', true, 'ClassNames', [-1,1], 'BoxConstraint', 1);
    otherwise
        error('Invalid svmModelType.');
end

supportVectors = SVMModel.SupportVectors;
fprintf('Number of support vectors: %d\n', size(supportVectors,1));

%% ==================== DISPLAY WEIGHTS / SLOPE ====================
if strcmp(SVMModel.KernelParameters.Function,'linear')
    w = SVMModel.Beta;
    b = SVMModel.Bias;
    fprintf('Weight vector w = [%.3f, %.3f]\n', w(1), w(2));
    fprintf('Bias b = %.3f\n', b);
    
    slope = -w(1)/w(2);
    intercept = -b/w(2);
    fprintf('Slope of decision boundary: %.3f\n', slope);
    fprintf('Intercept of decision boundary: %.3f\n', intercept);
    fprintf('Decision boundary formula: x2 = %.3f*x1 + %.3f\n', slope, intercept);
else
    disp('Non-linear kernel selected. Decision boundary is not a straight line.');
end

%% ==================== PLOT ====================
figure;
h1 = scatter(X(y==1,1), X(y==1,2), 50, 'ro', 'filled');
hold on;
h2 = scatter(X(y==-1,1), X(y==-1,2), 50, 'bo', 'filled');
h3 = scatter(supportVectors(:,1), supportVectors(:,2), 100, 'k', 'filled');

title(['SVM Classifier - ', svmModelType]);
xlabel('X1'); ylabel('X2');

%% ==================== DECISION BOUNDARY ====================
disp('Plotting decision boundary...');

d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(X(:,1))-1:d:max(X(:,1))+1, min(X(:,2))-1:d:max(X(:,2))+1);
XGrid = [x1Grid(:), x2Grid(:)];
[~, score] = predict(SVMModel, XGrid);
scoreGrid = reshape(score(:,2), size(x1Grid));

% Decision boundary
contour(x1Grid, x2Grid, scoreGrid, [0 0], 'k', 'LineWidth', 2);

% Margins for linear kernel
if strcmp(SVMModel.KernelParameters.Function,'linear')
    contour(x1Grid, x2Grid, scoreGrid, [1 1], 'k--', 'LineWidth', 1);
    contour(x1Grid, x2Grid, scoreGrid, [-1 -1], 'k--', 'LineWidth', 1);
end

grid on;
hold off;
legend([h1,h2,h3], {'Class 1','Class -1','Support Vectors'}, 'Location','Best');
disp('Plotting complete.');
