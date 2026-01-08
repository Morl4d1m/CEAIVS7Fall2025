% Cross-Validation Script with Explanations
% -------------------------------------------------------------
% Parameters Section: Modify these values to change settings

% Cross-validation settings
cv_folds = 10;  % Number of folds for cross-validation (e.g., 10)
model_type = 'svm'; % 'svm' for Support Vector Machine, 'knn' for k-NN, etc.

% Data Input Options:
data_source = 'random'; % 'file', 'manual', or 'random'
data_file = 'data.csv'; % Path to the file (if using 'file' option)
n_samples = 100; % Number of samples (used if generating random data)
n_features = 2; % Number of features (used if generating random data)

% Model hyperparameters (specific to the chosen model)
svm_kernel = 'linear'; % Only used if model_type = 'svm'
knn_k = 5; % Number of neighbors for k-NN (only used if model_type = 'knn')

% -------------------------------------------------------------
% Data Loading/Generation

disp('--- Data Loading/Generation ---');
% Load data from a file, generate random data, or input manually
if strcmp(data_source, 'file')
    % Load dataset from a CSV file
    disp(['Loading data from file: ', data_file]);
    data = readtable(data_file);
    X = data{:, 1:end-1}; % Assuming the last column is the label
    y = data{:, end};
elseif strcmp(data_source, 'manual')
    % Manual data input (prompt the user)
    disp('Input data manually...');
    n_samples = input('Enter number of samples: ');
    n_features = input('Enter number of features: ');
    X = randn(n_samples, n_features); % Randomly generated input data
    y = randi([0, 1], n_samples, 1); % Random binary labels
elseif strcmp(data_source, 'random')
    % Generate random data
    disp(['Generating random data: ', num2str(n_samples), ' samples, ', num2str(n_features), ' features']);
    X = randn(n_samples, n_features); % Randomly generated input data
    y = randi([0, 1], n_samples, 1); % Random binary labels
else
    error('Invalid data source specified.');
end

% Ensure labels are binary for classification tasks
y = double(y > 0); % Convert labels to 0 or 1

disp('Data generation complete.');
disp(['Number of samples: ', num2str(size(X, 1))]);
disp(['Number of features: ', num2str(size(X, 2))]);

% -------------------------------------------------------------
% Cross-validation procedure

disp('--- Starting Cross-Validation ---');

% Initialize cross-validation
cv = cvpartition(y, 'KFold', cv_folds);

% Store errors for each fold
errors = zeros(cv_folds, 1);

% Perform cross-validation
for i = 1:cv_folds
    % Training and testing data for the current fold
    train_idx = cv.training(i);
    test_idx = cv.test(i);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Print fold information
    disp(['\nFold #', num2str(i), ' of ', num2str(cv_folds)]);
    disp('Training data size: ');
    disp(size(X_train));
    disp('Testing data size: ');
    disp(size(X_test));

    % Choose the model based on user input
    if strcmp(model_type, 'svm')
        % Train SVM model
        disp('Training SVM model...');
        model = fitcsvm(X_train, y_train, 'KernelFunction', svm_kernel);
        disp(['Using SVM kernel: ', svm_kernel]);
        y_pred = predict(model, X_test);
    elseif strcmp(model_type, 'knn')
        % Train k-NN model
        disp('Training k-NN model...');
        model = fitcknn(X_train, y_train, 'NumNeighbors', knn_k);
        disp(['Using k-NN with k = ', num2str(knn_k)]);
        y_pred = predict(model, X_test);
    else
        error('Unsupported model type');
    end
    
    % Calculate error for the current fold
    fold_error = sum(y_pred ~= y_test) / length(y_test);
    errors(i) = fold_error;
    
    % Print the fold error
    disp(['Fold error (misclassification rate) = ', num2str(fold_error)]);
end

% Display cross-validation results
mean_error = mean(errors);
disp('\n--- Cross-Validation Results ---');
disp(['Cross-validation mean error: ', num2str(mean_error)]);

% -------------------------------------------------------------
% Plotting Results

% Cross-validation error plot
figure;
plot(1:cv_folds, errors, '-o', 'LineWidth', 2, 'MarkerSize', 6);
title('Cross-Validation Error for Each Fold');
xlabel('Fold Number');
ylabel('Error');
grid on;

% Histogram of errors
figure;
histogram(errors, 10);
title('Distribution of Cross-Validation Errors');
xlabel('Error');
ylabel('Frequency');
grid on;

% Boxplot of errors
figure;
boxplot(errors);
title('Boxplot of Cross-Validation Errors');
ylabel('Error');
grid on;

% Model Performance (Confusion Matrix for the last fold)
disp('\n--- Final Model Training and Performance Evaluation ---');
train_idx = cv.training(cv_folds);
test_idx = cv.test(cv_folds);
X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

% Final model training
if strcmp(model_type, 'svm')
    disp('Training final SVM model...');
    model = fitcsvm(X_train, y_train, 'KernelFunction', svm_kernel);
elseif strcmp(model_type, 'knn')
    disp('Training final k-NN model...');
    model = fitcknn(X_train, y_train, 'NumNeighbors', knn_k);
else
    error('Unsupported model type');
end

% Prediction and confusion matrix
y_pred = predict(model, X_test);
disp('Final model performance evaluation:');
disp('Confusion Matrix:');
cm = confusionmat(y_test, y_pred);
disp(cm);

% Display confusion matrix plot
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix for Final Model');

disp('--- Script Complete ---');
