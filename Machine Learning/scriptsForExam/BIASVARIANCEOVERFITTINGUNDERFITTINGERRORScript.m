%% Bias-Variance Tradeoff, Overfitting, Underfitting Example Script
% Expanded script with more detailed visualizations and breakdowns of bias, variance, and error.

clear;
clc;

%% Parameters (modify these values)
n = 100;  % Number of data points
d = 1;    % Number of features (1 for univariate)
poly_degree = 10;  % Polynomial degree (affects model complexity)
noise_level = 0.3; % Noise level for data generation
model_complexity = 5;  % Control the complexity of the model
train_test_split = 0.7; % Fraction of data used for training (rest is test)
data_type = 'random'; % Options: 'random', 'manual', 'file'

% Options for data generation
% For 'manual' input: x = [values]; y = [values];
% For 'file' input: Use a CSV with x in first column and y in second column.

%% Data Generation or Loading
disp('=== Data Generation or Loading ===');
switch data_type
    case 'random'
        % Generate synthetic data (x, y) with noise
        x = linspace(-5, 5, n)';
        y_true = sin(x); % True underlying function
        y = y_true + noise_level * randn(n, 1); % Add noise to the true function
        disp('Generated random synthetic data (x, y) with noise.');

    case 'manual'
        % Manually input your data here (x, y)
        % Example: x = [values]; y = [values];
        x = linspace(-5, 5, n)';
        y = sin(x) + noise_level * randn(n, 1); % Add noise
        disp('Manually generated data (x, y) with noise.');
        
    case 'file'
        % Load data from a CSV file (x, y format)
        filename = 'your_data.csv'; % Replace with your file path
        data = csvread(filename);
        x = data(:, 1);
        y = data(:, 2);
        disp(['Loaded data from file: ', filename]);
end

disp('Data Generation or Loading complete.');
disp('Data Summary:');
disp(['Number of Data Points: ', num2str(n)]);
disp(['First few points of x: ', num2str(x(1:5)')]);
disp(['First few points of y: ', num2str(y(1:5)')]);

% Split data into training and test sets
split_idx = round(train_test_split * n);
x_train = x(1:split_idx);
y_train = y(1:split_idx);
x_test = x(split_idx+1:end);
y_test = y(split_idx+1:end);

disp(['Training Data Size: ', num2str(length(x_train))]);
disp(['Test Data Size: ', num2str(length(x_test))]);
disp('--- Data Splitting Complete ---');

%% Model Training (Polynomial Regression)
disp('=== Model Training: Polynomial Regression ===');
train_errors = [];
test_errors = [];
degrees = 1:poly_degree;

for deg = degrees
    disp(['Training Polynomial Model with Degree: ', num2str(deg)]);
    
    % Fit polynomial regression model of degree 'deg'
    p = polyfit(x_train, y_train, deg);
    
    % Display the polynomial coefficients
    disp('Polynomial Coefficients:');
    disp(p);
    
    % Evaluate the model
    y_train_pred = polyval(p, x_train);
    y_test_pred = polyval(p, x_test);
    
    % Calculate train and test errors (Mean Squared Error)
    train_error = mean((y_train - y_train_pred).^2);
    test_error = mean((y_test - y_test_pred).^2);
    
    train_errors = [train_errors, train_error];
    test_errors = [test_errors, test_error];
    
    disp(['Train Error (Degree ', num2str(deg), '): ', num2str(train_error)]);
    disp(['Test Error (Degree ', num2str(deg), '): ', num2str(test_error)]);
end

disp('--- Model Training Complete ---');

%% Bias and Variance
disp('=== Bias and Variance Calculations ===');
bias_train = train_errors - min(train_errors);
variance_train = train_errors - mean(train_errors);
bias_test = test_errors - min(test_errors);
variance_test = test_errors - mean(test_errors);

disp('Bias and Variance for Train and Test Sets Calculated.');

%% Plot Results
disp('=== Plotting Results ===');

% Plotting Train and Test Errors vs Polynomial Degree
figure;
subplot(2, 2, 1);
plot(degrees, train_errors, 'b-', 'LineWidth', 2);
hold on;
plot(degrees, test_errors, 'r-', 'LineWidth', 2);
xlabel('Polynomial Degree');
ylabel('Error');
legend('Train Error', 'Test Error');
title('Train vs Test Error');
disp('Plot 1: Train vs Test Error');

% Plot Bias and Variance for Training Set
subplot(2, 2, 2);
plot(degrees, bias_train, 'b-', 'LineWidth', 2);
hold on;
plot(degrees, variance_train, 'g-', 'LineWidth', 2);
xlabel('Polynomial Degree');
ylabel('Error');
legend('Bias', 'Variance');
title('Bias and Variance (Train Set)');
disp('Plot 2: Bias and Variance (Train Set)');

% Plot Bias and Variance for Test Set
subplot(2, 2, 3);
plot(degrees, bias_test, 'b-', 'LineWidth', 2);
hold on;
plot(degrees, variance_test, 'g-', 'LineWidth', 2);
xlabel('Polynomial Degree');
ylabel('Error');
legend('Bias', 'Variance');
title('Bias and Variance (Test Set)');
disp('Plot 3: Bias and Variance (Test Set)');

% Plot Polynomial Fits
subplot(2, 2, 4);
hold on;
plot(x, y, 'k.', 'MarkerSize', 10); % True noisy data
for deg = degrees
    p = polyfit(x_train, y_train, deg);
    y_fit = polyval(p, x);
    plot(x, y_fit, 'LineWidth', 2);
end
xlabel('x');
ylabel('y');
legend('Data', 'Degree 1', 'Degree 2', 'Degree 3', 'Degree 4', 'Degree 5', 'Location', 'Best');
title('Polynomial Fits for Various Degrees');
disp('Plot 4: Polynomial Fits for Various Degrees');

%% Additional Plots to Illustrate Bias, Variance, and Error
% Bias vs Variance (General Concept)
figure;
subplot(2, 2, 1);
bias = linspace(0.1, 0.5, 10);
variance = linspace(0.1, 0.6, 10);
plot(bias, variance, 'b-', 'LineWidth', 2);
xlabel('Bias');
ylabel('Variance');
title('Bias vs Variance Tradeoff');
disp('Plot 5: Bias vs Variance Tradeoff (General Concept)');

% Train Error, Test Error, and Total Error Decomposition
subplot(2, 2, 2);
total_error = train_errors + test_errors;
plot(degrees, train_errors, 'b-', 'LineWidth', 2);
hold on;
plot(degrees, test_errors, 'r-', 'LineWidth', 2);
plot(degrees, total_error, 'g-', 'LineWidth', 2);
xlabel('Polynomial Degree');
ylabel('Error');
legend('Train Error', 'Test Error', 'Total Error');
title('Error Decomposition');
disp('Plot 6: Train Error, Test Error, and Total Error');

% Error Decomposition: Bias, Variance, Irreducible Error
subplot(2, 2, 3);
irreducible_error = 0.1 * ones(1, poly_degree);
plot(degrees, bias_train.^2, 'b-', 'LineWidth', 2);
hold on;
plot(degrees, variance_train, 'g-', 'LineWidth', 2);
plot(degrees, irreducible_error, 'r-', 'LineWidth', 2);
xlabel('Polynomial Degree');
ylabel('Error');
legend('Bias^2', 'Variance', 'Irreducible Error');
title('Error Decomposition');
disp('Plot 7: Bias, Variance, and Irreducible Error');
