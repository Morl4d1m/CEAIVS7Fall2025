%% Initialization: Input Parameters
% These can be modified by you directly in the script.

% Parameters
true_class = [1, 2];  % True class labels (e.g., 1 for class 1, 2 for class 2)
prior_class_1 = 0.6;  % Prior probability of class 1
prior_class_2 = 0.4;  % Prior probability of class 2
mu_1 = 2;             % Mean of class 1 (normal distribution)
sigma_1 = 1;          % Standard deviation of class 1
mu_2 = 4;             % Mean of class 2 (normal distribution)
sigma_2 = 1.5;        % Standard deviation of class 2
cost_1 = 1;           % Cost of classifying class 1 as class 1
cost_2 = 1;           % Cost of classifying class 2 as class 2
observation_range = [-10, 10]; % Range of x-axis (observation space)

% Option for loading data (0: manual, 1: load from file, 2: random data generation)
load_option = 2;  % Change this value to load data or generate random

%% Data Generation
if load_option == 1
    % Load data from file (CSV, .mat, etc.)
    % Example: data = load('datafile.mat');
    disp('Loading data from file...');
    % Make sure the file is in the MATLAB path
elseif load_option == 0
    % Manual Input (for small-scale examples)
    disp('Manually input data:');
    % Here you can add code to input data manually or use a GUI to capture data
    % Example:
    % data = input('Enter data matrix [x1, x2, ..., xn]: ');
elseif load_option == 2
    % Generate Random Data for two classes
    disp('Generating random data...');
    n_points = 200;  % Number of points to generate
    x1 = mu_1 + sigma_1 * randn(n_points, 1);  % Class 1 data
    x2 = mu_2 + sigma_2 * randn(n_points, 1);  % Class 2 data
    data = [x1; x2];
    labels = [ones(n_points, 1); 2*ones(n_points, 1)];
    disp(['Generated ' num2str(n_points) ' points for Class 1 and Class 2']);
end

%% Likelihood Functions (Normal Distribution)
% Likelihood for class 1 and class 2
pdf_class_1 = @(x) (1 / (sigma_1 * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_1) / sigma_1).^2);
pdf_class_2 = @(x) (1 / (sigma_2 * sqrt(2 * pi))) * exp(-0.5 * ((x - mu_2) / sigma_2).^2);

% Display the likelihood functions
disp('Likelihood Functions:');
disp(['Class 1 Likelihood: N(' num2str(mu_1) ', ' num2str(sigma_1) '^2)']);
disp(['Class 2 Likelihood: N(' num2str(mu_2) ', ' num2str(sigma_2) '^2)']);

%% Posterior Calculations
disp('Calculating Posterior Distributions...');
posterior_class_1 = @(x) prior_class_1 * pdf_class_1(x);
posterior_class_2 = @(x) prior_class_2 * pdf_class_2(x);

% Display posterior equations
disp(['Posterior for Class 1: P(C1|x) = P(C1) * P(x|C1)']);
disp(['Posterior for Class 2: P(C2|x) = P(C2) * P(x|C2)']);

% Vectorized decision boundary computation for plotting
x_vals = linspace(observation_range(1), observation_range(2), 1000);
decision_vals = arrayfun(@(x) (posterior_class_1(x) * cost_1) < (posterior_class_2(x) * cost_2), x_vals);

% Display Posterior Evaluation at a few points
disp('Posterior Evaluation at selected points:');
for i = 1:5
    test_point = x_vals(i * 200); % Display at intervals
    disp(['At x = ' num2str(test_point) ':']);
    disp(['  P(C1|x) = ' num2str(posterior_class_1(test_point))]);
    disp(['  P(C2|x) = ' num2str(posterior_class_2(test_point))]);
end

%% Decision Rule (Minimize Expected Loss)
% The decision rule is based on comparing the posterior probabilities
% and minimizing the expected loss for each observation.

disp('Applying Decision Rule:');
disp('Minimize expected loss: Compare P(C1|x) * Cost(C1) vs P(C2|x) * Cost(C2)');

% Generate the decision boundary for plotting
decision_boundary = @(x) (posterior_class_1(x) * cost_1) < (posterior_class_2(x) * cost_2);

% Display the decision rule at a few points
disp('Evaluating Decision Rule at selected points:');
for i = 1:5
    test_point = x_vals(i * 200); % Test at intervals
    decision = decision_boundary(test_point);
    disp(['At x = ' num2str(test_point) ':']);
    disp(['  Decision: Class ' num2str(decision + 1)]);  % Class 1 for 0, Class 2 for 1
end

%% Plots

% 1. Plot the prior distributions
figure;
hold on;
fplot(pdf_class_1, observation_range, 'r', 'LineWidth', 2);
fplot(pdf_class_2, observation_range, 'b', 'LineWidth', 2);
title('Prior Distributions');
xlabel('x');
ylabel('Probability Density');
legend('Class 1', 'Class 2');
hold off;

% 2. Plot the posterior distributions
figure;
hold on;
fplot(posterior_class_1, observation_range, 'r', 'LineWidth', 2);
fplot(posterior_class_2, observation_range, 'b', 'LineWidth', 2);
title('Posterior Distributions');
xlabel('x');
ylabel('Posterior Probability');
legend('Class 1', 'Class 2');
hold off;

% 3. Plot the decision boundary
figure;
plot(x_vals, decision_vals, 'k', 'LineWidth', 2);
title('Decision Boundary');
xlabel('x');
ylabel('Decision Rule');
hold off;

% 4. Plot random generated data points with decision boundary
figure;
hold on;
scatter(x1, ones(n_points, 1), 'r', 'filled');
scatter(x2, ones(n_points, 1), 'b', 'filled');
plot(x_vals, decision_vals, 'k--', 'LineWidth', 2);
title('Class Data Points with Decision Boundary');
xlabel('x');
ylabel('Decision');
legend('Class 1', 'Class 2', 'Decision Boundary');
hold off;

%% Confusion Matrix and Performance Metrics (Optional)
% Compute confusion matrix, accuracy, etc., if applicable

disp('Script execution complete.');
