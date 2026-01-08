% Maximum Likelihood Estimation (MLE) - MATLAB Script
% Parameters can be defined here:
clear; clc;

% Define the true parameters for the distribution (e.g., normal distribution)
mu_true = 5;     % True mean
sigma_true = 2;  % True standard deviation

% Options for input data: 
% 1 -> Load data from file
% 2 -> Manually input data
% 3 -> Generate random data
input_option = 3; % Change this to 1, 2, or 3 based on your choice

% File path for data (used if input_option = 1)
file_path = 'data.txt'; % Path to your data file

% Number of data points if generating random data
num_data_points = 1000;

% Initialize data variable
data = [];

%% Load or Generate Data
fprintf('\nStarting Data Input...\n');
if input_option == 1
    % Load data from a file
    data = load(file_path); % Assuming a single-column dataset
    disp('Data loaded from file.');
elseif input_option == 2
    % Manually input data
    disp('Please input your data (press Enter to stop):');
    data = input('Data: ');
elseif input_option == 3
    % Generate random data (normal distribution)
    data = mu_true + sigma_true * randn(num_data_points, 1);
    disp('Random data generated.');
end

% Display the first few data points
fprintf('First few data points:\n');
disp(data(1:5));

%% Plot histogram of the data
figure;
histogram(data, 30, 'Normalization', 'pdf');
title('Histogram of the Data');
xlabel('Value');
ylabel('Probability Density');
grid on;

%% Maximum Likelihood Estimation (MLE) for parameters
fprintf('\n--- Starting MLE Calculations ---\n');

n = length(data);  % Number of data points

% MLE for the mean (mu) and standard deviation (sigma) of normal distribution
mu_MLE = mean(data);
sigma_MLE = sqrt(sum((data - mu_MLE).^2) / n);

% Display the intermediate calculations
fprintf('\nThe MLE for the mean (mu) is calculated as the sample mean:\n');
fprintf('mu_MLE = mean(data) = %.4f\n', mu_MLE);

fprintf('\nThe MLE for the standard deviation (sigma) is calculated as:\n');
fprintf('sigma_MLE = sqrt(sum((data - mu_MLE).^2) / n) = %.4f\n', sigma_MLE);

disp('--- MLE Estimates ---');
disp(['Estimated mu (MLE): ', num2str(mu_MLE)]);
disp(['Estimated sigma (MLE): ', num2str(sigma_MLE)]);

%% Plot the Likelihood Function for mu
fprintf('\n--- Plotting Likelihood Function for mu ---\n');
mu_range = linspace(min(data) - 3*sigma_true, max(data) + 3*sigma_true, 500);
likelihood_mu = zeros(size(mu_range));

for i = 1:length(mu_range)
    likelihood_mu(i) = sum(log(normpdf(data, mu_range(i), sigma_MLE)));
end

figure;
plot(mu_range, likelihood_mu, 'LineWidth', 2);
hold on;
plot(mu_true, sum(log(normpdf(data, mu_true, sigma_MLE))), 'ro', 'MarkerFaceColor', 'r');
title('Likelihood Function for \mu');
xlabel('\mu');
ylabel('Log-Likelihood');
legend('Likelihood Function', 'True \mu');
grid on;

%% Plot the Likelihood Function for sigma
fprintf('\n--- Plotting Likelihood Function for sigma ---\n');
sigma_range = linspace(0.1, 3 * sigma_true, 500);
likelihood_sigma = zeros(size(sigma_range));

for i = 1:length(sigma_range)
    likelihood_sigma(i) = sum(log(normpdf(data, mu_MLE, sigma_range(i))));
end

figure;
plot(sigma_range, likelihood_sigma, 'LineWidth', 2);
hold on;
plot(sigma_true, sum(log(normpdf(data, mu_MLE, sigma_true))), 'ro', 'MarkerFaceColor', 'r');
title('Likelihood Function for \sigma');
xlabel('\sigma');
ylabel('Log-Likelihood');
legend('Likelihood Function', 'True \sigma');
grid on;

%% Confidence Intervals for MLE
fprintf('\n--- Confidence Intervals for MLE ---\n');
% Approximate 95% confidence interval using normal approximation
z = 1.96; % 95% confidence level
CI_mu = [mu_MLE - z * sigma_MLE / sqrt(n), mu_MLE + z * sigma_MLE / sqrt(n)];
CI_sigma = [sigma_MLE - z * sigma_MLE / sqrt(2 * n), sigma_MLE + z * sigma_MLE / sqrt(2 * n)];

disp('Confidence Intervals:');
disp(['95% CI for \mu: [', num2str(CI_mu(1)), ', ', num2str(CI_mu(2)), ']']);
disp(['95% CI for \sigma: [', num2str(CI_sigma(1)), ', ', num2str(CI_sigma(2)), ']']);

fprintf('\nThe confidence interval for mu is based on the formula:\n');
fprintf('CI_mu = [mu_MLE - z*sigma_MLE/sqrt(n), mu_MLE + z*sigma_MLE/sqrt(n)]\n');
fprintf('For \sigma, CI_sigma is computed similarly using a factor of sqrt(2*n) for the standard deviation.\n');

%% Plot the fitted normal distribution on the histogram
fprintf('\n--- Plotting Fitted Normal Distribution vs Histogram ---\n');
x_vals = linspace(min(data) - 3*sigma_true, max(data) + 3*sigma_true, 500);
pdf_fitted = normpdf(x_vals, mu_MLE, sigma_MLE);

figure;
hold on;
histogram(data, 30, 'Normalization', 'pdf');
plot(x_vals, pdf_fitted, 'r-', 'LineWidth', 2);
title('Fitted Normal Distribution vs Histogram');
xlabel('Value');
ylabel('Probability Density');
legend('Histogram', 'Fitted Normal PDF');
grid on;
hold off;

%% Conclusion
fprintf('\n--- Conclusion ---\n');
disp('MLE estimation complete.');
disp(['The true parameters were mu = ', num2str(mu_true), ' and sigma = ', num2str(sigma_true)]);
disp(['The estimated parameters are mu = ', num2str(mu_MLE), ' and sigma = ', num2str(sigma_MLE)]);
disp(' ');
disp('The confidence intervals give us a range within which the true parameter values are likely to lie.');
