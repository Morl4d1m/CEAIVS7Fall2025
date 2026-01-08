% ================================================================
% Maximum a Posteriori Estimation (MAP) - Detailed Version
% ================================================================

%% User-Defined Parameters (Adjust as needed)
% Prior distribution parameters
prior_mean = 0;         % Mean of prior (for Gaussian prior)
prior_variance = 1;     % Variance of prior (for Gaussian prior)

% Likelihood parameters (for Gaussian likelihood)
likelihood_mean = 0;    % Mean of likelihood (for Gaussian likelihood)
likelihood_variance = 1; % Variance of likelihood (for Gaussian likelihood)

% Observation data options
data_source = 'manual'; % Options: 'manual', 'file', 'random'
file_name = 'data.txt'; % Data file name (for loading data from file)
n_points = 100;         % Number of random data points (for generating random data)

% ================================================================
% 1. Data Input Handling
% ================================================================
disp('==============================');
disp('STEP 1: Data Input Handling');
disp('==============================');
switch data_source
    case 'manual'
        % Manually input data
        disp('Input the observed data (e.g., [1 2 3]):');
        data = input('Enter data as a vector: ');
        disp(['Data: ', num2str(data)]);
        
    case 'file'
        % Load data from file (assuming it's a column vector of data)
        if exist(file_name, 'file')
            data = load(file_name);
            disp('Data loaded from file.');
            disp(['Data: ', num2str(data')]);
        else
            error('File not found.');
        end
        
    case 'random'
        % Generate random data (Gaussian noise)
        data = likelihood_mean + sqrt(likelihood_variance) * randn(1, n_points);
        disp('Random data generated.');
        disp(['Data: ', num2str(data)]);
        
    otherwise
        error('Invalid data source.');
end

disp(' '); % Blank line for better readability

%% ================================================================
% 2. Define Prior, Likelihood, and Posterior Distributions
% ================================================================
disp('=============================================');
disp('STEP 2: Define Prior, Likelihood, and Posterior');
disp('=============================================');

% Prior distribution (Gaussian)
disp('Prior distribution: P(Theta)');
disp(['P(Theta) = N(Theta; ', num2str(prior_mean), ', ', num2str(prior_variance), ')']);
prior_dist = @(theta) (1 / sqrt(2*pi*prior_variance)) * exp(-(theta - prior_mean).^2 / (2 * prior_variance));

% Likelihood function (Gaussian)
disp('Likelihood function: P(Data | Theta)');
disp(['P(Data | Theta) = N(Data; Theta, ', num2str(likelihood_variance), ')']);
likelihood_dist = @(theta) prod((1 / sqrt(2*pi*likelihood_variance)) * exp(-(data - theta).^2 / (2 * likelihood_variance)));

% Posterior distribution (proportional to likelihood * prior)
disp('Posterior distribution: P(Theta | Data)');
disp('P(Theta | Data) = P(Data | Theta) * P(Theta) (up to a normalization constant)');
posterior_dist = @(theta) likelihood_dist(theta) * prior_dist(theta);

disp(' '); % Blank line for better readability

%% ================================================================
% 3. MAP Estimation (Find theta that maximizes posterior)
% ================================================================
disp('==============================================');
disp('STEP 3: MAP Estimation');
disp('==============================================');
disp('We seek the value of Theta that maximizes the posterior distribution P(Theta | Data).');
disp('This is equivalent to minimizing the negative log of the posterior.');

% Define the negative log-posterior function (for optimization)
neg_log_posterior = @(theta) -log(posterior_dist(theta));

% Show the negative log-posterior function for Theta
disp(['Negative log-posterior function: -log(P(Theta | Data))']);
disp('We will minimize this function to find the MAP estimate.');

% Use fminunc to find the MAP estimate (minimizes the negative log-posterior)
options = optimset('Display', 'iter', 'TolX', 1e-6);
MAP_estimate = fminunc(neg_log_posterior, 0, options); % Initial guess = 0

disp(['MAP Estimate (Theta): ', num2str(MAP_estimate)]);

disp(' '); % Blank line for better readability

%% ================================================================
% 4. Plots for Visualization
% ================================================================
disp('=====================');
disp('STEP 4: Plots for Visualization');
disp('=====================');

theta_vals = linspace(min(data) - 3, max(data) + 3, 500);

% Plot prior distribution
figure;
subplot(2,2,1);
plot(theta_vals, prior_dist(theta_vals), 'b', 'LineWidth', 2);
title('Prior Distribution P(Theta)');
xlabel('Theta');
ylabel('Density');

% Plot likelihood function
likelihood_vals = arrayfun(@(theta) likelihood_dist(theta), theta_vals);
subplot(2,2,2);
plot(theta_vals, likelihood_vals, 'r', 'LineWidth', 2);
title('Likelihood Function P(Data | Theta)');
xlabel('Theta');
ylabel('Likelihood');

% Plot posterior distribution
posterior_vals = arrayfun(@(theta) posterior_dist(theta), theta_vals);
subplot(2,2,3);
plot(theta_vals, posterior_vals, 'g', 'LineWidth', 2);
title('Posterior Distribution P(Theta | Data)');
xlabel('Theta');
ylabel('Posterior');

% Plot MAP estimate
subplot(2,2,4);
plot(theta_vals, posterior_vals, 'g', 'LineWidth', 2);
hold on;
plot(MAP_estimate, posterior_dist(MAP_estimate), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('MAP Estimate');
xlabel('Theta');
ylabel('Posterior');
legend('Posterior', 'MAP Estimate');

disp(' '); % Blank line for better readability

%% ================================================================
% 5. Output Results
% ================================================================
disp('====================================');
disp('STEP 5: Output Results');
disp('====================================');
disp(['MAP Estimate: ', num2str(MAP_estimate)]);
disp(['Maximum Posterior Value: ', num2str(posterior_dist(MAP_estimate))]);
disp(' ');
disp('The MAP estimate corresponds to the value of Theta that maximizes the posterior distribution,');
disp('which is the most probable value of Theta given the observed data and the prior knowledge.');
disp(' ');

