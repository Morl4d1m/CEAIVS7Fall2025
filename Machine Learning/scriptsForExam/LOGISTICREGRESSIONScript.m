%% LOGISTIC REGRESSION DEMO SCRIPT (Verbose)
clear; clc; close all;

%% ================= USER PARAMETERS =================
DATA_SOURCE = 'random';
DATA_FILE = 'data.mat';     % Only used if DATA_SOURCE = 'file'

X_manual = [1 2; 2 1; 2 3; 3 2; 4 3; 3 4];
y_manual = [0; 0; 0; 1; 1; 1];

N = 200;            
mu0 = [2 2];        
mu1 = [4 4];        
sigma = 0.6;        

alpha = 0.1;        
num_iters = 500;    
add_intercept = true;

%% ================= LOAD / GENERATE DATA =================
switch DATA_SOURCE
    case 'file'
        load(DATA_FILE); % expects X and y
        fprintf('Loaded data from file: %s\n', DATA_FILE);

    case 'manual'
        X = X_manual;
        y = y_manual;
        fprintf('Using manual data with %d samples.\n', length(y));

    case 'random'
        X0 = mvnrnd(mu0, sigma^2 * eye(2), N/2);
        X1 = mvnrnd(mu1, sigma^2 * eye(2), N/2);
        X = [X0; X1];
        y = [zeros(N/2,1); ones(N/2,1)];
        fprintf('Generated random data with %d samples.\n', N);
end

m = length(y);
fprintf('Number of features: %d\n', size(X,2));
fprintf('Number of samples: %d\n', m);

%% ================= DATA VISUALIZATION =================
figure; hold on;
scatter(X(y==0,1), X(y==0,2), 'ro','filled');
scatter(X(y==1,1), X(y==1,2), 'bo','filled');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1');
title('Training Data');
grid on;

%% ================= ADD INTERCEPT =================
if add_intercept
    X = [ones(m,1) X];
    fprintf('Added intercept term to X.\n');
end

n = size(X,2);
theta = zeros(n,1);
fprintf('Initialized theta as zeros: [%s]\n', num2str(theta'));

%% ================= SIGMOID FUNCTION =================
sigmoid = @(z) 1 ./ (1 + exp(-z));

% Plot sigmoid
z = linspace(-10,10,200);
figure;
plot(z, sigmoid(z),'LineWidth',2);
xlabel('z'); ylabel('sigmoid(z)');
title('Sigmoid Function');
grid on;

%% ================= COST FUNCTION =================
costFunction = @(theta) ...
    (-1/m) * sum(y .* log(sigmoid(X*theta)) + ...
    (1-y) .* log(1 - sigmoid(X*theta)));

fprintf('Initial cost (theta=0): %.4f\n', costFunction(theta));

%% ================= GRADIENT DESCENT =================
J_history = zeros(num_iters,1);

fprintf('\nStarting gradient descent...\n');
for i = 1:num_iters
    h = sigmoid(X * theta);                  % Hypothesis
    gradient = (1/m) * (X' * (h - y));      % Gradient
    theta = theta - alpha * gradient;       % Update theta
    J_history(i) = costFunction(theta);     % Cost
    
    % Display intermediate updates every 50 iterations
    if mod(i,50)==0 || i==1
        fprintf('Iteration %d:\n', i);
        fprintf('   Cost J = %.4f\n', J_history(i));
        fprintf('   Theta = [%s]\n', num2str(theta'));
        fprintf('   Gradient = [%s]\n', num2str(gradient'));
    end
end
fprintf('Gradient descent finished.\nFinal theta: [%s]\n', num2str(theta'));

%% ================= COST VS ITERATIONS =================
figure;
plot(1:num_iters, J_history,'LineWidth',2);
xlabel('Iteration'); ylabel('Cost J');
title('Cost Function Convergence');
grid on;

%% ================= DECISION BOUNDARY =================
figure; hold on;
scatter(X(y==0,2), X(y==0,3), 'ro','filled');
scatter(X(y==1,2), X(y==1,3), 'bo','filled');

x_vals = linspace(min(X(:,2)), max(X(:,2)), 100);
y_vals = -(theta(1) + theta(2)*x_vals) / theta(3);
plot(x_vals, y_vals, 'k','LineWidth',2);

xlabel('Feature 1'); ylabel('Feature 2');
legend('Class 0','Class 1','Decision Boundary');
title('Logistic Regression Decision Boundary');
grid on;

%% ================= PROBABILITY CONTOUR =================
[x1Grid, x2Grid] = meshgrid(linspace(min(X(:,2)),max(X(:,2)),100), ...
                            linspace(min(X(:,3)),max(X(:,3)),100));
Xgrid = [ones(numel(x1Grid),1), x1Grid(:), x2Grid(:)];
probs = sigmoid(Xgrid * theta);
probs = reshape(probs, size(x1Grid));

figure;
contourf(x1Grid, x2Grid, probs, 20);
colorbar;
hold on;
scatter(X(:,2), X(:,3), 20, y,'filled');
title('Predicted Probability Contours');
xlabel('Feature 1'); ylabel('Feature 2');
grid on;

%% ================= PREDICTIONS =================
p = sigmoid(X * theta) >= 0.5;
fprintf('\nPredicted probabilities for first 5 samples:\n');
disp([sigmoid(X(1:5,:) * theta), y(1:5)]);

%% ================= CLASSIFICATION RESULTS =================
figure; hold on;
scatter(X(p==y,2), X(p==y,3), 'g','filled');
scatter(X(p~=y,2), X(p~=y,3), 'r','filled');
legend('Correct','Incorrect');
xlabel('Feature 1'); ylabel('Feature 2');
title('Classification Results');
grid on;

%% ================= CONFUSION MATRIX =================
confMat = confusionmat(y, double(p));
figure;
confusionchart(confMat, {'Class 0','Class 1'});
title('Confusion Matrix');
fprintf('Confusion Matrix:\n');
disp(confMat);

%% ================= ACCURACY =================
accuracy = mean(double(p == y)) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracy);
