%% Factor Graph Parameters - Input at the top of the script
clear; clc;

% Option to load data, enter manually, or generate random
inputOption = 'random';  % Choices: 'file', 'manual', 'random'

% Parameters for the factor graph
numVariables = 6;  % Number of variable nodes
numFactors = 4;    % Number of factor nodes
maxConnections = 3; % Max number of variables connected to a factor

% Data file (if inputOption is 'file')
dataFile = 'data.txt';  % Specify the data file (if using file input)

% Random Generation Parameters (for 'random' option)
maxValue = 10;  % Maximum value for random data

%% Load or Generate Data
switch inputOption
    case 'file'
        % Load data from file
        data = load(dataFile);
        variables = data(:, 1:numVariables);
        factors = data(:, numVariables+1:end);
    case 'manual'
        % Manual input (you can prompt for data or hard-code here)
        variables = [1 2 3 4 5 6];  % Example input for 6 variables
        factors = [7 8 9 10];        % Example input for factors
    case 'random'
        % Generate random data
        variables = rand(1, numVariables) * maxValue;
        factors = rand(1, numFactors) * maxValue;
end

%% Create Factor Graph
% Create empty adjacency matrix for factor graph (factors x variables)
adjMatrix = zeros(numFactors, numVariables);

% Randomly connect variables to factors (can adjust as needed)
for i = 1:numFactors
    numConnections = randi([1 maxConnections], 1);
    connectedVars = randperm(numVariables, numConnections);
    adjMatrix(i, connectedVars) = 1;  % Create edges between factors and variables
end

%% Plot 1: Factor Graph Structure
figure;
hold on;

% Plot variable nodes
theta = linspace(0, 2*pi, numVariables + 1);
variablePos = [cos(theta(1:end-1)); sin(theta(1:end-1))];
scatter(variablePos(1,:), variablePos(2,:), 100, 'filled', 'r');

% Plot factor nodes (arranged in a circle)
factorPos = [cos(theta(1:end-1) + pi/2); sin(theta(1:end-1) + pi/2)];
scatter(factorPos(1,:), factorPos(2,:), 100, 'filled', 'b');

% Plot edges (connections between factors and variables)
for i = 1:numFactors
    for j = 1:numVariables
        if adjMatrix(i, j) == 1
            plot([variablePos(1,j) factorPos(1,i)], [variablePos(2,j) factorPos(2,i)], 'k-', 'LineWidth', 1);
        end
    end
end

% Annotate nodes
for i = 1:numVariables
    text(variablePos(1,i), variablePos(2,i), sprintf('x%d', i), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'r');
end
for i = 1:numFactors
    text(factorPos(1,i), factorPos(2,i), sprintf('f%d', i), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'b');
end

title('Factor Graph Structure');
axis equal;
hold off;

%% Belief Propagation (Simple Example)
% Initialize beliefs (using random values for illustration)
beliefs = rand(numVariables, 1);

% Initialize messages (from factors to variables)
messages = rand(numFactors, numVariables);

% Perform a simple belief propagation iteration (for demonstration)
iterations = 10;
beliefsHistory = zeros(numVariables, iterations);  % Store beliefs for plotting
for iter = 1:iterations
    newBeliefs = beliefs;
    for i = 1:numVariables
        % Update belief based on connected factors (simplified for demonstration)
        connectedFactors = find(adjMatrix(:, i) == 1);
        newBeliefs(i) = sum(beliefs(connectedFactors)) / length(connectedFactors);
    end
    beliefs = newBeliefs;  % Update beliefs
    beliefsHistory(:, iter) = beliefs;  % Save beliefs for later plotting
end

%% Plot 2: Belief Propagation Results
figure;
plot(1:numVariables, beliefs, 'o-', 'LineWidth', 2);
title('Belief Propagation Results');
xlabel('Variable Index');
ylabel('Belief Value');
grid on;

%% Plot 3: Factor Marginals
% In a factor graph, marginals show the marginal probability distribution
% of each factor. For simplicity, we can assume uniform marginals or use
% belief propagation results to derive marginals (e.g., summing over variables).
factorMarginals = rand(numFactors, 1);  % Dummy marginals for illustration

figure;
bar(factorMarginals);
title('Factor Marginals');
xlabel('Factor Index');
ylabel('Marginal Value');
grid on;

%% Plot 4: Variable Marginals (Using Beliefs)
% Variable marginals are the final beliefs of each variable
variableMarginals = beliefs;

figure;
bar(variableMarginals);
title('Variable Marginals (Beliefs)');
xlabel('Variable Index');
ylabel('Marginal Value');
grid on;

%% Plot 5: Messages Between Variables and Factors
% Plot the magnitude of messages passed between variables and factors
figure;
imagesc(messages);
colorbar;
title('Messages Between Variables and Factors');
xlabel('Variable Index');
ylabel('Factor Index');
grid on;

%% Plot 6: Convergence of Belief Propagation
% Track the change in beliefs over iterations to show convergence
beliefChanges = zeros(iterations, 1);
for iter = 2:iterations
    beliefChanges(iter) = norm(beliefsHistory(:, iter) - beliefsHistory(:, iter - 1));
end

figure;
plot(1:iterations, beliefChanges, 'o-', 'LineWidth', 2);
title('Convergence of Belief Propagation');
xlabel('Iteration');
ylabel('Change in Beliefs');
grid on;
