% Define the input data
X = [2 3; 4 3; 2 4; 4 2];  % Input data points (x1, x2, x3, x4)
y = [-1, 1, 1, -1];        % Labels (y)

% Train the SVM model using a linear kernel with hard margin (high BoxConstraint)
SVMModel = fitcsvm(X, y, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [-1, 1], 'BoxConstraint', Inf);

% Get the support vectors
supportVectors = SVMModel.SupportVectors;

% Extract the weights and bias from the SVM model
w = SVMModel.Beta;  % The weight vector (w1, w2)
b = SVMModel.Bias;  % The bias term

% Calculate the slope of the decision boundary
slope = -w(1) / w(2);  % Slope = -w1 / w2

% Construct the formula for the decision boundary
boundary_formula = sprintf('x_2 = %.2fx_1 + %.2f', slope, -b/w(2));

% Display the slope and the decision boundary formula in the command window
disp(['Slope of the decision boundary: ', num2str(slope)]);
disp(['Decision Boundary Formula: ', boundary_formula]);

% Plot the data points and the support vectors
figure;
h1 = gscatter(X(:,1), X(:,2), y, 'rb', 'xo');  % Scatter plot of the data points
hold on;
h2 = scatter(supportVectors(:,1), supportVectors(:,2), 100, 'k', 'filled');  % Plot support vectors
title('Hard Margin SVM Classifier with Linear Kernel');
xlabel('X1');
ylabel('X2');

% Plot the decision boundary and margins
d = 0.02;  % Grid spacing
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), min(X(:,2)):d:max(X(:,2)));
XGrid = [x1Grid(:), x2Grid(:)];
[~, score] = predict(SVMModel, XGrid);
scoreGrid = reshape(score(:,2), size(x1Grid));

% Plot the decision boundary (where the score is zero)
contour(x1Grid, x2Grid, scoreGrid, [0 0], 'k', 'LineWidth', 2);

% Plot the margins (where the score is ±1)
contour(x1Grid, x2Grid, scoreGrid, [1 1], 'k--', 'LineWidth', 1); % Upper margin
contour(x1Grid, x2Grid, scoreGrid, [-1 -1], 'k--', 'LineWidth', 1); % Lower margin

% Enable gridlines
grid on;

% Add the custom legend
legend([h1(1), h1(2)], {'Class 1', 'Class -1'}, 'Location', 'Best');

hold off;

% Display the support vectors
disp('Support Vectors:');
disp(supportVectors);
