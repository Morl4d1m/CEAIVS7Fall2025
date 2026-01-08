%% Markov Random Field (MRF) Parameters - Edit these as needed
clear;
clc;

% Grid size (dimensions of MRF)
gridSize = [10, 10]; % 10x10 grid (can adjust)

% Interaction parameters
beta = 1; % interaction strength (pairwise interaction coefficient)

% Data input options
loadDataFlag = false;  % Load data from file (true/false)
manualInputFlag = false;  % Manual data input (true/false)
generateRandomDataFlag = true; % Generate random data (true/false)

% For manual input (example: input size grid, 1 or 0 for each point)
manualData = NaN; % leave as NaN for no manual input

% Filepath for data (if loading from file)
filePath = 'data.mat'; % Adjust to your file location

% Plotting options
plotEnergyFlag = true; % Plot energy curve during sampling
plotGridFlag = true;   % Show grid plot of samples

%% Generate/Load MRF Data
% Option 1: Generate Random MRF Grid
if generateRandomDataFlag
    disp('Generating random MRF grid...');
    MRFGrid = randi([0, 1], gridSize); % Random binary grid (0 or 1)
end

% Option 2: Load data from file (if set to true)
if loadDataFlag
    if exist(filePath, 'file')
        data = load(filePath);
        MRFGrid = data.MRFGrid; % Assuming the file contains 'MRFGrid' variable
        disp(['Loaded MRF grid from: ', filePath]);
    else
        disp('File not found!');
    end
end

% Option 3: Manual Input
if manualInputFlag
    if isnan(manualData)
        disp('Enter the grid data manually (0 or 1 for each cell in the grid):');
        MRFGrid = zeros(gridSize); % Initialize grid as all zeros
        for i = 1:gridSize(1)
            for j = 1:gridSize(2)
                MRFGrid(i,j) = input(sprintf('Enter value for (%d,%d) [0/1]: ', i, j));
            end
        end
    else
        MRFGrid = manualData; % If manual data is provided
    end
end

% Ensure MRFGrid is binary (0 or 1)
MRFGrid = double(MRFGrid == 1);

%% Visualization - Plot Initial Grid (MRF)
if plotGridFlag
    figure;
    imagesc(MRFGrid);
    title('Initial MRF Grid');
    colormap(gray);
    axis equal;
    axis off;
end

%% Energy Function for MRF (Ising Model Energy)
% Assuming a simple Ising model (pairwise interactions)
% Energy function with periodic boundary conditions

disp('Calculating initial energy of the grid...');
energy = 0;
for i = 1:gridSize(1)
    for j = 1:gridSize(2)
        % Sum over neighbors (periodic boundary conditions)
        neighbors = [
            MRFGrid(mod(i, gridSize(1)) + 1, j), ... % Right neighbor
            MRFGrid(mod(i-2, gridSize(1)) + 1, j), ... % Left neighbor
            MRFGrid(i, mod(j, gridSize(2)) + 1), ... % Down neighbor
            MRFGrid(i, mod(j-2, gridSize(2)) + 1) ... % Up neighbor
        ];
        
        % Compute pairwise interaction energy (Ising model)
        for k = 1:length(neighbors)
            energy = energy - beta * MRFGrid(i,j) * neighbors(k);
        end
    end
end
disp(['Initial Energy: ', num2str(energy)]);

%% Gibbs Sampling for Inference (Updating MRF configuration)
iterations = 1000;  % Number of Gibbs sampling iterations
energyHistory = zeros(iterations, 1); % Record energy at each iteration

disp('Starting Gibbs sampling...');
for it = 1:iterations
    if mod(it, 100) == 0
        disp(['Iteration ', num2str(it), ' of ', num2str(iterations)]);
    end
    
    for i = 1:gridSize(1)
        for j = 1:gridSize(2)
            % Update grid point using conditional probability (for Ising model)
            neighbors = [
                MRFGrid(mod(i, gridSize(1)) + 1, j), ... % Right neighbor
                MRFGrid(mod(i-2, gridSize(1)) + 1, j), ... % Left neighbor
                MRFGrid(i, mod(j, gridSize(2)) + 1), ... % Down neighbor
                MRFGrid(i, mod(j-2, gridSize(2)) + 1) ... % Up neighbor
            ];
            
            % Energy calculation before and after flipping
            energy0 = -beta * MRFGrid(i,j) * sum(neighbors); % current energy
            energy1 = -beta * (1 - MRFGrid(i,j)) * sum(neighbors); % new energy (flipped)
            
            % Display the energy calculations
            if mod(it, 100) == 0
                disp(['Energy before flipping (at ', num2str(i), ',', num2str(j), '): ', num2str(energy0)]);
                disp(['Energy after flipping (at ', num2str(i), ',', num2str(j), '): ', num2str(energy1)]);
            end
            
            % Metropolis-Hastings or simple probabilistic update
            if energy1 < energy0
                MRFGrid(i,j) = 1 - MRFGrid(i,j); % Flip the value (0 -> 1 or 1 -> 0)
                if mod(it, 100) == 0
                    disp(['Flipped value at ', num2str(i), ',', num2str(j)]);
                end
            end
        end
    end
    
    % Calculate and store the energy after updating the grid
    energy = 0;
    for i = 1:gridSize(1)
        for j = 1:gridSize(2)
            neighbors = [
                MRFGrid(mod(i, gridSize(1)) + 1, j), ...
                MRFGrid(mod(i-2, gridSize(1)) + 1, j), ...
                MRFGrid(i, mod(j, gridSize(2)) + 1), ...
                MRFGrid(i, mod(j-2, gridSize(2)) + 1)
            ];
            
            for k = 1:length(neighbors)
                energy = energy - beta * MRFGrid(i,j) * neighbors(k);
            end
        end
    end
    
    energyHistory(it) = energy; % Store energy at this iteration
end

%% Plotting Energy Curve
if plotEnergyFlag
    figure;
    plot(1:iterations, energyHistory);
    title('Energy vs. Iterations');
    xlabel('Iterations');
    ylabel('Energy');
end

%% Final MRF Grid Visualization
figure;
imagesc(MRFGrid);
title('Final MRF Grid (After Gibbs Sampling)');
colormap(gray);
axis equal;
axis off;

disp('Gibbs sampling completed.');
disp(['Final Energy: ', num2str(energy)]);
