%% Bayesian Network MATLAB Script

clear; clc; close all;

%% ====================== USER INPUT PARAMETERS =========================
dataSource = 'random';  % 'file', 'manual', 'random'
dataFile = 'data.csv';  % if using 'file'
numVars = 5;            % number of variables for random generation
numSamples = 1000;      % number of samples for random generation
varNames = {'A','B','C','D','E'}; % variable names

%% ========================== LOAD DATA =================================
switch lower(dataSource)
    case 'file'
        data = readmatrix(dataFile);
        if isempty(varNames)
            varNames = strcat("Var", string(1:size(data,2)));
        end
        
    case 'manual'
        data = [0 1 0 1 1;
                1 0 1 0 0;
                0 1 0 1 0];
        
    case 'random'
        data = randi([0 1], numSamples, numVars);
end

numVars = size(data,2); % ensure consistency
if isempty(varNames)
    varNames = strcat("Var", string(1:numVars));
end

%% ======================== CREATE BAYESIAN NETWORK =====================
% Random DAG generation
adjMatrix = zeros(numVars);
rng(0); % reproducibility

for i = 2:numVars
    parents = randsample(1:i-1, randi([0 i-1]), false);
    adjMatrix(parents,i) = 1;
end

% Learn CPTs from data
CPTs = cell(1,numVars);
for i = 1:numVars
    parents = find(adjMatrix(:,i));
    if isempty(parents)
        % No parents: marginal probability
        p = mean(data(:,i));
        CPTs{i} = [1-p, p]; % P(X=0), P(X=1)
    else
        % Parents exist: conditional probabilities
        numParents = length(parents);
        numComb = 2^numParents;
        CPT = zeros(numComb,2);
        for j = 0:numComb-1
            comb = bitget(j, numParents:-1:1);
            idx = ismember(data(:,parents), comb, 'rows');
            if sum(idx)==0
                CPT(j+1,:) = [0.5 0.5]; % default if no data
            else
                p = mean(data(idx,i));
                CPT(j+1,:) = [1-p, p];
            end
        end
        CPTs{i} = CPT;
    end
end

%% ======================== PLOTS =======================================
% Plot network graph
G = digraph(adjMatrix, varNames);
figure('Name','Bayesian Network Structure');
plot(G,'Layout','layered','NodeColor','cyan','MarkerSize',7,'LineWidth',1.5);
title('Bayesian Network Structure');

% Plot marginal probabilities
figure('Name','Marginal Probabilities');
for i = 1:numVars
    subplot(ceil(numVars/2),2,i);
    if isempty(find(adjMatrix(:,i),1))
        probs = CPTs{i};
    else
        % Average over all parent configurations
        probs = mean(CPTs{i},1);
    end
    bar(probs);
    title(['P(' varNames{i} ')']);
    xlabel('State'); ylabel('Probability');
end

% Display CPTs
for i = 1:numVars
    disp(['CPT of ' varNames{i} ':']);
    disp(CPTs{i});
end

%% ======================== JOINT DISTRIBUTION (first 10 rows) =========
allComb = dec2bin(0:2^numVars-1)-'0';  % All binary combinations of variables
jointProbs = zeros(2^numVars,1);

% Compute joint probabilities
for i = 1:2^numVars
    prob = 1;
    for j = 1:numVars
        parents = find(adjMatrix(:,j));
        if isempty(parents)
            prob = prob * CPTs{j}(allComb(i,j)+1);
        else
            parentIdx = 0;
            for k = 1:length(parents)
                parentIdx = parentIdx + allComb(i,parents(k)) * 2^(length(parents)-k);
            end
            prob = prob * CPTs{j}(parentIdx+1, allComb(i,j)+1);
        end
    end
    jointProbs(i) = prob;
end

% Display first 10 joint probabilities
disp('First 10 joint probabilities:');
jointTable = array2table([allComb(1:min(10,end),:), jointProbs(1:min(10,end))], ...
    'VariableNames', [varNames, {'JointProb'}]);

disp(jointTable);
