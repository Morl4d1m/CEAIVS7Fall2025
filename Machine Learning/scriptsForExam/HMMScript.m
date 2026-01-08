%% ====================== HIDDEN MARKOV MODEL SCRIPT (VERBOSE) ======================
% This version shows detailed explanations and intermediate calculations.

clear; close all; clc;

%% ====================== USER PARAMETERS ======================

N = 3;  % hidden states
M = 4;  % observation symbols
input_mode = 'manual';  % 'file', 'manual', 'random'
T = 10;  % sequence length for random generation

% HMM parameters
A = [0.7 0.2 0.1;
     0.3 0.4 0.3;
     0.2 0.3 0.5];        % Transition matrix
B = [0.5 0.2 0.2 0.1;
     0.1 0.3 0.4 0.2;
     0.2 0.3 0.2 0.3];   % Observation matrix
pi_init = [0.5 0.3 0.2];  % Initial state distribution

% Choose algorithm: 'forward', 'backward', 'baumwelch', 'viterbi'
algorithm = 'forward';

%% ====================== LOAD OR GENERATE DATA ======================
switch input_mode
    case 'file'
        [filename, pathname] = uigetfile('*.txt','Select Observation File');
        obs = load(fullfile(pathname, filename));
    case 'manual'
        obs = input('Enter observation sequence as a vector [1 2 3 ...]: ');
    case 'random'
        obs = zeros(1,T);
        state = zeros(1,T);
        state(1) = find(mnrnd(1,pi_init));
        obs(1) = find(mnrnd(1,B(state(1),:)));
        for t = 2:T
            state(t) = find(mnrnd(1,A(state(t-1),:)));
            obs(t) = find(mnrnd(1,B(state(t),:)));
        end
        fprintf('Random observation sequence generated:\n'); disp(obs);
    otherwise
        error('Invalid input mode.');
end

%% ====================== RUN SELECTED ALGORITHM ======================
switch lower(algorithm)
    case 'forward'
        [alpha, c] = forward_pass_verbose(obs,A,B,pi_init);
        fprintf('\nSequence likelihood P(O|HMM) = %.6f\n', prod(c));
        
    case 'backward'
        [alpha, c] = forward_pass_verbose(obs,A,B,pi_init);
        beta = backward_pass_verbose(obs,A,B,c);
        fprintf('\nSequence likelihood P(O|HMM) = %.6f\n', sum(alpha(end,:)));
        
    case 'baumwelch'
        % Initialize guesses
        A_init = rand(N); A_init = A_init ./ sum(A_init,2);
        B_init = rand(N,M); B_init = B_init ./ sum(B_init,2);
        pi_init_train = ones(1,N)/N;
        max_iter = 50; tol = 1e-4;
        
        [A_est,B_est,pi_est,loglik] = hmm_train_verbose(obs,N,M,A_init,B_init,pi_init_train,max_iter,tol);
        
        disp('Estimated Transition Matrix A:'); disp(A_est);
        disp('Estimated Observation Matrix B:'); disp(B_est);
        disp('Estimated Initial Distribution pi:'); disp(pi_est);
        fprintf('Final Log-Likelihood = %.6f\n', loglik(end));
        
    case 'viterbi'
        [seq, seq_prob, delta, psi] = hmmviterbi_verbose(obs,A,B,pi_init);
        disp('Most likely hidden state sequence (Viterbi):'); disp(seq);
        fprintf('Probability of most likely sequence = %.6f\n', seq_prob);
        
    otherwise
        error('Invalid algorithm selection.');
end

%% ====================== FUNCTIONS ======================

function [alpha,c] = forward_pass_verbose(obs,A,B,pi_est)
    T = length(obs); N = size(A,1);
    alpha = zeros(T,N); c = zeros(T,1);
    
    fprintf('\n=== Forward Pass ===\n');
    
    % Initial step
    fprintf('\nStep 1: Initialize alpha_1(i) = pi(i)*B(i,O1)\n');
    alpha(1,:) = pi_est .* B(:,obs(1))';
    fprintf('Unscaled alpha(1,:) = '); disp(alpha(1,:));
    c(1) = sum(alpha(1,:));
    alpha(1,:) = alpha(1,:) / c(1);
    fprintf('Scaled alpha(1,:) = '); disp(alpha(1,:));
    fprintf('Scaling factor c(1) = %.4f\n', c(1));
    
    % Recursion
    for t = 2:T
        fprintf('\nStep %d: alpha_t(j) = [sum_i alpha_{t-1}(i)*A(i,j)]*B(j,O_t)\n', t);
        alpha(t,:) = (alpha(t-1,:) * A) .* B(:,obs(t))';
        fprintf('Unscaled alpha(%d,:) = ', t); disp(alpha(t,:));
        c(t) = sum(alpha(t,:));
        alpha(t,:) = alpha(t,:) / c(t);
        fprintf('Scaled alpha(%d,:) = ', t); disp(alpha(t,:));
        fprintf('Scaling factor c(%d) = %.4f\n', t, c(t));
    end
end

function beta = backward_pass_verbose(obs,A,B,c)
    T = length(obs); N = size(A,1);
    beta = zeros(T,N);
    
    fprintf('\n=== Backward Pass ===\n');
    fprintf('\nStep T: Initialize beta_T(i) = 1/c(T)\n');
    beta(T,:) = ones(1,N)/c(T);
    fprintf('beta(%d,:) = ', T); disp(beta(T,:));
    
    for t = T-1:-1:1
        fprintf('\nStep %d: beta_t(i) = sum_j A(i,j)*B(j,O_{t+1})*beta_{t+1}(j) / c(t)\n', t);
        beta(t,:) = (beta(t+1,:) .* B(:,obs(t+1))') * A' / c(t);
        fprintf('beta(%d,:) = ', t); disp(beta(t,:));
    end
end

function [A,B,pi_est,loglik] = hmm_train_verbose(obs,N,M,A,B,pi_est,max_iter,tol)
    T = length(obs); loglik = zeros(1,max_iter);
    fprintf('\n=== Baum-Welch Training ===\n');
    
    for iter = 1:max_iter
        fprintf('\n--- Iteration %d ---\n', iter);
        [alpha, c] = forward_pass_verbose(obs,A,B,pi_est);
        beta = backward_pass_verbose(obs,A,B,c);
        
        % Gamma
        fprintf('\nCompute gamma_t(i) = alpha_t(i) * beta_t(i) / sum(alpha_t * beta_t)\n');
        gamma = alpha .* beta; gamma = gamma ./ sum(gamma,2);
        fprintf('Gamma probabilities:\n'); disp(gamma);
        
        % Xi
        fprintf('\nCompute xi_t(i,j) = alpha_t(i)*A(i,j)*B(j,O_{t+1})*beta_{t+1}(j) / sum_{i,j}(...)\n');
        xi = zeros(N,N,T-1);
        for t = 1:T-1
            denom = sum(sum(alpha(t,:)' .* A .* B(:,obs(t+1))' .* beta(t+1,:)));
            for i = 1:N
                for j = 1:N
                    xi(i,j,t) = alpha(t,i) * A(i,j) * B(j,obs(t+1)) * beta(t+1,j) / denom;
                end
            end
        end
        fprintf('Xi probabilities at t=1:\n'); disp(xi(:,:,1)); % show first time step as example
        
        % Re-estimate
        fprintf('\nRe-estimate pi, A, B using gamma and xi\n');
        pi_est = gamma(1,:);
        for i = 1:N
            for j = 1:N
                A(i,j) = sum(xi(i,j,:)) / sum(gamma(1:end-1,i));
            end
            for k = 1:M
                B(i,k) = sum(gamma(obs==k,i)) / sum(gamma(:,i));
            end
        end
        
        loglik(iter) = sum(log(c));
        fprintf('Log-likelihood = %.6f\n', loglik(iter));
        if iter>1 && abs(loglik(iter)-loglik(iter-1)) < tol
            loglik = loglik(1:iter);
            break;
        end
    end
end

function [seq, seq_prob, delta, psi] = hmmviterbi_verbose(obs,A,B,pi_est)
    T = length(obs); N = size(A,1);
    delta = zeros(T,N); psi = zeros(T,N);
    
    fprintf('\n=== Viterbi Algorithm ===\n');
    fprintf('Step 1: Initialize delta_1(i) = log(pi(i)*B(i,O1))\n');
    delta(1,:) = log(pi_est) + log(B(:,obs(1))');
    fprintf('delta(1,:) = '); disp(delta(1,:));
    
    for t = 2:T
        fprintf('\nStep %d: delta_t(j) = max_i[delta_{t-1}(i)+log(A(i,j))] + log(B(j,O_t))\n', t);
        for j = 1:N
            [delta(t,j), psi(t,j)] = max(delta(t-1,:) + log(A(:,j)'));
            delta(t,j) = delta(t,j) + log(B(j,obs(t)));
        end
        fprintf('delta(%d,:) = ', t); disp(delta(t,:));
        fprintf('psi(%d,:) = ', t); disp(psi(t,:));
    end
    
    % Backtracking
    seq = zeros(1,T);
    [seq_prob, seq(T)] = max(delta(T,:));
    fprintf('\nBacktracking to find most likely state sequence\n');
    for t = T-1:-1:1
        seq(t) = psi(t+1,seq(t+1));
    end
end
