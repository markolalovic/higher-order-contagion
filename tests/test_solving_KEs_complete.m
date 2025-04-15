%%
% --- Solving the ODEs using ode45 in Matlab ---
%
% clear all
clear; clc; close all; 

% --- Setup ---
N = 4;
I0 = 1;
time_max = 10;

beta1 = 2 / N;       % pairwise infection rate
beta2 = 4 / (N^2);   % hyperedge contagion rate
mu    = 1;           % recovery rate

fprintf('Setup: \n');
fprintf('\tH = Complete Hypergraph, N = %d, I0 = %d\n', N, I0);
fprintf('\tbeta1 = %f, beta2 = %f, mu = %f\n\n', beta1, beta2, mu);

M = N + 1; % number of states k = 0, 1, ..., N

% --- initial condition ---
% p0 is a column vector 
% p0(k+1) corresponds to the probability of state k
p0 = zeros(M, 1);

p0(I0 + 1) = 1.0; % NOTE: state k = I0 is at index I0 + 1
fprintf('p0 = \n');
disp(p0');

% --- time range and times where we evaluate solution ---
nsteps = 101;
t_span = [0.0, time_max];
t_eval = linspace(t_span(1), t_span(2), nsteps);

% --- solve KEs ---
% function handle for the ODE system
ode_system_handle = @(t, p) ode_system_complete(t, p, N, beta1, beta2, mu);

% run ode45 
% returns solution `sol`
% TODO: set `options`?
sol = ode45(ode_system_handle, t_eval, p0);

% solution p_k(t) is in sol.y 
% each column corresponds to a time point in sol.x
% sol.y(k + 1, i) is the probability of state k at time sol.x(i)

% calculate and plot expected values
expected_values = calculate_expected_values(sol);

figure;
scatter(sol.x, expected_values, 10, 'k', 'filled'); % 'filled' makes points solid like Python default
xlabel('Time t');
ylabel('E[p_k(t)]');
title(sprintf('H = Complete Hypergraph, N = %d', N));
grid on;

% --- save the figure ---
output_dir = '../figures/solutions-kolmogorov/debug';
if ~exist(output_dir, 'dir')
   mkdir(output_dir);
end
filename = fullfile(output_dir, 'expected-values_matlab_N=4.pdf');
saveas(gcf, filename);
fprintf('Figure saved to %s\n', filename);


function [s1, s2] = total_SI_pairs_and_SII_triples(N, k)
    s1 = k * (N - k);
    if k >= 2
        s2 = nchoosek(k, 2) * (N - k);
    else
        s2 = 0; % if k < 2, no 3-node edges can fire, i.e. no pairs of infected nodes
    end
end

function dpdt = ode_system_complete(t, p, N, beta1, beta2, mu)
    % system of fwd Kolmogorov equations
    M = N + 1;
    dpdt = zeros(M, 1); % initialize solution column vector

    % for k_idx, k in 0, 1, ..., N
    % k_idx corresponds to the index in the vector p (1 to M)
    % k corresponds to the number of infected nodes (0 to N)
    for k_idx = 1:M
        k = k_idx - 1; % current number of infected 

        % calculate rates
        if k >= 0 && k <= N
            [s1K, s2K] = total_SI_pairs_and_SII_triples(N, k);
            infection_out_rate = beta1 * s1K + beta2 * s2K;
            recovery_out_rate = mu * k;
        else
             infection_out_rate = 0;
             recovery_out_rate = 0;
        end

        % to state k from state k - 1 (infection)
        if k > 0
            [s1M, s2M] = total_SI_pairs_and_SII_triples(N, k - 1);
            infection_in_rate = beta1 * s1M + beta2 * s2M;
            inflow_from_k_minus_1 = infection_in_rate * p(k_idx - 1);
        else
            inflow_from_k_minus_1 = 0;
        end

        % to state k from state k + 1 (recovery)
        if k < N
             recovery_in_rate = mu * (k + 1);
             inflow_from_k_plus_1 = recovery_in_rate * p(k_idx + 1);
        else
             inflow_from_k_plus_1 = 0;
        end

        % stay in state k
        dpdt(k_idx) = inflow_from_k_minus_1 + inflow_from_k_plus_1 ...
                       - (infection_out_rate + recovery_out_rate) * p(k_idx);
    end
end


function expected_values = calculate_expected_values(sol)
    p_vals = sol.y;
    [M, ntimes] = size(p_vals);
    N = M - 1;
    k_vector = (0:N)'; % of size M x 1
    expected_values = k_vector' * p_vals;
end