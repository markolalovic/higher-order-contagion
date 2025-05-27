""" Estimation of:
  * state-wise total rates a_k, b_k, i.e.: per-state k rates
  * underlying global rates beta_1, beta_2, i.e.: per-interaction rates  
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from scipy.special import comb

from simulate_gillespie import *
from solve_kolmogorov import *
import birdepy as bd
import scipy.linalg
from scipy.sparse import diags
import pickle

def calculate_mle_beta1_beta2_complete(X_sims, N):
    r"""
    Calculates the Maximum Likelihood Estimates (MLEs): 
      * for global beta1 and beta2
      * for an SIS model on a COMPLETE simplicial complex
      * given continuous observations
      * where event types (PW, HO, RC) are distinguished
    
    NOTE: recovery rate mu can also be estimated, altough we asume it's known
    """
    total_U_events = 0.0 # sum of all pair wise birth events (sum over k of U_k in draft)
    total_V_events = 0.0 # sum of all higher-order birth events (sum over k of V_k)
    total_D_events = 0.0 # sum of all recovery events (sum over k of D_k)

    # denominators for MLEs
    # sum over k of [ S_k^(1) * T_k ] and [ S_k^(2) * T_k ]
    # sum over k of [ k * T_k ] for mu
    sum_S1_Tk = 0.0
    sum_S2_Tk = 0.0
    sum_k_Tk = 0.0

    for sim_idx, X_t in enumerate(X_sims):
        if X_t.shape[1] < 3:
            print(f"Skipping sim {sim_idx}: has less than 3 recorded steps!")
            continue

        # iterate over actual events
        for j in range(1, X_t.shape[1] - 1):
            k_before = int(X_t[2, j-1])    # state *before* the j-th event
            waiting_time = X_t[1, j]       # time spent in state k_before
            event_type = X_t[3, j]         # type of the j-th event

            # this should not happen for actual events
            if waiting_time is None or waiting_time <= 0:
                continue

            # accumulate time spent in state k_before, weighted by structural opportunity
            # for complete graph, S_k^(1) = k (N - k)
            s1_k_before = k_before * (N - k_before)
            sum_S1_Tk += s1_k_before * waiting_time

            # for complete graph, S_k^(2) = binom(k,2)(N-k)
            s2_k_before = 0.0
            if k_before >= 2:
                s2_k_before = comb(k_before, 2, exact=False) * (N - k_before)
            sum_S2_Tk += s2_k_before * waiting_time

            # for recovery rate
            sum_k_Tk += k_before * waiting_time

            # count event types
            if event_type == 'PW':
                total_U_events += 1
            elif event_type == 'HO':
                total_V_events += 1
            elif event_type == 'RC':
                total_D_events += 1

    # --- Calculate MLEs ---
    beta1_hat = total_U_events / sum_S1_Tk if sum_S1_Tk > 1e-9 else 0.0
    beta2_hat = total_V_events / sum_S2_Tk if sum_S2_Tk > 1e-9 else 0.0
    mu_hat_calc = total_D_events / sum_k_Tk if sum_k_Tk > 1e-9 else 0.0

    return beta1_hat, beta2_hat, mu_hat_calc

def calculate_estimates(X_sims, N, beta1_true, beta2_true, min_Tk_threshold=1e-9):
    r""" Calculates estimates of `a_k`, `b_k`:
      - Total rate MLEs: `a_k_hat`, `b_k_hat`
      - Empirical estimators with empirical counts: `a_k_tilde`, `b_k_tilde`
      - Total birth rate for both `lambda_k = a_k + b_k`
    
    # Returns dict of full M = N + 1 length arrays a_k_hat, b_k_hat, ...

    NOTE:
      - To get valid_k indices threshold using Tk  for reliable estimates:
        `valid_k_idx = estimates["T_k"] > min_Tk_threshold`
    TODO:
      - tune min_Tk_threshold, 1e-9, ..., 1e-6
    """    
    # initialize aggregated stats for MLEs
    T_k = np.zeros(N + 1, dtype=np.float64) # time spent in states k
    U_k = np.zeros(N + 1, dtype=np.float64) # PW births FROM state k
    V_k = np.zeros(N + 1, dtype=np.float64) # HO births FROM state k
    D_k = np.zeros(N + 1, dtype=np.float64) # RC FROM state k

    # initialize accumulators for time-weighted structural counts for tilde estimates
    # sum of (total_pw_count * waiting_time) for each state k
    # sum of (total_ho_count * waiting_time) for each state k
    sum_pw_counts_x_time_k = np.zeros(N + 1, dtype=np.float64)
    sum_ho_counts_x_time_k = np.zeros(N + 1, dtype=np.float64)

    total_events_processed = 0
    for sim_idx, X_t in enumerate(X_sims):
        # simulation needs to have: initial state, at least one event, final state
        if X_t.shape[1] < 3:
            print(f"Skipping: {sim_idx} has less than 3 events")
            continue

        durations = X_t[1, 1:-1]
        states_during_interval = X_t[2, :-2].astype(int)
        events_ending_interval = X_t[3, 1:-1]
        counts_pw = X_t[4, :-2].astype(int)
        counts_ho = X_t[5, :-2].astype(int)

        for i in range(len(durations)):
            k_before = states_during_interval[i]
            duration = durations[i]
            event_type = events_ending_interval[i]

            pw_count_in_k = counts_pw[i]
            ho_count_in_k = counts_ho[i]

            T_k[k_before] += duration
            total_events_processed += 1
            if event_type == 'PW':
                U_k[k_before] += 1
            elif event_type == 'HO':
                V_k[k_before] += 1
            elif event_type == 'RC':
                D_k[k_before] += 1

            # accumulate for tilde estimates
            sum_pw_counts_x_time_k[k_before] += pw_count_in_k * duration
            sum_ho_counts_x_time_k[k_before] += ho_count_in_k * duration

    print(f"Total events processed: {total_events_processed}")
    print(f"Total time T_k accumulated across all states: {np.sum(T_k):.2f}")

    # --- Calculate MLEs: a_k, b_k hats ---
    a_k_hat = np.zeros(N + 1, dtype=float)
    b_k_hat = np.zeros(N + 1, dtype=float)
    c_k_hat = np.zeros(N + 1, dtype=float)
    
    for k in range(N + 1):
        if T_k[k] >= min_Tk_threshold:
            a_k_hat[k] = U_k[k] / T_k[k]
            b_k_hat[k] = V_k[k] / T_k[k]
            c_k_hat[k] = D_k[k] / T_k[k]

    lambda_k_hat = a_k_hat + b_k_hat # total birth rate from MLEs

    # --- Calculate empirical estimates: a_k, b_k tildes ---
    a_k_tilde = np.zeros(N + 1, dtype=float)
    b_k_tilde = np.zeros(N + 1, dtype=float)

    for k in range(N + 1):
        if T_k[k] >= min_Tk_threshold:
            avg_pw_struct_potential = sum_pw_counts_x_time_k[k] / T_k[k]
            avg_ho_struct_potential = sum_ho_counts_x_time_k[k] / T_k[k]
            a_k_tilde[k] = beta1_true * avg_pw_struct_potential
            b_k_tilde[k] = beta2_true * avg_ho_struct_potential

    lambda_k_tilde = a_k_tilde + b_k_tilde # total birth rate from tildes

    return {
        "a_k_hat": a_k_hat,
        "b_k_hat": b_k_hat,
        "c_k_hat": c_k_hat,
        "lambda_k_hat": lambda_k_hat,
        "a_k_tilde": a_k_tilde,
        "b_k_tilde": b_k_tilde,
        "lambda_k_tilde": lambda_k_tilde,
        "T_k": T_k,
        "U_k": U_k,
        "V_k": V_k,
        "D_k": D_k,
        "sum_pw_counts_x_time_k": sum_pw_counts_x_time_k, # for diagnostics
        "sum_ho_counts_x_time_k": sum_ho_counts_x_time_k
    }

def complete_birth_rate(z, p):
    N_fixed = 100 # TODO: write a wrapper
    k = int(z) # integer state
    # use these directly without dividing by N or N^2 first
    beta1_unscaled, beta2_unscaled = p # e.g. p = [2.4, 4.4]
    
    rate_a = (beta1_unscaled / N_fixed) * k * (N_fixed - k)
    rate_b = 0.0
    if k >= 2:
        rate_b = (beta2_unscaled / (N_fixed**2)) * comb(k, 2, exact=False) * (N_fixed - k)
    
    return max(0.0, rate_a + rate_b)

def complete_death_rate(z, p):
    mu_fixed = 1.0 # TODO: write a wrapper
    k = int(z) # integer state
    # set recovery rate c_k = mu * k
    rate_c = mu_fixed * k

    return max(0.0, rate_c)

def estimate_em(t_data_bd, p_data_bd, p0_guess, p_bounds, max_iter=100):
    # estimating beta1, beta2
    est_em_custom = bd.estimate(
        t_data=t_data_bd,
        p_data=p_data_bd,
        p0=p0_guess,
        p_bounds=p_bounds,
        model='custom',
        framework='em',
        scheme='discrete',
        b_rate=complete_birth_rate,
        d_rate=complete_death_rate,
        max_it=max_iter,
        i_tol=1e-6,
        se_type='asymptotic',
        display=True
    )
    print(f"Estimated parameters [beta1, beta2]: {est_em_custom.p}")
    print(f"Standard errors: {est_em_custom.se}")
    print(f"Log-likelihood: {est_em_custom.val}")
    print(f"Compute time: {est_em_custom.compute_time:.0f} seconds")    
    return est_em_custom

def estimate_dnm(t_data_bd, p_data_bd, p0_guess, p_bounds):
    # estimating beta1, beta2
    est_dnm_custom = bd.estimate(
        t_data=t_data_bd,
        p_data=p_data_bd,
        p0=p0_guess,
        p_bounds=p_bounds,
        model='custom',
        framework='dnm',
        opt_method='SLSQP',    
        scheme='discrete',
        b_rate=complete_birth_rate,
        d_rate=complete_death_rate,
        max_it=100,
        i_tol=1e-6,
        se_type='asymptotic',
        display=True
    )
    print(f"Estimated parameters [beta1, beta2]: {est_dnm_custom.p}")
    print(f"Standard errors: {est_dnm_custom.se}")
    print(f"Log-likelihood: {est_dnm_custom.val}")
    print(f"Compute time: {est_dnm_custom.compute_time:.0f} seconds")    
    return est_dnm_custom

def di_lauro_ak_model(k, N_val, C_a, p_a, alpha_a):
    r"""
    Model fit for pairwise rates a_k, based on:
    Di Lauro et al. "Network inference from population-level observation of epidemics." 
    Scientific Reports 10.1 (2020): 18779.
    """
    k_arr = np.asarray(k, dtype=float)
    rate = np.zeros_like(k_arr)
    inner_idx = (k_arr > 1e-9) & (k_arr < N_val - 1e-9)
    if np.any(inner_idx):
        k_in = k_arr[inner_idx]
        term_k = np.maximum(1e-9, k_in)
        term_N_minus_k = np.maximum(1e-9, N_val - k_in)
        boundary_shape = (term_k**p_a) * (term_N_minus_k**p_a)
        norm_factor = N_val / 2.0
        skew_exponent = alpha_a * (k_in - norm_factor) / (norm_factor if abs(norm_factor) > 1e-9 else 1.0)
        skew_term = np.exp(skew_exponent)
        rate[inner_idx] = np.abs(C_a) * boundary_shape * skew_term
    rate[k_arr <= 1e-9] = 0.0
    rate[k_arr >= N_val - 1e-9] = 0.0
    return rate

def di_lauro_bk_model(k, N_val, C_b, p_b, alpha_b):
    r"""
    Generalized model fit for higher-order rates b_k, based on:
    Di Lauro et al. "Network inference from population-level observation of epidemics." 
    Scientific Reports 10.1 (2020): 18779.
    """
    k_arr = np.asarray(k, dtype=float)
    rate = np.zeros_like(k_arr)
    inner_idx = (k_arr > 1.0 + 1e-9) & (k_arr < N_val - 1e-9)
    if np.any(inner_idx):
        k_in = k_arr[inner_idx]
        term_k = np.maximum(1e-9, k_in)
        term_k_minus_1 = np.maximum(1e-9, k_in - 1)
        term_N_minus_k = np.maximum(1e-9, N_val - k_in)
        boundary_shape = term_k * (term_k_minus_1**p_b) * (term_N_minus_k**p_b)
        norm_factor = N_val / 2.0
        skew_exponent = alpha_b * (k_in - norm_factor) / (norm_factor if abs(norm_factor) > 1e-9 else 1.0)
        skew_term = np.exp(skew_exponent)
        rate[inner_idx] = np.abs(C_b) * boundary_shape * skew_term
    rate[k_arr <= 1.0 + 1e-9] = 0.0
    rate[k_arr >= N_val - 1e-9] = 0.0
    return rate
