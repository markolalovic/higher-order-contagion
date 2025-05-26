""" Estimate rates a_k, b_k. """

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

def estimate_em(t_data_bd, p_data_bd, p0_guess, p_bounds):
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
        max_it=100,
        i_tol=1e-6,
        se_type='asymptotic',
        display=True
    )
    print(f"Estimated parameters [beta1, beta2]: {est_em_custom.p}")
    print(f"Standard errors: {est_em_custom.se}")
    print(f"Log-likelihood: {est_em_custom.val}")
    print(f"Compute time: {est_em_custom.compute_time:.0f} seconds")    
    return est_em_custom

def estimate_em_2(t_data_bd, p_data_bd, p0_guess, p_bounds):
    # estimating beta1, beta2
    est_em_custom = bd.estimate(
        t_data=t_data_bd,
        p_data=p_data_bd,
        p0=p0_guess,
        p_bounds=p_bounds,
        model='custom',
        # framework='em',
        scheme='discrete',
        b_rate=complete_birth_rate,
        d_rate=complete_death_rate,
        max_it=100,
        i_tol=1e-6,
        # se_type='asymptotic',
        display=True
    )
    print(f"Estimated parameters [beta1, beta2]: {est_em_custom.p}")
    print(f"Standard errors: {est_em_custom.se}")
    print(f"Log-likelihood: {est_em_custom.val}")
    print(f"Compute time: {est_em_custom.compute_time:.0f} seconds")    
    return est_em_custom

def estimate_em_3(t_data_bd, p_data_bd, p0_guess, p_bounds):
    # estimating beta1, beta2
    est_em_custom = bd.estimate(
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
    print(f"Estimated parameters [beta1, beta2]: {est_em_custom.p}")
    print(f"Standard errors: {est_em_custom.se}")
    print(f"Log-likelihood: {est_em_custom.val}")
    print(f"Compute time: {est_em_custom.compute_time:.0f} seconds")    
    return est_em_custom
