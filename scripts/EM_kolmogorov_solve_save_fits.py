# ./scripts/EM_kolmogorov_solve_save_fits.py
# Estimates beta1, beta2 from discrete observations using EM.
# Solves Kolmogorov forward equations with these EM estimates and with true beta1, beta2.
# Calculates the Gillespie average curve.
# Saves all resulting trajectories to a CSV file in ./data/demos/ for later plotting.

import numpy as np
import pandas as pd
import os
import sys
from scipy.integrate import solve_ivp
import pickle
import time

sys.path.append('../src/')

from simulate_gillespie import gillespie_sim_complete, discretize_sequence, get_average
from estimate_total_rates import estimate_em
from solve_kolmogorov import list_all_ODEs_complete, calculate_expected_values

np.random.seed(123) # seed for reproducibility

if __name__ == "__main__":
    run_all_calculations = True

    # --- configuration ---
    test_name = "em_data_quantity"
    N = 100
    I0 = 10
    time_max = 10.0
    beta1_s_val_true, beta2_s_val_true = (2.4, 4.4)
    mu_true = 1.0

    # --- directory setup ---
    em_estimates_storage_dir = "../results/estimation/demos/"
    output_csv_filename = "../data/demos/kolmogorov_EM_fits.csv"
    gillespie_avg_csv_filename = "../data/demos/kolmogorov_EM_gillespie_avg.csv"    

    # discretization parameters
    t_obs_start = 0.0
    t_obs_end = time_max
    num_intervals_discrete = 100 # for discretize_sequence

    # true (SCALED) parameters for data generation
    beta1_orig_true = beta1_s_val_true / N
    beta2_orig_true = beta2_s_val_true / (N**2)

    # EM algorithm parameters
    beta1_guess_em_scaled = 1.0
    beta2_guess_em_scaled = 1.0
    p0_em_guess_scaled = [beta1_guess_em_scaled, beta2_guess_em_scaled]

    max_beta1_em_bound_scaled = 10.0
    max_beta2_em_bound_scaled = 50.0
    p_em_bounds_scaled = [[1e-9, max_beta1_em_bound_scaled], [1e-9, max_beta2_em_bound_scaled]]
    em_max_iterations = 100 # max iterations for each EM run
    
    # data richness parameter for EM, how many nsims_pooled
    nsims_per_em = 10

    # for average Gillespie curve plot
    nsims_for_avg_display = 100

    print(f"--- Test: {test_name}, preparing data for ../figures/combined/kolmogorov_EM_fits_comparison.pdf ---")
    print(f"  N={N}, I0={I0}, time_max={time_max}, mu_true={mu_true}")
    print(f"  True beta1_scaled={beta1_s_val_true:.2f}, True beta2_scaled={beta2_s_val_true:.2f}")


    def perform_em_estimation(nsims_pooled):
        """ Returns EM estimates given data richness parameter, how many nsims_pooled to use. """
        print(f"\tGenerating data and running EM for N = {N}, nsims_pooled = {nsims_pooled} ...\n")
        X_sims_current = [gillespie_sim_complete(N, beta1_orig_true, beta2_orig_true, mu_true, I0, time_max) for _ in range(nsims_pooled)]
        Y_sequence_all = []
        for X_t_cont in X_sims_current:
            if X_t_cont.shape[1] < 2: continue
            times_c, states_c = X_t_cont[0,:].astype(float), X_t_cont[2,:].astype(int)
            if times_c[0] <= t_obs_start and times_c[-1] >= t_obs_end:
                Y_sequence_all.extend(discretize_sequence((times_c, states_c), t_obs_start, t_obs_end, num_intervals_discrete))
        
        if not Y_sequence_all: print("STOP: No discrete data generated."); return None
        
        dt_int = Y_sequence_all[0][2]; num_tot_int = len(Y_sequence_all)
        t_data_bd = [np.linspace(0, num_tot_int*dt_int, num_tot_int+1).tolist()]
        p_data_bd = [[Y_sequence_all[0][0]] + [interval[1] for interval in Y_sequence_all]]
        
        # --- call estimate EM ---
        est_result = estimate_em(N, mu_true, t_data_bd, p_data_bd, p0_em_guess_scaled, p_em_bounds_scaled,
                                    max_iter=em_max_iterations, print_metrics=True)
        return est_result.p
    
    # --- get EM estimates ---
    est_params = perform_em_estimation(nsims_per_em) # EM estimates of beta1, beta2

    # --- KE solver setup ---
    # solve ode_system over time
    t_span_ke = (0.0, time_max)
    steps_eval_ke = 201 # for smooth KE curves
    # times t_i to evaluate in, get saved in sol.t
    t_eval_ke = np.linspace(t_span_ke[0], t_span_ke[1], int(steps_eval_ke)) 
    p0_ke = np.zeros(N + 1, dtype=np.float64)
    p0_ke[I0] = 1.0  # all other states have prob 0 at time 0

    # --- solve KEs with true parameters ---
    print("\nSolving KE with True Parameters...")
    # use original (unscaled) beta1, beta2
    ode_system_true = list_all_ODEs_complete(N, beta1_orig_true, beta2_orig_true, mu_true)
    sol_true = solve_ivp(lambda t, p: ode_system_true(t, p), # inline fun
                        t_span_ke, p0_ke, t_eval=t_eval_ke, method="LSODA")
    k_expected_true = calculate_expected_values(sol_true)
    print("Done.\n")

    # --- solve KEs with EM estimates ---
    print("Solving KE with EM Estimates...")
    k_expected_em = np.full_like(t_eval_ke, np.nan)
    beta1_orig_est = est_params[0] / N
    beta2_orig_est = est_params[1] / (N**2)
    ode_system_em = list_all_ODEs_complete(N, beta1_orig_est, beta2_orig_est, mu_true)
    sol_em = solve_ivp(lambda t, p: ode_system_em(t, p),
                                t_span_ke, p0_ke, t_eval=t_eval_ke, method="LSODA")
    k_expected_em = calculate_expected_values(sol_em)
    print("Done.\n")

    # --- generate average gillespie curve ---
    print(f"\nGenerating average gillespie curve (N={N}, {nsims_for_avg_display} sims)...")
    X_sims_for_avg = [gillespie_sim_complete(N, beta1_orig_true, beta2_orig_true, mu_true, I0, time_max) for _ in range(nsims_for_avg_display)]
    avg_curve_gillespie, times_gillespie_avg = get_average(X_sims_for_avg, time_max, nsims_for_avg_display, delta_t=0.01)
    print("Done.\n")

    # --- save all trajectories to CSV ---
    df_data_for_plot = pd.DataFrame({'time': t_eval_ke})
    df_data_for_plot['k_expected_true'] = k_expected_true
    df_data_for_plot['k_expected_em'] = k_expected_em

    df_data_for_plot.to_csv(output_csv_filename, index=False, float_format='%.8g')
    print(f"\nAll KE plot data saved to: {output_csv_filename}")

    # save raw Gillespie average separately (has different time grid)
    df_gillespie_avg_raw = pd.DataFrame({'time_gillespie_avg': times_gillespie_avg, 'k_avg_gillespie_raw': avg_curve_gillespie})
    df_gillespie_avg_raw.to_csv(gillespie_avg_csv_filename, index=False, float_format='%.8g')
    print(f"Raw Gillespie average data saved to: {gillespie_avg_csv_filename}")

    print("\nDone.")

'''
--- Preparing Data for Plot: em_data_quantity ---
  N=100, I0=10, time_max=10.0, mu_true=1.0
  True beta1_scaled=2.40, True beta2_scaled=4.40
	Generating data and running EM for N = 100, nsims_pooled = 10 ...

Iteration  1  estimate is:  [2.00902364 1.53754556]
Iteration  2  estimate is:  [2.48094396 4.44879725]
Iteration  3  estimate is:  [2.02253129 5.85839556]
Iteration  4  estimate is:  [2.0370158  5.81844661]
Iteration  5  estimate is:  [2.02325885 5.86051921]
Iteration  6  estimate is:  [2.02325885 5.86051921]
   
# EM beta1, beta2 hats on num_intervals_discrete * nsims_pooled = 1000 observations:

   Estimated parameters [beta1, beta2]: [2.02325885, 5.86051921]

Standard errors: [np.float64(0.20283040463518895), np.float64(0.6824876571617775)]
Log-likelihood: -2713.78172021913
Compute time: 43 seconds

Solving KE with True Parameters...
Done.

Solving KE with EM Estimates...
Done.


Generating average gillespie curve (N=100, 100 sims)...
Done.


All KE plot data saved to: ../data/demos/kolmogorov_EM_fits.csv
Raw Gillespie average data saved to: ../data/demos/kolmogorov_EM_gillespie_avg.csv

Done.
'''