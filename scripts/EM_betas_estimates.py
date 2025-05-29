# ./scripts/EM_betas_estimates.py
# Computes/loads EM beta estimates, calculates metrics,
# and saves a scatter plot to: 
# ./figures/combined/beta_estimates_scatter_EM_balanced_case.pdf

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
sys.path.append('../src/')

from simulate_gillespie import gillespie_sim_complete, discretize_sequence
from estimate_total_rates import estimate_em
# from higher_order_structures import Complete

if __name__ == "__main__":
    r"""
    This script tests the EM algorithm for beta1 and beta2 on complete SCs,
    for a more complex scenario where we are given: 
      * discrete observations
      * and event types are unknown (PW and HO events cannot be distinguished)
    
    It runs multiple independent estimations (each from pooled Gillespie runs)
    to assess the distribution and stability of the EM estimates.

    It saves the raw estimates and produces a scatter plot and key metrics.
    """
    # --- config ---
    zoom_in = True

    run_estimations = False    # set to False for plot modifications!!
    num_estimation_runs = 1000  # number of independent EM estimations to perform <- TODO: increase to 1000
    nsims_per_em_run = 10      # number of pooled Gillespie sims for each EM run

    test_name = "EM_complete_discrete_unknown_event_types"
    N = 100
    I0 = 10
    time_max_gillespie = 10.0
    mu_true = 1.0

    # discretization parameters
    t_obs_start = 0.0
    t_obs_end = time_max_gillespie
    num_intervals_discrete = 100 # number of intervals for `discretize_sequence`

    # true parameters for data generation 
    # Case 1: "Balanced"
    beta1_s_val_true, beta2_s_val_true = (2.4, 4.4)

    # Case 2: "Hard Case"
    # hand-picked pair = hard case: low beta1 (PW), high beta2 (HO) rate
    # beta1_s_val_true, beta2_s_val_true = (1.1, 8.0)

    # EM algorithm parameters
    beta1_guess_em = 1.0    # initial guess for SCALED beta1 * N
    beta2_guess_em = 1.0    # initial guess for SCALED beta2 * N^2

    # p0_guess for estimate_em should be SCALED.
    # complete_birth_rate expects unscaled p0 = [beta1_unscaled, beta2_unscaled]
    # which it then scales
    p0_em_guess = [beta1_guess_em, beta2_guess_em]

    # TODO: perhaps choose tighter bounds
    max_beta1_em_bound = 10.0    # bound for SCALED beta1 * N
    max_beta2_em_bound = 50.0    # bound for SCALED beta2 * N^2
    p_em_bounds = [[1e-9, max_beta1_em_bound], [1e-9, max_beta2_em_bound]]
    
    em_max_iterations = 100 # max iterations for each EM run

    # --- directory setup ---
    dir_figure = "../figures/combined/"
    dir_data = "../results/estimation/demos/"

    file_name_estimates = f"EM_beta_estimates_{N}_I0{I0}_runs{num_estimation_runs}.pkl"
    estimates_filename = os.path.join(dir_data, file_name_estimates)

    # for data generation, convert SCALED true betas to ORIGINAL per-interaction true betas
    beta1_orig_true = beta1_s_val_true / N
    beta2_orig_true = beta2_s_val_true / (N**2)

    print(f"--- EM Estimation Test: {test_name} ---")
    print(f"N={N}, I0={I0}, time_max_gillespie={time_max_gillespie}")
    print(f"\t True beta1_scaled={beta1_s_val_true:.2f}, True beta2_scaled={beta2_s_val_true:.2f}")
    print(f"\t Each EM estimation based on {nsims_per_em_run} Gillespie runs, discretized into {num_intervals_discrete} intervals each.")
    print(f"\t Total EM estimation repetitions: {num_estimation_runs}\n")

    if run_estimations:
        all_beta1_hats_scaled = []
        all_beta2_hats_scaled = []
        estimation_times = []

        print(f"Running {num_estimation_runs} EM estimations...")
        for est_run_idx in range(num_estimation_runs):
            print(f"  EM Estimation Run {est_run_idx + 1}/{num_estimation_runs}...")
            start_single_em_run_time = time.time()

            # Step 1: Generate continuous observations
            X_sims_current = []
            for _ in range(nsims_per_em_run):
                X_t = gillespie_sim_complete(N, beta1_orig_true, beta2_orig_true, mu_true,
                                             I0, time_max_gillespie)
                X_sims_current.append(X_t)

            # Step 2: Discretize them
            Y_sequence_all_current = []
            for X_t_cont in X_sims_current:
                if X_t_cont.shape[1] < 2: continue # need at least start and end
                times_c_run = X_t_cont[0, :].astype(float)
                states_c_run = X_t_cont[2, :].astype(int)
                if times_c_run[0] <= t_obs_start and times_c_run[-1] >= t_obs_end:
                    Y_sequence_run = discretize_sequence((times_c_run, states_c_run),
                                                         t_obs_start, t_obs_end, num_intervals_discrete)
                    Y_sequence_all_current.extend(Y_sequence_run)

            if not Y_sequence_all_current:
                print(f"Warning: No valid discrete data for EM run {est_run_idx + 1}. Storing NaNs.")
                all_beta1_hats_scaled.append(np.nan)
                all_beta2_hats_scaled.append(np.nan)
                estimation_times.append(time.time() - start_single_em_run_time)
                continue

            # prepare data for EM algorithm: list of times and list of states
            dt_interval = Y_sequence_all_current[0][2]
            num_total_intervals = len(Y_sequence_all_current)
            t_data_bd_em = [np.linspace(0, num_total_intervals * dt_interval, num_total_intervals + 1).tolist()]
            p_data_bd_em = [[Y_sequence_all_current[0][0]] + [interval[1] for interval in Y_sequence_all_current]]
            
            # ---------------------------------------
            # --- call the `estimate_em` function ---
            # ---------------------------------------
            est_result = estimate_em(N, mu_true, t_data_bd_em, p_data_bd_em, p0_em_guess, p_em_bounds, 
                                     max_iter=em_max_iterations, print_metrics=False)
            
            if est_result and hasattr(est_result, 'p') and len(est_result.p) == 2:
                # est_result.p are the SCALED estimates: [beta1 N_hat, beta2 N^2_hat]
                all_beta1_hats_scaled.append(est_result.p[0])
                all_beta2_hats_scaled.append(est_result.p[1])
                print(f"\n\tRun {est_run_idx+1} Estimates (scaled): b1N = {est_result.p[0]:.3f}, b2N^2 = {est_result.p[1]:.3f}\n")
            else:
                print(f"\n\tEM failed! Run: {est_run_idx+1}. Storing NaNs.\n")
                all_beta1_hats_scaled.append(np.nan)
                all_beta2_hats_scaled.append(np.nan)
            estimation_times.append(time.time() - start_single_em_run_time)

        # save the estimates
        with open(estimates_filename, 'wb') as f:
            pickle.dump({'beta1_hats_scaled': all_beta1_hats_scaled,
                         'beta2_hats_scaled': all_beta2_hats_scaled,
                         'estimation_times': estimation_times}, f)
        print(f"\nAll EM estimates saved to {estimates_filename}")

    else: 
        # load existing estimates
        if os.path.exists(estimates_filename):
            print(f"Loading EM estimates from {estimates_filename}...")
            with open(estimates_filename, 'rb') as f:
                loaded_data = pickle.load(f)
                all_beta1_hats_scaled = loaded_data['beta1_hats_scaled']
                all_beta2_hats_scaled = loaded_data['beta2_hats_scaled']
                if 'estimation_times' in loaded_data:
                    estimation_times = loaded_data['estimation_times']
                    print(f"\t Average estimation time per run: {np.mean(estimation_times):.2f}s \n")
            print(f"Loading complete. Found {len(all_beta1_hats_scaled)} estimates.")
            if len(all_beta1_hats_scaled) != num_estimation_runs:
                 print(f"Warning: Loaded data has {len(all_beta1_hats_scaled)} runs, script expected {num_estimation_runs}.")
        else:
            print(f"Error: Estimates file {estimates_filename} not found. Please run with run_estimations=True.")
            sys.exit(1)

    # ---------------------------------------
    # --- process and plot results ----------
    # ---------------------------------------
    beta1_hats_s_np = np.array(all_beta1_hats_scaled)
    beta2_hats_s_np = np.array(all_beta2_hats_scaled)

    valid_mask = ~np.isnan(beta1_hats_s_np) & ~np.isnan(beta2_hats_s_np)
    beta1_hats_s_valid = beta1_hats_s_np[valid_mask]
    beta2_hats_s_valid = beta2_hats_s_np[valid_mask]
    num_valid_estimates = len(beta1_hats_s_valid)

    if num_valid_estimates == 0:
        print("STOP: no valid EM estimates to plot.")
        sys.exit(1)

    mean_b1_s_hat = np.mean(beta1_hats_s_valid)
    std_b1_s_hat = np.std(beta1_hats_s_valid)
    mean_b2_s_hat = np.mean(beta2_hats_s_valid)
    std_b2_s_hat = np.std(beta2_hats_s_valid)

    bias_b1_s = mean_b1_s_hat - beta1_s_val_true
    bias_b2_s = mean_b2_s_hat - beta2_s_val_true
    mse_b1_s = np.mean((beta1_hats_s_valid - beta1_s_val_true)**2)
    mse_b2_s = np.mean((beta2_hats_s_valid - beta2_s_val_true)**2)

    print("\n--- Key Metrics for SCALED EM Beta Estimates ---")
    print(f"Number of Valid EM Estimation Runs: {num_valid_estimates}/{num_estimation_runs}")
    print(f"True Scaled (beta1N, beta2N^2): ({beta1_s_val_true:.3f}, {beta2_s_val_true:.3f})")

    print(f"Mean Estimated beta1_hat*N: {mean_b1_s_hat:.3f} (StdDev: {std_b1_s_hat:.3f})")
    print(f"Mean Estimated beta2_hat*N^2: {mean_b2_s_hat:.3f} (StdDev: {std_b2_s_hat:.3f})")

    print(f"Bias for beta1_hat*N: {bias_b1_s:.3f}")
    print(f"Bias for beta2_hat*N^2: {bias_b2_s:.3f}")

    print(f"MSE for beta1_hat*N: {mse_b1_s:.4f}")
    print(f"MSE for beta2_hat*N^2: {mse_b2_s:.4f}")

    # ---------------------------------------
    # --- scatter plot of estimates ---------
    # ---------------------------------------
    plt.figure(figsize=(7, 6), dpi=150)

    # TODO: adjust alpha / size / color (dodgerblue or darkorange)
    plt.scatter(beta1_hats_s_valid, beta2_hats_s_valid,
                alpha=0.9, s=25, label=f'EM Estimates ({num_valid_estimates} runs)', color='dodgerblue') # darkorange
    
    plt.scatter([beta1_s_val_true], [beta2_s_val_true],
                marker='X', color='red', s=150, edgecolor='black', linewidth=1,
                alpha=1, label='True Parameters', zorder=5)

    plt.xlabel(r'Estimated Scaled Pairwise Rate ($\widehat{\beta_1 N}$)')
    plt.ylabel(r'Estimated Scaled Higher-Order Rate ($\widehat{\beta_2 N^2}$)')


    # TODO: use the title for a figure caption rather than on the plot itself
    # plt.title(f'Distribution of EM estimates for (beta1 N, beta2 N^2)\n(N = {N}, I0 = {I0}, True beta1 N={beta1_s_val_true}, True beta2 N^2 ={beta2_s_val_true})')    
    plt.axhline(beta2_s_val_true, color='grey', linestyle=':', linewidth=0.8, zorder=1)
    plt.axvline(beta1_s_val_true, color='grey', linestyle=':', linewidth=0.8, zorder=1)

    # TODO: set legend position
    plt.legend(loc='lower left')
    # plt.grid(True, linestyle=':', alpha=0.5)  # TODO: grid Yes / No?

    if zoom_in:
        zoom_factor = 3.5
        x_lim_min_plot = beta1_s_val_true - std_b1_s_hat * zoom_factor
        x_lim_max_plot = beta1_s_val_true + std_b1_s_hat * zoom_factor
        y_lim_min_plot = beta2_s_val_true - std_b2_s_hat * zoom_factor
        y_lim_max_plot = beta2_s_val_true + std_b2_s_hat * zoom_factor
        plt.xlim(max(0, x_lim_min_plot), x_lim_max_plot if x_lim_max_plot > x_lim_min_plot + 0.1 else x_lim_min_plot + 0.1)
        plt.ylim(max(0, y_lim_min_plot), y_lim_max_plot if y_lim_max_plot > y_lim_min_plot + 0.1 else y_lim_min_plot + 0.1)

    # plt.show()
    # plot_filename = f"EM_beta_estimates_scatter_N{N}_b1s{beta1_s_val_true}_b2s{beta2_s_val_true}.pdf"
    plot_filename = "beta_estimates_scatter_EM_balanced_case.pdf"
    plot_filepath = os.path.join(dir_figure, plot_filename)
    plt.savefig(plot_filepath, format='pdf', bbox_inches='tight')
    print(f"\nScatter plot of EM estimates saved to {plot_filepath}")
    plt.close()


'''
# Test 1: balanced case:
--- EM Estimation Test: EM_complete_discrete_unknown_event_types ---
  EM Estimation Run 100/100...
Iteration  1  estimate is:  [2.06953986 1.26912633]
Iteration  2  estimate is:  [2.54340975 4.09862126]
Iteration  3  estimate is:  [2.08843038 5.49081336]
Iteration  4  estimate is:  [2.10196029 5.45329727]
Iteration  5  estimate is:  [2.08909352 5.49246801]
Iteration  6  estimate is:  [2.08909352 5.49246801]

	Run 100 Estimates (scaled): b1N = 2.089, b2N^2 = 5.492

All EM estimates saved to ../results/estimation/demos/EM_beta_estimates_100_I010_runs100_b1s2.4_b2s4.4.pkl

--- Key Metrics for SCALED EM Beta Estimates ---

Number of Valid EM Estimation Runs: 100/100
True Scaled (beta1N, beta2N^2): (2.400, 4.400)

Mean Estimated beta1_hat*N: 2.076 (StdDev: 0.164)
Mean Estimated beta2_hat*N^2: 5.278 (StdDev: 0.605)

Bias for beta1_hat*N: -0.324
Bias for beta2_hat*N^2: 0.878

MSE for beta1_hat*N: 0.1316
MSE for beta2_hat*N^2: 1.1372

# Test 2: hard imbalanced case has too many breaks, dying out Gillespie simulation trajectories:
--- EM Estimation Test: EM_complete_discrete_unknown_event_types ---
N=100, I0=10, time_max_gillespie=10.0
	 True beta1_scaled=1.10, True beta2_scaled=8.00
	 Each EM estimation based on 10 Gillespie runs, discretized into 100 intervals each.
	 Total EM estimation repetitions: 5

Running 5 EM estimations...
  EM Estimation Run 1/5...
break: k = 0
Iteration  1  estimate is:  [1. 1.]
	Run 1 Estimates (scaled): b1N = 1.000, b2N^2 = 1.000

  EM Estimation Run 2/5...
Iteration  1  estimate is:  [1.35988332 2.90678074]
Iteration  2  estimate is:  [1.54708061 6.027405  ]
Iteration  3  estimate is:  [1.27670509 6.93928138]
Iteration  4  estimate is:  [1.28491873 6.91595092]
Iteration  5  estimate is:  [1.27726973 6.94140486]
Iteration  6  estimate is:  [1.27726973 6.94140486]

	Run 2 Estimates (scaled): b1N = 1.277, b2N^2 = 6.941

  EM Estimation Run 3/5...
break: k = 0
break: k = 0
break: k = 0
Iteration  1  estimate is:  [1. 1.]


'''