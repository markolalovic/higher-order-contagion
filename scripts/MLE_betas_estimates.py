# ./scripts/MLE_betas_estimates.py
# Computes/loads MLE beta estimates, calculates metrics,
# and saves a scatter plot to either:
# ./figures/combined/beta_estimates_scatter_MLE_balanced_case.pdf
#  or:
# ./figures/combined/beta_estimates_scatter_MLE_hard_case.pdf

import numpy as np
import matplotlib.pyplot as plt
import pickle # using np.savez/load
import os
import sys
sys.path.append('../src/')

from simulate_gillespie import gillespie_sim_complete
from estimate_total_rates import calculate_mle_beta1_beta2_complete
# from higher_order_structures import Complete

np.random.seed(123) # seed for reproducibility

if __name__ == "__main__":
    r"""
    This script tests the MLE framework for beta1 and beta2 on complete SCs.
    
    It runs multiple independent estimations (each from a single Gillespie run)
    to assess the distribution and stability of the estimates.

    It saves the raw estimates and produces a scatter plot and key metrics.
    """
    # --- Plot settings ---
    plt_DPI = 200
    fig_w, fig_h = 7, 6  # 8:5 ratio goes well with beamer
    plt_legend_fontsize = 16
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14  # for tick labels

    scatter_size = 25
    scatter_alpha = 0.8
    marker_size = 250
    zoom_in_spread = 4

    # --- config -----
    balanced_case = False
    run_estimations = True    # set to False for plot modifications

    zoom_in = True
    num_estimation_runs = 1000 # number of independent Gillespie runs for estimation

    test_name = "demos"
    N = 1000
    I0 = 50
    time_max = 20.0
    mu_true = 1.0

    # true parameters for data generation 
    if balanced_case:
        # Case 1: "Balanced"
        beta1_s_val_true, beta2_s_val_true = (2.4, 4.4)
        file_name_estimates = f"mle_beta_estimates_N{N}_I0{I0}_runs{num_estimation_runs}_balanced.npz"
    else: 
        # Case 2: "Hard Case"
        # hand-picked pair = hard case: low beta1 (PW), high beta2 (HO) rate
        beta1_s_val_true, beta2_s_val_true = (1.1, 8.0)
        file_name_estimates = f"mle_beta_estimates_N{N}_I0{I0}_runs{num_estimation_runs}_hard.npz"

    # --- directory setup ---
    figure_dir = "../figures/combined/"

    output_base_dir_data = "../results/estimation/"
    data_dir = os.path.join(output_base_dir_data, test_name)
    
    estimates_filename = os.path.join(data_dir, file_name_estimates)

    # --- convert SCALED true betas to ORIGINAL per-interaction true betas! ---
    beta1_orig_true = beta1_s_val_true / N
    beta2_orig_true = beta2_s_val_true / (N**2)

    if run_estimations:
        print(f"Targeting True Parameters (Hard Case):")
        print(f"\tbeta1_orig_true = {beta1_orig_true:.6f} (scaled beta1N = {beta1_s_val_true})")
        print(f"\tbeta2_orig_true = {beta2_orig_true:.10f} (scaled beta2N^2 = {beta2_s_val_true})")

        print(f"Running {num_estimation_runs} independent estimations from single Gillespie runs...")

        all_beta1_hats_orig = []
        all_beta2_hats_orig = []

        for run_idx in range(num_estimation_runs):
            if (run_idx + 1) % (num_estimation_runs // 10 or 1) == 0:
                print(f"Processing estimation run {run_idx + 1}/{num_estimation_runs}...")

            X_t_single_run = gillespie_sim_complete(N, beta1_orig_true, beta2_orig_true, mu_true, I0, time_max)
            X_sims_for_this_estimation = [X_t_single_run]

            b1_hat, b2_hat, _ = calculate_mle_beta1_beta2_complete(X_sims_for_this_estimation, N)
            all_beta1_hats_orig.append(b1_hat)
            all_beta2_hats_orig.append(b2_hat)

        print(f"... {len(all_beta1_hats_orig)} estimations complete.")

        # save the computed estimates
        np.savez_compressed(estimates_filename,
                            beta1_hats_orig=np.array(all_beta1_hats_orig),
                            beta2_hats_orig=np.array(all_beta2_hats_orig))
        print(f"Estimates saved to {estimates_filename}")

        beta1_hats_np_orig = np.array(all_beta1_hats_orig)
        beta2_hats_np_orig = np.array(all_beta2_hats_orig)
    else:
        print(f"Loading pre-computed estimates from {estimates_filename}...")
        data = np.load(estimates_filename)
        beta1_hats_np_orig = data['beta1_hats_orig']
        beta2_hats_np_orig = data['beta2_hats_orig']
        print(f"... {len(beta1_hats_np_orig)} estimates loaded.")
        if len(beta1_hats_np_orig) != num_estimation_runs:
            print(f"STOP: Loaded data contains {len(beta1_hats_np_orig)} runs, but script expected {num_estimation_runs}.")

    # --- calculate key metrics to report for SCALED parameters ---
    beta1_hats_np_scaled = beta1_hats_np_orig * N
    beta2_hats_np_scaled = beta2_hats_np_orig * (N**2)

    mean_beta1_hat_s = np.mean(beta1_hats_np_scaled)
    mean_beta2_hat_s = np.mean(beta2_hats_np_scaled)

    # using population std dev as if all runs are the population
    # TODO: could use ddof=1 for sample std dev since the runs are a sample
    std_beta1_hat_s = np.std(beta1_hats_np_scaled)
    std_beta2_hat_s = np.std(beta2_hats_np_scaled)

    bias_beta1_s = mean_beta1_hat_s - beta1_s_val_true
    bias_beta2_s = mean_beta2_hat_s - beta2_s_val_true

    mse_beta1_s = np.mean((beta1_hats_np_scaled - beta1_s_val_true)**2)
    mse_beta2_s = np.mean((beta2_hats_np_scaled - beta2_s_val_true)**2)

    print("\n--- Key Metrics for SCALED MLE Beta Estimates ---")
    print(f"Number of Estimation Runs: {len(beta1_hats_np_scaled)}")
    print(f"True Scaled (beta1*N, beta2*N^2): ({beta1_s_val_true:.3f}, {beta2_s_val_true:.3f})")

    print(f"Mean Estimated beta1_hat*N: {mean_beta1_hat_s:.3f} (StdDev: {std_beta1_hat_s:.3f})")
    print(f"Mean Estimated beta2_hat*N^2: {mean_beta2_hat_s:.3f} (StdDev: {std_beta2_hat_s:.3f})")

    print(f"Bias for beta1_hat*N: {bias_beta1_s:.3f}")
    print(f"Bias for beta2_hat*N^2: {bias_beta2_s:.3f}")

    print(f"MSE for beta1_hat*N: {mse_beta1_s:.4f}")
    print(f"MSE for beta2_hat*N^2: {mse_beta2_s:.4f}")

    # ----------------------------------
    # --- Scatter Plot of Estimates ----
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)
    
    # Scatter plot of estimates    
    ax.scatter(beta1_hats_np_scaled, beta2_hats_np_scaled,
               alpha=scatter_alpha, s=scatter_size, 
               label=f'MLE estimates ({len(beta1_hats_np_scaled)} runs)', 
               color='dodgerblue')
    
    # True parameters
    label_true = f'True parameters ({beta1_s_val_true:.1f}, {beta2_s_val_true:.1f})'
    ax.scatter([beta1_s_val_true], [beta2_s_val_true],
               marker='X', color='red', s=marker_size, edgecolor='black', linewidth=1.5,
               alpha=1, label=label_true, zorder=10)

    # Labels
    ax.set_xlabel(r'Estimated Pairwise Rate ($\widehat{\beta_1 N}$)', fontsize=plt_labels_fontsize)
    ax.set_ylabel(r'Estimated Higher-Order Rate ($\widehat{\beta_2 N^2}$)', fontsize=plt_labels_fontsize)

    # TODO: use the title for a figure caption rather than on the plot itself
    # plt.title(f'Distribution of MLEs for (beta1 N, beta2 N^2)
    # \n(N = {N}, I0 = {I0}, 
    # True beta1 N={beta1_s_val_true}, 
    # True beta2 N^2 ={beta2_s_val_true})')

    # Reference lines
    ax.axhline(beta2_s_val_true, color='grey', linestyle=':', linewidth=1, alpha=0.8, zorder=1)
    ax.axvline(beta1_s_val_true, color='grey', linestyle=':', linewidth=1, alpha=0.8, zorder=1)

    # Grid
    # ax.grid(True, linestyle=':', alpha=0.4)

    # Legend
    ax.legend(fontsize=plt_legend_fontsize, loc='lower left',
              frameon=True, fancybox=True, shadow=False,
              framealpha=0.9, edgecolor='gray')

    if zoom_in:
        # set axis limits to zoom, to not show the full spread
        x_lim_min = beta1_s_val_true - std_beta1_hat_s * zoom_in_spread
        x_lim_max = beta1_s_val_true + std_beta1_hat_s * zoom_in_spread
        y_lim_min = beta2_s_val_true - std_beta2_hat_s * zoom_in_spread
        y_lim_max = beta2_s_val_true + std_beta2_hat_s * zoom_in_spread
        ax.set_xlim(max(0, x_lim_min), x_lim_max)
        ax.set_ylim(max(0, y_lim_min), y_lim_max)

    # Tick parameters
    # ax.tick_params(axis='both', which='major', linestyle='dashed', labelsize=plt_tick_fontsize)
    # TODO: draw one vertical, one horizontal dashed line
    ax.axvline(x=beta1_s_val_true, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=beta2_s_val_true, color='gray', linestyle='--', linewidth=1)

    # Presentation-friendly plot
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    # plt.show()
    if balanced_case:
        plot_filename = os.path.join(figure_dir, f"beta_estimates_scatter_MLE_balanced_case.pdf")
    else:
        plot_filename = os.path.join(figure_dir, f"beta_estimates_scatter_MLE_hard_case.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    print(f"\nScatter plot of estimates saved to {plot_filename}")
    plt.close()

'''
# Results for `num_estimation_runs = 1000`:

True Scaled (beta1*N, beta2*N^2): (1.100, 8.000)
Mean Estimated beta1_hat*N: 1.099 (StdDev: 0.019)
Mean Estimated beta2_hat*N^2: 7.997 (StdDev: 0.113)
Bias for beta1_hat*N: -0.001
Bias for beta2_hat*N^2: -0.003
MSE for beta1_hat*N: 0.0004
MSE for beta2_hat*N^2: 0.0129

Number of Estimation Runs: 1000
True Scaled (beta1*N, beta2*N^2): (2.400, 4.400)
Mean Estimated beta1_hat*N: 2.401 (StdDev: 0.025)
Mean Estimated beta2_hat*N^2: 4.404 (StdDev: 0.059)
Bias for beta1_hat*N: 0.001
Bias for beta2_hat*N^2: 0.004
MSE for beta1_hat*N: 0.0006
MSE for beta2_hat*N^2: 0.0035


# Test 1: balanced case:
--- Key Metrics for SCALED MLE Beta Estimates ---
  Number of Estimation Runs: 1000
  True Scaled (beta1*N, beta2*N^2): (2.400, 4.400)

  Mean Estimated beta1_hat*N: 2.400 (StdDev: 0.025)
  Mean Estimated beta2_hat*N^2: 4.403 (StdDev: 0.057)

  Bias for beta1_hat*N: -0.000
  Bias for beta2_hat*N^2: 0.003

  MSE for beta1_hat*N: 0.0006
  MSE for beta2_hat*N^2: 0.0032


# Key Metrics for SCALED Beta Estimates:

  Number of Estimation Runs: 1000

  True Scaled (beta1*N, beta2*N^2): (1.100, 8.000)

  Mean Estimated beta1_hat*N: 1.099 (StdDev: 0.019)
  Mean Estimated beta2_hat*N^2: 8.000 (StdDev: 0.089)

  Bias for beta1_hat*N: -0.001
  Bias for beta2_hat*N^2: -0.000

  MSE for beta1_hat*N: 0.0004
  MSE for beta2_hat*N^2: 0.0080

----

This is first strong result to present, to show that:
    IF we have detailed continuous data,
    THEN our framework can precisely recover:
    both pairwise and higher-order infection rates

This is a baseline before moving to more complex scenarios like:
    1. When only discrete observations (EM algorithm)
    2. When PW and HO events cannot be distinguished (EM algorithm)
    3. Non-complete structures 
    * where structural counts $S_K^{(1)}$ and $S_K^{(2)}$ terms become approximations

This script, for a number of independent Gillespie runs for estimation, where each run produces one
    * (beta1_hat, beta2_hat) pair
    * produces scatter plot of estimates
    * to show the joint distribution of and how they cluster around true value
    * and any correlation between the MLE estimates beta1_hat, beta2_hat
Also, it calculates key metrics to report:
    * mean beta1_hat across all runs vs true beta1
    * mean beta2_hat across all runs vs true beta2
    * StdDev of beta1_hat and beta2_hat
    * Bias: mean - true 
    * and MSE: mean squared error  
'''