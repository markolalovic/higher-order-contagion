# ./scripts/gillespie_sims_complete.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append('../src/')
from simulate_gillespie import gillespie_sim_complete, get_average

np.random.seed(123) # seed for reproducibility

if __name__ == "__main__":
    # --- Plot settings (consistent with other slides) ---
    plt_DPI = 200
    fig_w, fig_h = 8, 5  # 8:5 ratio goes well with beamer
    plt_legend_fontsize = 16
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14  # for tick labels
    alpha_run = 0.9
    
    # --- config -----
    balanced_case = False
    test_name = "demos"
    N = 1000
    I0 = 50
    time_max = 20.0
    mu_true = 1.0
    nsims = 20  # number of realizations
    
    # parameters for data generation
    if balanced_case:
        # Case 1: "Balanced"
        beta1_s_val_true, beta2_s_val_true = (2.4, 4.4)
    else:
        # Case 2: "Hard Case"
        # hand-picked pair = hard case: low beta1 (PW), high beta2 (HO) rate
        beta1_s_val_true, beta2_s_val_true = (1.1, 8.0)
    
    # --- directory setup ---
    figure_path_name = "../figures/combined/gillespie_sims_complete.pdf"
    
    # --- convert SCALED true betas to ORIGINAL per-interaction true betas! ---
    beta1_orig_true = beta1_s_val_true / N
    beta2_orig_true = beta2_s_val_true / (N**2)
    
    ## --- Run Gillespie ---
    X_sims = []
    for _ in range(nsims):
        X_t = gillespie_sim_complete(N, beta1_orig_true, beta2_orig_true, mu_true, I0, time_max)
        X_sims.append(X_t)
    
    ## calculate average Gillespie trajectory
    avg_curve, times = get_average(X_sims, time_max, nsims, delta_t=0.01)
    
    ## plot the results
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)
    
    # Plot individual runs
    for i, X_t in enumerate(X_sims):
        if i == 0:  # Add label only for first run
            ax.plot(X_t[0], X_t[2], c='blue', alpha=alpha_run, linewidth=0.8, 
                   label='Gillespie run', rasterized=True)
        else:
            ax.plot(X_t[0], X_t[2], c='blue', alpha=alpha_run, linewidth=0.8, rasterized=True)
    
    # Plot average
    ax.plot(times, avg_curve, 'red', linewidth=3, label='Avg. Gillespie (20 runs)')
    
    # Target line
    ax.axhline(y=int(0.75 * N), color='black', lw=2, alpha=0.5, 
               linestyle='--', label=f'Target $k^* = {int(0.75 * N)}$')
    
    # Labels and formatting
    ax.set_xlabel('Time (t)', fontsize=plt_labels_fontsize)
    ax.set_ylabel('Number of Infected (k)', fontsize=plt_labels_fontsize)
    
    # Grid
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Axis limits
    ax.set_xlim(0, time_max)
    ax.set_ylim(0, N)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=plt_tick_fontsize)
    
    # Presentation-friendly plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Legend
    ax.legend(fontsize=plt_legend_fontsize, loc='lower right',
              frameon=True, fancybox=True, shadow=False,
              framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(figure_path_name), exist_ok=True)
    plt.savefig(figure_path_name, format='pdf', bbox_inches='tight')
    print(f"Figure saved to: {figure_path_name}")
    plt.close(fig)