# ./scripts/contour_plot.py
# saves contour plot to `../figures/estimation/demos/k_star_contour.pdf`

import numpy as np
import matplotlib.pylab as plt
import pickle
import os
# import sys # Not needed for plotting part

# for part to compute k_star_matrix_gillespie (commented out):
# from scipy.integrate import solve_ivp
# sys.path.append('../src/')
# sys.path.append('../scripts/')
# from solve_kolmogorov import *
# from simulate_gillespie import *
# from estimate_total_rates import *
# from higher_order_structures import Complete
# import time # Needed for commented out part

if __name__ == "__main__":
    # --- setup from original script ---
    test_name = "demos"
    N = 1000
    I0 = 50
    nsims = 20  # As per original title: "20 sims/pt"
    time_max = 20.0 # As per original title: "t_max=20.0"
    mu = 1.0    # As per original title: "mu=1.0"

    # --- Plot settings from MF_limit_ODE_decomposition.py ---
    plt_DPI = 200
    fig_w, fig_h = 8, 6.5  # Adjusted height slightly from 5 to 6.5 to better suit contour plot proportions
    plt_legend_fontsize = 16
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14
    # Linewidths for contour lines can be set directly in plt.contour if needed

    # --- output dirs ---
    # Adjusted path to be relative to a potential project root if this script is in './scripts/'
    # and figures are in './figures/'
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Assumes script is in a 'scripts' subdirectory
    
    # Output directory for figures, ensuring it matches the structure implied by the input slide
    # The input slide uses `figures/combined/k_star_contour.pdf`
    # The original script saved to `../figures/estimation/{test_name}/`
    # For consistency with the prompt, let's target a path like `figures/combined/`
    # or ensure `output_dir_figs` is flexible enough.
    # Given the prompt shows `figures/combined/k_star_contour.pdf` for the *unnamed* slide,
    # I will adapt to a similar structure.
    output_figure_dir = os.path.join(project_root, "figures", "combined")
    os.makedirs(output_figure_dir, exist_ok=True)

    # Data directory (assuming it's relative to project root as well, e.g., in a 'results' or 'data' folder)
    # Original script used: `../results/estimation/{test_name}/`
    data_dir_results = os.path.join(project_root, "results", "estimation", test_name)
    os.makedirs(data_dir_results, exist_ok=True) # Ensure data directory exists if running data generation

    # --- denser grid for scaled beta parameters ---
    beta1_s_min, beta1_s_max, beta1_s_steps = 0.5, 6.0, 36
    beta2_s_min, beta2_s_max, beta2_s_steps = 0.0, 12.0, 45

    beta1_scaled_vec = np.linspace(beta1_s_min, beta1_s_max, int(beta1_s_steps))
    beta2_scaled_vec = np.linspace(beta2_s_min, beta2_s_max, int(beta2_s_steps))

    # ------------------------
    # --- Contour plot -------
    # ------------------------
    # print(f"Plotting contours for: ") # Removed for cleaner output if used in automated pipeline
    # print(f"\tN = {N}, I0 = {I0}, time_max = {time_max}, nsims_for_avg = {nsims}, mu = {mu}\n")

    # --- load the k_star_matrix_gillespie ---
    # Ensure the path to the pickle file is correct relative to the script's execution location.
    # If `k_star_matrix_gillespie_...pkl` is in `data_dir_results`
    k_star_matrix_gillespie_path = f"k_star_matrix_gillespie_N{N}_I0{I0}_t{time_max}_nsims{nsims}.pkl"
    pickle_filename_gillespie = os.path.join(data_dir_results, k_star_matrix_gillespie_path)
    

    print(f"Loading k_star_matrix_gillespie from {pickle_filename_gillespie}...")
    with open(pickle_filename_gillespie, 'rb') as f:
        k_star_matrix_gillespie = pickle.load(f)
    print("Loading complete.")


    # --- hand-picked (beta1_scaled, beta2_scaled) pairs for the k*~750 regime ---
    points_for_750_regime = [
        (1.1, 8.0),   # low beta1_s, high beta2_s
        (2.4, 4.4),   # mid beta1_s, mid beta2_s
        (3.6, 1.0)    # high beta1_s, low beta2_s
    ]
    points_for_750_regime_np = np.array(points_for_750_regime)

    # --- Contour Plot ---
    B1_s, B2_s = np.meshgrid(beta1_scaled_vec, beta2_scaled_vec)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)

    # define specific contour levels
    contour_level_target = 0.75 * N
    contour_levels_specific = [0.25 * N, 0.5 * N, contour_level_target]

    contour_colors_specific = ['black', 'black', 'black'] # Kept black as per original

    k_min_plot = np.nanmin(k_star_matrix_gillespie)
    k_max_plot = np.nanmax(k_star_matrix_gillespie)
    # Ensure contourf_levels are sensible even if k_min_plot or k_max_plot are NaN (e.g. if all data is NaN)
    if np.isnan(k_min_plot) or np.isnan(k_max_plot):
        k_min_plot = 0
        k_max_plot = N
        print("Warning: k_star_matrix_gillespie contains NaNs, using default plot range.")

    contourf_levels = np.linspace(max(0, k_min_plot), min(N, k_max_plot), 21)


    contourf_plot = ax.contourf(B1_s, B2_s, k_star_matrix_gillespie.T,
                                levels=contourf_levels, cmap='viridis', alpha=0.75, extend='both')
    
    # Colorbar label (original was fine, matches scientific context)
    # The caption will contain: N=1000, I0=50, mu=1.0, t_max=20.0, 20 sims/pt
    # So the label can be more concise about what k* is.
    cbar = plt.colorbar(contourf_plot, ax=ax, label=r'Quasi-Steady State $k^*$')
    cbar.ax.tick_params(labelsize=plt_tick_fontsize) # Apply tick font size to colorbar
    cbar.set_label(r'Quasi-Steady State $k^*$', size=plt_labels_fontsize) # Apply label font size
    cbar.set_ticks(np.linspace(max(0, k_min_plot), min(N, k_max_plot), 6))

    contour_lines = ax.contour(B1_s, B2_s, k_star_matrix_gillespie.T,
                                levels=contour_levels_specific,
                                colors=contour_colors_specific,
                                linewidths=2.0, linestyles='solid', zorder=3)
    ax.clabel(contour_lines, inline=True, fontsize=plt_tick_fontsize - 2, fmt='%1.0f', colors='black') # clabel fontsize adjusted

    if points_for_750_regime_np.size > 0:
        ax.scatter(points_for_750_regime_np[:, 0],
                    points_for_750_regime_np[:, 1],
                    marker='X',
                    s=150, # Size of the marker
                    color='red',
                    edgecolors='black',
                    linewidth=0.75,
                    label=f'$k^* \\approx {contour_level_target:.0f}$ Test Points', # Simplified legend entry
                    zorder=5)

    ax.set_xlabel(r'Scaled Pairwise Rate ($\beta_1 N$)', fontsize=plt_labels_fontsize)
    ax.set_ylabel(r'Scaled Higher-Order Rate ($\beta_2 N^2$)', fontsize=plt_labels_fontsize)
    
    # Title removed as requested
    # Original title info for caption:
    # Quasi-Steady State k* (Gillespie Averages)
    # N=1000, I0=50, mu=1.0, t_max=20.0, 20 sims/pt

    ax.grid(True, linestyle=':', alpha=0.4)

    ax.set_xlim(beta1_s_min, 4.5)
    ax.set_ylim(beta2_s_min, 8.5)

    # Apply spine settings from MF_limit_ODE_decomposition.py
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Apply legend settings from MF_limit_ODE_decomposition.py
    if points_for_750_regime_np.size > 0 : # Only show legend if there are points
        ax.legend(fontsize=plt_legend_fontsize, loc='upper right',
                frameon=True, fancybox=True, shadow=False,
                framealpha=0.9, edgecolor='gray')


    ax.tick_params(axis='both', which='major', labelsize=plt_tick_fontsize)

    plt.tight_layout()

    # Use the output directory and filename specified for the slide
    plot_filename = "k_star_contour.pdf"
    plot_path = os.path.join(output_figure_dir, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Contour plot with test points saved to {plot_path}")
    # plt.show()
    plt.close()

'''
Part to compute_k_star_matrix_gillespie (original, needs time import and other dependencies if used):

    # --- This part needs to be uncommented and dependencies resolved if you need to generate the .pkl data ---
    # print(f"Running Gillespie simulations for {len(beta1_scaled_vec) * len(beta2_scaled_vec)} parameter pairs ...")
    # k_star_matrix_gillespie = np.zeros((len(beta1_scaled_vec), len(beta2_scaled_vec))) # Initialize
    # total_param_pairs = len(beta1_scaled_vec) * len(beta2_scaled_vec)
    # current_pair_count = 0
    # start_time_total_gillespie = time.time() # Make sure to import time
    # for i, beta1_s_val in enumerate(beta1_scaled_vec):
    #     for j, beta2_s_val in enumerate(beta2_scaled_vec):
    #         current_pair_count += 1
    #         print(f"Processing param pair {current_pair_count}/{total_param_pairs}: beta1_s={beta1_s_val:.2f}, beta2_s={beta2_s_val:.2f}")
            
    #         beta1_orig = beta1_s_val / N
    #         beta2_orig = beta2_s_val / (N**2)

    #         X_sims_current_pair = []
    #         # Ensure gillespie_sim_complete and get_average are defined/imported
    #         # from simulate_gillespie import gillespie_sim_complete 
    #         # from estimate_total_rates import get_average
    #         for _ in range(nsims):
    #             X_t = gillespie_sim_complete(N, beta1_orig, beta2_orig, mu, I0, time_max) 
    #             X_sims_current_pair.append(X_t)

    #         avg_delta_t = 0.1
    #         avg_curve, avg_times = get_average(
    #             X_sims_current_pair, time_max, nsims, delta_t=avg_delta_t, selected=2
    #         )

    #         if len(avg_curve) > 0:
    #             num_avg_points = min(max(1, int(0.1 * len(avg_curve))), 5)
    #             k_star_estimate = np.mean(avg_curve[-num_avg_points:])
    #             k_star_matrix_gillespie[i, j] = k_star_estimate
    #         else:
    #             k_star_matrix_gillespie[i, j] = np.nan # Handle cases with no curve data
        
    #     print(f"Finished row {i+1}/{len(beta1_scaled_vec)} for beta1_scaled. Current total time: {time.time() - start_time_total_gillespie:.2f}s")
    # end_time_total_gillespie = time.time()
    # print(f"Total Gillespie simulation and k* estimation time: {end_time_total_gillespie - start_time_total_gillespie:.2f}s")
    
    # # --- save the k_star_matrix_gillespie ---
    # pickle_filename_gillespie_output = os.path.join(data_dir_results, f"k_star_matrix_gillespie_N{N}_I0{I0}_t{time_max}_nsims{nsims}.pkl")
    # with open(pickle_filename_gillespie_output, 'wb') as f:
    #     pickle.dump(k_star_matrix_gillespie, f)
    # print(f"k_star_matrix_gillespie saved to {pickle_filename_gillespie_output}")
'''