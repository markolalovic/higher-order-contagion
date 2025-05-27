# ./scripts/contour_plot.py
# saves contour plot to `../figures/estimation/demos/k_star_contour.pdf`

import numpy as np
import matplotlib.pylab as plt
import pickle
import os
import sys

# for part to compute k_star_matrix_gillespie:
# from scipy.integrate import solve_ivp
# sys.path.append('../src/')
# sys.path.append('../scripts/')
# from solve_kolmogorov import *
# from simulate_gillespie import *
# from estimate_total_rates import *
# from higher_order_structures import Complete

if __name__ == "__main__":
    # --- setup ---
    test_name = "demos"
    N = 1000
    I0 = 50
    nsims = 20
    time_max = 20.0
    mu = 1.0

    # --- output dirs ---
    output_dir_figs = f"../figures/estimation/{test_name}/"
    os.makedirs(output_dir_figs, exist_ok=True)
    data_dir_results = f"../results/estimation/{test_name}/"

    # --- denser grid for scaled beta parameters ---
    beta1_s_min, beta1_s_max, beta1_s_steps = 0.5, 6.0, 36
    beta2_s_min, beta2_s_max, beta2_s_steps = 0.0, 12.0, 45

    beta1_scaled_vec = np.linspace(beta1_s_min, beta1_s_max, int(beta1_s_steps))
    beta2_scaled_vec = np.linspace(beta2_s_min, beta2_s_max, int(beta2_s_steps))

    # ------------------------
    # --- Contour plot -------
    # ------------------------
    print(f"Plotting contours for: ")
    print(f"\tN = {N}, I0 = {I0}, time_max = {time_max}, nsims_for_avg = {nsims}, mu = {mu}\n")

    # --- load the k_star_matrix_gillespie ---
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

    plt.figure(figsize=(8, 6.5), dpi=150)

    # define specific contour levels
    contour_level_target = 0.75 * N
    contour_levels_specific = [0.25 * N, 0.5 * N, contour_level_target]

    # TODO: choose colors for specific contour levels
    contour_colors_specific = ['black', 'black', 'black']

    k_min_plot = np.nanmin(k_star_matrix_gillespie)
    k_max_plot = np.nanmax(k_star_matrix_gillespie)
    contourf_levels = np.linspace(max(0, k_min_plot), min(N, k_max_plot), 21)

    # plot filled contours, clipped by axis limits later
    contourf_plot = plt.contourf(B1_s, B2_s, k_star_matrix_gillespie.T,
                                levels=contourf_levels, cmap='viridis', alpha=0.75, extend='both')
    cbar = plt.colorbar(contourf_plot, label=f'$k^* = E[X(t_{{{time_max:.0f}}})]$') # k^*_{{GillespieAvg}}
    cbar.set_ticks(np.linspace(max(0, k_min_plot), min(N, k_max_plot), 6))

    # plot specific contour lines (these will also be clipped by axis limits)
    contour_lines = plt.contour(B1_s, B2_s, k_star_matrix_gillespie.T,
                                levels=contour_levels_specific,
                                colors=contour_colors_specific,
                                linewidths=2.0, linestyles='solid', zorder=3) # zorder to ensuer lines are on top of contourf
    plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%1.0f', colors='black')

    # --- add points for the k*~750 regime ---
    if points_for_750_regime_np.size > 0:
        plt.scatter(points_for_750_regime_np[:, 0],
                    points_for_750_regime_np[:, 1],
                    marker='X', # 'X', 'P' for filled plus, or '*' for star
                    s=150,
                    color='red',
                    edgecolors='black',
                    linewidth=0.75,
                    label=f'Test Points ($k^* \\approx {contour_level_target:.0f}$)',
                    zorder=5)

    plt.xlabel(r'Scaled Pairwise Rate ($\beta_1 N$)')
    plt.ylabel(r'Scaled Higher-Order Rate ($\beta_2 N^2$)')
    plt_title = f'Quasi-Steady State $k^*$ (Gillespie Averages)\n'
    plt_title += f'$N={N}, I_0={I0}, mu={mu}, t_{{max}}={time_max}, {nsims}$ sims/pt'
    plt.title(plt_title)
    plt.grid(True, linestyle=':', alpha=0.4)

    # --- set the axis limits to restrict the view ---
    plt.xlim(beta1_s_min, 4.5)  # restricted beta1_scaled up to 4.5
    plt.ylim(beta2_s_min, 8.5)  # restricted beta2_scaled up to 8.5

    plt.legend(loc='upper right') # bbox_to_anchor=(0.01, 0.99)

    plot_filename = "k_star_contour.pdf"
    plot_path = os.path.join(output_dir_figs, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Contour plot with test points saved to {plot_path}")
    # plt.show()
    plt.close()

'''
Part to compute_k_star_matrix_gillespie:

    print(f"Running Gillespie simulations for {len(beta1_scaled_vec) * len(beta2_scaled_vec)} parameter pairs ...")
    total_param_pairs = len(beta1_scaled_vec) * len(beta2_scaled_vec)
    current_pair_count = 0
    start_time_total_gillespie = time.time()
    for i, beta1_s_val in enumerate(beta1_scaled_vec):
        for j, beta2_s_val in enumerate(beta2_scaled_vec):
            current_pair_count += 1
            print(f"Processing param pair {current_pair_count}/{total_param_pairs}: beta1_s={beta1_s_val:.2f}, beta2_s={beta2_s_val:.2f}")
            
            # scale back to original beta1, beta2 
            beta1_orig = beta1_s_val / N
            beta2_orig = beta2_s_val / (N**2)

            X_sims_current_pair = []
            for _ in range(nsims):
                X_t = gillespie_sim_complete(N, beta1_orig, beta2_orig, mu, I0, time_max)
                X_sims_current_pair.append(X_t)

            # calculate the average curve for this (beta1, beta2) pair
            avg_delta_t = 0.1
            avg_curve, avg_times = get_average(
                X_sims_current_pair, time_max, nsims, delta_t=avg_delta_t, selected=2
            )

            # estimate quasi-steady state k* from the end of the average curve
            # more robust is to average the last few points of avg_curve
            if len(avg_curve) > 0:
                # take average of last 10% of points or last 5 points, whichever is smaller
                num_avg_points = min(max(1, int(0.1 * len(avg_curve))), 5)
                k_star_estimate = np.mean(avg_curve[-num_avg_points:])
                k_star_matrix_gillespie[i, j] = k_star_estimate
        
        print(f"Finished row {i+1}/{len(beta1_scaled_vec)} for beta1_scaled. Current total time: {time.time() - start_time_total_gillespie:.2f}s")
    end_time_total_gillespie = time.time()
'''