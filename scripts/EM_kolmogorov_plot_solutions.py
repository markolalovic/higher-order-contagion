# ./scripts/EM_kolmogorov_plot_solutions.py
# Loads pre-computed KE solutions (true, EM sparse) and raw Gillespie average.
# Generates a single plot comparing these trajectories:
# ../figures/combined/kolmogorov_EM_fits_comparison.pdf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # --- configuration ---
    test_name = "em_data_quantity"

    # data richness parameter for EM, how many nsims_pooled
    nsims_pooled = 500 # NOTE: increased from 10 to 500

    # for average Gillespie curve plot
    # NOTE: ground truth, increased to a 1000
    nsims_for_avg_display = 1000

    N = 200
    I0 = 20
    time_max = 10.0
    beta1_s_val_true, beta2_s_val_true = (2.4, 4.4)
    mu_true = 1.0

    # --- range for denser EM markers during sharp rise ---
    sharp_rising_end = 4  # up to which markers are denser
    num_markers_sharp = 15  # number of markers in the sharp rising phase
    num_markers_later = 13  # number of markers in the later phase
    min_marker_time_gap = 0.05  # minimum time gap between consecutive markers
    marker_size = 80
    marker_alpha = 0.8

    # --- directory setup ---
    input_csv_filepath = "../data/demos/kolmogorov_EM_fits.csv"
    gillespie_avg_csv_filename = "../data/demos/kolmogorov_EM_gillespie_avg.csv"
    output_plot_filename = "../figures/combined/kolmogorov_EM_fits_comparison.pdf"

    print(f"--- Starting Plot Generation from Pre-solved EM / KE Data ---")
    print(f"For N={N}, I0={I0}, True Scaled Betas: ({beta1_s_val_true}, {beta2_s_val_true})")

    # --- load KE solutions data ---
    df_plot_data = pd.read_csv(input_csv_filepath)
    times_ke = df_plot_data['time'].to_numpy() # it is `time`  and not `time_ke`!
    k_expected_true = df_plot_data['k_expected_true'].to_numpy()
    k_expected_em = df_plot_data['k_expected_em'].to_numpy()
    print(f"Loaded main plot data from: {os.path.basename(input_csv_filepath)}")

    # --- load raw Gillespie average data ---
    df_g_avg_raw = pd.read_csv(gillespie_avg_csv_filename)
    times_gillespie_avg_raw = df_g_avg_raw['time_gillespie_avg'].to_numpy()
    k_gillespie_avg_raw = df_g_avg_raw['k_avg_gillespie_raw'].to_numpy()
    print(f"Loaded raw Gillespie average data from: {os.path.basename(gillespie_avg_csv_filename)}")

    # --- prepare data for EM scatter markers with varying density ---
    t_markers_em_list = []
    k_markers_em_list = []

    # 1. dense markers in sharp rising phase

    dense_mask_sharp = (times_ke >= 0.0) & (times_ke <= sharp_rising_end)
    times_sharp_phase = times_ke[dense_mask_sharp]
    k_sharp_phase = k_expected_em[dense_mask_sharp]

    if len(times_sharp_phase) > num_markers_sharp:
        marker_indices_sharp = np.linspace(0, len(times_sharp_phase) - 1, num_markers_sharp, dtype=int)
        if marker_indices_sharp[-1] == len(times_sharp_phase) - 1 and len(times_sharp_phase) > num_markers_sharp :
                marker_indices_sharp = marker_indices_sharp[:-1]
        t_markers_em_list.append(times_sharp_phase[marker_indices_sharp])
        k_markers_em_list.append(k_sharp_phase[marker_indices_sharp])
    elif len(times_sharp_phase) > 0:
        t_markers_em_list.append(times_sharp_phase) # take all if fewer than desired
        k_markers_em_list.append(k_sharp_phase)

    # 2. Sparser markers in later phase for EM Sparser
    dense_mask_later = times_ke > sharp_rising_end
    times_later_phase = times_ke[dense_mask_later]
    k_later_phase = k_expected_em[dense_mask_later]

    if len(times_later_phase) > num_markers_later:
        marker_indices_later = np.linspace(0, len(times_later_phase) - 1, num_markers_later, dtype=int)
        if t_markers_em_list and t_markers_em_list[0].size > 0 and marker_indices_later.size > 0:
            last_sharp_time = t_markers_em_list[0][-1]
            first_later_time_candidate = times_later_phase[marker_indices_later[0]]
            if first_later_time_candidate - last_sharp_time < min_marker_time_gap:
                marker_indices_later = marker_indices_later[1:] # skip first later marker if too close
        if marker_indices_later.size > 0:
            t_markers_em_list.append(times_later_phase[marker_indices_later])
            k_markers_em_list.append(k_later_phase[marker_indices_later])
    elif len(times_later_phase) > 0:
        if t_markers_em_list and t_markers_em_list[0].size > 0:
            last_sharp_time = t_markers_em_list[0][-1]
            valid_later_mask = times_later_phase > (last_sharp_time + min_marker_time_gap)
            if np.any(valid_later_mask):
                t_markers_em_list.append(times_later_phase[valid_later_mask])
                k_markers_em_list.append(k_later_phase[valid_later_mask])
        else: # no sharp markers, just add what's there
            t_markers_em_list.append(times_later_phase)
            k_markers_em_list.append(k_later_phase)
    
    t_markers_em_combined = np.concatenate(t_markers_em_list) if t_markers_em_list else np.array([])
    k_markers_em_combined = np.concatenate(k_markers_em_list) if k_markers_em_list else np.array([])

    # check for unique markers if min_marker_time_gap is very small
    if len(t_markers_em_combined) > 1:
        sorted_indices = np.argsort(t_markers_em_combined)
        t_markers_em_combined = t_markers_em_combined[sorted_indices]
        k_markers_em_combined = k_markers_em_combined[sorted_indices]
        unique_mask = np.concatenate([[True], np.diff(t_markers_em_combined) > (min_marker_time_gap / 2.0)])
        t_markers_em_combined = t_markers_em_combined[unique_mask]
        k_markers_em_combined = k_markers_em_combined[unique_mask]

    # ------------------
    # --- Plotting -----
    # ------------------
    print("\n--- Plotting ---")
    # TODO: set single plot figure size and DPI
    plt.figure(figsize=(8, 6), dpi=200)

    # 1. average Gillespie curve (red line)
    plt.plot(times_gillespie_avg_raw, k_gillespie_avg_raw, 'r-', linewidth=2.0, alpha=0.8,
             label=f'Avg. Gillespie ({nsims_for_avg_display} runs)', zorder=2)

    # 2. KE solution using true beta1, beta2 (blue dashed line)
    plt.plot(times_ke, k_expected_true, 'b--', linewidth=2.0,
             label=r'KE (True $\beta_1, \beta_2$)', zorder=3)

    # 3. KE solution using EM-estimated beta1, beta2 (black 'X' symbols)
    plt.scatter(t_markers_em_combined, k_markers_em_combined,
                marker='x', color='black', s=marker_size, linewidth=1.0, alpha=marker_alpha,
                label=r'KE (EM Estimates $\widehat{\beta}_1, \widehat{\beta}_2$)', zorder=4)

    # finalize
    plt.xlabel("Time (t)")
    plt.ylabel("Number of Infected (k)")
    
    # TODO: remove the title and write this rather in caption
    title = f"KE(EM Estimates) and Gillespie Average (N = {N}, I0 = {I0}, nsims_pooled = {nsims_pooled})\n"
    title += f"True Scaled: $\\beta_1N={beta1_s_val_true:.1f}, \\beta_2N^2={beta2_s_val_true:.1f}$"
    plt.title(title, fontsize=11)
    
    # TODO: adjust the fontsize and location
    plt.legend(loc='best', fontsize='medium')

    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Y-axis from 0 to N
    plt.ylim(bottom=0, top=N + N*0.05) 
    plt.xlim(left=-0.05*time_max, right=time_max*1.05)

    # plt.show()
    plt.savefig(output_plot_filename, format="pdf", bbox_inches="tight")
    print(f"\nPlot saved to: {output_plot_filename}")
    plt.close()
    print("\nDone.")