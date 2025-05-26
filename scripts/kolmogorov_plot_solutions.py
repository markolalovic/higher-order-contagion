# ./scripts/kolmogorov_plot_solutions.py
# reads KE solutions from CSV files: `./data/estimates/kolmogorov_{test_name}.csv`
# saves figure: `./figures/combined/kolmogorov_solutions.pdf`

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # --- Setup ---
    N_global = 1000
    I0_global = 50
    num_gillespie_runs_for_avg = 100
    time_max_global = 10.0

    test_names = ["complete", "random_ER", "regular", "scale_free"]
    plot_titles = {
        "complete": f"Complete",
        "random_ER": f"Erdős-Rényi",
        "regular": f"Regular",
        "scale_free": f"Scale-Free"
    }

    # --- ranges for denser MLE markers during sharp rise ---
    sharp_rising_ranges = {
        "complete":   [0.0, 4.5],
        "random_ER":  [0.0, 4.5],
        "regular":    [0.0, 5.5],
        "scale_free": [0.0, 4]
    }
    num_markers_mle_sharp = 15  # number of markers in the sharp rising phase
    num_markers_mle_later = 10  # number of markers in the later phase
    min_marker_time_gap = 0.1  # minimum time gap between consecutive markers    
    marker_size_mle = 60
    marker_alpha_mle = 0.8

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    solutions_csv_input_dir = os.path.join(project_root, "data", "estimates")
    gillespie_avg_base_dir = os.path.join(project_root, "data", "gillespie_sims")
    output_figure_dir = os.path.join(project_root, "figures", "combined")
    os.makedirs(output_figure_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    axes = axes.flatten()

    print(f"--- Starting Plot Generation from Pre-solved KE Data ---")
    print(f"N={N_global}, I0={I0_global}")

    for i, test_name in enumerate(test_names):
        ax = axes[i]
        print(f"--- Plotting: {plot_titles[test_name]} (Subplot {i+1}) ---")

        avg_curve_filename = os.path.join(gillespie_avg_base_dir, test_name, f"average_curve_{num_gillespie_runs_for_avg}.csv")
        df_avg = pd.read_csv(avg_curve_filename)
        times_gillespie_avg = df_avg['time'].to_numpy()
        k_gillespie_avg = df_avg['avg_infected_k'].to_numpy()
        print(f"Loaded average Gillespie curve from: {os.path.basename(avg_curve_filename)}")

        ke_solutions_filename = os.path.join(solutions_csv_input_dir, f"kolmogorov_solutions_{test_name}.csv")
        df_ke_sols = pd.read_csv(ke_solutions_filename)
        times_ke_dense = df_ke_sols['time'].to_numpy()
        expected_values_hat_dense = df_ke_sols['k_expected_hat'].to_numpy()
        expected_values_tilde_dense = df_ke_sols['k_expected_tilde'].to_numpy()
        print(f"Loaded KE solutions from: {os.path.basename(ke_solutions_filename)}")

        # --- prepare data for MLE (hat) scatter markers with varying density ---
        t_markers_hat_list = []
        k_markers_hat_list = []

        time_rise_start, time_rise_end = sharp_rising_ranges.get(test_name, [0, time_max_global / 3.0])

        # 1. dense markers in sharp rising phase
        dense_mask_sharp = (times_ke_dense >= time_rise_start) & (times_ke_dense <= time_rise_end)
        times_sharp_phase = times_ke_dense[dense_mask_sharp]
        k_sharp_phase = expected_values_hat_dense[dense_mask_sharp]

        if len(times_sharp_phase) > num_markers_mle_sharp:
            # select markers, but exclude the very last index to avoid boundary overlap
            marker_indices_sharp = np.linspace(0, len(times_sharp_phase) - 1, num_markers_mle_sharp, dtype=int)
            # remove the last marker if it's too close to the boundary
            if marker_indices_sharp[-1] == len(times_sharp_phase) - 1:
                marker_indices_sharp = marker_indices_sharp[:-1]
            t_markers_hat_list.append(times_sharp_phase[marker_indices_sharp])
            k_markers_hat_list.append(k_sharp_phase[marker_indices_sharp])
        elif len(times_sharp_phase) > 0:
            # if few points, exclude the last one to avoid boundary issues
            t_markers_hat_list.append(times_sharp_phase[:-1] if len(times_sharp_phase) > 1 else times_sharp_phase)
            k_markers_hat_list.append(k_sharp_phase[:-1] if len(k_sharp_phase) > 1 else k_sharp_phase)

        # 2. sparser markers in later phase
        dense_mask_later = times_ke_dense > time_rise_end
        times_later_phase = times_ke_dense[dense_mask_later]
        k_later_phase = expected_values_hat_dense[dense_mask_later]

        if len(times_later_phase) > num_markers_mle_later:
            # for the later phase, ensure the first marker isn't too close to the boundary
            marker_indices_later = np.linspace(0, len(times_later_phase) - 1, num_markers_mle_later, dtype=int)
            
            # check if we need to skip the first marker to maintain gap
            if len(t_markers_hat_list) > 0 and len(t_markers_hat_list[0]) > 0:
                last_sharp_time = t_markers_hat_list[0][-1] if len(t_markers_hat_list[0]) > 0 else 0
                first_later_time = times_later_phase[marker_indices_later[0]]
                
                # if the gap is too small, skip the first marker
                if first_later_time - last_sharp_time < min_marker_time_gap:
                    marker_indices_later = marker_indices_later[1:]
            
            t_markers_hat_list.append(times_later_phase[marker_indices_later])
            k_markers_hat_list.append(k_later_phase[marker_indices_later])
        elif len(times_later_phase) > 0:
            # for few points, check the gap with the last sharp marker
            if len(t_markers_hat_list) > 0 and len(t_markers_hat_list[0]) > 0:
                last_sharp_time = t_markers_hat_list[0][-1]
                # only add later phase markers that are sufficiently far from the last sharp marker
                valid_mask = times_later_phase > last_sharp_time + min_marker_time_gap
                t_markers_hat_list.append(times_later_phase[valid_mask])
                k_markers_hat_list.append(k_later_phase[valid_mask])
            else:
                t_markers_hat_list.append(times_later_phase)
                k_markers_hat_list.append(k_later_phase)
        
        # combine marker points (if any were generated)
        t_markers_hat_combined = np.concatenate(t_markers_hat_list) if t_markers_hat_list else np.array([])
        k_markers_hat_combined = np.concatenate(k_markers_hat_list) if k_markers_hat_list else np.array([])

        # additional safety check: remove any duplicate or near-duplicate time points
        if len(t_markers_hat_combined) > 1:
            unique_mask = np.concatenate([[True], np.diff(t_markers_hat_combined) > min_marker_time_gap])
            t_markers_hat_combined = t_markers_hat_combined[unique_mask]
            k_markers_hat_combined = k_markers_hat_combined[unique_mask]

        # --- Plotting for current subplot ---
        ax.plot(times_gillespie_avg, k_gillespie_avg, color="red", linewidth=2.0,
                label=f'Avg. Gillespie ({num_gillespie_runs_for_avg} runs)', alpha=0.9, zorder=1)

        ax.plot(times_ke_dense, expected_values_tilde_dense, color="blue", linestyle='--', linewidth=1.8,
                 label=r'KE (Empirical $\widetilde{a}_k, \widetilde{b}_k$)', alpha=0.8, zorder=2)

        if len(t_markers_hat_combined) > 0:
            ax.scatter(t_markers_hat_combined, k_markers_hat_combined, color="black", marker='x', s=marker_size_mle,
                       alpha=marker_alpha_mle, label=r'KE (MLE $\widehat{a}_k, \widehat{b}_k$)', zorder=3)

        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Number of Infected (k)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(f"{plot_titles[test_name]}", fontsize=12)
        ax.set_ylim(bottom=-N_global*0.02, top=N_global*1.05)

        if i == 0:
            ax.legend(fontsize='small', loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"KE Solutions with Estimated Rates vs. Average Gillespie Dynamics (N = {N_global}, I0 = {I0_global})", fontsize=14)

    output_figure_filename = os.path.join(output_figure_dir, "kolmogorov_solutions.pdf")
    plt.savefig(output_figure_filename, format="pdf", bbox_inches="tight")
    print(f"\nCombined figure saved to: {output_figure_filename}")

    # plt.show()
    plt.close(fig)

    print("\nDone.")