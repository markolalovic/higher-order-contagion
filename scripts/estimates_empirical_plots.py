# ./scripts/estimates_empirical_plots.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # setup
    N = 1000
    k_full_range = np.arange(N + 1)

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)

    
    test_names = ["complete", "random_ER", "regular", "scale_free"]
    # reverse it, to plot complete, random_ER, ... zorder on top
    test_names = list(reversed(test_names))

    # TODO: choose colors for each class
    class_colors = {
        "complete":   "red",
        "random_ER":  "green",
        "regular":    "blue",
        "scale_free": "darkorange"
    }
    # TODO: choose symbols for scatter points, tilde data
    class_markers = { #
        "complete": "o",
        "random_ER": "s",
        "regular": "^",
        "scale_free": "D"
    }
    # ---------------------
    # --- Plot settings ---
    # ---------------------
    marker_size = 50
    marker_alpha = 0.85
    plt_line_width = 1.5
    legend_location = "upper right"
    plt_font_size = "small"
    
    remove_y_ticks_and_labels = True
    sample_size = 25 # TODO: 25 or lower, a number that looks good on the plot
    
    plot_zero_estimates = False  # to remove near-zero estimates in scatters
    zero_threshold = 1e-9       # and filter values below this

    plot_labels = {
        "complete": "Complete", 
        "random_ER": "Erdős-Rényi", 
        "regular": "Regular",
        "scale_free": "Scale-Free"
    }

    csv_input_dir = os.path.join(project_root_dir, 'data', 'estimates')
    fig_output_dir = os.path.join(project_root_dir, 'figures', 'combined', 'estimates')
    os.makedirs(fig_output_dir, exist_ok=True)

    # --- 3 figures of combined plots for all 4 classes ---
    fig_ak, ax_ak = plt.subplots(figsize=(8, 6))
    fig_bk, ax_bk = plt.subplots(figsize=(8, 6))
    fig_lk, ax_lk = plt.subplots(figsize=(8, 6))

    for test_name in test_names:
        print(f"\n--- Class: {test_name} ---")
        csv_path = os.path.join(csv_input_dir, f'estimates_{test_name}.csv')
        df = pd.read_csv(csv_path)

        k_full_range = df['k'].values
        a_k_tilde = df['a_k_tilde'].values
        b_k_tilde = df['b_k_tilde'].values
        lambda_k_tilde = df['lambda_k_tilde'].values

        ak_fitted = df['ak_fitted'].values
        bk_fitted = df['bk_fitted'].values
        lambda_k_fitted = df['lambda_k_fitted'].values

        # --- plot_zero_estimates: True / False ---
        def get_scatter_samples(k_all, tilde_values_all, sample_size_target, plot_zeros, zero_thresh):
            valid_tilde_idx_initial = ~np.isnan(tilde_values_all)
            k_base = k_all[valid_tilde_idx_initial]
            tilde_base = tilde_values_all[valid_tilde_idx_initial]

            if not plot_zeros:
                non_zero_idx = np.abs(tilde_base) > zero_thresh
                k_for_sample = k_base[non_zero_idx]
                tilde_for_sample = tilde_base[non_zero_idx]
            else:
                k_for_sample = k_base
                tilde_for_sample = tilde_base

            if len(k_for_sample) > sample_size_target:
                sample_indices = np.linspace(0, len(k_for_sample) - 1, sample_size_target, dtype=int)
                k_s = k_for_sample[sample_indices]
                tilde_s = tilde_for_sample[sample_indices]
            elif len(k_for_sample) > 0:
                k_s = k_for_sample
                tilde_s = tilde_for_sample
            else:
                k_s, tilde_s = [], []
            return k_s, tilde_s
        

        # --------------------------------------------------------------------------------
        # Sample from ak_tilde_reliable, bk_tilde_reliable, lambda_k_tilde_reliable
        # --------------------------------------------------------------------------------
        # valid_tilde_idx = ~np.isnan(a_k_tilde)
        # k_for_scatter_base = k_full_range[valid_tilde_idx]
        # # sample equally spaced indices to from the reliable data
        # sample_indices = np.linspace(0, len(k_for_scatter_base) - 1, sample_size, dtype=int)
        # k_scatter = k_for_scatter_base[sample_indices]
        # ak_tilde_sample = a_k_tilde[valid_tilde_idx][sample_indices]
        # bk_tilde_sample = b_k_tilde[valid_tilde_idx][sample_indices]
        # lambda_k_tilde_sample = lambda_k_tilde[valid_tilde_idx][sample_indices]
        k_scatter_ak, ak_tilde_sample = get_scatter_samples(k_full_range, a_k_tilde, sample_size, plot_zero_estimates, zero_threshold)
        k_scatter_bk, bk_tilde_sample = get_scatter_samples(k_full_range, b_k_tilde, sample_size, plot_zero_estimates, zero_threshold)
        k_scatter_lk, lambda_k_tilde_sample = get_scatter_samples(k_full_range, lambda_k_tilde, sample_size, plot_zero_estimates, zero_threshold)

        # default black color
        default_color_line = "black" # For the fitted line
        default_marker = "."

        current_color = class_colors.get(test_name, "black")
        current_marker = class_markers.get(test_name, ".")

        # ----------------
        # --- Plotting ---
        # --- -------- ---
        # skipping labels for estimates: label=f'{plot_labels[test_name]}'
        # a_k plot
        ax_ak.plot(k_full_range, ak_fitted, color=default_color_line, linewidth=plt_line_width, alpha=0.7)
        ax_ak.scatter(k_scatter_ak, ak_tilde_sample, label=f'{plot_labels[test_name]}',
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)

        # b_k plot
        ax_bk.plot(k_full_range, bk_fitted, color=default_color_line, linewidth=plt_line_width, alpha=0.7)
        ax_bk.scatter(k_scatter_bk, bk_tilde_sample, label=f'{plot_labels[test_name]}',
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)

        # lambda_k plot
        ax_lk.plot(k_full_range, lambda_k_fitted, color=default_color_line, linewidth=plt_line_width, alpha=0.7)
        ax_lk.scatter(k_scatter_lk, lambda_k_tilde_sample, label=f'{plot_labels[test_name]}',
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)
        print(f"  Plotted data for {test_name}")
    
    plt_version = ""
    if plot_zero_estimates:
        plt_version = "_with_zeros"
    # --- ---------- ---
    # --- Figure a_k ---
    # --- ---------- ---
    ax_ak.set_xlabel("Number of Infected (k)")
    ax_ak.set_ylabel("Rate")
    ax_ak.set_title(f"Pairwise Rate a_k (N = {N})")

    # reverse the legend b/c of reversed test_names
    handles_ak, labels_ak = ax_ak.get_legend_handles_labels()
    ax_ak.legend(handles_ak[::-1], labels_ak[::-1], fontsize=plt_font_size, loc=legend_location)

    ax_ak.set_ylim(bottom=0) # Start y-axis at 0
    if remove_y_ticks_and_labels:
        ax_ak.set_yticklabels([])
        ax_ak.set_yticks([])
    fig_ak.savefig(os.path.join(fig_output_dir, f"empirical_a_k{plt_version}.pdf"), bbox_inches='tight')
    plt.close(fig_ak)

    # ------------------
    # --- Figure b_k ---
    # ------------------
    ax_bk.set_xlabel("Number of Infected (k)")
    ax_bk.set_ylabel("Rate")
    ax_bk.set_title(f"Higher-Order Rate b_k (N = {N})")

    # reverse the legend b/c of reversed test_names
    handles_bk, labels_bk = ax_bk.get_legend_handles_labels()
    ax_bk.legend(handles_bk[::-1], labels_bk[::-1], fontsize=plt_font_size, loc=legend_location)

    ax_bk.set_ylim(bottom=0)
    if remove_y_ticks_and_labels:
        ax_bk.set_yticklabels([])
        ax_bk.set_yticks([])
    fig_bk.savefig(os.path.join(fig_output_dir, f"empirical_b_k{plt_version}.pdf"), bbox_inches='tight')
    plt.close(fig_bk)

    # -----------------------
    # --- Figure lambda_k ---
    # -----------------------
    ax_lk.set_xlabel("Number of Infected (k)")
    ax_lk.set_ylabel("Rate")
    ax_lk.set_title(f"Total Birth Rate lambda_k (N = {N})")

    # reverse the legend b/c of reversed test_names
    handles_lk, labels_lk = ax_lk.get_legend_handles_labels()
    ax_lk.legend(handles_lk[::-1], labels_lk[::-1], fontsize=plt_font_size, loc=legend_location)

    ax_lk.set_ylim(bottom=0)
    if remove_y_ticks_and_labels:    
        ax_lk.set_yticklabels([])
        ax_lk.set_yticks([])
    fig_lk.savefig(os.path.join(fig_output_dir, f"empirical_lambda_k{plt_version}.pdf"), bbox_inches='tight')
    plt.close(fig_lk)

    print("\nDone.")