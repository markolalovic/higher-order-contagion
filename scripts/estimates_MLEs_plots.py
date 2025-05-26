# ./scripts/estimates_MLEs_plots.py
# produces 4 figures for each class: `./figures/combined/estimates/MLEs_{test_name}.pdf
# of a_k_hats, b_k_hats estimates scattered on top of fitted rate curves
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

if __name__ == "__main__":
    # --- setup ---
    N = 1000

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)

    test_names = ["complete", "random_ER", "regular", "scale_free"]
    
    # --- Plot settings ---
    marker_size_mle = 100
    marker_alpha_mle = 1
    plt_line_width_fitted = 1.5
    legend_location = "upper right"
    plt_font_size = "small"
    remove_y_ticks_and_labels = False

    sample_size_mle = 201

    plot_zero_mle_estimates = False
    zero_mle_threshold = 1e-9

    plot_labels = {
        "complete": "Complete", 
        "random_ER": "Erdős-Rényi", 
        "regular": "Regular",
        "scale_free": "Scale-Free"
    }
    plot_titles = {
        "complete": f"MLE Rate Estimates vs True Rates - Complete (N = {N})",
        "random_ER": f"MLE Rate Estimates vs Fitted Rates - Erdős-Rényi (N = {N})",
        "regular": f"MLE Rate Estimates vs Fitted Rates - Regular (N = {N})",
        "scale_free": f"MLE Rate Estimates vs Fitted Rates - Scale-Free (N = {N})"
    }

    csv_input_dir = os.path.join(project_root_dir, 'data', 'estimates')
    fig_output_dir = os.path.join(project_root_dir, 'figures', 'combined', 'estimates')
    os.makedirs(fig_output_dir, exist_ok=True)


    for test_name in test_names:
        print(f"\n--- Generating MLE Plot for Class: {test_name} ---")
        csv_path = os.path.join(csv_input_dir, f'estimates_{test_name}.csv')
        df = pd.read_csv(csv_path)

        k_full_range = df['k'].values
        a_k_hat_all = df['a_k_hat'].values # Load MLEs
        b_k_hat_all = df['b_k_hat'].values # Load MLEs
        # lambda_k_hat_all = df['lambda_k_hat'].values

        ak_fitted = df['ak_fitted'].values
        bk_fitted = df['bk_fitted'].values
        # lambda_k_fitted = df['lambda_k_fitted'].values

        def get_mle_scatter_samples(k_all, hat_values_all, sample_size_target, plot_zeros, zero_thresh):
            valid_hat_idx_initial = ~np.isnan(hat_values_all)
            k_base = k_all[valid_hat_idx_initial]
            hat_base = hat_values_all[valid_hat_idx_initial]

            if not plot_zeros:
                non_zero_idx = np.abs(hat_base) > zero_thresh
                k_for_sample = k_base[non_zero_idx]
                hat_for_sample = hat_base[non_zero_idx]
            else:
                k_for_sample = k_base
                hat_for_sample = hat_base

            if len(k_for_sample) == 0:
                return [], []
            
            if len(k_for_sample) > sample_size_target:
                sample_indices = np.linspace(1, len(k_for_sample) - 10, sample_size_target, dtype=int)
                k_s = k_for_sample[sample_indices]
                hat_s = hat_for_sample[sample_indices]
            else:
                k_s = k_for_sample
                hat_s = hat_for_sample
            return k_s, hat_s

        k_scatter_ak_hat, ak_hat_sample = get_mle_scatter_samples(k_full_range, a_k_hat_all, sample_size_mle, plot_zero_mle_estimates, zero_mle_threshold)
        k_scatter_bk_hat, bk_hat_sample = get_mle_scatter_samples(k_full_range, b_k_hat_all, sample_size_mle, plot_zero_mle_estimates, zero_mle_threshold)

        # ---a figure with two subplots one for a_k, one for b_k ---
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True) # that share x-axis
        
        fig.suptitle(plot_titles[test_name], fontsize=14)

        # --- subplot for a_k_hat ---
        axes[0].plot(k_full_range, ak_fitted, color='black', linewidth=plt_line_width_fitted,
                     alpha=0.9, zorder=1)
        if len(k_scatter_ak_hat) > 0:
            axes[0].scatter(k_scatter_ak_hat, ak_hat_sample, label=r'MLEs $\widehat{a}_k$',
                            alpha=marker_alpha_mle, s=marker_size_mle, color='red', marker='.', zorder=2)
        axes[0].set_ylabel(r"Pairwise Rate $a_k$")
        axes[0].legend(fontsize=plt_font_size, loc=legend_location)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].set_ylim(bottom=0)
        if remove_y_ticks_and_labels:
            axes[0].set_yticklabels([])
            axes[0].set_yticks([])


        # --- subplot for b_k_hat ---
        axes[1].plot(k_full_range, bk_fitted, color='black', linewidth=plt_line_width_fitted,
                     alpha=0.9, zorder=1)
        if len(k_scatter_bk_hat) > 0:
            axes[1].scatter(k_scatter_bk_hat, bk_hat_sample, label=r'MLEs $\widehat{b}_k$',
                            alpha=marker_alpha_mle, s=marker_size_mle, color='blue', marker='.', zorder=2)
        axes[1].set_xlabel("Number of Infected (k)")
        axes[1].set_ylabel(r"Higher-Order Rate $b_k$")
        axes[1].legend(fontsize=plt_font_size, loc=legend_location)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].set_ylim(bottom=0)
        if remove_y_ticks_and_labels:
            axes[1].set_yticklabels([])
            axes[1].set_yticks([])

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # TODO: adjust or just remove the suptitle

        output_filename = os.path.join(fig_output_dir, f"MLEs_{test_name}.pdf")
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved MLE plot to: {output_filename}")

    print("\nDone.")