# ./scripts/fit_rate_curves.py
# fits rate curves given many empirical estimates
# script for testing, no need to run

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from estimate_total_rates import di_lauro_ak_model, di_lauro_bk_model
from scipy.special import comb as nchoosek
from scipy.optimize import curve_fit

if __name__ == "__main__":
    # setup
    N = 1000
    k_full_range = np.arange(N + 1)

    # complete class true parameters for for actual rates a_k, b_k, lambda_k
    true_beta1_for_complete = 2.4 / N
    true_beta2_for_complete = 4.4 / (N**2)

    test_names = ["complete", "random_ER", "regular", "scale_free"]
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
    marker_size = 50
    marker_alpha = 0.85

    plot_labels = {
        "complete": "Complete", 
        "random_ER": "Erdős-Rényi", 
        "regular": "Regular",
        "scale_free": "Scale-Free"
    }

    fig_output_dir = os.path.join(project_root, 'figures', 'combined', 'fits')
    os.makedirs(fig_output_dir, exist_ok=True)

    # --- 3 figures of combined plots for all 4 classes ---
    fig_ak, ax_ak = plt.subplots(figsize=(8, 6))
    fig_bk, ax_bk = plt.subplots(figsize=(8, 6))
    fig_lk, ax_lk = plt.subplots(figsize=(8, 6))

    for test_name in test_names:
        print(f"\n --- Class: {test_name} --- ")
        estimates_path = os.path.join(project_root, 'results', 'estimates', f'{test_name}.npz')
        estimates = np.load(estimates_path, allow_pickle=True)

        a_k_tilde_all = estimates["a_k_tilde"]
        b_k_tilde_all = estimates["b_k_tilde"]
        lambda_k_tilde_all = estimates["lambda_k_tilde"]
        T_k_all = estimates["T_k"]

        # min_Tk_threshold = 0.1 # TODO: tune it
        min_Tk_threshold = 1e-6 # TODO: tune it
        valid_k_idx = T_k_all > min_Tk_threshold
        
        k_observed_reliable = k_full_range[valid_k_idx]
        ak_tilde_reliable = a_k_tilde_all[valid_k_idx]
        bk_tilde_reliable = b_k_tilde_all[valid_k_idx]
        lambda_k_tilde_reliable = lambda_k_tilde_all[valid_k_idx]
        print(f"Number of reliable data points for fitting: {len(k_observed_reliable)}")

        # --------------------------------------------------------------------------------
        # Sample from ak_tilde_reliable, bk_tilde_reliable, lambda_k_tilde_reliable
        # --------------------------------------------------------------------------------
        sample_size = 25 # TODO: 25 or 10, a number that looks good on the plot
        # sample equally spaced indices to from the reliable data
        sample_indices = np.linspace(0, len(k_observed_reliable) - 1, sample_size, dtype=int)
        k_scatter = k_observed_reliable[sample_indices]
        ak_tilde_sample = ak_tilde_reliable[sample_indices]
        bk_tilde_sample = bk_tilde_reliable[sample_indices]
        lambda_k_tilde_sample = lambda_k_tilde_reliable[sample_indices]

        # default black color and dot marker
        default_color = "black"
        default_marker = "."

        current_color = class_colors.get(test_name, "black")
        current_marker = class_markers.get(test_name, ".")

        # --------------------
        # --- PW rates a_k ---
        # --------------------
        ak_peak_val = np.max(ak_tilde_reliable) if len(ak_tilde_reliable) > 0 else 0.01
        p_a_init = 1.5 if test_name == "scale_free" else 1.0
        alpha_a_init = -1.0 if test_name == "scale_free" else 0.0
        C_a_denom = ((N/2)**p_a_init * (N/2)**p_a_init) if N > 0 else 1
        C_a_init = ak_peak_val / C_a_denom if C_a_denom > 1e-9 else ak_peak_val
        p0_ak = [C_a_init, p_a_init, alpha_a_init]
        bounds_ak = ([0, 0.1, -5], [np.inf, 5, 2])

        ak_fitted = np.full_like(k_full_range, np.nan)
        popt_ak, _ = curve_fit(lambda k, C, p, alpha: di_lauro_ak_model(k, N, C, p, alpha),
                            k_observed_reliable, ak_tilde_reliable, p0=p0_ak, bounds=bounds_ak, maxfev=10000)
        ak_fitted = di_lauro_ak_model(k_full_range, N, *popt_ak)
        print(f"\na_k fit params ({test_name}): C={popt_ak[0]:.2e}, p={popt_ak[1]:.2f}, alpha={popt_ak[2]:.2f}")
        
        # --------------------
        # --- HO rates b_k ---
        # --------------------
        bk_peak_val = np.max(bk_tilde_reliable) if len(bk_tilde_reliable) > 0 else 1e-6
        
        # TODO: adjust for scale-free
        p_b_init = 1.5 if test_name == "scale_free" else 1.0
        alpha_b_init = -0.5 if test_name == "scale_free" else 0.0

        k_peak_b_approx = 2*N/3
        denom_cb = k_peak_b_approx * (k_peak_b_approx-1)**p_b_init * (N-k_peak_b_approx)**p_b_init if k_peak_b_approx > 1 and N-k_peak_b_approx > 0 else 1
        C_b_init = bk_peak_val / denom_cb if denom_cb > 1e-9 else bk_peak_val
        p0_bk = [C_b_init, p_b_init, alpha_b_init]
        bounds_bk = ([0, 0.1, -5], [np.inf, 5, 2])
        
        k_obs_for_bk_fit = k_observed_reliable[k_observed_reliable >= 2]
        bk_tilde_for_bk_fit = bk_tilde_reliable[k_observed_reliable >= 2]

        bk_fitted = np.full_like(k_full_range, np.nan)
        popt_bk, _ = curve_fit(lambda k, C, p, alpha: di_lauro_bk_model(k, N, C, p, alpha),
                                k_obs_for_bk_fit, bk_tilde_for_bk_fit, p0=p0_bk, bounds=bounds_bk, maxfev=10000)
        bk_fitted = di_lauro_bk_model(k_full_range, N, *popt_bk)
        print(f"b_k fit params ({test_name}): C={popt_bk[0]:.2e}, p={popt_bk[1]:.2f}, alpha={popt_bk[2]:.2f}")
        
        # ----------------------------
        # --- Total rates lambda_k ---
        # ----------------------------
        lambda_k_fitted = ak_fitted + bk_fitted

        # ----------------
        # --- Plotting ---
        # --- -------- ---
        # skipping labels for fits: label=f'{plot_labels[test_name]}'
        # a_k plot
        ax_ak.plot(k_full_range, ak_fitted, color=default_color, linewidth=1.5)
        ax_ak.scatter(k_scatter, ak_tilde_sample, label=f'{plot_labels[test_name]}', 
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)

        # b_k plot
        ax_bk.plot(k_full_range, bk_fitted, color=default_color, linewidth=1.5)
        ax_bk.scatter(k_scatter, bk_tilde_sample, label=f'{plot_labels[test_name]}', 
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)
        
        # lambda_k plot
        ax_lk.plot(k_full_range, lambda_k_fitted, color=default_color, linewidth=1.5)
        ax_lk.scatter(k_scatter, lambda_k_tilde_sample, label=f'{plot_labels[test_name]}', 
                        alpha=marker_alpha, s=marker_size, color=current_color, marker=current_marker, zorder=10)
        
    # --- ---------- ---
    # --- Figure a_k ---
    # --- ---------- ---
    # if test_name == "complete":
    #     # for complete we can add true theoretical curve
    #     a_k_true_comp = true_beta1_for_complete * k_full_range * (N - k_full_range)
    #     ax_ak.plot(k_full_range, a_k_true_comp, label='ak_true (Complete)', color='black', linestyle=':', linewidth=1.5, zorder=0)
    ax_ak.set_xlabel("Number of Infected (k)")
    ax_ak.set_ylabel("Rate")
    ax_ak.set_title(f"Pairwise Rate a_k (N = {N})")
    ax_ak.legend(fontsize='small')
    # ax_ak.grid(True, linestyle=':') # TODO: no grid?
    ax_ak.set_ylim(bottom=-0.05 * np.nanmax(ax_ak.get_ylim()[-1] if ax_ak.get_ylim()[-1]>0 else 1.0))
    ax_ak.set_yticklabels([]) # TODO: decide y-axis tick labels (the numbers) should be there or not
    ax_ak.set_yticks([])
    # save and close
    # fig_ak.savefig(os.path.join(fig_output_dir, "fits_a_k.pdf"), bbox_inches='tight')
    # plt.close(fig_ak)

    # ------------------
    # --- Figure b_k ---
    # ------------------
    # if test_name == "complete":
    #     # for complete we can add true theoretical curve
    #     binom_k_2 = np.zeros_like(k_full_range, dtype=float)
    #     valid_binom_idx = k_full_range >= 2
    #     binom_k_2[valid_binom_idx] = nchoosek(k_full_range[valid_binom_idx], 2, exact=False)
    #     b_k_true_comp = true_beta2_for_complete * binom_k_2 * (N - k_full_range)
    #     ax_bk.plot(k_full_range, b_k_true_comp, label='bk_true (Complete)', color='black', linestyle=':', linewidth=1.5, zorder=0)
    ax_bk.set_xlabel("Number of Infected (k)")
    ax_bk.set_ylabel("Rate")
    ax_bk.set_title(f"Higher-Order Rate b_k (N = {N})")
    ax_bk.legend(fontsize='small')
    # ax_bk.grid(True, linestyle=':') # TODO: no grid?
    ax_bk.set_ylim(bottom=-0.05 * np.nanmax(ax_bk.get_ylim()[-1] if ax_bk.get_ylim()[-1]>0 else 1.0))
    ax_bk.set_yticklabels([]) # TODO: decide y-axis tick labels (the numbers) should be there or not
    ax_bk.set_yticks([])
    # save and close
    # fig_bk.savefig(os.path.join(fig_output_dir, "fits_b_k.pdf"), bbox_inches='tight')
    # plt.close(fig_bk)

    # -----------------------
    # --- Figure lambda_k ---
    # -----------------------
    # if test_name == "complete":
    #     # for complete we can add true theoretical curve
    #     lambda_k_true_comp = a_k_true_comp + b_k_true_comp
    #     ax_lk.plot(k_full_range, lambda_k_true_comp, label='lk_true (Complete)', color='black', linestyle=':', linewidth=1.5, zorder=0)
    ax_lk.set_xlabel("Number of Infected (k)")
    ax_lk.set_ylabel("Rate")
    ax_lk.set_title(f"Total Birth Rate lambda_k (N = {N} )")
    ax_lk.legend(fontsize='small')
    # ax_lk.grid(True, linestyle=':') # TODO: no grid?
    ax_lk.set_ylim(bottom=-0.05 * np.nanmax(ax_lk.get_ylim()[-1] if ax_lk.get_ylim()[-1]>0 else 1.0))
    ax_lk.set_yticklabels([]) # TODO: decide y-axis tick labels (the numbers) should be there or not
    ax_lk.set_yticks([])
    # save and close
    # fig_lk.savefig(os.path.join(fig_output_dir, "fits_lambda_k.pdf"), bbox_inches='tight')
    # plt.close(fig_lk)

    print("\nDone.")


'''
Results:

a_k fit params (complete): C=2.40e-03, p=1.00, alpha=0.00
b_k fit params (complete): C=2.20e-06, p=1.00, alpha=0.00

 --- Class: random_ER ---
Number of reliable data points for fitting: 766

a_k fit params (random_ER): C=1.85e-03, p=0.98, alpha=-0.07
b_k fit params (random_ER): C=2.73e-05, p=0.83, alpha=-0.24

 --- Class: regular ---
Number of reliable data points for fitting: 787

a_k fit params (regular): C=1.53e-03, p=1.00, alpha=0.08
b_k fit params (regular): C=3.17e-05, p=0.81, alpha=0.13

 --- Class: scale_free ---
Number of reliable data points for fitting: 753

a_k fit params (scale_free): C=3.01e-02, p=0.77, alpha=-0.60
b_k fit params (scale_free): C=1.13e-02, p=0.36, alpha=-1.02

Done.
'''