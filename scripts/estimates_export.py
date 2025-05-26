import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import curve_fit
from scipy.special import comb as nchoosek

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from estimate_total_rates import di_lauro_ak_model, di_lauro_bk_model

def calculate_and_save_estimates_csv(N_nodes, test_name_list, project_root_dir):
    r"""
    Loads estimates, re-calculates fitted curves, and saves all to CSV files.
    """
    k_full_range = np.arange(N_nodes + 1)
    estimates_input_dir = os.path.join(project_root_dir, 'results', 'estimates')
    csv_output_dir = os.path.join(project_root_dir, 'data', 'estimates')
    os.makedirs(csv_output_dir, exist_ok=True)

    for test_name in test_name_list:
        print(f"\n--- Class: {test_name} ---")
        estimates_path = os.path.join(estimates_input_dir, f'{test_name}.npz')

        estimates = np.load(estimates_path, allow_pickle=True)

        # extract estimates from saved files
        a_k_hat = estimates.get("a_k_hat", np.full(N_nodes + 1, np.nan))
        b_k_hat = estimates.get("b_k_hat", np.full(N_nodes + 1, np.nan))
        lambda_k_hat = estimates.get("lambda_k_hat", np.full(N_nodes + 1, np.nan))
        a_k_tilde = estimates.get("a_k_tilde", np.full(N_nodes + 1, np.nan))
        b_k_tilde = estimates.get("b_k_tilde", np.full(N_nodes + 1, np.nan))
        lambda_k_tilde = estimates.get("lambda_k_tilde", np.full(N_nodes + 1, np.nan))
        T_k_all = estimates.get("T_k", np.zeros(N_nodes + 1)) # T_k for valid_k_idx

        min_Tk_threshold = 1e-6 # TODO: same threshold as in fit_rate_curves.py
        valid_k_idx = T_k_all > min_Tk_threshold
        k_observed_reliable = k_full_range[valid_k_idx]

        # recalculate fitted curves 
        ak_fitted = np.full_like(k_full_range, np.nan, dtype=float)
        bk_fitted = np.full_like(k_full_range, np.nan, dtype=float)

        # -----------
        # Fit a_k
        # -----------
        ak_tilde_reliable = a_k_tilde[valid_k_idx]
        # TODO: same p0 and bounds as in fit_rate_curves.py for consistency
        ak_peak_val = np.max(ak_tilde_reliable) if len(ak_tilde_reliable) > 0 else 0.01
        p_a_init = 1.5 if test_name == "scale_free" else 1.0
        alpha_a_init = -1.0 if test_name == "scale_free" else 0.0
        C_a_denom = ((N_nodes/2)**p_a_init * (N_nodes/2)**p_a_init) if N_nodes > 0 else 1
        C_a_init = ak_peak_val / C_a_denom if C_a_denom > 1e-9 else ak_peak_val
        p0_ak = [C_a_init, p_a_init, alpha_a_init]
        bounds_ak = ([0, 0.1, -5], [np.inf, 5, 2])

        popt_ak, _ = curve_fit(lambda k, C, p, alpha: di_lauro_ak_model(k, N_nodes, C, p, alpha),
                            k_observed_reliable, ak_tilde_reliable, p0=p0_ak, bounds=bounds_ak, maxfev=10000)
        ak_fitted = di_lauro_ak_model(k_full_range, N_nodes, *popt_ak)
        print(f"\na_k fit params ({test_name}): C={popt_ak[0]:.2e}, p={popt_ak[1]:.2f}, alpha={popt_ak[2]:.2f}")

        # -----------
        # Fit b_k
        # -----------
        bk_tilde_reliable = b_k_tilde[valid_k_idx]
        k_obs_for_bk_fit = k_observed_reliable[k_observed_reliable >= 2]
        bk_tilde_for_bk_fit = bk_tilde_reliable[k_observed_reliable >= 2] # apply same mask

        bk_peak_val = np.max(bk_tilde_for_bk_fit) if len(bk_tilde_for_bk_fit) > 0 else 1e-6
        p_b_init = 1.5 if test_name == "scale_free" else 1.0
        alpha_b_init = -0.5 if test_name == "scale_free" else 0.0
        k_peak_b_approx = 2*N_nodes/3
        denom_cb = k_peak_b_approx * (k_peak_b_approx-1)**p_b_init * (N_nodes-k_peak_b_approx)**p_b_init
        C_b_init = bk_peak_val / denom_cb if denom_cb > 1e-9 else bk_peak_val
        p0_bk = [C_b_init, p_b_init, alpha_b_init]
        bounds_bk = ([0, 0.1, -5], [np.inf, 5, 2])

        popt_bk, _ = curve_fit(lambda k, C, p, alpha: di_lauro_bk_model(k, N_nodes, C, p, alpha),
                            k_obs_for_bk_fit, bk_tilde_for_bk_fit, p0=p0_bk, bounds=bounds_bk, maxfev=10000)
        bk_fitted = di_lauro_bk_model(k_full_range, N_nodes, *popt_bk)
        print(f"\nb_k fit params ({test_name}): C={popt_bk[0]:.2e}, p={popt_bk[1]:.2f}, alpha={popt_bk[2]:.2f}")

        # ----------- 
        # lambda_k
        # -----------
        lambda_k_fitted = ak_fitted + bk_fitted

        # create DataFrame and save it
        df_estimates = pd.DataFrame({
            'k': k_full_range,
            'a_k_hat': a_k_hat,
            'b_k_hat': b_k_hat,
            'lambda_k_hat': lambda_k_hat,
            'a_k_tilde': a_k_tilde,
            'b_k_tilde': b_k_tilde,
            'lambda_k_tilde': lambda_k_tilde,
            'ak_fitted': ak_fitted,
            'bk_fitted': bk_fitted,
            'lambda_k_fitted': lambda_k_fitted
        })

        # format floats nicely
        output_csv_path = os.path.join(csv_output_dir, f'estimates_{test_name}.csv')
        df_estimates.to_csv(output_csv_path, index=False, float_format='%.8g')
        print(f"\nEstimates and fits saved to: {output_csv_path}")

    print("\nAll processing done.")

if __name__ == "__main__":
    # setup
    N_global = 1000
    test_name_list_global = ["complete", "random_ER", "regular", "scale_free"]
    calculate_and_save_estimates_csv(N_global, test_name_list_global, project_root)
