# ./scripts/kolmogorov_solve_save_fits.py
# solves kolmogorov forward equations with empirical estimates (tildes) and MLEs (hats) 
# and saves the solutions to: `./data/estimates/kolmogorov_{test_name}.csv`

import numpy as np
import pandas as pd
import os
import sys
from scipy.integrate import solve_ivp
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from solve_kolmogorov import calculate_expected_values, list_all_ODEs_using_estimates

if __name__ == "__main__":
    # --- Setup ---
    N_global = 1000
    I0_global = 50
    mu_global = 1.0
    time_max_global = 10.0

    test_names = ["complete", "random_ER", "regular", "scale_free"]

    # directories: input estimates, output KE solutions
    estimates_csv_input_dir = os.path.join(project_root, "data", "estimates")
    solutions_csv_output_dir = os.path.join(project_root, "data", "estimates")
    os.makedirs(solutions_csv_output_dir, exist_ok=True)

    # KE solver setup
    t_span = (0.0, time_max_global)

    # NOTE: using dense evaluation grid for saving, plotting script can subsample if needed!
    steps_eval_ke = 201 # or more
    t_eval_ke = np.linspace(t_span[0], t_span[1], int(steps_eval_ke))
    M_global = N_global + 1
    p0_global = np.zeros(M_global, dtype=np.float64) # float64 for solver!
    if 0 <= I0_global <= N_global:
        p0_global[I0_global] = 1.0
    else:
        print(f"Warning: I0_global={I0_global} invalid. Using p0[0]=1.0 for KEs.")
        p0_global[0] = 1.0
    
    print(f"--- Starting KE Solves & Saving to CSV ---")
    print(f"N={N_global}, I0={I0_global}, mu={mu_global}, t_max={time_max_global}")

    # --- for each network class ---
    for test_name in test_names:
        print(f"\n--- Processing class: {test_name} --- ")
        start_time_class = time.time()

        # --- load estimated rates ---
        estimates_filename = os.path.join(estimates_csv_input_dir, f"estimates_{test_name}.csv")
        df_est = pd.read_csv(estimates_filename)
        a_k_hat = df_est['a_k_hat'].to_numpy()
        b_k_hat = df_est['b_k_hat'].to_numpy()
        a_k_tilde = df_est['a_k_tilde'].to_numpy()
        b_k_tilde = df_est['b_k_tilde'].to_numpy()
        print(f"\tLoaded estimated rates from: {os.path.basename(estimates_filename)}")

        # --- Solve KEs with MLEs (hat estimates) ---
        print(f"\tSolving KEs with MLE (hat) rates...")
        ode_system_hat = list_all_ODEs_using_estimates(N_global, a_k_hat, b_k_hat, mu_global)
        sol_hat = solve_ivp(lambda t, p: ode_system_hat(t, p),
                            t_span, p0_global, t_eval=t_eval_ke, method="LSODA",
                            rtol=1e-6, atol=1e-9)
        if sol_hat.status != 0:
             print(f"\tWarning!: Solver for hat estimates failed with status {sol_hat.status}: {sol_hat.message}")
        expected_values_hat = calculate_expected_values(sol_hat)

        # --- Solve KEs with Empirical (tilde estimates) ---
        print(f"  Solving KEs with Empirical (tilde) rates...")
        ode_system_tilde = list_all_ODEs_using_estimates(N_global, a_k_tilde, b_k_tilde, mu_global)
        sol_tilde = solve_ivp(lambda t, p: ode_system_tilde(t, p),
                              t_span, p0_global, t_eval=t_eval_ke, method="LSODA",
                              rtol=1e-6, atol=1e-9)
        if sol_tilde.status != 0:
             print(f"\tWarning!: Solver for tilde estimates failed with status {sol_tilde.status}: {sol_tilde.message}")
        expected_values_tilde = calculate_expected_values(sol_tilde)
        
        # --- save solutions to CSV ---
        df_solutions = pd.DataFrame({
            'time': sol_hat.t, # sol_hat.t and sol_tilde.t are identical, they have the same t_eval_ke
            'k_expected_hat': expected_values_hat,
            'k_expected_tilde': expected_values_tilde
        })
        
        output_csv_filename = os.path.join(solutions_csv_output_dir, f"kolmogorov_solutions_{test_name}.csv")
        df_solutions.to_csv(output_csv_filename, index=False, float_format='%.8g')
        end_time_class = time.time()
        print(f"\tKE solutions for {test_name} saved to: {os.path.basename(output_csv_filename)}")
        print(f"\tTime taken for {test_name}: {end_time_class - start_time_class:.2f} seconds.")

    print("\nDone. KE solves and saves completed.")

'''
# Test finished without warnings:

--- Starting KE Solves & Saving to CSV ---
N=1000, I0=50, mu=1.0, t_max=10.0

--- Processing class: complete ---
	Loaded estimated rates from: estimates_complete.csv
	Solving KEs with MLE (hat) rates...
  Solving KEs with Empirical (tilde) rates...
	KE solutions for complete saved to: kolmogorov_solutions_complete.csv
	Time taken for complete: 59.37 seconds.

--- Processing class: random_ER ---
	Loaded estimated rates from: estimates_random_ER.csv
	Solving KEs with MLE (hat) rates...
  Solving KEs with Empirical (tilde) rates...
	KE solutions for random_ER saved to: kolmogorov_solutions_random_ER.csv
	Time taken for random_ER: 42.34 seconds.

--- Processing class: regular ---
	Loaded estimated rates from: estimates_regular.csv
	Solving KEs with MLE (hat) rates...
  Solving KEs with Empirical (tilde) rates...
	KE solutions for regular saved to: kolmogorov_solutions_regular.csv
	Time taken for regular: 48.15 seconds.

--- Processing class: scale_free ---
	Loaded estimated rates from: estimates_scale_free.csv
	Solving KEs with MLE (hat) rates...
  Solving KEs with Empirical (tilde) rates...
	KE solutions for scale_free saved to: kolmogorov_solutions_scale_free.csv
	Time taken for scale_free: 72.10 seconds.

Done. KE solves and saves completed.
'''