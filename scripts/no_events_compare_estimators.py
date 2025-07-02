# ./scripts/no_events_compare_estimators.py
# comparison of 
# 1. grid search GS
# 2. direct likelihood maximization DLM
# 3. two-stage regression TSR
# 4. expectation-maximization EM

import numpy as np
import time
import sys

sys.path.append('../src/')
sys.path.append('../scripts/')

from estimate_no_event_types import *
from simulate_gillespie import gillespie_sim_complete

if __name__ == "__main__":
    # --- Setup ---
    N = 100
    I0 = 10
    time_max = 10.0  # TODO: try to increase to 50
    mu_true = 1.0
    nsims = 100      # number of simulations per regime
    
    # define parameter regimes to test (using scaled betas)
    regimes = {
        "Balanced":        (2.4, 4.4),
        "Low PW, High HO": (1.4, 8.0),
        "High PW, Low HO": (3.7, 1.0),
    }

    # store results
    results = {regime: {method: {'beta1': [], 'beta2': [], 'time': []} 
                        for method in ['GS', 'DLM', 'TSR', 'EM']} 
               for regime in regimes}

    # initial guess for the optimizers (original, not scaled)
    initial_guess = [0.01 / N, 0.01 / (N**2)]
    
    # loop over regimes and simulations
    for regime_name, (b1_s_true, b2_s_true) in regimes.items():
        print(f"\n---- Regime: {regime_name} ----")
        print(f"True scaled (beta1, beta2): ({b1_s_true}, {b2_s_true})")

        # convert scaled betas to original per-interaction betas
        b1_orig_true = b1_s_true / N
        b2_orig_true = b2_s_true / (N**2)
        
        for i in range(nsims):
            # generate a new trajectory for each simulation
            # unique seed for each run for independence
            rng = np.random.default_rng(seed=(hash(regime_name) % (2**32)) + i)
            X_t = gillespie_sim_complete(N, b1_orig_true, b2_orig_true, mu_true, I0, time_max, rng)
            
            # extract stats
            stats = get_sufficient_stats(X_t, N)

            # run and time each estimator
            estimators = {
                'GS':  (grid_search, (stats, N, mu_true)),
                'DLM': (estimate_dlm, (stats, N, initial_guess)),
                'TSR': (estimate_tsr, (stats, N, initial_guess)),
                'EM':  (estimate_em, (stats, N, initial_guess))
            }
            
            for method_name, (func, args) in estimators.items():
                start_time = time.perf_counter()
                b1_est, b2_est = func(*args)
                end_time = time.perf_counter()
                
                results[regime_name][method_name]['beta1'].append(b1_est)
                results[regime_name][method_name]['beta2'].append(b2_est)
                results[regime_name][method_name]['time'].append(end_time - start_time)

            if (i + 1) % (nsims // 10 if nsims >= 10 else 1) == 0:
                print(f"\t Completed run {i + 1}/{nsims}")

    # ------------------------------
    # Process and print results
    # ------------------------------
    print("\n\n" + "-"*70)
    print("Results")
    print("-"*70)
    
    for regime_name, regime_results in results.items():
        b1_s_true, b2_s_true = regimes[regime_name]
        
        print(f"\n--- REGIME: {regime_name} (True Scaled Betas: {b1_s_true:.2f}, {b2_s_true:.2f}) ---\n")
        header = f"{'Method':<8} | {'Mean B1 (scaled)':<18} | {'SD B1 (scaled)':<16} | {'Mean B2 (scaled)':<18} | {'SD B2 (scaled)':<16} | {'Avg Time (ms)':<15}"
        # TODO: add SD for Time
        print(header)
        print("-" * len(header))
        
        for method_name, data in regime_results.items():
            # Convert original beta estimates back to scaled for easy comparison
            b1_est = np.array(data['beta1']) #* N
            b2_est = np.array(data['beta2']) #* N**2
            
            # Calculate summary statistics
            mean_b1 = np.nanmean(b1_est)
            sd_b1 = np.nanstd(b1_est)

            mean_b2 = np.nanmean(b2_est)
            sd_b2 = np.nanstd(b2_est)

            mean_time_ms = np.nanmean(data['time']) * 1000 # to milliseconds

            print(f"{method_name:<8} | {mean_b1:<18.3f} | {sd_b1:<16.3f} | {mean_b2:<18.3f} | {sd_b2:<16.3f} | {mean_time_ms:<15.2f}")
            
    print("\n" + "-"*70)
