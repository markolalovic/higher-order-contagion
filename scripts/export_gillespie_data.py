# ./scripts/export_gillespie_data.py
import numpy as np
import pandas as pd
import pickle
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from solve_kolmogorov import *
from simulate_gillespie import *
from estimate_total_rates import *

if __name__ == "__main__":
    # --- setup ---
    N = 1000
    I0 = 50
    nsims = 100
    time_max = 10.0

    # sample part of runs and plot it including average curve and save the data
    num_samples = 10
    csv_base_dir = "../data/gillespie_sims/"
    
    test_names = ["complete", "random_ER", "regular", "scale_free"]
    for test_name in test_names:
        # --- load simulation results ---
        sim_results = np.load(f'../results/gillespie_sims/{test_name}.npz', allow_pickle=True)
        X_sims = [sim_results[f'sim_{i}'] for i in range(nsims)]
        
        ## --- export to csv ---
        csv_dir = os.path.join(csv_base_dir, test_name)
        os.makedirs(csv_dir, exist_ok=True)

        # save average curve for all nsims
        avg_curve, times = get_average(X_sims, time_max, nsims, delta_t = 0.01)
        avg_curve_file_name = os.path.join(csv_dir, f"average_curve_{nsims}.csv")
        df_average = pd.DataFrame({
            'time': times,
            'avg_infected_k': avg_curve # avg number of infected
        })
        df_average.to_csv(avg_curve_file_name, index=False, header=True)

        ## --- sample some number of X_sims and save ---
        X_sims_sampled = random.sample(X_sims, num_samples)

        # save average curve for num_samples of sims
        avg_curve, times = get_average(X_sims_sampled, time_max, num_samples, delta_t = 0.01)
        avg_curve_file_name = os.path.join(csv_dir, f"average_curve_{num_samples}.csv")
        df_average = pd.DataFrame({
            'time': times,
            'avg_infected_k': avg_curve # avg number of infected
        })
        df_average.to_csv(avg_curve_file_name, index=False, header=True)

        ## --- save the sample of X_sims ---
        csv_header = ["t", "waiting_time", "total_infected", "event_type", "total_pw_count", "total_ho_count"]
        for i, X_t_s in enumerate(X_sims_sampled):
            data_for_df = []
            for row_idx in range(X_t_s.shape[0]):
                current_row = X_t_s[row_idx, :]
                if row_idx == 3:
                    processed_row = ["" if x is None else x for x in current_row]
                else:
                    try:
                        temp_row = np.array(current_row, dtype=float)
                        processed_row = temp_row
                    except ValueError:
                        processed_row = [np.nan if x is None else x for x in current_row]
                data_for_df.append(processed_row)

            df_sampled_run = pd.DataFrame(dict(zip(csv_header, data_for_df)))
            df_sampled_run.head()

            sample_number = i + 1
            sampled_run_csv_filename = os.path.join(csv_dir, f"continuous_observations_{sample_number}.csv")
            df_sampled_run.to_csv(sampled_run_csv_filename, index=False, header=True)
        
        print(f"Saved {num_samples} X_sims, average_curve_100 and average_curve_10 to {csv_base_dir}.")

