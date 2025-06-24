# decompositions_comparison.py

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt
import os

# copy paste from gillespie_decomposition.py
def gillespie_sim_complete_decomposed(N, beta1, beta2, mu, I0, time_max, rng):
    """
    Gillespie for complete SC adapted to track the origin PW / HO: k_pw and k_ho.
    """
    # initial conditions: half half
    k_total = I0

    k_pw = int(I0 / 2)
    k_ho = int(I0 / 2)

    t = 0.0
    history = [[t, None, k_total, None, k_pw, k_ho]]

    while t < time_max:
        if k_total == 0: break
        num_s = N - k_total
        rate_a = beta1 * k_total * num_s
        rate_b = beta2 * 0.5 * k_total * (k_total - 1) * num_s if k_total >= 2 else 0.0
        rate_c = mu * k_total
        total_event_rate = rate_a + rate_b + rate_c
        if total_event_rate <= 1e-15: break
        waiting_time = -np.log(rng.random()) / total_event_rate
        t_next = t + waiting_time
        if t_next >= time_max: break
        rand_event_draw = rng.random() * total_event_rate
        event_type = None

        if rand_event_draw < rate_a:
            event_type = "PW"
            k_total += 1
            k_pw += 1
        elif rand_event_draw < rate_a + rate_b:
            event_type = "HO"
            k_total += 1
            k_ho += 1
        else:
            event_type = "RC"
            k_total -= 1
            if k_pw > 0 and k_ho > 0:
                if rng.random() < k_pw / (k_pw + k_ho):
                    k_pw -= 1
                else:
                    k_ho -= 1
            elif k_pw > 0:
                k_pw -= 1
            elif k_ho > 0:
                k_ho -= 1

        t = t_next
        history.append([t, waiting_time, k_total, event_type, k_pw, k_ho])

    history.append([time_max, None, k_total, None, k_pw, k_ho])
    return np.array(history, dtype=object).transpose()

def get_average_decomposed(X_sims_list, t_max, num_sims, delta_t=0.01, selected_indices=[2, 4, 5]):
    """
    Calculates average for specified columns over X_sims_list.
    """
    times_grid = np.arange(0, t_max + delta_t, delta_t)
    avg_curves_list = [np.zeros(len(times_grid)) for _ in selected_indices]
    for X_t_data in X_sims_list:
        sim_times = X_t_data[0, :].astype(float)
        for curve_idx, data_col_idx in enumerate(selected_indices):
            sim_values = X_t_data[data_col_idx, :].astype(float)
            interp_func = np.interp(times_grid, sim_times, sim_values, left=sim_values[0], right=sim_values[-1])
            avg_curves_list[curve_idx] += interp_func
    for curve_idx in range(len(avg_curves_list)):
        if num_sims > 0:
            avg_curves_list[curve_idx] /= num_sims
    return avg_curves_list, times_grid

if __name__ == "__main__":
    ## --- Setup ---
    N = 100
    I0 = 20
    time_max = 10.0
    beta1_scaled = 2.0
    beta2_scaled = 6.0
    mu = 1.0
    beta1 = beta1_scaled / N
    beta2 = beta2_scaled / (N**2)
    nsims = 1000

    # Initialize random number generator for Gillespie
    rng = np.random.default_rng(123)

    # ===================================================================
    # avg Gillespie
    # ===================================================================
    print("--- Running Gillespie Simulations ---")
    X_sims_decomposed = []
    for i in range(nsims):
        if (i + 1) % (nsims // 10 if nsims >= 10 else 1) == 0:
            print(f"  Run {i + 1} / {nsims}")
        X_t = gillespie_sim_complete_decomposed(N, beta1, beta2, mu, I0, time_max, rng)
        X_sims_decomposed.append(X_t)

    print("\n--- Calculating Gillespie Averages ---")
    # indices are [2, 4, 5] for [k_total, k_pw, k_ho]
    gillespie_avg_curves, common_times = get_average_decomposed(
        X_sims_decomposed, time_max, nsims, delta_t=0.01, selected_indices=[2, 4, 5])
    g_avg_k_total, g_avg_k_pw, g_avg_k_ho = gillespie_avg_curves
    print("Done.\n")

    # ===================================================================
    # solve system E
    # ===================================================================
    print("--- solving Kolmogorov System (E) ---")
    # --- State space mapping ---
    state_to_idx, idx_to_state = {}, []
    idx = 0
    for k in range(N + 1):
        for k1 in range(k + 1):
            k2 = k - k1
            idx_to_state.append((k1, k2))
            state_to_idx[(k1, k2)] = idx
            idx += 1
    M = len(idx_to_state)
    print(f"N={N}, I0={I0}, Total states M={M}")

    # --- constructing Q matrix ---
    print("constructing Q matrix...")
    Q = lil_matrix((M, M), dtype=np.float64)
    for i in range(M):
        k1, k2 = idx_to_state[i]
        k = k1 + k2
        if k < N:
            a_rate = beta1 * k * (N - k)
            b_rate = beta2 * (k * (k - 1) / 2) * (N - k) if k >= 2 else 0.0
            dest_idx_a = state_to_idx.get((k1 + 1, k2))
            if dest_idx_a is not None: Q[dest_idx_a, i] = a_rate
            dest_idx_b = state_to_idx.get((k1, k2 + 1))
            if dest_idx_b is not None: Q[dest_idx_b, i] = b_rate
        else:
            a_rate, b_rate = 0, 0
        rec1_rate = mu * k1
        rec2_rate = mu * k2
        if k1 > 0:
            dest_idx_r1 = state_to_idx.get((k1 - 1, k2))
            if dest_idx_r1 is not None: Q[dest_idx_r1, i] = rec1_rate
        if k2 > 0:
            dest_idx_r2 = state_to_idx.get((k1, k2 - 1))
            if dest_idx_r2 is not None: Q[dest_idx_r2, i] = rec2_rate
        Q[i, i] = -(a_rate + b_rate + rec1_rate + rec2_rate)
    Q_csc = csc_matrix(Q)

    ##
    # initial condition p(0)
    # y(0) = 50 of unspecified origin
    # let's put uniform prob dist over all (k1, k2) where k1 + k2 = 50
    p0 = np.zeros(M)
    initial_states_indices = []
    for k1_init in range(I0 + 1):
        k2_init = I0 - k1_init
        initial_states_indices.append(state_to_idx[(k1_init, k2_init)])

    # let it be uniform prob dist
    if initial_states_indices:
        p0[initial_states_indices] = 1.0 / len(initial_states_indices)

    # --- solve the ODE system ---
    print("Solving ODE system...")
    sol = solve_ivp(lambda t, p: Q_csc.dot(p), [0, time_max], p0, t_eval=common_times, method='BDF')

    # --- calculate expected values ---
    k1_vals = np.array([s[0] for s in idx_to_state])
    k2_vals = np.array([s[1] for s in idx_to_state])
    ke_E_k1 = sol.y.T @ k1_vals
    ke_E_k2 = sol.y.T @ k2_vals
    ke_E_k_total = ke_E_k1 + ke_E_k2
    print("Done.\n")

    # ===================================================================
    # Comparison Plot
    # ===================================================================
    print("--- Plotting Comparison ---")
    plt_DPI = 200
    fig_w, fig_h = 8, 5
    plt_legend_fontsize = 14
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)

    # system (E) in color
    ax.plot(sol.t, ke_E_k_total, color='black', linestyle='-', linewidth=3,
            label=r'KE: $E[k_{total}]$', zorder=5)
    ax.plot(sol.t, ke_E_k1, color='red', linestyle='--', linewidth=2.5,
            label=r'KE: $E[k_{pw}]$', zorder=4)
    ax.plot(sol.t, ke_E_k2, color='blue', linestyle=':', linewidth=3,
            label=r'KE: $E[k_{ho}]$', zorder=4)

    # Gillespie averages in gray
    ax.plot(common_times, g_avg_k_total, color='gray', linestyle='-', linewidth=5, alpha=0.9,
            label=r'Gillespie: Avg $k_{total}$', zorder=3)
    ax.plot(common_times, g_avg_k_pw, color='darkgray', linestyle='--', linewidth=2, alpha=0.9,
            label=r'Gillespie: Avg $k_{pw}$', zorder=2)
    ax.plot(common_times, g_avg_k_ho, color='lightgray', linestyle=':', linewidth=4, alpha=0.9,
            label=r'Gillespie: Avg $k_{ho}$', zorder=1)

    ax.set_xlabel("Time (t)", fontsize=plt_labels_fontsize)
    ax.set_ylabel("Number of Infected", fontsize=plt_labels_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    
    # ax.legend(fontsize=plt_legend_fontsize, loc='lower right')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=time_max)
    ax.tick_params(axis='both', which='major', labelsize=plt_tick_fontsize)
    plt.tight_layout()

    # save the figure
    output_folder = "../figures/combined"
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, "comparison_gillespie_vs_kolmogorov.pdf")
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

