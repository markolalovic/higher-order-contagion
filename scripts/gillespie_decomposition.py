# gillespie_decomposition.py

import numpy as np
import matplotlib.pylab as plt

import os
import sys
sys.path.append('../src/')
sys.path.append('../scripts/')

from simulate_gillespie import gillespie_sim_complete, get_average

rn = np.random.randint(100, size=1)[0]
rng = np.random.default_rng(rn)

def gillespie_sim_complete_decomposed(N, beta1, beta2, mu, I0, time_max):
    """
    Gillespie for complete SC adapted to track the origin PW / HO: k_pw and k_ho.
    Returns:  
      [time, waiting_time, k_total, event_type, 
      k_pw_after_event, k_ho_after_event,              <- new returns
      total_pw_struct_count, total_ho_struct_count]
    """
    k_total = I0

    # k_pw = I0 # assuming intital infections are "from pairwise"
    k_pw = 0 # assuming initial infections are from "unspecified"

    k_ho = 0
    t = 0.0

    # store initial state
    num_s_init = N - k_total
    pw_struct_init = k_total * num_s_init
    ho_struct_init = 0.5 * k_total * (k_total - 1) * num_s_init if k_total >= 2 else 0
    
    history = [[t, None, k_total, None, k_pw, k_ho, pw_struct_init, ho_struct_init]]

    while t < time_max:
        if k_total == 0:
            break

        num_s = N - k_total
        rate_a = beta1 * k_total * num_s
        rate_b = beta2 * 0.5 * k_total * (k_total - 1) * num_s if k_total >= 2 else 0.0
        rate_c = mu * k_total
        total_event_rate = rate_a + rate_b + rate_c

        if total_event_rate <= 1e-15:
            break

        waiting_time = -np.log(rng.random()) / total_event_rate
        t_next = t + waiting_time

        if t_next >= time_max:
            break

        # determine event type
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
            # now we need to decide from which pool to recover from
            if k_pw > 0 and k_ho > 0:
                rand_recovery_draw = rng.random()
                # either from PW pool
                if rand_recovery_draw < k_pw / k_total:
                    k_pw -= 1
                else:
                    # recover from HO pool
                    k_ho -= 1
            # else:
            # if k_total > 0: # this is always true
            #     k_total -= 1
            k_total -= 1
        
        t = t_next

        # calculate structural counts for the new state
        num_s_new = N - k_total
        pw_struct_now = k_total * num_s_new
        ho_struct_now = 0.5 * k_total * (k_total - 1) * num_s_new if k_total >= 2 else 0.0

        history.append([t, waiting_time, k_total, event_type, k_pw, k_ho, pw_struct_now, ho_struct_now])

    # final record at time_max
    num_s_final = N - k_total
    pw_struct_final = k_total * num_s_final
    ho_struct_final = 0.5 * k_total * (k_total - 1) * num_s_final if k_total >= 2 else 0.0
    history.append([time_max, None, k_total, None, k_pw, k_ho, pw_struct_final, ho_struct_final])
    return np.array(history, dtype=object).transpose()


def get_average_decomposed(X_sims_list, t_max, num_sims, delta_t=0.01, selected_indices=[2]):
    """
    Avarage adapted to for decomposed Gillespie: origin PW / HO: k_pw and k_ho, average:
      * k_total (index = 2)
      * k_pw (index = 4)
      * k_ho (index = 5)
    
    Calculates average for specified `selected_indices` columns over `X_sims_list`.
    Returns: 
      - list of average curves
      - and a single time array
    """
    times_grid = np.arange(0, t_max + delta_t, delta_t)
    avg_curves_list = [np.zeros(len(times_grid)) for _ in selected_indices]

    for X_t_data in X_sims_list:
        sim_times = X_t_data[0, :].astype(float)
        
        for curve_idx, data_col_idx in enumerate(selected_indices):
            sim_values = X_t_data[data_col_idx, :].astype(float)
            
            # interpolate onto the common time grid
            interp_func = np.interp(times_grid, sim_times, sim_values, left=sim_values[0], right=sim_values[-1])
            avg_curves_list[curve_idx] += interp_func
            
    for curve_idx in range(len(avg_curves_list)):
        if num_sims > 0:
            avg_curves_list[curve_idx] /= num_sims
            
    return avg_curves_list, times_grid




if __name__ == "__main__":
    # --- Setup - as for MF_limit_ODE_decomposition ---
    N = 1000
    I0 = 50

    time_max = 10.0
    beta1_scaled = 2.0
    beta2_scaled = 6.0
    mu = 1.0

    beta1 = beta1_scaled / N
    beta2 = beta2_scaled / (N**2)

    nsims = 1000 # TODO: increased for smoother averages

    # --- run gillespie_sim_complete_decomposed ---
    X_sims_decomposed = []
    print("Running simulations...")
    for i in range(nsims):
        if (i+1) % (nsims//10 if nsims >=10 else 1) == 0:
            print(f"Run {i + 1} / {nsims}")
        X_t = gillespie_sim_complete_decomposed(N, beta1, beta2, mu, I0, time_max)
        X_sims_decomposed.append(X_t)
    print("Done.")


    # --- call get_average_decomposed ----
    print("Calculating averages...")
    # indices are [2, 4, 5]  for  [k_total, k_pw, k_ho]
    avg_curves_list, common_times = get_average_decomposed(
        X_sims_decomposed, time_max, nsims, delta_t=0.01, selected_indices = [2, 4, 5])
    print("Done.")

    # --- correctness check ---
    avg_k_total = avg_curves_list[0]
    avg_k_pw = avg_curves_list[1]
    avg_k_ho = avg_curves_list[2]

    max_diff = np.max(np.abs(avg_k_total - (avg_k_pw + avg_k_ho)))
    print(f"Max absolute difference between avg_k_total and sum avg_k_pw + avg_k_ho is: {max_diff:.2e}")
    

    # --- plot decomposed average curves ---
    # --- Plot settings ---
    plt_DPI = 200
    fig_w, fig_h = 8, 5  # 8:5 ratio goes well with beamer
    plt_legend_fontsize = 16
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14  # for tick labels

    plt_linewidth_total = 3  # thicker for visibility
    plt_linewidth_pw = 2.5       
    plt_linewidth_ho = 3     # dotted line needs to be thicker


    # -----------------
    # ---- Plot it ----
    # -----------------
    # TODO: adjusted the size slightly for slides    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)

    # Plot y(t) = Total Infected
    ax.plot(common_times, avg_k_total, color='black', linestyle='-', linewidth=plt_linewidth_total,
            label=r'$y(t)$ (Total)', zorder=3)
    
    # Plot p(t) = Pairwise Contribution (Red Dashed)
    ax.plot(common_times, avg_k_pw, color='red', linestyle='--', linewidth=plt_linewidth_pw,
            label=r'$p(t)$ (Pairwise)', zorder=2)
    
    # Plot h(t) = Higher-Order Contribution (Blue Dotted)
    ax.plot(common_times, avg_k_ho, color='blue', linestyle=':', linewidth=plt_linewidth_ho,
            label=r'$h(t)$ (Higher-Order)', zorder=2)
    
    ax.set_xlabel("Time (t)", fontsize=plt_labels_fontsize)
    ax.set_ylabel("Number of Infected (k)", fontsize=plt_labels_fontsize)

    # presentation-friendly plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # adjusting legend for better visibility
    ax.legend(fontsize=plt_legend_fontsize, loc='lower right', 
            frameon=True, fancybox=True, shadow=False, 
            framealpha=0.9, edgecolor='gray')
    
    # ax.legend(fontsize=plt_legend_fontsize, loc='lower right')
    # ax.grid(True, linestyle=':', alpha=0.5) # TODO: grid to see the crossover at 4.1
    ax.set_ylim(bottom=0, top=max(0.01, np.max(avg_k_total) * 1.05))
    ax.set_xlim(left=0, right=time_max)

    # TODO: ensure ticks are visible
    ax.tick_params(axis='both', which='major', labelsize=plt_tick_fontsize)

    plt.tight_layout()    

    # plt.show()
    
    # -----------------------
    # --- save the figure ---
    # -----------------------    
    output_filename = "../figures/combined/gillespie_decomposition.pdf"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")


'''
(.venv) marko@mb scripts % python3 gillespie_decomposition.py
Running simulations...
Run 100 / 1000
Run 200 / 1000
Run 300 / 1000
Run 400 / 1000
Run 500 / 1000
Run 600 / 1000
Run 700 / 1000
Run 800 / 1000
Run 900 / 1000
Run 1000 / 1000
Done.
Calculating averages...
Done.
Max absolute difference between avg_k_total and sum avg_k_pw + avg_k_ho is: 5.00e+01
Plot saved to ../figures/combined/gillespie_decomposition.pdf
'''