
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # for counting degrees

def plot_degree_distribution_ErdosRenyiSC(g):
    N = g.N
    degrees_pw_instance = np.zeros(N, dtype=int)
    degrees_ho_instance = np.zeros(N, dtype=int)

    for i in range(N):
        degrees_pw_instance[i] = len(g.neighbors(i, 1))
        degrees_ho_instance[i] = len(g.neighbors(i, 2))

    mean_d1_realized = np.mean(degrees_pw_instance)
    mean_d2_realized = np.mean(degrees_ho_instance)
    # print(f"realized d1:  {mean_d1_realized:.2f}")
    # print(f"realized d2:  {mean_d2_realized:.2f}\n")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) # on a single figure
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    title = f"Degree distributions of single ER-SC Ver 2 instance on "
    title += f"N = {N} nodes, target d1 = {g.d1_target}, d2 = {g.d2_target})"
    fig.suptitle(title, fontsize=14)

    # PW degree distribution
    degree_counter_pw = Counter(degrees_pw_instance)
    k_vals_pw_plot = sorted(degree_counter_pw.keys())
    norm_d1_plot = np.array([degree_counter_pw[k] for k in k_vals_pw_plot]) / float(N)

    ax.plot(k_vals_pw_plot, norm_d1_plot, 'o-', label=r'Realized PW Degree',
            clip_on=True, mfc='white', color=u'#1f77b4', markersize=7, alpha=0.8)
    ax.axvline(mean_d1_realized, ymax=1,
                linewidth=1.5, linestyle='--', color=u'#1f77b4',
                label=f'Avg PW Degree = {mean_d1_realized:.2f}')

    # HO degree distribution
    degree_counter_ho = Counter(degrees_ho_instance)
    k_vals_ho_plot = sorted(degree_counter_ho.keys())
    norm_d2_plot = np.array([degree_counter_ho[k] for k in k_vals_ho_plot]) / float(N)

    # ax.plot(k_vals_ho_plot, norm_d2_plot, 'o-', label=r'Realized HO Degree',
    #         clip_on=True, mfc='white', color=u'#ff7f0e', markersize=5, alpha=0.8)
    ax.plot(k_vals_ho_plot, norm_d2_plot, marker='^', linestyle='-', label=r'Realized HO Degree',
            clip_on=True, mfc='white', color=u'#ff7f0e', markersize=10, alpha=0.8)

    ax.axvline(mean_d2_realized, ymax=1,
                linewidth=1.5, linestyle=':', color=u'#ff7f0e',
                label=f'Avg HO Degree = {mean_d2_realized:.2f}')

    # plot settings
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('Realized Degree d', size=14)
    ax.set_ylabel('Density', size=14)
    ax.set_ylim(bottom=0)

    # Determine appropriate xlim based on both distributions
    max_deg_overall = 0
    if len(degrees_pw_instance) > 0:
        max_deg_overall = max(max_deg_overall, np.max(degrees_pw_instance))
    if len(degrees_ho_instance) > 0:
        max_deg_overall = max(max_deg_overall, np.max(degrees_ho_instance))
    ax.set_xlim(left=-1, right=max_deg_overall + 1 if max_deg_overall > 0 else 10)

    ax.legend(fontsize=11, loc='upper right', handlelength=1.5, frameon=True) # frameon might look better
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plt.savefig(f"../figures/higher_order_structures/degree_distributions_ER_SC.pdf", 
                format='pdf', bbox_inches='tight')
    plt.show()
