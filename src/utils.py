import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter # for counting degrees

from scipy.stats import zipf # for power-law Zeta distribution
from scipy.stats import powerlaw # more direct power-law draw
from scipy.optimize import curve_fit # for plotting fits

def test_generate_sf_sc(pw_degrees_sc, ho_degrees_sc, kgi_generated,
                        N, gamma_sc, m_sc, figure_fname=None):
    # TODO: plot_degree_distribution_ScaleFreeSC

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    title = f"Degree Distributions: ScaleFreeSC with N = {N}, target gamma_ho = {gamma_sc}, target m_ho = {m_sc}"
    fig.suptitle(title, fontsize=16)

    # -------------
    # PW degrees
    ax = axes[0]

    realized_avg_pw = np.mean(pw_degrees_sc)
    realized_max_pw = np.max(pw_degrees_sc)
    min_val_pw = np.min(pw_degrees_sc)

    bins_pw = np.logspace(np.log10(min_val_pw), np.log10(realized_max_pw + 1), 25)
    counts_pw, _ = np.histogram(pw_degrees_sc, bins=bins_pw, density=True)
    bin_centers_pw = (bins_pw[:-1] + bins_pw[1:]) / 2
    valid_pw = counts_pw > 0

    ax.loglog(bin_centers_pw[valid_pw], counts_pw[valid_pw], 'x', color='red',
                markersize=7, alpha=0.9, label='Realized PW Degrees')

    # power-law fit for realized PW degrees

    def power_law_func_log(log_k, log_C, gamma_fit_param): return log_C - gamma_fit_param * log_k

    # some reasonable initial guess for gamma_pw_fit: gamma_sc or 2.5
    popt_pw, _ = curve_fit(power_law_func_log, 
                        np.log10(bin_centers_pw[valid_pw]), 
                        np.log10(counts_pw[valid_pw]), 
                        p0=[0, gamma_sc], 
                        maxfev=3000)

    gamma_pw_fit_val = popt_pw[1]
    k_fit_range = np.logspace(np.log10(bin_centers_pw[valid_pw][0]), np.log10(bin_centers_pw[valid_pw][-1]), 50)

    # theoretical fit as solid black line
    ax.plot(k_fit_range, (10**popt_pw[0]) * k_fit_range**(-gamma_pw_fit_val), 'k-', alpha=0.8,
                label=f'Fit: P(k_pw) ~ k_pw^{{-{gamma_pw_fit_val:.2f}}}')


    ax.set_title(f'PW Degrees, Realized Avg = {realized_avg_pw:.2f}, Realized Max = {realized_max_pw}')
    ax.set_xlabel('Pairwise Degree k_pw')
    ax.set_ylabel('P(k_pw)')
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.7)

    # -------------
    # HO degrees
    ax = axes[1]
    realized_avg_ho = 0
    realized_max_ho = 0

    realized_avg_ho = np.mean(ho_degrees_sc)
    realized_max_ho = np.max(ho_degrees_sc)
    min_val_ho = np.min(ho_degrees_sc)

    bins_ho_actual = np.logspace(np.log10(min_val_ho), np.log10(realized_max_ho + 1), 25)
    counts_ho_actual, _ = np.histogram(ho_degrees_sc, bins=bins_ho_actual, density=True)
    bin_centers_ho_actual = (bins_ho_actual[:-1] + bins_ho_actual[1:]) / 2
    valid_ho_actual = counts_ho_actual > 0

    # realized HO degrees as red crosses
    ax.loglog(bin_centers_ho_actual[valid_ho_actual], counts_ho_actual[valid_ho_actual], 'x', color='red',
                markersize=7, alpha=0.9, label='Realized HO Degrees (k_ho)')

    # plot the target kgi distribution, sequence of HO degrees

    min_val_kgi = np.min(kgi_generated[kgi_generated > 0]) if np.any(kgi_generated > 0) else 1
    max_val_kgi = np.max(kgi_generated)

    bins_kgi = np.logspace(np.log10(min_val_kgi), np.log10(max_val_kgi + 1), 25)
    counts_kgi, _ = np.histogram(kgi_generated, bins=bins_kgi, density=True)
    bin_centers_kgi = (bins_kgi[:-1] + bins_kgi[1:]) / 2
    valid_kgi = counts_kgi > 0

    # target generalized degrees as blue dots
    ax.loglog(bin_centers_kgi[valid_kgi], counts_kgi[valid_kgi], 'o', color='blue', markersize=5,
                alpha=0.7, label='Target Gen. Degree (k_gi) Dist.')

    # theoretical line P(k) ~ k^-gamma_sc for k_gi
    k_plot = np.logspace(np.log10(max(m_sc,1 if min_val_kgi==0 else min_val_kgi)), 
                        np.log10(max_val_kgi if max_val_kgi > m_sc else m_sc +1.01), 50)

    idx_min_kgi_bin = np.where(bin_centers_kgi[valid_kgi] >= m_sc)[0]

    k_min_plot_for_norm = bin_centers_kgi[valid_kgi][idx_min_kgi_bin[0]]
    P_k_min_plot = counts_kgi[valid_kgi][idx_min_kgi_bin[0]]
    C_norm = P_k_min_plot * (k_min_plot_for_norm**(gamma_sc))

    # theoretical fit as solid black line
    ax.loglog(k_plot, C_norm * (k_plot**(-gamma_sc)),
                'k-', alpha=0.8, label=f'Target P(k_gi) ~ k_gi^{{-{gamma_sc:.2f}}}')

    ax.set_title(f'HO Degrees, Realized Avg = {realized_avg_ho:.2f}, Realized Max = {realized_max_ho}')
    ax.set_xlabel('Degree (k_ho or k_gi)')
    ax.set_ylabel('P(degree)')
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    if figure_fname != None:
        save_dir = "../figures/higher_order_structures/"
        save_path = os.path.join(save_dir, f"{figure_fname}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()
    
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

def save_hypergraph(g, file_path):
    r"""
    Saves a hypergraph object to a file using pickle, 
    e.g. to: `../data/random_graph.pkl`
    """
    with open(file_path, "wb") as f:
        pickle.dump(g, f)

def load_hypergraph(file_path):
    r"""
    Loads a hypergraph object to a file using pickle,
    e.g. to: `../data/random_graph.pkl`
    """
    with open(file_path, "rb") as f:
        g = pickle.load(f)
    return g

def load_c_output_edges(filename):
    r"""
    Loads edges of SF-SC generated by `../tests/SC_d2_mod`
    as a unique list of sorted tuples. 
    """
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append(tuple(sorted((u, v))))
    return list(set(edges))

def load_c_output_triangles(filename):
    r"""
    Loads triangles of SF-SC generated by `../tests/SC_d2_mod`
    as a unique list of sorted tuples. 
    """
    triangles = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                u, v, w = int(parts[0]), int(parts[1]), int(parts[2])
                triangles.append(tuple(sorted((u, v, w))))
    return list(set(triangles))