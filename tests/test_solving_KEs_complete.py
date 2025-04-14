""" Isolated solving KEs for complete hypergraphs only. """

import numpy as np
import pickle
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp

def total_SI_pairs_and_SII_triples(N, k):
    r"""
    Given number of nodes N and current number of infected nodes k returns:  
        s1 = total SI pairs
        s2 = total SII triples
    
    Since g is complete:
        s1 = k (N - k)
        s2 = \binom{k}{2} (N - k)    
    """
    s1, s2 = k * (N - k), (1/2) * k * (k - 1) * (N - k)
    # casting to integers since totals s1, s2 are counts
    return int(s1), int(s2)

def list_all_ODEs_complete(N, beta1, beta2, mu):
    r"""Returns the list of forward Kolmogorov equations dp_{k}(t)/dt = ...
        for all states k = 0, 1, ..., N.
    """
    M = N + 1 # number of possible states

    # TODO: no need to precompute it is all O(1)
    s12_cache = {}
    for state_k_ in range(M):
        s1_, s2_ = total_SI_pairs_and_SII_triples(N, state_k_)
        s12_cache[state_k_] = (s1_, s2_)
    
    def ode_system_complete(t, p):
      r"""Given p = p(t) a vector of length M, returns dp / dt where p[i] is:

        * In case of complete hypergraph p[i] is simply p[k]:
          p[i] = p[k] = p_{k}(t) for state_k, such that: all_states[i] = state_k = k
      """
      dpdt = np.zeros(M, dtype=float)
      for k in range(M):
          if k == 0:
              dpdt[0] = mu * p[1]
          elif k == N:
              # infections from state N - 1, infecting the remaining v
              s1M, s2M = s12_cache[N - 1]
              infection_rate = beta1 * s1M + beta2 * s2M
              dpdt[N] += infection_rate * p[N - 1] # NOTE: bug fixed N - 1 instead of N!

              # or no event, that is none of N infected nodes recovers
              dpdt[N] -= (N * mu) * p[N]
          else: 
              # I. infection from state k - 1 to k
              s1M, s2M = s12_cache[k - 1]
              infection_rate = beta1 * s1M + beta2 * s2M
              dpdt[k] += infection_rate * p[k - 1]

              # II. recovery from state k + 1 to k
              dpdt[k] += mu * (k + 1) * p[k + 1]

              # III. no event
              s1K, s2K = s12_cache[k]
              outflow_rate = beta1 * s1K + beta2 * s2K + mu * k
              dpdt[k] -= outflow_rate * p[k]
      return dpdt
    return ode_system_complete

def calculate_expected_values(sol):
    r"""Given solution `sol` of forward Kolmogorov equations at times t_i in sol.t, returns:
       E[X(t_i)] = \sum_{k = 0}^{M - 1} k * p_{k}(t)
       where k = number of infected, that si |K| in general.
    """
    # t_vals = sol.t # vector of times of length ntimes
    p_vals = sol.y # matrix of shape (M x ntimes)
    M, ntimes = p_vals.shape # M = N + 1 for complete case

    expected_values = np.zeros(ntimes, dtype=float)
    for i in range(ntimes):
        expected_values[i] = np.sum([k * p_vals[k, i] for k in range(M)])
    return expected_values

if __name__ == "__main__":
    """Complete case test."""
    # setup
    # TODO: increase these values
    N = 100
    I0 = 10 
    time_max = 20

    beta1 = 2 / N       # pairwise infection rate
    beta2 = 4 / (N**2)  # hyperedge contagion rate
    mu    = 1           # recovery rate

    print(f"Setup: \n")
    print(f"\tH = Complete Hypergraph, N = {N}, I0 = {I0}\n")
    print(f"\tbeta1 = {beta1}, beta2 = {beta2}, mu = {mu}\n")

    # set mesh size
    i_max = 25
    j_max = 25

    # initialize
    k_star = np.zeros((i_max, j_max)) # to store the values k^* = E[X(t_max)]    
    eps = 1e-1 # shift for esp to not start with 0 !
    beta1_vec = (np.array(list(range(i_max))) + eps) / N
    beta2_vec = (np.array(list(range(j_max))) + eps) / (N**2)
    print(f"beta1: {beta1_vec[:5]}, ..., {beta1_vec[-3:-1]}")
    print(f"beta2: {beta2_vec[:5]}, ..., {beta2_vec[-3:-1]}")

    M = N + 1 # number of all states

    # set the initial condition
    p0 = np.zeros(M)
    p0[I0] = 1.0 # all other states have prob 0 at time 0
    print(f"p0 = {p0[:20]} ...")

    # time range and times to evaluate solution
    nsteps = 101
    t_span = (0.0, time_max)
    t_eval = np.linspace(t_span[0], t_span[1], nsteps)

    solve_for_betas = False
    # save solutions to run it once
    file_path = '../results/solutions_stationary_state_25x25.pickle'

    if solve_for_betas:
        solutions = {}
        for i, beta1 in enumerate(beta1_vec):
            for j, beta2 in enumerate(beta2_vec):
                ode_system_complete = list_all_ODEs_complete(N, beta1, beta2, mu)

                def f_ode(t, p):
                    return ode_system_complete(t, p)

                sol = solve_ivp(f_ode, 
                                t_span, 
                                p0, 
                                t_eval=t_eval,
                                method="LSODA")
                solutions[str((i, j))] = sol
        # save the solutions
        with open(file_path, "wb") as f:
            pickle.dump(solutions, f)    
    else:
        # load solutions
        with open(file_path, "rb") as f:
            solutions = pickle.load(f)
    
    # plot expected values of p_{k}(t) over time t
    k_star = np.zeros((i_max, j_max)) # to store the values k^* = E[X(t_max)]

    plt.figure()
    for i, beta1 in enumerate(beta1_vec):
        for j, beta2 in enumerate(beta2_vec):
            sol = solutions[str((i, j))]
            expected_values = calculate_expected_values(sol)
            
            k_star[i, j] = expected_values[-1]

            plt.plot(sol.t, expected_values, color="k")

    plt.xlabel("Time")
    plt.ylabel("Number of Infected")
    plt.grid(True)
    plt.savefig("../figures/solutions-kolmogorov/complete/stationary-state_25x25.pdf", 
                format='pdf', bbox_inches='tight')
    plt.show()

    B1, B2 = np.meshgrid(beta1_vec, beta2_vec)
    plt.figure()
    levels = np.linspace(k_star.min(), k_star.max(), 15) # TODO: set levels

    contourf_plot = plt.contourf(B1, B2, k_star.T, levels=levels, cmap='viridis') # use k_star.T!!
    contour_plot = plt.contour(B1, B2, k_star.T, levels=contourf_plot.levels, colors='k', linewidths=0.5)

    plt.clabel(contour_plot, inline=True, fontsize=8, fmt='%.1f')

    cbar = plt.colorbar(contourf_plot)
    cbar.set_label(r'')

    plt_title = r'Stationary state $k^{*} = E[X(t_{max})]$, '
    plt_title += f"\nwhere: time_max = {time_max}, N = {N}, I0 = {I0}, mu = {mu}"
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title(plt_title)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig("../figures/solutions-kolmogorov/complete/stationary-state-contour_25x25.pdf", 
                format='pdf', bbox_inches='tight')
    plt.show()