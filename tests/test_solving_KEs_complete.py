""" Isolated solving KEs for complete hypergraphs only. """

import numpy as np
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
    # setup
    nsims = 10 # number of simulation runs
    time_max = 10   # maximum time duration

    # TODO: increase these values
    N = 4 
    I0 = 1 
    t_max = 10

    beta1 = 2 / N       # pairwise infection rate
    beta2 = 4 / (N**2)  # hyperedge contagion rate
    mu    = 1           # recovery rate
    print(f"Setup: \n")
    print(f"\tH = Complete Hypergraph, N = {N}, I0 = {I0}\n")
    print(f"\tbeta1 = {beta1}, beta2 = {beta2}, mu = {mu}\n")

    M = N + 1 # number of all states
    # set the initial condition
    p0 = np.zeros(M)
    p0[I0] = 1.0 # all other states have prob 0 at time 0
    print(f"p0 = {p0}")

    # time range and times to evaluate solution
    nsteps = 101
    t_span = (0.0, time_max)
    t_eval = np.linspace(t_span[0], t_span[1], nsteps) 

    # Solve KEs
    ode_system_complete = list_all_ODEs_complete(N, beta1, beta2, mu)

    def f_ode(t, p):
        return ode_system_complete(t, p)

    sol = solve_ivp(f_ode, 
                    t_span, 
                    p0, 
                    t_eval=t_eval,
                    method="RK45"
    )

    # plot the expected values of p_{k}(t) over time t
    expected_values = calculate_expected_values(sol)
    plt.figure()
    plt.scatter(sol.t, expected_values, s = 10, color="k", 
                label=r"Expected values $E[p_{k}(t)]$")
    plt.xlabel("Time t")
    plt.ylabel(r"$E[p_{k}(t)]$")
    # plt.legend()
    plt.title(f"H = Complete Hypergraph, N = {N}")
    plt.savefig("../figures/solutions-kolmogorov/debug/bump-behavior.pdf", format='pdf', bbox_inches='tight')
    print("figure saved")    
    plt.show()
