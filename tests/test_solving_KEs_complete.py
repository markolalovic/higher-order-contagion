""" Isolated solving KEs for complete hypergraphs only. """

import numpy as np
import matplotlib.pylab as plt

from scipy.integrate import solve_ivp
# from scipy.special import comb

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
              dpdt[N] += infection_rate * p[N]

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

