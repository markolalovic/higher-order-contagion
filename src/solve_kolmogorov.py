"""
Solving system of forward Kolmogorov equations on 
  * general
  * complete 
  * and other special types of hypergraphs
"""

import numpy as np
import pandas as pd
from itertools import combinations

from Hypergraphs import CompleteHypergraph

import simulate_gillespie
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

treat_as_general = False # TODO: this is for testing only

def list_all_states(g):
    r"""Returns `all_states` = (ordered) list of states Markov chain X(t) can be in,
    given hypergraph g with number of nodes = `N`.

      * If g is symmetric type of hypergraph (e.g. complete) where rates depend only on 
    the number k of infected nodes,
          all_states = {k, where k = 0, 1, ..., N}.
    
      * Else, for a general hypergraph g, 
          all_states = {subset K_i, where i = 1, ..., M, and M = 2^N}.
    """
    N = g.number_of_nodes()
    all_states = []
    if type(g) == CompleteHypergraph and not treat_as_general:
        all_states = list(range(N + 1))
    else:
        # treat_as_general
        all_states = []
        for k in range(N + 1):
            for set_K in combinations(range(N), k):
                all_states.append(frozenset(set_K))
    return all_states

def total_SI_pairs_and_SII_triples(g, current_state):
    r"""Given a hypergraph `g` and current state, it returns:
        s1 = total SI pairs
        s2 = total SII triples

      * If g is symmetric type, current state is number of infected nodes k, and:
        s1 = s_{k}^{(1)} = k (N - k)
        s2 = s_{k}^{(2)} = \binom{k}{2} (N - k)

      * Else, for a general hypergraph g, current state is infected set `set_K`, and:
        s1 = S_{K}^{(1)} = count depending on the structure of g 
        s2 = S_{K}^{(2)} = count depending on the structure of g
    
    """
    s1, s2 = 0, 0
    if type(g) == CompleteHypergraph and not treat_as_general:
        k = current_state
        N = g.number_of_nodes()
        s1, s2 = k * (N - k), (1/2) * k * (k - 1) * (N - k)
    else:
        # infect the nodes from current_state = set_K 
        states = np.zeros(g.number_of_nodes(), dtype=int)
        states[list(current_state)] = 1
        for i, state in enumerate(states):
            g.nodes[i]["state"] = state

        # calculate s1, s2 given infected nodes
        for node_i in range(g.number_of_nodes()):
            if g.nodes[node_i]["state"] == 0:
                # for pairwise infection, num(nbs(i)) * beta1
                nbs_pw = g.neighbors(node_i, 1)
                inf_pw = sum(1 for node_j in nbs_pw \
                                if g.nodes[node_j]["state"] == 1)

                # for hyperedge HO infection, only if both neighbors are infected
                nbs_ho = g.neighbors(node_i, 2)
                inf_ho = sum(1 for node_j, node_k in nbs_ho if \
                                g.nodes[node_j]["state"] == 1 and \
                                g.nodes[node_k]["state"] == 1)
                s1 += inf_pw
                s2 += inf_ho
    # casting to integers since totals s1, s2 are counts
    return int(s1), int(s2)

def create_generator_matrix(g, beta1, beta2, mu):
    N = g.number_of_nodes()
    all_states = list_all_states(g)
    M = len(all_states)
    
    set_V = frozenset(list(range(N)))
    set_0 = frozenset([])

    # state index as a dictionary, keys are frozensets K from powerset(V)
    state_index = {st: i for i, st in enumerate(all_states)}

    # precompute (cache it as a dictionary to speed things up): 
    # {state: (s1, s2) = total_SI_pairs_and_SII_triples(g, set_K)}
    # for each state = subset K
    s12_cache = {}
    for st in all_states:
        s1_, s2_ = total_SI_pairs_and_SII_triples(g, st)
        s12_cache[st] = (s1_, s2_)

    # fill the generator matrix A
    # TODO: make it sparse for performance!
    matrixA = np.zeros((M, M), dtype=float)

    # for all (j, K); e.g. (j, K) = (5, frozenset({0, 1}))
    # compute "outrate", "inrate"
    for j, set_K in enumerate(all_states):
        # handle set_0 boundary state
        if set_K == set_0:
            # only recovery from singleton sets {v} are possible
            # TODO: duplicated code ~ see @# sum 2:            
            for v in range(N):
                set_K_plus_v = frozenset(set_K.union({v}))
                i = state_index[set_K_plus_v]
                matrixA[j, i] += mu

        # handle set_V boundary state
        elif set_K == set_V:
            # only infections from sets V \ {v}, infecting the remaining v are possible
            # TODO: duplicated code ~ see @# sum 1:
            for v in set_K:
                set_K_minus_v = frozenset(set_K - {v})
                i = state_index[set_K_minus_v]
                s1M, s2M = s12_cache[set_K_minus_v]
                infection_rate = beta1 * s1M + beta2 * s2M
                matrixA[j, i] += infection_rate
            # or no event, that is none of N infected nodes recovers:
            matrixA[j, i] -= (N * mu)
        
        # handle set_K is proper subset of set_V, i.e. inner states
        else:
            # I. sum: infection from smaller set to larger: K \ {v} to K
            for v in set_K:
                set_K_minus_v = frozenset(set_K - {v})
                i = state_index[set_K_minus_v]
                s1M, s2M = s12_cache[set_K_minus_v]
                infection_rate = beta1 * s1M + beta2 * s2M
                matrixA[j, i] += infection_rate
            
            # II. sum: recovery from larger set K \cup {v} to smaller set K
            for v in range(N):
                if v not in set_K:
                    set_K_plus_v = frozenset(set_K.union({v}))
                    i = state_index[set_K_plus_v]
                    size_K_plus_v = len(set_K_plus_v)
                    matrixA[j, i] += (mu * size_K_plus_v)

            # III. last term: no event
            s1K, s2K = s12_cache[set_K]
            outflow_rate = beta1 * s1K + beta2 * s2K + mu * len(set_K)
            matrixA[j, i] -= outflow_rate
    return matrixA

def list_all_ODEs(g, beta1, beta2, mu):
    r"""Returns the list of forward Kolmogorov equations dp_{K}(t)/dt = ... 
        for all subsets K of V(H).
        This is the ODE system, a function that given p_{K}(t) as a vector of length M = 2^N,
        returns dp / dt according to derived fwd KEs.

        Indexing is as follows: If `all_states[i] = K` then p[i] = p_{K}(t), 
        that is the probability of being in state (subset) K at time t.
    """
    N = g.number_of_nodes()
    all_states = list_all_states(g)
    M = len(all_states)
    
    set_V = frozenset(list(range(N)))
    set_0 = frozenset([])

    # state index as a dictionary, keys are frozensets K from powerset(V)
    state_index = {set_K_: idx for idx, set_K_ in enumerate(all_states)}

    # precompute (cache it as a dictionary to speed things up): 
    # {state: (s1, s2) = total_SI_pairs_and_SII_triples(g, set_K)}
    # for each state = subset K
    s12_cache = {}
    for set_K_ in all_states:
        s1_, s2_ = total_SI_pairs_and_SII_triples(g, set_K_)
        s12_cache[set_K_] = (s1_, s2_)
        
    def ode_system(t, p):
        r"""Given p = p(t) a vector of length M, it returns dp / dt where:
            p[i] = p_{K}(t) for subset K of V and `all_states[i] = K`.
        """
        dpdt = np.zeros(M, dtype=float)
        for i, set_K in enumerate(all_states):
            # handle set_0 boundary state
            if set_K == set_0:
                # only recovery from singleton sets {v} are possible
                # TODO: duplicated code ~ see @# II. sum:            
                for v in range(N):
                    set_K_plus_v = frozenset(set_K.union({v}))
                    idx_plus = state_index[set_K_plus_v]
                    dpdt[i] += mu * p[idx_plus]

            # handle set_V boundary state
            elif set_K == set_V:
                # only infections from sets V \ {v}, infecting the remaining v are possible
                # TODO: duplicated code ~ see @# I. sum:
                for v in set_K:
                    set_K_minus_v = frozenset(set_K - {v})
                    idx_minus = state_index[set_K_minus_v]
                    s1M, s2M = s12_cache[set_K_minus_v]
                    infection_rate = beta1 * s1M + beta2 * s2M
                    ## TODO: exploding, dividing by |set_K| = N
                    dpdt[i] += infection_rate * p[idx_minus]
                
                # or no event, that is none of N infected nodes recovers
                # with np.errstate(subtract="ignore"):
                dpdt[i] -= (N * mu) * p[i]
            
            # handle set_K is proper subset of set_V, i.e. inner states
            else:
                # I. sum: infection from smaller set to larger: K \ {v} to K
                for v in set_K:
                    set_K_minus_v = frozenset(set_K - {v})
                    idx_minus = state_index[set_K_minus_v]
                    s1M, s2M = s12_cache[set_K_minus_v]
                    infection_rate = beta1 * s1M + beta2 * s2M
                    dpdt[i] += infection_rate * p[idx_minus]
                
                # II. sum: recovery from larger set K \cup {v} to smaller set K
                for v in range(N):
                    if v not in set_K:
                        set_K_plus_v = frozenset(set_K.union({v}))
                        idx_plus = state_index[set_K_plus_v]
                        dpdt[i] += mu * p[idx_plus]

                # III. last term: no event
                s1K, s2K = s12_cache[set_K]
                outflow_rate = beta1 * s1K + beta2 * s2K + mu * len(set_K)
                dpdt[i] -= outflow_rate * p[i]
        return dpdt
    return ode_system

def list_all_ODEs_complete(g, beta1, beta2, mu):
    r"""Returns the list of forward Kolmogorov equations dp_{k}(t)/dt = ...
        for all states k = 0, 1, ..., N.
    """
    N = g.number_of_nodes()
    all_states = list_all_states(g)
    M = len(all_states)

    # precompute (cache it as a dictionary to speed things up):
    # {state k: (s1, s2) = total_SI_pairs_and_SII_triples(g, state_k)}
    # for example if g = CompleteHypergraph(N = 5) then s12_cache is:
    # {0: (0, 0), 1: (4, 0), 2: (6, 3), 3: (6, 6), 4: (4, 6), 5: (0, 0)}
    s12_cache = {}
    for state_k_ in all_states:
        s1_, s2_ = total_SI_pairs_and_SII_triples(g, state_k_)
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

def test_on_general():    
    g = example45()
    N = g.number_of_nodes()
    beta1, beta2, mu = 1.0, 0.5, 1.0 # TODO: what rates?
    time_max = 10   # maximum time duration
    initial_infections = list([1]) # which nodes are infected at t=0

def test_on_complete():
    """Complete case test."""
    # setup
    nsims = 10
    N = 100
    beta1 = 2 / N       # pairwise infection rate
    beta2 = 4 / (N**2)  # hyperedge contagion rate
    mu    = 1           # recovery rate
    I0 = 10 # number of initial infected
    time_max = 10   # maximum time duration
    g = CompleteHypergraph(N)
    print(f"Setup: \n")
    print(f"\tH = {g.__class__.__name__}, N = {N}, I0 = {I0}\n")
    print(f"\tbeta1 = {beta1}, beta2 = {beta2}, mu = {mu}\n")

    # load the data and unpak into X_sims list of X_t lists 
    df = pd.read_csv("../data/sim_complete.csv")
    X_sims = [[list(group["time"]), list(group["time_to_event"]), 
               list(group["infected"]), list(group["event_type"]),
               list(group["total_pw"]), list(group["total_ho"])]
        for _, group in df.groupby("nsim")]
    
    # plot X_t curves in gray and average curve in red
    fig = plt.figure()
    ax  = plt.subplot()
    for X_t in X_sims:
        ax.plot(X_t[0], X_t[2], c="gray", alpha=0.5)
    avg_curve, times = simulate_gillespie.get_average(X_sims, time_max, nsims, delta_t = 0.01)
    plt.plot(times, avg_curve, "red")
    plt.xlabel("Time")
    plt.ylabel("Number of Infected")
    plt.title(f"H = {g.__class__.__name__}, N = {N}, nsims = {nsims}")
    plt.savefig("../figures/solutions-kolmogorov/complete/gillespie-sims.pdf", 
                format='pdf', bbox_inches='tight')    
    plt.show()

    # solve KEs and compare
    all_states = list_all_states(g)
    M = len(all_states)
    ode_system_complete = list_all_ODEs_complete(g, beta1, beta2, mu)

    # set the initial condition
    p0 = np.zeros(M)
    i_set0 = all_states.index(I0)
    p0[i_set0] = 1.0 # all other states have prob 0 at time 0

    # solve ode_system_complete over time
    # e.g. from 0 to 10
    t_span = (0.0, 10.0)

    # times t_i to evaluate in, get saved in sol.t
    t_eval = np.linspace(t_span[0], t_span[1], 101)
    
    def f_ode(t, p):
        return ode_system_complete(t, p)
    
    sol = solve_ivp(f_ode, 
                    t_span, 
                    p0, 
                    t_eval=t_eval,
                    method="LSODA")

    # plot the results: probability of each state over time
    plt.figure()
    for k in all_states:
        # plt.scatter(sol.t, sol.y[k], s = 1, label=str(k))
        plt.scatter(sol.t, sol.y[k], s = 1)
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Probability $p_{k}(t)$")
    # plt.legend()
    plt.title(f"H = {g.__class__.__name__}, N = {N}")
    plt.savefig("../figures/solutions-kolmogorov/complete/probabilities.pdf", 
            format='pdf', bbox_inches='tight')
    plt.show()

    # plot the expected values of p_{k}(t) over time t
    expected_values = calculate_expected_values(sol)
    plt.figure()
    plt.scatter(sol.t, expected_values, s = 10, color="k", 
                label=r"Expected values $E[p_{k}(t)]$")
    plt.xlabel("Time t")
    plt.ylabel(r"$E[p_{k}(t)]$")
    # plt.legend()
    plt.title(f"H = {g.__class__.__name__}, N = {N}")
    plt.savefig("../figures/solutions-kolmogorov/complete/expected-values.pdf", 
                format='pdf', bbox_inches='tight')       
    plt.show()

    # plot both the expected values of p_{k}(t) 
    # and the Gillespie average curve
    # on the same figure
    plt.figure()
    plt.scatter(sol.t, expected_values, s = 10, color="k", 
                label=r"Expected values $E[p_{k}(t)]$")
    plt.plot(sol.t, expected_values, color="k",
             label="Solution of KEs")
    plt.plot(times, avg_curve, 'red', label="Gillespie average curve")
    plt.xlabel("Time")
    plt.ylabel("Number of Infected")
    plt.legend()
    plt.grid(True)
    plt.title(f"H = {g.__class__.__name__}, N = {N}, nsims = {nsims}")
    plt.savefig("../figures/solutions-kolmogorov/complete/solution-vs-gillespie-avg.pdf", 
                format='pdf', bbox_inches='tight') 
    plt.show()

if __name__ == "__main__":
    test_on_complete()
