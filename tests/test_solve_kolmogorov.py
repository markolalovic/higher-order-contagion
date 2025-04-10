"""
Tests of solving system of forward Kolmogorov equations on:

  * complete
  * general 

"""

import numpy as np
import pandas as pd

from itertools import combinations, chain
from scipy.integrate import solve_ivp

import sys
sys.path.append('../src/')
import Hypergraphs
import solve_kolmogorov
import simulate_gillespie



if __name__ == "__main__":
    ###
    ## Examples
    #
    print("Example complete case")
    N = 5
    g = Hypergraphs.CompleteHypergraph(N)
    print(solve_kolmogorov.list_all_states(g))
    #> [0, 1, 2, 3, 4, 5]
    print()
    
    print("Example general case")
    g = Hypergraphs.example45()
    for state_K in solve_kolmogorov.list_all_states(g):
        print(list(state_K), end=", ")
    #> [], [0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], \
    #> [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]    
    print()

    ### 
    ## Tests
    #
    print("Test complete case")
    N = 5
    g = Hypergraphs.CompleteHypergraph(N)
    all_states = solve_kolmogorov.list_all_states(g)
    s12_cache = {}
    for state_k_ in all_states:
        s1_, s2_ = solve_kolmogorov.total_SI_pairs_and_SII_triples(g, state_k_)
        s12_cache[state_k_] = (s1_, s2_)
    print(s12_cache)
    #> {0: (0, 0), 1: (4, 0), 2: (6, 3), 3: (6, 6), 4: (4, 6), 5: (0, 0)}
    print()

    print("Test general case")
    g = Hypergraphs.example45()
    states_of_size_2 = list(combinations(g.nodes, 2))
    # > [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for state_of_size_2 in states_of_size_2:
        set_K = frozenset(state_of_size_2)
        s1, s2 = solve_kolmogorov.total_SI_pairs_and_SII_triples(g, set_K)
        print(f"set_K {state_of_size_2}: {s1}, {s2}")
    #> set_K (0, 1): 2, 0
    #> set_K (0, 2): 3, 0
    #> set_K (0, 3): 3, 0
    #> set_K (1, 2): 3, 1
    #> set_K (1, 3): 3, 1
    #> set_K (2, 3): 2, 1    
    print()

# TODO: rewrite it as tests
def test_list_all_states_complete():
    N = 5
    g = Hypergraphs.CompleteHypergraph(N)
    assert solve_kolmogorov.list_all_states(g) == [0, 1, 2, 3, 4, 5]

def test_list_all_states_general():
    g = Hypergraphs.example45()
    nodes = list(g.nodes)

    # chain together combinations of all sizes
    all_combinations_iter = chain.from_iterable(
        combinations(nodes, i) for i in range(len(nodes) + 1))
    # convert to frozensets
    expected_states = [frozenset(combo) for combo in all_combinations_iter]
    print(expected_states)
    #> [], [0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], \
    #> [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]       

    actual_states = list(solve_kolmogorov.list_all_states(g))
    assert len(actual_states) == len(expected_states)
    assert set(actual_states) == set(expected_states)

def test_total_pairs_triples_complete():
    N = 5
    g = Hypergraphs.CompleteHypergraph(N)
    all_states = solve_kolmogorov.list_all_states(g)
    s12_cache = {}
    for state_k_ in all_states:
        s1_, s2_ = solve_kolmogorov.total_SI_pairs_and_SII_triples(g, state_k_)
        s12_cache[state_k_] = (s1_, s2_)
    #> {0: (0, 0), 1: (4, 0), 2: (6, 3), 3: (6, 6), 4: (4, 6), 5: (0, 0)}
    ###
    expected_cache = {0: (0, 0), 1: (4, 0), 2: (6, 3), 3: (6, 6), 4: (4, 6), 5: (0, 0)}
    assert s12_cache == expected_cache

def test_total_pairs_triples_general():
    g = Hypergraphs.example45()
    states_of_size_2 = list(combinations(g.nodes, 2))
    expected_results = {
        (0, 1): (2, 0), (0, 2): (3, 0), (0, 3): (3, 0),
        (1, 2): (3, 1), (1, 3): (3, 1), (2, 3): (2, 1)
    }
    #> set_K (0, 1): 2, 0
    #> set_K (0, 2): 3, 0
    #> set_K (0, 3): 3, 0
    #> set_K (1, 2): 3, 1
    #> set_K (1, 3): 3, 1
    #> set_K (2, 3): 2, 1     
    actual_results = {}
    for state_tuple in states_of_size_2:
        set_K = frozenset(state_tuple)
        s1, s2 = solve_kolmogorov.total_SI_pairs_and_SII_triples(g, set_K)
        actual_results[state_tuple] = (s1, s2)

    assert actual_results == expected_results    