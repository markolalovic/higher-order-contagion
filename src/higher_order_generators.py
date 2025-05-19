"""higher_order_generators.py
Generators of simplicial complexes:

- Erdos-Renyi like simplicial complex generator:
  * Aims to achieve target average pairwise (d1) and higher-order (d2) degrees

- Regular simplicial complex: TODO
- Scale-free simplicial complex: TODO
"""

from scipy.special import comb
import numpy as np
import networkx as nx
import random
from itertools import combinations

# -----------------------------------------------------------------------------------
# Erdos-Renyi like random SC
# -----------------------------------------------------------------------------------
def generate_er_sc_components(N_target, p1_initial, p2_triangles):
    r"""
    Adapted from: 
      - Iacopini et al., 'Simplicial models of social contagion', Nat. Commun., 2019
      - https://arxiv.org/pdf/1810.07031
      - https://github.com/iaciac/simplagion/blob/master/utils_simplagion_on_RSC.py

    NOTE: we allow for the presence of 3-cliques which are not 2-simplices, 
    i.e. simplicial complexes having both "empty" and "full" triangles.

    Algorithm to generates Erdos-Renyi like simplicial complex:
      - First, creates G(N, p1_initial) random graph
      - Then goes over all possible triples and adds them as 2-simplices (triangles)
       with probability p2_triangles
        * If 2-simplex {i, j, k} is added, it ensures its faces (edges {i,j}, {j,k}, {i,k})
       are also added, NOTE: here using sets
        * So faces are added, only if they were not already in the initial G(N, p1_initial)
     
    Returns the `N_realized`, and SC components: 
      - Unique list of 1-simplices (edges)
      - Unique list of unique 2-simplices (triangles)
    """
    G = nx.fast_gnp_random_graph(N_target, p1_initial, seed=None)
    N_realized = N_target
    if not nx.is_connected(G):
        connected_components = list(nx.connected_components(G))
        if not connected_components: return 0, [], []
        # take the largest connected component
        giant_nodes_set = max(connected_components, key=len)
        G_gc = G.subgraph(giant_nodes_set).copy()
        N_realized = G_gc.order()
        # re-label nodes of the giant component to be 0 to N_realized - 1
        mapping = { old_label: new_label for new_label, old_label in enumerate(G_gc.nodes()) }
        G = nx.relabel_nodes(G_gc, mapping) # G is now the re-labeled giant component
    else:
        # ensure nodes are 0 to N-1 if G was already connected but had arbitrary labels
        if set(G.nodes()) != set(range(N_target)):
            mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

    # store initially generated edges (1-simplices)
    # use a set to ensure uniqueness 
    # and store sorted tuples
    edges_set = {tuple(sorted(edge)) for edge in G.edges()}
    triangles_set = set()
    nodes_for_triangles = list(G.nodes()) # these are 0 to N_realized - 1
    if len(nodes_for_triangles) >= 3:
        for tri_nodes_edge in combinations(nodes_for_triangles, 3):
            if random.random() < p2_triangles:
                sorted_triple = tuple(sorted(tri_nodes_edge))
                triangles_set.add(sorted_triple)
                # close the triangle by adding its faces (edges) if not already present
                # NOTE: this modifies the set of 1-simplices
                edges_set.add(tuple(sorted((sorted_triple[0], sorted_triple[1]))))
                edges_set.add(tuple(sorted((sorted_triple[0], sorted_triple[2]))))
                edges_set.add(tuple(sorted((sorted_triple[1], sorted_triple[2]))))
    return N_realized, list(edges_set), list(triangles_set)

def get_p1_p2_for_target_degrees_precise(d1, d2, N):
    r"""
    Calculates p_G for initial G ~ G(N, p_G) and p_delta (prob of adding 2-simplices)
    to achieve target average degrees: d1 and d2 more precisely.

    NOTE: 
      * `denominator_pG` being close to 0 means that $p_{\Delta}$ is close to 1 
      * That is all possible triangles are formed
      * If also `target_p1` is close to 1, That is when `d_1` is close to `N - 1` 
      * Then `p_G -> 0/0`
      * In this case `p_G` can be anything as all edges are formed by triangles anyway
    """
    p_delta = (2.0 * d2) / ((N - 1.0) * (N - 2.0)) # this is the same
    p_delta = np.clip(p_delta, 0.0, 1.0) # clip p_delta s.t. it's valid probability

    target_p1 = d1 / (N - 1.0)
    target_p1 = np.clip(target_p1, 0.0, 1.0) # clip p_1 s.t. it's valid probability
    
    # if prob_induced_by_triangles is high, then p_G can be 0
    prob_induced_by_triangles = 1.0 - (1.0 - p_delta)**(N - 2.0)

    denominator_pG = (1.0 - p_delta)**(N - 2.0) # TODO: this shouldn't be close to 0
    numerator_pG = target_p1 - prob_induced_by_triangles
    p_G = numerator_pG / denominator_pG
    p_G = np.clip(p_G, 0.0, 1.0)
    return p_G, p_delta









''' approximation
def get_p1_p2_for_target_degrees(d1, d2, N):
    r"""
    Returns estimates: 
      - p1_initial for G(N, p1_initial) 
      - p2_triangles for adding 2-simplices
    Given: 
      - d1_target: Target average pairwise degree
      - d2_target: Target average number of 2-simplices (triangles) a node is part of 
      - N_target: Target number of nodes (NOTE: N_target must be > 3)
        
    NOTE: 
      - These estimates achieve given target average degrees d1_target and d2_target.
      - This p1_initial is for initial E-R graph, before additionla edges from triangles.
    
    - Derivation of p2:
      * p2 = Pr(forming a triangle (2-simplex) from any triple of nodes)
      * Average number of triangles per node: N * d2
      * Each triangle has 3 nodes
      * Number of possible triangles: N choose 3
      * So, p2 = (N * d2 / 3) / (N choose 3) = (2 * d2) / (N - 1)(N - 2)

    - Derivation of p1: 
      * p1 = Pr(of an edge in the initial G(N, p1) graph)
      * We need to set p1 such that after triangles are added, the total average degree is d1
      * Each of d2 triangles a node is part of, contributes 2 new edges to its degree, on average
      * Maximum possible degree: N - 1
      * So, "subtracting triangle degrees": p1 = (d1 - 2 * d2) / ((N - 1) - 2 * d2)
      TODO: this is just an approximation
    """
    p2 = (2. * d2) / ((N - 1.) * (N - 2.))
    p1 = (d1 - 2. * d2) / ((N - 1.) - 2. * d2)
    if (p1 >= 0) and (p2 >= 0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')
'''