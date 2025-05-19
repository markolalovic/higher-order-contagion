"""higher_order_generators.py
Generators of simplicial complexes:
  - Erdos-Renyi like simplicial complex
  - Regular simplicial complex TODO
  - Scale free simplicial complex TODO
"""

from higher_order_structures import HigherOrderStructure
from scipy.special import comb
import numpy as np

import networkx as nx
import random
from itertools import combinations

# -----------------------------------------------------------------------------------
# Erdos-Renyi like random SC
# -----------------------------------------------------------------------------------
def generate_my_simplicial_complex_d2(N,p1,p2):
    r"""
    -----------------------------------------------------------------------------------
    Adapted from: Iacopini et al.
    'Simplicial models of social contagion' Nat. Commun. (2019)  
    https://arxiv.org/pdf/1810.07031
    https://github.com/iaciac/simplagion/blob/master/utils_simplagion_on_RSC.py 

    Note: we allow for the presence of 3-cliques which are not 2-simplices, 
    i.e. simplicial complexes having both "empty" and "full" triangles.
    -----------------------------------------------------------------------------------
    """
    # """Our model"""
    
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    N_realized = N
    if not nx.is_connected(G):
        giant = list(nx.connected_components(G))[0]
        G = nx.subgraph(G, giant)
        print('not connected, but GC has order %i ans size %i'%(len(giant), G.size())) 
        max_component = max(nx.connected_components(G), key=len)
        max_component = G.subgraph(max_component).copy()        
        N_realized = max_component.order()

    triangles_list = []
    G_copy = G.copy()
    
    #Now I run over all the possible combinations of three elements:
    for tri in combinations(list(G.nodes()),3):
        #And I create the triangle with probability p2
        if random.random() <= p2:
            #I close the triangle.
            triangles_list.append(tri)
            
            #Now I also need to add the new links to the graph created by the triangle
            G_copy.add_edge(tri[0], tri[1])
            G_copy.add_edge(tri[1], tri[2])
            G_copy.add_edge(tri[0], tri[2])
            
    G = G_copy
             
    #Creating a dictionary of neighbors
    node_neighbors_dict = {}
    for n in list(G.nodes()):
        node_neighbors_dict[n] = G[n].keys()           
                
    #print len(triangles_list), 'triangles created. Size now is', G.size()
        
    #avg_n_triangles = 3.*len(triangles_list)/G.order()
    
    #return node_neighbors_dict, node_triangles_dict, avg_n_triangles
    #return node_neighbors_dict, triangles_list, avg_n_triangles

    edges_list = [tuple(sorted(edge)) for edge in G.edges()]    
    return N_realized, edges_list, triangles_list

def get_p1_and_p2(d1,d2,N):
    p2 = (2.*d2)/((N-1.)*(N-2.))
    p1 = (d1 - 2.*d2)/((N-1.)- 2.*d2)
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

if __name__ == "__main__":
    # E-R SC
    # input: 
    # N = number of nodes
    # d1, d2 = average PW degree, average HO degree (number of triangles a node is part of)
    N = 1000
    d1, d2 = (20, 6)
    print(f"Using (d1, d2) = {(d1, d2)}")

    # first, calculate p1, p2 based on input parameters
    p1, p2 = get_p1_and_p2(d1,d2,N)
    print(f"target p1: {p1:.4f}")
    print(f"target p2: {p2:.8f}\n")

    max_pw_edges = comb(N, 2, exact=True) # N * (N - 1) / 2
    max_ho_edges = comb(N, 3, exact=True) # N * (N - 1) * (N - 2) / 6
    print(f"target pw edges: {p1 * max_pw_edges:.2f}/{max_pw_edges}")
    print(f"target ho edges: {p2 * max_ho_edges:.2f}/{max_ho_edges}\n")

    # generate connected SC using N, p1, p2
    attempts = 1000
    for _ in range(attempts):
        N_realized, edges, triangles = generate_my_simplicial_complex_d2(N,p1,p2)
        if N_realized == N:
            print(f"Found connected SC of size {N_realized}.")
            break
    
    # turn it into instance of HigherOrderStructure
    g_edges = []
    all_edges = edges + triangles
    for edge in all_edges:
        g_edges.append(tuple(edge))
    print(f"g_edges: {g_edges[:5]}, ..., {g_edges[-5:]}")
    g = HigherOrderStructure(N)
    g.name = "E-R"
    g.set_edges(g_edges)
    g.print()

    # check what are realized average degrees, and p1, p2
    d1_sim = np.mean([len(g.neighbors(i, 1)) for i in list(g.nodes.keys())])
    d2_sim = np.mean([len(g.neighbors(i, 2)) for i in list(g.nodes.keys())])
    print(f"realized d1:  {d1_sim:.2f}")
    print(f"realized d2:  {d2_sim:.2f}\n")

    p1_est = len(edges) / max_pw_edges
    p2_est = len(triangles) / max_ho_edges
    print(f"realized p1: {p1_est:.4f}")
    print(f"realized p2: {p2_est:.8f}\n")

    print(f"realized pw edges:  {len(edges)}/{max_pw_edges}")
    print(f"realized ho edges:  {len(triangles)}/{max_ho_edges}\n")


"""output venv➜  src git:(main) ✗ python3 higher_order_generators.py:
Using (d1, d2) = (20, 6)
target p1: 0.0081
target p2: 0.00001204

target pw edges: 4048.63/499500
target ho edges: 2000.00/166167000

Found connected SC of size 1000.
g_edges: [(0, 184), (0, 552), (0, 606), (0, 617), (0, 750)], ..., [(873, 874, 926), (898, 910, 942), (921, 950, 975), (944, 948, 988), (961, 973, 980)]
        E-R on 1000 nodes with 11878 edges.

realized d1:  19.79
realized d2:  5.95

realized p1: 0.0198
realized p2: 0.00001193

realized pw edges:  9895/499500
realized ho edges:  1983/166167000
"""