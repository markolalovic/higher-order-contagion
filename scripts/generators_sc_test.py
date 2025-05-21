"""
Testing simplicial complex generators:
  * `ER_like_simplicial_complex` and `inter_order_overlap` check
  * `regular_maximum_overlapped_simplicial_complex`
  * `scale_free_simplicial_complex_bianconi_courtney`
"""

import numpy as np
import networkx as nx
import random
from itertools import combinations

# -----------------------------------------------------------------------------------
# Regular Maximum overlapped SC
# -----------------------------------------------------------------------------------
def regular_maximum_overlapped_simplicial_complex(n, k, seed=None):
    """
    -----------------------------------------------------------------------------------
    Thanks to Luca Gallo - https://scholar.google.com/citations?hl=it&user=sKiSU9AAAAAJ
    -----------------------------------------------------------------------------------
    
    Returns a regular simplicial complex of order 2 with maximum value of intra-order hyperedge overlap.

    Each node has exactly
    - 1-degree = k
    - 2-degree = (k-1)(k-2)/2
    We first fix a value of initial number of nodes n, the final structure will be will be composed of N = k*n nodes.
    
    Parameters
    ----------
    k : int
      The degree of each node.
    n : int
      The number of nodes for the starting k-regular random graph. 
      The value of $N \times k$ must be even.
      The numer of nodes of the simplicial complex will be k*N.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        
    Output
    ----------
    N : int
      The number of nodes in the simplicial complex.
    edges_list : ndarray
      Array composed by a list of tuples [(0,1),(2,3),...,etc] composing the set of 1-simplices
    triangles_list : ndarray
      Array composed by a list of tuples [(0,1,2),(2,3,4),...,etc] representing the set of 2-simplices
    """
    

    M = nx.random_regular_graph(k,n,seed=seed)
    
    graph_to_simplex = {ii:[ii*k+r for r in range(k)] for ii in range(n)}
    
    G = nx.disjoint_union_all([nx.complete_graph(k) for ii in range(n)])
    
    triangle_list = []
    for clique in nx.enumerate_all_cliques(G):
        if len(clique)==3:
            triangle_list.append(clique)
        
    for edge in M.edges():
        m1 = edge[0]
        m2 = edge[1]
        n1 = random.choice(graph_to_simplex[m1])
        n2 = random.choice(graph_to_simplex[m2])
        
        G.add_edge(n1,n2)
        
        graph_to_simplex[m1].remove(n1)
        graph_to_simplex[m2].remove(n2)
        
    edges_list = np.array(G.edges())
    N = n*k
    return N,np.array(edges_list), np.array(triangle_list)

# -----------------------------------------------------------------------------------
# Scale-free SC (Bianconi, Courtney)
# TODO: missing `convert_adjacency_matrix_to_edges_list`
# -----------------------------------------------------------------------------------
def choose_bianconi_courtney(x, kgi):
    r"""
    Randomly select an unmatched stub based on generalized degree distribution.
    
    Args:
        x (float): Random number between 0 and total stubs
        kgi (np.ndarray): Generalized degree array
    
    Returns:
        int: Index of the selected node
    """
    for i, deg in enumerate(kgi):
        x -= deg
        if x < 0:
            return i
    return len(kgi) - 1  # Fallback to last node

def check_triangle_bianconi_courtney(i1, i2, i3, tri):
    r"""
    Check if a triangle already exists between three nodes.
    
    Args:
        i1, i2, i3 (int): Node indices
        tri (list): Triangles list
    
    Returns:
        bool: True if triangle exists, False otherwise
    """
    for existing_node in range(len(tri[i1])):
        if {tri[i1][existing_node][0], tri[i1][existing_node][1]} == {i2, i3}:
            return True
    return False

def create_triangle_bianconi_courtney(i1, i2, i3, tri, kg):
    r"""
    Create a triangle between three nodes.
    
    Args:
        i1, i2, i3 (int): Node indices
        tri (list): Triangles list to be updated
        kg (np.ndarray): Generalized degree array
    """
    tri[i1].append([i2, i3])
    tri[i2].append([i1, i3])
    tri[i3].append([i1, i2])

def scale_free_simplicial_complex_bianconi_courtney(N,m,gamma2,AVOID=1,NX = 15,FIGURE=0):
    r"""
    -----------------------------------------------------------------------------------
    Adapted from: Bianconi, Courtney https://arxiv.org/abs/1602.04110
    -----------------------------------------------------------------------------------

    Generate a random simplicial complex with scale-free generalized degree distribution.
    
    Returns:
        tuple: Adjacency matrix, triangles list, node degrees
    """
    # Set random seed
    random.seed(time.time())
    np.random.seed(int(time.time()))

    # Initialize arrays
    kgi = np.zeros(N, dtype=int)  # Generalized degrees
    kg = np.zeros(N, dtype=int)  # Current generalized degrees
    k = np.zeros(N, dtype=int)  # Node degrees
    a = np.zeros((N, N), dtype=int)  # Adjacency matrix
    tri = [[] for _ in range(N)]  # Triangles list
    triangles_list = []
    # Generate initial generalized degrees
    for i in range(N):
        # Scale-free degree distribution
        while True:
            kgi[i] = int(m * math.pow(random.random(), -1.0 / (gamma2 - 1.0)))
            
            # Cut off if exceeds maximum possible generalized degree
            if kgi[i] <= (N-1)*(N-2)*0.5:
                break

    # Calculate total stubs
    xaus = np.sum(kgi)
    naus = 0  # Backtrack counter

    # Matching process
    while xaus > 3 and naus < 1 + AVOID * NX:
        # Randomly select three nodes proportional to unmatched stubs
        x = xaus * random.random()
        i1 = choose_bianconi_courtney(x, kgi)
        kg[i1] += 1
        kgi[i1] -= 1
        xaus -= 1

        x = xaus * random.random()
        i2 = choose_bianconi_courtney(x, kgi)
        kg[i2] += 1
        kgi[i2] -= 1
        xaus -= 1

        x = xaus * random.random()
        i3 = choose_bianconi_courtney(x, kgi)
        kg[i3] += 1
        kgi[i3] -= 1
        xaus -= 1

        # Check proposed matching is legal
        if (i1 != i2 and i2 != i3 and i3 != i1 and not check_triangle_bianconi_courtney(i1, i2, i3, tri)):
            # Create triangle and links
            create_triangle_bianconi_courtney(i1, i2, i3, tri, kg)
            
            triangles_list.append(np.sort([i1,i2,i3]))
            a[i1, i2] = a[i2, i1] = 1
            a[i1, i3] = a[i3, i1] = 1
            a[i2, i3] = a[i3, i2] = 1
        else:
            # Backtrack if matching is illegal
            naus += 1
            if AVOID == 1:
                kg[i1] -= 1
                kgi[i1] += 1
                kg[i2] -= 1
                kgi[i2] += 1
                kg[i3] -= 1
                kgi[i3] += 1

    # Calculate final node degrees
    for i in range(N):
        for j in range(i+1, N):
            if a[i, j] > 0:
                k[i] += 1
                k[j] += 1

    # Optionally write edges to file
    if FIGURE == 1:
        with open('SCd2figure.edges', 'w') as fp:
            for i in range(N):
                for j in range(i+1, N):
                    if a[i, j] == 1:
                        fp.write(f"{i} {j}\n")

    edges_list = convert_adjacency_matrix_to_edges_list(a)
    return a, edges_list, np.array(triangles_list), k
