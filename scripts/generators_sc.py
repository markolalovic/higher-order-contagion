"""
Simplicial complex generators:
  * `ER_like_simplicial_complex` and `inter_order_overlap` check
  * `regular_maximum_overlapped_simplicial_complex`
  * `scale_free_simplicial_complex_bianconi_courtney`
"""

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

def get_p1_and_p2(k1,k2,N):
    p2 = (2.*k2)/((N-1.)*(N-2.))
    p1 = (k1 - 2.*k2)/((N-1.)- 2.*k2)
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

# from Federico:
def ER_like_simplicial_complex(N,p1,p2):
    """Our model"""
    #I first generate a standard ER graph with edges connected with probability p1

    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    giant_order = N
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        giant = G.subgraph(giant).copy()
        # print('not connected, but GC has order ', giant.order(), 'and size', giant.size())
        giant_order = giant.order()
        G = giant
        N=G.order()
        mapping = {nn:0 for nn in list(G.nodes())}
        for iidx,ii in enumerate(list(G.nodes())):
            mapping[ii] = iidx
        G = nx.relabel_nodes(G,mapping)

    num_triangles = int(np.random.normal(((N*(N-1)*(N-2))/6)*p2,np.sqrt((N*(N-1)*(N-2)/6)*p2*(1-p2)),size=None))# np.random.binomial(N*(N-1)*(N-2)/6, p2, size=None)
    # NOTE: This stops working with N >~ 2000, as N*(N-1)*(N-2)/6 is interpreted as int32
    #       use instead np.random.normal(N*(N-1)*(N-2)/6*p2,np.sqrt(N*(N-1)*(N-2)/6*p2(1-p2)),size=None)
    
    Triangles = np.zeros((num_triangles,3))
    for i in range(num_triangles):
        
        # we select randomly 3 distinct nodes of the network (we order the nodes in increasing index)
        tri = sorted(random.sample(range(N), 3))
        
        # we verify that the triplet of nodes was not already generated
        while np.sum(np.all(Triangles==tri,axis=1))!=0:
            tri = sorted(random.sample(range(N), 3))
        Triangles[i,:3]=tri
        
        #We add the new links to the graph created by the triangle (in a simplicial complex, all the lower
        # interactions are also present)
        G.add_edge(tri[0], tri[1])
        G.add_edge(tri[1], tri[2])
        G.add_edge(tri[0], tri[2])
    
    # 2-body interactions
    m = G.number_of_edges() # number of edges in the network
    edges = np.zeros((m,2))
    edges[:,:2] = np.array([e for e in G.edges])
    edges = np.array(edges, dtype=int)
    
    # 3-body interactions
    Triangles = np.sort(Triangles)
    triangles = np.ones((len(Triangles),3), dtype=int)
        
    triangles[:,:3]=Triangles
    
    return giant_order, edges, triangles

def p1_p2_ER_like_simplicial_complex(k1,k2,N):
    p2 = (2.*k2)/((N-1.)*(N-2.))
    p1 = (k1 - 2.*k2)/((N-1.)- 2.*k2) # <- this part here
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

def sort_edge(edge):
    return (min(edge[0], edge[1]), max(edge[0], edge[1]))

def inter_order_overlap(edges_list, triangles_list):
    """ Should give 1. """

    # Use a set to store needed links for fast lookup
    needed_links = set()
    
    # Iterate over triangles to add all unique undirected edges (sorted)
    for triple in triangles_list:
        i, j, k = triple
        needed_links.add(sort_edge((i, j)))
        needed_links.add(sort_edge((i, k)))
        needed_links.add(sort_edge((j, k)))
    
    # Initialize set to track existing needed links and count duplicates
    existing_needed_links = set()
    repes = 0

    # Iterate over the edges list and check for needed links
    for pair in edges_list:
        edge = sort_edge(pair)
        
        if edge in needed_links:
            if edge not in existing_needed_links:
                existing_needed_links.add(edge)
            else:
                repes += 1

    # Calculate inclusiveness (inter_order_overlap)
    inter_order_overlap = len(existing_needed_links) / len(needed_links)
    
    return inter_order_overlap


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
