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
# Scale-Free SC
# -----------------------------------------------------------------------------------
def generate_sf_sc_components(N_nodes, m_min_kgi, gamma_kgi,
                                      max_retries_for_stub_set=10, # N // 100, for N = 1000
                                      max_initial_stub_gen_attempts=100):
    r"""
    NOTE: This is the version with full-triangle property.

    Generates a Simplicial Complex (up to 2-simplices) with scale-free
    generalized degrees for 2-simplices k_gi = target number of triangles a node is part of.

    Given:
        * m_min_kgi: minimum number of target triangles per node
        * gamma_kgi: exponent of P(k_gi) ~ k_gi^(-gamma_kgi)
        * max_retries_for_stub_set: max retries if current 3 stubs form an illegal triangle
        * max_initial_stub_gen_attempts: max attempts to generate valid total stub sum

    Adapted from:
        * O.T. Courtney and G. Bianconi
        * "Generalized network structures: the configuration model and the canonical ensemble of
        * simplicial complexes"
        * Phys. Rev. E 93, 062311 (2016)
        * http://dx.doi.org/10.1103/PhysRevE.93.062311
        * https://github.com/ginestrab/Ensembles-of-Simplicial-Complexes/blob/8c32d11a281f31813c8e0693cd010b07e2c823b3/SC_d2.c    
    """
    ## Step 1: Generate generalized degree sequence
    kgi_stubs_generated = np.zeros(N_nodes, dtype=int)
    
    for _ in range(max_initial_stub_gen_attempts):
        # draw from scale-free distribution
        for i in range(N_nodes):
            u = random.random()
            while u == 0.0: 
                u = random.random()
            
            # scale-free draw
            drawn_kgi = int(m_min_kgi * (u**(-1.0 / (gamma_kgi - 1.0))))
            
            # apply constraints
            max_possible_triangles = (N_nodes - 1) * (N_nodes - 2) // 2
            kgi_stubs_generated[i] = max(m_min_kgi, min(drawn_kgi, max_possible_triangles))
        
        # check if total stubs divisible by 3
        total_stubs = np.sum(kgi_stubs_generated)
        if total_stubs % 3 == 0 and total_stubs >= 3:
            break
    else:
        # fix stub sum if not divisible by 3
        total_stubs = np.sum(kgi_stubs_generated)
        remainder = total_stubs % 3
        if remainder != 0:
            # remove excess stubs
            nodes_to_adjust = np.where(kgi_stubs_generated > m_min_kgi)[0]
            if len(nodes_to_adjust) >= remainder:
                for i in range(remainder):
                    kgi_stubs_generated[nodes_to_adjust[i]] -= 1
            total_stubs = np.sum(kgi_stubs_generated)
        
        if total_stubs < 3 or total_stubs % 3 != 0:
            print(f"Failed, returning empty SC components. Final stubs:  {total_stubs}")
            return N_nodes, [], [], kgi_stubs_generated
    
    ## Step 2: Create stub lists
    # number of triangles (now factor nodes):
    M = total_stubs // 3
    
    # create node stub list
    node_stubs = []
    for node_id, degree in enumerate(kgi_stubs_generated):
        node_stubs.extend([node_id] * degree)
    
    ## Step 3: Configuration model matching
    triangles_set = set()
    max_global_attempts = 10 # TODO: adjust
    
    for _ in range(max_global_attempts):
        random.shuffle(node_stubs)
        triangles_set.clear()
        stub_index = 0
        success = True
        
        # try to form M triangles
        for triangle_id in range(M):
            if stub_index + 2 >= len(node_stubs):
                success = False
                break
            
            # try to form a triangle with next 3 stubs
            found_valid = False
            for local_retry in range(max_retries_for_stub_set):
                if stub_index + 2 >= len(node_stubs):
                    break
                
                # get three nodes
                nodes = [node_stubs[stub_index], 
                        node_stubs[stub_index + 1], 
                        node_stubs[stub_index + 2]]
                
                # check if valid triangle (all nodes should be distinct)
                if len(set(nodes)) == 3:
                    triangle = tuple(sorted(nodes))
                    if triangle not in triangles_set:
                        triangles_set.add(triangle)
                        stub_index += 3
                        found_valid = True
                        break
                
                # if invalid, try reshuffling remaining stubs
                if stub_index + 3 < len(node_stubs):
                    remaining = node_stubs[stub_index:]
                    random.shuffle(remaining)
                    node_stubs[stub_index:] = remaining
            
            if not found_valid:
                # skip these stubs if can't form valid triangle
                # stub_index += 3  # <- the problem !!
                # if stub_index >= len(node_stubs):
                #     success = False
                #     break
                success = False
                break                
        
        # if success or len(triangles_set) >= 0.9 * M:
        #     # accept if successful or got most triangles
        #     break
        if len(triangles_set) == M:  # Not >= 0.9 * M
            break        
    
    ## Step 4: Extract edges from (full) triangles
    edges_set = set()
    for i, j, k in triangles_set:
        edges_set.add((min(i, j), max(i, j)))
        edges_set.add((min(i, k), max(i, k)))
        edges_set.add((min(j, k), max(j, k)))
    
    return N_nodes, list(edges_set), list(triangles_set), kgi_stubs_generated

def generate_sf_sc_components_simple(N_nodes, m_min_kgi, gamma_kgi,
                              max_retries_for_stub_set=10, # N // 100, for N = 1000
                              max_initial_stub_gen_attempts=100):
    r"""
    Generates a Simplicial Complex (up to 2-simplices) with scale-free
    generalized degrees for 2-simplices k_gi = target number of triangles a node is part of.

    Given:
        * m_min_kgi: minimum number of target triangles per node
        * gamma_kgi: exponent of P(k_gi) ~ k_gi^(-gamma_kgi)
        * max_retries_for_stub_set: max retries if current 3 stubs form an illegal triangle
        * max_initial_stub_gen_attempts: max attempts to generate valid total stub sum

    Adapted from:
        * O.T. Courtney and G. Bianconi
        * "Generalized network structures: the configuration model and the canonical ensemble of
        * simplicial complexes"
        * Phys. Rev. E 93, 062311 (2016)
        * http://dx.doi.org/10.1103/PhysRevE.93.062311
        * https://github.com/ginestrab/Ensembles-of-Simplicial-Complexes/blob/8c32d11a281f31813c8e0693cd010b07e2c823b3/SC_d2.c    
    """
    kgi_stubs_generated = np.zeros(N_nodes, dtype=int)
    for _ in range(max_initial_stub_gen_attempts):
        for i in range(N_nodes):
            u = random.random()
            while u == 0.0: u = random.random()
            drawn_kgi = int(m_min_kgi * (u**(-1.0 / (gamma_kgi - 1.0))))
            kgi_stubs_generated[i] = max(m_min_kgi, drawn_kgi)
            max_poss_tri_node = (N_nodes - 1) * (N_nodes - 2) / 2.0
            kgi_stubs_generated[i] = min(kgi_stubs_generated[i], int(max_poss_tri_node))
            if kgi_stubs_generated[i] < 0: kgi_stubs_generated[i] = 0

        total_stubs = np.sum(kgi_stubs_generated)
        if total_stubs % 3 == 0 and total_stubs >=3 : break
    else:
        remainder = total_stubs % 3
        if remainder != 0:
            eligible_nodes = np.where(kgi_stubs_generated > m_min_kgi)[0]
            if len(eligible_nodes) < remainder: eligible_nodes = np.where(kgi_stubs_generated > 0)[0]
            for _i_dec in range(remainder):
                if not eligible_nodes.size: break
                node_to_dec = random.choice(eligible_nodes)
                if kgi_stubs_generated[node_to_dec] > 0: kgi_stubs_generated[node_to_dec] -= 1
            total_stubs = np.sum(kgi_stubs_generated)
        if total_stubs < 3 or total_stubs % 3 != 0:
            print(f"Failed, returning empty SC components. Final stubs: {total_stubs}")
            return 0, [], []

    stub_list = []
    for node_idx, num_s in enumerate(kgi_stubs_generated):
        stub_list.extend([node_idx] * num_s)
    random.shuffle(stub_list)

    triangles_set = set()
    edges_set = set()
    
    current_stub_idx = 0
    while current_stub_idx + 2 < len(stub_list):
        retries_for_current_set = 0
        while retries_for_current_set < max_retries_for_stub_set:
            if current_stub_idx + 2 >= len(stub_list): break # ran out of stubs

            node1 = stub_list[current_stub_idx]
            node2 = stub_list[current_stub_idx + 1]
            node3 = stub_list[current_stub_idx + 2]

            is_legal = False
            if len({node1, node2, node3}) == 3: # nodes should be distinct
                proposed_triangle = tuple(sorted((node1, node2, node3)))
                if proposed_triangle not in triangles_set:
                    is_legal = True
            
            if is_legal:
                triangles_set.add(proposed_triangle)
                edges_set.add(tuple(sorted((node1, node2))))
                edges_set.add(tuple(sorted((node1, node3))))
                edges_set.add(tuple(sorted((node2, node3))))
                current_stub_idx += 3 # used all 3 stubs
                break # break from retry for this set
            else:
                retries_for_current_set += 1
                if retries_for_current_set < max_retries_for_stub_set:
                    # NOTE: simple backtrack: 
                    # reshuffle remaining stubs and retry current position
                    if len(stub_list[current_stub_idx:]) >=3:
                         # reshuffle the tail from current_stub_idx to the end
                         # next attempt at current_stub_idx should pick different stubs
                        temp_tail = stub_list[current_stub_idx:]
                        random.shuffle(temp_tail)
                        stub_list[current_stub_idx:] = temp_tail
                    else: 
                        # not enough stubs to retry
                        current_stub_idx +=3 # give up on these 3
                        # break from retry
                        break 
                else: 
                    # max retries for this specific set of 3 stubs
                    # print(f"max retries for stubs at index {current_stub_idx}, moving on")
                    current_stub_idx += 3 # give up on these 3
                    break # break from retry
        else: 
            # retry loop finisheed due to max_retries and didn't break before
            if current_stub_idx + 2 >= len(stub_list): 
                break
        
    # returning kgi_stubs_generated for plots 
    return N_nodes, list(edges_set), list(triangles_set), kgi_stubs_generated

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
      The numer of nodes of the simplicial complex will be k * n <-- 
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
    # Kim & Vu (2003) Generating Random Regular Graphs 
    # NOTE: asympt uniform if d << n^(1/3), which holds since we want sparser graphs
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
    return N, np.array(edges_list), np.array(triangle_list)

def check_full_triangle_property(edges, triangles):
    r"""
    Checks if the "full triangle property" holds for a given set of
    edges and triangles:
    
    This means that if three nodes {i, j, k} form a 3-clique 
    (all pairwise edges {i, j}, {j, k}, {i, k} exist),
    then the 2-simplex {i, j, k} must also exist in the triangles list
    """
    # convert to sorted tuples for lookup
    edge_set = {tuple(sorted(edge)) for edge in edges if len(edge) == 2}
    triangle_set = {tuple(sorted(triangle)) for triangle in triangles if len(triangle) == 3}

    # determine the node set    
    all_nodes_in_edges = set(node for edge in edge_set for node in edge)
    all_nodes_in_triangles = set(node for triangle in triangle_set for node in triangle)
    all_nodes = all_nodes_in_edges.union(all_nodes_in_triangles)
    node_list = sorted(list(all_nodes)) # Iterate over sorted nodes

    empty_triangles_found = []
    # go over all possible 3-cliques based on node_list
    for i_idx, node_i in enumerate(node_list):
        # consider j > i and k > j to avoid duplicates
        for j_idx in range(i_idx + 1, len(node_list)):
            node_j = node_list[j_idx]
            # check if edge {i, j} exists
            edge_ij = tuple(sorted((node_i, node_j)))
            if edge_ij not in edge_set:
                continue # not even a path i, j
            for k_idx in range(j_idx + 1, len(node_list)):
                node_k = node_list[k_idx]
                # check if unique triple forms a 3-clique
                edge_ik = tuple(sorted((node_i, node_k)))
                edge_jk = tuple(sorted((node_j, node_k)))
                # all pairwise edges exist
                is_3_clique = (edge_ij in edge_set and
                               edge_ik in edge_set and
                               edge_jk in edge_set)
                if is_3_clique:
                    # now check if it is a "full triangle"
                    potential_empty_triangle = tuple(sorted((node_i, node_j, node_k)))
                    if potential_empty_triangle not in triangle_set:
                        empty_triangles_found.append(potential_empty_triangle)
                        print(f"Found an empty triangle: {potential_empty_triangle}")
                        return False
    return True   


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
