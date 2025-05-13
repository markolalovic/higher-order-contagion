""" Hypergraph generators from Federico:
  * Erdos-Renyi like random hypergraph
  * configuration model TODO: draw degrees scale-free k^(-\gamma) 
  * random regular hypergraph TODO: ensure uniqueness of in-group and groups of 3-nodes picked
"""

import numpy as np
import networkx as nx
import random
from scipy.stats import nbinom

# from Federico:
def ER_like_random_hypergraph(N,p1,p2):
    """Our model"""
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        giant = G.subgraph(giant).copy()
        print('not connected, but GC has order ', giant.order(), 'and size', giant.size())
        G = giant
        N=G.order()
        mapping = {nn:0 for nn in list(G.nodes())}
        for iidx,ii in enumerate(list(G.nodes())):
            mapping[ii] = iidx
        G = nx.relabel_nodes(G,mapping)

    num_triangles = int(np.random.normal((N*(N-1)*(N-2))/6*p2,np.sqrt((N*(N-1)*(N-2)/6)*p2*(1-p2)),size=None))# np.random.binomial(N*(N-1)*(N-2)/6, p2, size=None)
    # NOTE: This stops working with N >~ 2000, as N*(N-1)*(N-2)/6 is interpreted as int32
    #       use instead np.random.normal(N*(N-1)*(N-2)/6*p2,np.sqrt(N*(N-1)*(N-2)/6*p2(1-p2)),size=None)
    #print(num_triangles)
    Triangles = np.zeros((num_triangles,3))
    for i in range(num_triangles):
        
        # we select randomly 3 distinct nodes of the network (we order the nodes in increasing index)
        tri = sorted(random.sample(range(N), 3))
        
        # we verify that the triplet of nodes was not already generated
        while np.sum(np.all(Triangles==tri,axis=1))!=0:
            tri = sorted(random.sample(range(N), 3))
        Triangles[i,:3]=tri
    
    # 2-body interactions
    m = G.number_of_edges() # number of edges in the network
    edges = np.zeros((m,2))
    edges[:,:2] = np.array([e for e in G.edges])
    edges = np.array(edges, dtype=int)
    
    # 3-body interactions
    Triangles = np.sort(Triangles)
    triangles = np.ones((len(Triangles),3), dtype=int)
        
    triangles[:,:3]=Triangles
    
    return G, edges, triangles

def p1_p2_ER_like_uncorrelated_hypergraph(k1,k2,N):
    p2 = (2*k2)/((N-1)*(N-2))
    p1 = (k1)/((N-1))
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

# from Federico:
# configuration model (negative-binomial distributed) 
def configuration_model_edges(degrees):
    # Create stubs
    stubs = np.repeat(np.arange(len(degrees)), degrees)
    np.random.shuffle(stubs)
    
    # Pair stubs to create edges
    edges = []
    i = 0
    while i < len(stubs) - 1:
        u, v = stubs[i], stubs[i+1]
        
        # Avoid self-loops and multiple edges
        if u != v and [u, v] not in edges and [v, u] not in edges:
            edges.append([min(u, v), max(u, v)])
            i += 2
        else:
            # If invalid, continue searching
            i += 1
    
    return np.array(edges)

def configuration_model_triangles(degrees):
    # Create stubs
    stubs = np.repeat(np.arange(len(degrees)), degrees)
    np.random.shuffle(stubs)
    
    # Create triangles
    triangles = []
    i = 0
    while i < len(stubs) - 2:
        u, v, w = stubs[i], stubs[i+1], stubs[i+2]
        
        # Ensure unique vertices and no repeated triangles
        if len({u, v, w}) == 3 and u != v and u!= w and v!=w and\
           tuple(sorted([u,v,w])) not in [tuple(sorted(t)) for t in triangles]:
            
            triangles.append(np.sort([u, v, w]))
            i += 3
        else:
            # If invalid, continue searching
            i += 1
    
    return np.array(triangles)    

# from Federico:
def random_regular_hypergraph(k_1,k_2, N, seed=None):
    """
    N*k_2/3 must be an int!
    
    
    Returns a k-regular random hypergraph, here each node has exactly
    - 1-degree = k_1
    - 2-degree = k_2
    The graph will be composed of k*N nodes.
    
    Parameters
    ----------
    k : int
      The degree of each node.
    N : int
      The number of nodes for the starting k-regular random graph. 
      The value of $N \times k$ must be even.
      The numer of nodes of the hypergraph will be k*N.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        
    """
    if k_2*N % 3 != 0:
        raise ValueError('k_2 * N must be a multiple of 3!')

    else:
        
        if N*(N-1)*(N-2)/6 < k_2*N/3:
            raise ValueError('You cannot obtain a regular set of 2-hyperedges with this set of parameters!')

            #raise ValueError('Number of possible triangles is larger then the number of possible triples!')
        elif N*(N-1)*(N-2)/6 >= k_2*N/3:

            G = nx.random_regular_graph(k_1,N,seed=seed)

            triangles_list = []
            list_nodes = list(range(N))
            dict_k2 = {j:0 for j in range(N)}
            counter = 0
            while sum(list(dict_k2.values())) != k_2*N:    
                counter+=1
                #n1 = np.random.choice(dict_k2[j])
                a = np.array(list(dict_k2.values())) 
                b = np.array(list(dict_k2.keys()))
                n1,n2,n3 = sorted(np.random.choice(b[a!=k_2],3,replace=False))
               # if dict_k2[n1] != k_2 and dict_k2[n2] != k_2 and dict_k2[n3] != k_2:
                if (n1,n2,n3) not in triangles_list:
                    triangles_list.append((n1,n2,n3))
                    dict_k2[n1] += 1 
                    dict_k2[n2] +=1 
                    dict_k2[n3] += 1
                else:
                    pass
                if counter == k_2*N*10:
                    raise ValueError("Too many attempts, try again")
                    
    return G, np.array(triangles_list)

"""
### generate edges_list 

N=2000

var = 200#640#602
r = (k1**2)/(var - k1)
p = k1/var

for aaa in range(1000):
    degrees = nbinom.rvs(r, p, size=N)+1
    if sum(degrees)%2 == 0:
        break

print('Edges set generated')

edges_list = configuration_model_edges(degrees)


### generate triangles_list
var = 600#650#620
r = (k2**2)/(var - k1)
p = k2/var

for aaa in range(1000):
    degrees = nbinom.rvs(r, p, size=N)+1
    if sum(degrees)%3 == 0:
        break
triangles_list_variance_600 = configuration_model_triangles(degrees)
"""

