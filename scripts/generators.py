""" Hypergraph generators from Federico:
  * Erdos Renyi like random hypergraph  
"""

import numpy as np
import networkx as nx
import random

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